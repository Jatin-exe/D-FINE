"""TTNN mirror of the HybridEncoder used by DFINE."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn
import ttnn


from .mha_ttnn import TTNNMHA
from .weight_store import ModuleKeyRegistry

# helpers and wrappers
from .hgnetv2_ttnn_manual import (
    TTActivation,
    TTNNConv2d,
    _activation_to_nhwc,
    _activation_from_nhwc,
    _concat_activations,
    fold_bn_to_conv,
)


def _load_tensor_from_store(weight_store, key: str, device, dtype, layout):
    tt = weight_store.load_tensor(key, device=device)
    if tt.get_layout() != layout:
        tt = ttnn.to_layout(tt, layout)
    if tt.dtype != dtype:
        tt = ttnn.typecast(tt, dtype)
    return tt


def _load_linear_params(weight_store, registry, linear_pt, device, dtype):
    if weight_store is None:
        raise RuntimeError("TTNN encoder requires a weight store (set TTNN_WEIGHT_DIR).")
    key = registry.name_of(linear_pt) if registry is not None else None
    if key is None:
        raise RuntimeError("Missing weight store key for linear module.")
    if weight_store.mode == "load":
        W = _load_tensor_from_store(weight_store, f"{key}.weight_t", device, dtype, ttnn.TILE_LAYOUT)
        b = _load_tensor_from_store(weight_store, f"{key}.bias", device, dtype, ttnn.TILE_LAYOUT)
        return W, b
    if weight_store.mode == "save":
        W = linear_pt.weight.detach().t().contiguous()
        b = linear_pt.bias.detach().reshape(1, 1, -1)
        W_tt = ttnn.from_torch(W, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        weight_store.save_tensor(f"{key}.weight_t", W_tt)
        weight_store.save_tensor(f"{key}.bias", b_tt)
        return W_tt, b_tt
    raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")


def _load_norm_params(weight_store, registry, norm_pt, device, dtype):
    if weight_store is None:
        raise RuntimeError("TTNN encoder requires a weight store (set TTNN_WEIGHT_DIR).")
    key = registry.name_of(norm_pt) if registry is not None else None
    if key is None:
        raise RuntimeError("Missing weight store key for norm module.")
    if weight_store.mode == "load":
        w = _load_tensor_from_store(weight_store, f"{key}.weight", device, dtype, ttnn.TILE_LAYOUT)
        b = _load_tensor_from_store(weight_store, f"{key}.bias", device, dtype, ttnn.TILE_LAYOUT)
        return w, b
    if weight_store.mode == "save":
        w_pt = norm_pt.weight.detach()
        b_pt = norm_pt.bias.detach()
        w_tt = ttnn.from_torch(w_pt, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b_pt, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        weight_store.save_tensor(f"{key}.weight", w_tt)
        weight_store.save_tensor(f"{key}.bias", b_tt)
        return w_tt, b_tt
    raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")


@dataclass
class _TTNNConfig:
    dtype: object
    layout: object          # input layout
    layout_out: object      # conv output layout


class _TTNNConvBN(nn.Module):
    """Conv2d with BN folded into weights/bias using TTNNConv2d"""

    def __init__(self, device, conv_pt: nn.Conv2d, bn_pt: nn.BatchNorm2d, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        weight_key = None
        if registry is not None:
            conv_name = registry.name_of(conv_pt)
            if conv_name:
                weight_key = f"{conv_name}.fused"
        weight_t = None
        bias_t = None
        if weight_store is None:
            raise RuntimeError("TTNN encoder requires a weight store (set TTNN_WEIGHT_DIR).")
        if weight_key is None:
            raise RuntimeError("Missing weight store key for ConvBN module.")
        if weight_store.mode == "save":
            weight_t, bias_t = fold_bn_to_conv(conv_pt, bn_pt)
        padding = tuple(int(p) for p in conv_pt.padding)
        self.conv = TTNNConv2d(
            device,
            weight_t,
            bias_t,
            stride=tuple(conv_pt.stride),
            padding=padding,
            dilation=tuple(conv_pt.dilation),
            groups=int(conv_pt.groups),
            dtype=cfg.dtype,
            layout=cfg.layout_out,
            weight_store=weight_store,
            weight_key=weight_key,
        )

    def forward(self, act: TTActivation) -> TTActivation:
        return self.conv(act)

    def enable_l1(self):
        if hasattr(self.conv, "use_l1_output"):
            self.conv.use_l1_output()


class _TTNNTransformerEncoderLayer(nn.Module):
    def __init__(self, layer_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.dtype = cfg.dtype
        self.trace_mode = False
        # Force TILE layout for Transformer internals (layernorm/linear/matmul)
        # perf diff btw TILE and Row major ? 
        self.layout = ttnn.TILE_LAYOUT
        self.normalize_before = getattr(layer_pt, "normalize_before", False)

        # MHA
        self.mha = TTNNMHA(
            layer_pt.self_attn,
            device,
            dtype=self.dtype,
            layout=self.layout,
            weight_store=weight_store,
            registry=registry,
        )

        # Feed-forward
        self.W1, self.b1 = _load_linear_params(weight_store, registry, layer_pt.linear1, device, self.dtype)
        self.W2, self.b2 = _load_linear_params(weight_store, registry, layer_pt.linear2, device, self.dtype)

        # Norms
        self.ln1_weight, self.ln1_bias = _load_norm_params(weight_store, registry, layer_pt.norm1, device, self.dtype)
        self.ln2_weight, self.ln2_bias = _load_norm_params(weight_store, registry, layer_pt.norm2, device, self.dtype)

        # Activation
        if isinstance(layer_pt.activation, nn.GELU):
            self._act = ttnn.gelu
        else:
            self._act = ttnn.relu

    def _ln(self, x_tt, weight, bias):
        # Use FP32 for layer norm computation for better precision (unless low-precision is forced)
        math_fidelity = self.ttnn.MathFidelity.HiFi4
        fp32_ok = True
        compute_cfg = self.ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_ok,
            packer_l1_acc=True,
        )
        return self.ttnn.layer_norm(x_tt, weight=weight, bias=bias, epsilon=1e-5, compute_kernel_config=compute_cfg)

    def _linear(self, x_tt, W, b):
        # PHASE 2 FIX: Add FP32 accumulation to linear layers for precision (unless low-precision forced)
        math_fidelity = self.ttnn.MathFidelity.HiFi4
        fp32_ok = True
        compute_cfg = self.ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_ok,
            packer_l1_acc=True,
        )
        return self.ttnn.linear(x_tt, W, bias=b, compute_kernel_config=compute_cfg)

    def forward_ttnn(self, x_tt, pos_embed_tt=None):
        """TTNN-only forward (expects TTNN tensors, returns TTNN tensor)."""
        ttnn = self.ttnn
        trace_mode = getattr(self, "trace_mode", False)
        def _ensure_tile(tensor):
            if tensor.get_layout() != ttnn.TILE_LAYOUT:
                if trace_mode:
                    raise RuntimeError("trace_mode expects TILE layout for encoder transformer")
                return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
            return tensor
        # Residual 1: Self-attention block
        residual_tt = x_tt
        if self.normalize_before:
            x_tt = self._ln(x_tt, self.ln1_weight, self.ln1_bias)

        if pos_embed_tt is None:
            qk_tt = _ensure_tile(x_tt)
        else:
            qk_tt = ttnn.add(_ensure_tile(x_tt), _ensure_tile(pos_embed_tt))
        attn_tt = self.mha(qk_tt, x_k=qk_tt, x_v=x_tt, return_ttnn=True)

        x_tt = ttnn.add(_ensure_tile(residual_tt), _ensure_tile(attn_tt))
        if not self.normalize_before:
            x_tt = self._ln(x_tt, self.ln1_weight, self.ln1_bias)

        # Residual 2: Feed-forward block
        residual_tt = x_tt
        if self.normalize_before:
            x_tt = self._ln(x_tt, self.ln2_weight, self.ln2_bias)
        x_tt = self._linear(x_tt, self.W1, self.b1)
        x_tt = self._act(x_tt)
        x_tt = self._linear(x_tt, self.W2, self.b2)
        x_tt = ttnn.add(_ensure_tile(residual_tt), _ensure_tile(x_tt))
        if not self.normalize_before:
            x_tt = self._ln(x_tt, self.ln2_weight, self.ln2_bias)

        return x_tt


# FPN/PAN section
def _act_op_from_module(act_mod):
    if act_mod is None:
        return None
    name = act_mod.__class__.__name__
    if name.lower().startswith("identity"):
        return None
    if name.lower().startswith("silu") or name.lower().startswith("swish"):
        return getattr(ttnn, "silu", getattr(ttnn, "hardswish", ttnn.gelu))
    if name.lower().startswith("gelu"):
        return ttnn.gelu
    if name.lower().startswith("relu6"):
        return getattr(ttnn, "relu6", ttnn.relu)
    if name.lower().startswith("relu"):
        return ttnn.relu
    if name.lower().startswith("leakyrelu"):
        return getattr(ttnn, "leaky_relu", ttnn.relu)
    if name.lower().startswith("hardsigmoid"):
        return getattr(ttnn, "hardsigmoid", ttnn.sigmoid)
    return None


class _TTNNConvBNAct(nn.Module):
    """TTNN-native conv block that accepts and returns TTNN tensors (no conversions)."""
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        conv_pt = getattr(module_pt, "conv")
        bn_pt = getattr(module_pt, "norm")
        self.conv = _TTNNConvBN(device, conv_pt, bn_pt, cfg, weight_store=weight_store, registry=registry)
        self.act_op = _act_op_from_module(getattr(module_pt, "act", None))
        self.device = device
        self.cfg = cfg
        # Prefer L1 outputs for these small convs.
        if hasattr(self.conv, "enable_l1"):
            self.conv.enable_l1()

    def _apply_act(self, y: TTActivation) -> TTActivation:
        if self.act_op is None:
            return y
        trace_mode = getattr(self, "trace_mode", False)
        if trace_mode:
            tens = y.tensor
        else:
            if y.tensor.get_layout() != ttnn.TILE_LAYOUT:
                tens = ttnn.to_layout(y.tensor, ttnn.TILE_LAYOUT)
            else:
                tens = y.tensor
        z = self.act_op(tens)
        return TTActivation(z, y.batch, y.height, y.width, y.channels)

    def forward(self, x_tt: TTActivation) -> TTActivation:
        """Forward pass - TTNN input, TTNN output (no conversions)."""
        y = self.conv(x_tt)
        y = self._apply_act(y)
        return y


class _TTNNVGGBlock(nn.Module):
    """TTNN-native VGG block - stays in TTNN throughout for better performance."""
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        self.conv1 = _TTNNConvBNAct(module_pt.conv1, device, cfg, weight_store=weight_store, registry=registry)
        self.conv2 = _TTNNConvBNAct(module_pt.conv2, device, cfg, weight_store=weight_store, registry=registry)
        self.act_op = _act_op_from_module(getattr(module_pt, "act", None))
        self.device = device
        self.cfg = cfg

    def forward(self, x_tt: TTActivation) -> TTActivation:
        """Forward pass - TTNN input, TTNN output (minimal conversions)."""
        y1_tt = self.conv1(x_tt)
        y2_tt = self.conv2(x_tt)
        # Add directly in TTNN - avoid conversion overhead
        y1_tensor = y1_tt.tensor
        y2_tensor = y2_tt.tensor
        # Ensure TILE for binary op, then restore original layout
        trace_mode = getattr(self, "trace_mode", False)
        y1_layout = y1_tensor.get_layout()
        if trace_mode:
            if y1_layout != y2_tensor.get_layout():
                raise RuntimeError("trace_mode expects matching layouts for encoder add")
            y_tensor = ttnn.add(y1_tensor, y2_tensor)
        else:
            if y1_layout == ttnn.TILE_LAYOUT:
                y1_tile = y1_tensor
            else:
                y1_tile = ttnn.to_layout(y1_tensor, ttnn.TILE_LAYOUT)
            if y2_tensor.get_layout() == ttnn.TILE_LAYOUT:
                y2_tile = y2_tensor
            else:
                y2_tile = ttnn.to_layout(y2_tensor, ttnn.TILE_LAYOUT)
            y_tensor = ttnn.add(y1_tile, y2_tile)
            if y1_layout != ttnn.TILE_LAYOUT:
                y_tensor = ttnn.to_layout(y_tensor, y1_layout)
        y_tt = TTActivation(y_tensor, y1_tt.batch, y1_tt.height, y1_tt.width, y1_tt.channels)
        if self.act_op is not None:
            if trace_mode:
                z = self.act_op(y_tt.tensor)
            else:
                tens = ttnn.to_layout(y_tt.tensor, ttnn.TILE_LAYOUT)
                z = self.act_op(tens)
            y_tt = TTActivation(z, y_tt.batch, y_tt.height, y_tt.width, y_tt.channels)
        return y_tt


def _slice_activation_channels(act: TTActivation, start: int, end: int) -> TTActivation:
    """Slice NHWC activation along channel dim."""
    if start < 0 or end > act.channels or end <= start:
        raise ValueError(f"Invalid channel slice [{start}, {end}) for C={act.channels}")
    nhwc = _activation_to_nhwc(act)
    sliced = ttnn.slice(nhwc, [0, 0, 0, start], [act.batch, act.height, act.width, end])
    return _activation_from_nhwc(sliced, act.batch, act.height, act.width, end - start)


class _TTNNCSPLayer(nn.Module):
    """TTNN-native CSP layer (no torch ops inside)."""
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        self.conv1 = _TTNNConvBNAct(module_pt.conv1, device, cfg, weight_store=weight_store, registry=registry)
        self.conv2 = _TTNNConvBNAct(module_pt.conv2, device, cfg, weight_store=weight_store, registry=registry)
        self.bottlenecks = nn.ModuleList([
            _TTNNVGGBlock(b, device, cfg, weight_store=weight_store, registry=registry) for b in module_pt.bottlenecks
        ])
        self.has_conv3 = not isinstance(module_pt.conv3, nn.Identity)
        self.conv3 = (
            _TTNNConvBNAct(module_pt.conv3, device, cfg, weight_store=weight_store, registry=registry) if self.has_conv3 else nn.Identity()
        )

    def forward(self, x_tt: TTActivation) -> TTActivation:
        x1 = self.conv1(x_tt)
        for b in self.bottlenecks:
            x1 = b(x1)
        x2 = self.conv2(x_tt)
        x1_tensor = x1.tensor
        x2_tensor = x2.tensor
        trace_mode = getattr(self, "trace_mode", False)
        x1_layout = x1_tensor.get_layout()
        if trace_mode:
            if x1_layout != x2_tensor.get_layout():
                raise RuntimeError("trace_mode expects matching layouts for encoder add")
            out_tensor = ttnn.add(x1_tensor, x2_tensor)
        else:
            if x1_layout == ttnn.TILE_LAYOUT:
                x1_tile = x1_tensor
            else:
                x1_tile = ttnn.to_layout(x1_tensor, ttnn.TILE_LAYOUT)
            if x2_tensor.get_layout() == ttnn.TILE_LAYOUT:
                x2_tile = x2_tensor
            else:
                x2_tile = ttnn.to_layout(x2_tensor, ttnn.TILE_LAYOUT)
            out_tensor = ttnn.add(x1_tile, x2_tile)
            if x1_layout != ttnn.TILE_LAYOUT:
                out_tensor = ttnn.to_layout(out_tensor, x1_layout)
        out = TTActivation(out_tensor, x1.batch, x1.height, x1.width, x1.channels)
        out = self.conv3(out) if self.has_conv3 else out
        return out


class _TTNNRepNCSPELAN4(nn.Module):
    """TTNN-native RepNCSPELAN4 block."""
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        self.cv1 = _TTNNConvBNAct(module_pt.cv1, device, cfg, weight_store=weight_store, registry=registry)
        self.cv2_csp = _TTNNCSPLayer(module_pt.cv2[0], device, cfg, weight_store=weight_store, registry=registry)
        self.cv2_post = _TTNNConvBNAct(module_pt.cv2[1], device, cfg, weight_store=weight_store, registry=registry)
        self.cv3_csp = _TTNNCSPLayer(module_pt.cv3[0], device, cfg, weight_store=weight_store, registry=registry)
        self.cv3_post = _TTNNConvBNAct(module_pt.cv3[1], device, cfg, weight_store=weight_store, registry=registry)
        self.cv4 = _TTNNConvBNAct(module_pt.cv4, device, cfg, weight_store=weight_store, registry=registry)
        self.c = module_pt.c

    def forward(self, x_tt: TTActivation) -> TTActivation:
        y = self.cv1(x_tt)
        a = _slice_activation_channels(y, 0, self.c)
        b = _slice_activation_channels(y, self.c, self.c * 2)
        y2 = self.cv2_post(self.cv2_csp(b))
        y3 = self.cv3_post(self.cv3_csp(y2))
        y_cat = _concat_activations([a, b, y2, y3])
        return self.cv4(y_cat)


class _TTNNSCDown(nn.Module):
    """TTNN-native SCDown block."""
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig, weight_store=None, registry=None):
        super().__init__()
        self.cv1 = _TTNNConvBNAct(module_pt.cv1, device, cfg, weight_store=weight_store, registry=registry)
        self.cv2 = _TTNNConvBNAct(module_pt.cv2, device, cfg, weight_store=weight_store, registry=registry)

    def forward(self, x_tt: TTActivation) -> TTActivation:
        return self.cv2(self.cv1(x_tt))


class HybridEncoderTTNN(nn.Module):
    """TTNN port of the HybridEncoder used in DFINE."""

    def __init__(
        self,
        encoder_pt: nn.Module,
        device=None,
        device_id: int = 0,
        return_stage: str = "final",
        weight_store=None,
        registry: ModuleKeyRegistry | None = None,
    ):
        super().__init__()
        self.ttnn = ttnn
        self.weight_store = weight_store
        if weight_store is None:
            raise RuntimeError("TTNN encoder requires a weight store (set TTNN_WEIGHT_DIR).")
        if registry is None:
            registry = ModuleKeyRegistry(encoder_pt)
        self.registry = registry
        self.trace_mode = False

        # Open a device if a shared one is not provided (prefer sharing backbone device)
        self._owns_device = device is None
        if device is None:
            try:
                device = ttnn.open_device(device_id=device_id, l1_small_size=655360)
            except TypeError:
                device = ttnn.open_device(device_id=device_id)
        self.device = device
        # Controls what forward() returns: "proj" for just input projections (unit tests),
        # "final" for full FPN+PAN outputs (deployment/benchmarks).
        assert return_stage in ("proj", "final"), "return_stage must be 'proj' or 'final'"
        self._return_stage = return_stage

        # Prefer ROW_MAJOR inputs but TILE outputs for better parity across encoder convs
        cfg_dtype = ttnn.bfloat16
        self.cfg = _TTNNConfig(dtype=cfg_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, layout_out=ttnn.TILE_LAYOUT)

        # Mirror key attrs from PyTorch encoder for consistent behavior later
        self.in_channels: List[int] = list(getattr(encoder_pt, "in_channels"))
        self.feat_strides: List[int] = list(getattr(encoder_pt, "feat_strides"))
        self.hidden_dim: int = int(getattr(encoder_pt, "hidden_dim"))
        self.use_encoder_idx: List[int] = list(getattr(encoder_pt, "use_encoder_idx"))
        self.num_encoder_layers: int = int(getattr(encoder_pt, "num_encoder_layers"))
        self.pe_temperature: float = float(getattr(encoder_pt, "pe_temperature"))
        self.eval_spatial_size = getattr(encoder_pt, "eval_spatial_size")
        self._pos_embed_ttnn_cache: Dict[Tuple[int, int], "ttnn.Tensor"] = {}

        # 2.1 Input projections: build TTNN convs from PyTorch conv+bn
        self.input_proj_tt = nn.ModuleList()
        for proj_pt in encoder_pt.input_proj:
            # Each proj is an OrderedDict inside nn.Sequential with named modules
            conv_pt = getattr(proj_pt, "conv")
            bn_pt = getattr(proj_pt, "norm")
            self.input_proj_tt.append(_TTNNConvBN(self.device, conv_pt, bn_pt, self.cfg, weight_store=weight_store, registry=self.registry))

        # Lateral convs (1x1) for FPN
        self.lateral_convs_tt = nn.ModuleList()
        for lat in encoder_pt.lateral_convs:
            conv_pt = getattr(lat, "conv")
            bn_pt = getattr(lat, "norm")
            self.lateral_convs_tt.append(_TTNNConvBN(self.device, conv_pt, bn_pt, self.cfg, weight_store=weight_store, registry=self.registry))

        # FPN fusion blocks (RepNCSPELAN4)
        self.fpn_blocks_tt_native = nn.ModuleList(
            [_TTNNRepNCSPELAN4(m, self.device, self.cfg, weight_store=weight_store, registry=self.registry) for m in encoder_pt.fpn_blocks]
        )

        # PAN blocks
        self.downsample_convs_tt_native = nn.ModuleList(
            [_TTNNSCDown(seq[0], self.device, self.cfg, weight_store=weight_store, registry=self.registry) for seq in encoder_pt.downsample_convs]
        )
        self.pan_blocks_tt_native = nn.ModuleList(
            [_TTNNRepNCSPELAN4(m, self.device, self.cfg, weight_store=weight_store, registry=self.registry) for m in encoder_pt.pan_blocks]
        )

        # 2.2 Encoder layers (optional)
        self._encoder_layers: Optional[nn.ModuleList] = None

        if self.num_encoder_layers > 0:
            # Build TTNN encoder stacks matching selected indices
            self._encoder_layers = nn.ModuleList()
            for i, enc_ind in enumerate(self.use_encoder_idx):
                enc_stack = nn.ModuleList()
                # encoder_pt.encoder[i] is a TransformerEncoder with layers list
                enc_block_pt = encoder_pt.encoder[i]
                for lyr_pt in enc_block_pt.layers:
                    enc_stack.append(_TTNNTransformerEncoderLayer(lyr_pt, self.device, self.cfg, weight_store=weight_store, registry=self.registry))
                self._encoder_layers.append(enc_stack)

        # Precompute positional embeddings when eval spatial size is fixed.
        if self.eval_spatial_size is not None:
            h, w = int(self.eval_spatial_size[0]), int(self.eval_spatial_size[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid eval_spatial_size: {(h, w)}")
            self._pos_embed_ttnn_cache[(w, h)] = self._build_2d_sincos_position_embedding_ttnn(
                w, h, self.hidden_dim, self.pe_temperature
            )


    def _upsample2x_ttnn_act(self, act: TTActivation) -> TTActivation:
        nhwc = _activation_to_nhwc(act)
        if nhwc.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            nhwc = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT)
        up = ttnn.upsample(input_tensor=nhwc, scale_factor=2, mode="nearest")
        return _activation_from_nhwc(up, act.batch, act.height * 2, act.width * 2, act.channels)

    def _slice_activation_spatial(self, act: TTActivation, target_h: int, target_w: int) -> TTActivation:
        if act.height == target_h and act.width == target_w:
            return act
        nhwc = _activation_to_nhwc(act)
        sliced = ttnn.slice(nhwc, [0, 0, 0, 0], [act.batch, target_h, target_w, act.channels])
        return _activation_from_nhwc(sliced, act.batch, target_h, target_w, act.channels)

    def _project_inputs_ttnn(self, feats: Sequence[TTActivation]) -> List[TTActivation]:
        projected: List[TTActivation] = []
        for idx, feat in enumerate(feats):
            if not isinstance(feat, TTActivation):
                raise TypeError("HybridEncoderTTNN expects TTActivation inputs from backbone")
            proj_act = self.input_proj_tt[idx](feat)
            projected.append(proj_act)
        return projected

    def _build_2d_sincos_position_embedding_ttnn(
        self, w: int, h: int, embed_dim: int, temperature: float
    ):
        """TTNN-only 2D sine-cosine positional embedding (returns TTNN tensor [1, H*W, C])."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        ttnn = self.ttnn
        pos_dim = embed_dim // 4

        grid_w = ttnn.arange(0, w, 1, device=self.device, dtype=ttnn.float32)
        grid_h = ttnn.arange(0, h, 1, device=self.device, dtype=ttnn.float32)

        grid_w = ttnn.reshape(grid_w, (w, 1))
        grid_w = ttnn.repeat(grid_w, (1, h))
        grid_h = ttnn.reshape(grid_h, (1, h))
        grid_h = ttnn.repeat(grid_h, (w, 1))

        grid_w = ttnn.reshape(grid_w, (w * h,))
        grid_h = ttnn.reshape(grid_h, (w * h,))

        omega = ttnn.arange(0, pos_dim, 1, device=self.device, dtype=ttnn.float32)
        omega = ttnn.div(omega, float(pos_dim))
        base = ttnn.full((pos_dim,), float(temperature), device=self.device, dtype=ttnn.float32)
        omega = ttnn.pow(base, omega)
        omega = ttnn.reciprocal(omega)

        grid_w = ttnn.reshape(grid_w, (w * h, 1))
        grid_h = ttnn.reshape(grid_h, (w * h, 1))
        omega = ttnn.reshape(omega, (1, pos_dim))

        grid_w = ttnn.to_layout(grid_w, ttnn.TILE_LAYOUT)
        grid_h = ttnn.to_layout(grid_h, ttnn.TILE_LAYOUT)
        omega = ttnn.to_layout(omega, ttnn.TILE_LAYOUT)

        out_w = ttnn.matmul(grid_w, omega)
        out_h = ttnn.matmul(grid_h, omega)

        pos = ttnn.concat(
            [ttnn.sin(out_w), ttnn.cos(out_w), ttnn.sin(out_h), ttnn.cos(out_h)], dim=1
        )
        pos = ttnn.reshape(pos, (1, w * h, embed_dim))
        return pos

    def _get_pos_embed_ttnn(self, w: int, h: int) -> "ttnn.Tensor":
        cache = self._pos_embed_ttnn_cache
        key = (int(w), int(h))
        if key not in cache:
            cache[key] = self._build_2d_sincos_position_embedding_ttnn(
                w, h, self.hidden_dim, self.pe_temperature
            )
        return cache[key]

    def _run_encoder_layers_ttnn(self, proj_acts: List[TTActivation]) -> List[TTActivation]:
        if not self._encoder_layers:
            return proj_acts
        encoded_acts = list(proj_acts)
        for stack_id, enc_ind in enumerate(self.use_encoder_idx):
            act = proj_acts[enc_ind]
            b, h, w, c = act.batch, act.height, act.width, act.channels
            x_tt = _activation_to_nhwc(act)
            if x_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                if self.trace_mode:
                    if x_tt.get_layout() == ttnn.TILE_LAYOUT:
                        x_tt = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
                    else:
                        raise RuntimeError("trace_mode expects ROW_MAJOR/TILE encoder inputs")
                x_tt = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
            x_tt = ttnn.reshape(x_tt, (b, h * w, c))
            if x_tt.get_layout() != ttnn.TILE_LAYOUT:
                if self.trace_mode:
                    x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
                x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)

            pos_embed_tt = self._get_pos_embed_ttnn(w, h)
            if b != 1:
                pos_embed_tt = ttnn.repeat(pos_embed_tt, (b, 1, 1))
            if pos_embed_tt.get_layout() != ttnn.TILE_LAYOUT:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects TILE pos embeddings")
                pos_embed_tt = ttnn.to_layout(pos_embed_tt, ttnn.TILE_LAYOUT)

            for layer in self._encoder_layers[stack_id]:
                x_tt = layer.forward_ttnn(x_tt, pos_embed_tt)

            x_tt = ttnn.reshape(x_tt, (b, h, w, c))
            encoded_acts[enc_ind] = _activation_from_nhwc(x_tt, b, h, w, c)
        return encoded_acts

    def _run_fpn(self, encoded_feats: List[TTActivation]) -> List[TTActivation]:
        return self._run_fpn_ttnn(encoded_feats)

    def _run_pan(self, fpn_feats: List[TTActivation]) -> List[TTActivation]:
        return self._run_pan_ttnn(fpn_feats)

    def _run_fpn_ttnn(self, encoded_feats: List[TTActivation]) -> List[TTActivation]:
        inner: List[TTActivation] = [encoded_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            hi = inner[0]
            lo = encoded_feats[idx - 1]
            lateral = self.lateral_convs_tt[len(self.in_channels) - 1 - idx]
            hi = lateral(hi)
            inner[0] = hi
            upsampled = self._upsample2x_ttnn_act(hi)
            if upsampled.height != lo.height or upsampled.width != lo.width:
                upsampled = self._slice_activation_spatial(upsampled, lo.height, lo.width)
            fused = _concat_activations([upsampled, lo])
            block = self.fpn_blocks_tt_native[len(self.in_channels) - 1 - idx]
            inner.insert(0, block(fused))
        return inner

    def _run_pan_ttnn(self, fpn_feats: List[TTActivation]) -> List[TTActivation]:
        outs = [fpn_feats[0]]
        for idx in range(len(self.in_channels) - 1):
            low = outs[-1]
            high = fpn_feats[idx + 1]
            down = self.downsample_convs_tt_native[idx](low)
            if down.height != high.height or down.width != high.width:
                down = self._slice_activation_spatial(down, high.height, high.width)
            fused = _concat_activations([down, high])
            outs.append(self.pan_blocks_tt_native[idx](fused))
        return outs

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        for module in self.modules():
            if module is self:
                continue
            setattr(module, "trace_mode", bool(enabled))
            if hasattr(module, "enable_trace_mode"):
                module.enable_trace_mode(enabled)

    def forward(self, feats: Sequence[TTActivation]) -> List[TTActivation]:
        proj_acts = self._project_inputs_ttnn(feats)
        if self._return_stage == "proj":
            return proj_acts
        encoded_acts = self._run_encoder_layers_ttnn(proj_acts) if self._encoder_layers else proj_acts
        fpn_acts = self._run_fpn_ttnn(encoded_acts)
        pan_acts = self._run_pan_ttnn(fpn_acts)
        return pan_acts

    # clean
    def close(self):
        if self._owns_device:
            self.ttnn.close_device(self.device)
