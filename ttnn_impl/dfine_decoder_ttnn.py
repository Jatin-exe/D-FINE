"""TTNN port of the D-FINE transformer decoder stack."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass
import os
import time

import torch
import torch.nn as nn
import ttnn
from .hgnetv2_ttnn_manual import TTActivation, _activation_to_nhwc
from .weight_store import ModuleKeyRegistry


from src.zoo.dfine.dfine_utils import weighting_function


def _softmax_lastdim_ttnn(ttnn_mod, x_tt, dim: int = -1, pad_tensor=None, trace_mode: bool = False):
    """Apply TTNN softmax on the last dim with padding to tile width."""
    if x_tt.get_layout() != ttnn_mod.TILE_LAYOUT:
        if trace_mode:
            raise RuntimeError("trace_mode expects TILE layout in _softmax_lastdim_ttnn")
        x_tt = ttnn_mod.to_layout(x_tt, ttnn_mod.TILE_LAYOUT)
    orig_shape = list(x_tt.shape)
    last = int(orig_shape[-1])
    pad = (32 - (last % 32)) % 32
    if pad:
        if pad_tensor is not None:
            x_tt = ttnn_mod.concat([x_tt, pad_tensor], dim=dim)
        else:
            padding = [(0, 0)] * (len(orig_shape) - 1) + [(0, pad)]
            x_tt = ttnn_mod.pad(x_tt, padding, value=-1.0e4)
    y = ttnn_mod.softmax(x_tt, dim=dim)
    if pad:
        start = [0] * len(orig_shape)
        y = ttnn_mod.slice(y, start, orig_shape)
    return y


@dataclass
class _DecoderTraceContext:
    softmax_pad: Optional["ttnn.Tensor"] = None
    lqe_pad: Optional["ttnn.Tensor"] = None
    topk_score_pad: Optional["ttnn.Tensor"] = None
    topk_mask_rng: Optional["ttnn.Tensor"] = None
    topk_tie_break: Optional["ttnn.Tensor"] = None


def _load_tensor_from_store(weight_store, key: str, device, dtype, layout):
    tt = weight_store.load_tensor(key, device=device)
    if tt.get_layout() != layout:
        tt = ttnn.to_layout(tt, layout)
    if dtype is not None and tt.dtype != dtype:
        tt = ttnn.typecast(tt, dtype)
    return tt


def _module_key(registry: ModuleKeyRegistry | None, module) -> Optional[str]:
    if registry is None:
        return None
    name = registry.name_of(module)
    if not name:
        return None
    return name


def _load_linear_params(weight_store, registry, linear_pt, device, dtype):
    if weight_store is None:
        raise RuntimeError("TTNN decoder requires a weight store (set TTNN_WEIGHT_DIR).")
    key = _module_key(registry, linear_pt)
    if key is None:
        raise RuntimeError("Missing weight store key for linear module.")
    if weight_store.mode == "load":
        W = _load_tensor_from_store(weight_store, f"{key}.weight_t", device, dtype, ttnn.TILE_LAYOUT)
        b = _load_tensor_from_store(weight_store, f"{key}.bias", device, dtype, ttnn.TILE_LAYOUT)
        return W, b
    if weight_store.mode == "save":
        W = linear_pt.weight.detach().t().contiguous()
        b = linear_pt.bias.detach().reshape(1, 1, -1)
        if dtype == ttnn.float32:
            W = W.to(torch.float32)
            b = b.to(torch.float32)
        W_tt = ttnn.from_torch(W, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        weight_store.save_tensor(f"{key}.weight_t", W_tt)
        weight_store.save_tensor(f"{key}.bias", b_tt)
        return W_tt, b_tt
    raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")


def _load_norm_params(weight_store, registry, norm_pt, device, dtype):
    if weight_store is None:
        raise RuntimeError("TTNN decoder requires a weight store (set TTNN_WEIGHT_DIR).")
    key = _module_key(registry, norm_pt)
    if key is None:
        raise RuntimeError("Missing weight store key for norm module.")
    if weight_store.mode == "load":
        w = _load_tensor_from_store(weight_store, f"{key}.weight", device, dtype, ttnn.TILE_LAYOUT)
        b = _load_tensor_from_store(weight_store, f"{key}.bias", device, dtype, ttnn.TILE_LAYOUT)
        return w, b
    if weight_store.mode == "save":
        w_pt = norm_pt.weight.detach()
        b_pt = norm_pt.bias.detach()
        if dtype == ttnn.float32:
            w_pt = w_pt.to(torch.float32)
            b_pt = b_pt.to(torch.float32)
        w_tt = ttnn.from_torch(w_pt, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b_pt, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        weight_store.save_tensor(f"{key}.weight", w_tt)
        weight_store.save_tensor(f"{key}.bias", b_tt)
        return w_tt, b_tt
    raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")


def _is_ttnn_tensor(x) -> bool:
    return isinstance(x, ttnn.Tensor)


def _is_tt_activation(x) -> bool:
    return isinstance(x, TTActivation)


def _ttnn_inverse_sigmoid(ttnn_mod, x_tt, eps: float = 1e-5, trace_mode: bool = False):
    def _ensure_tile(tensor):
        if tensor.get_layout() != ttnn_mod.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _ttnn_inverse_sigmoid")
            return ttnn_mod.to_layout(tensor, ttnn_mod.TILE_LAYOUT)
        return tensor
    if x_tt.dtype != ttnn_mod.float32:
        x_tt = ttnn_mod.typecast(x_tt, ttnn_mod.float32)
    x_tt = _ensure_tile(x_tt)
    x_tt = ttnn_mod.clip(x_tt, min=eps, max=1.0 - eps)
    one = ttnn_mod.full_like(x_tt, 1.0)
    denom = ttnn_mod.subtract(one, x_tt)
    ratio = ttnn_mod.div(x_tt, denom)
    return ttnn_mod.log(ratio)


def _ttnn_box_xyxy_to_cxcywh(ttnn_mod, x_tt, trace_mode: bool = False):
    def _ensure_tile(tensor):
        if tensor.get_layout() != ttnn_mod.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _ttnn_box_xyxy_to_cxcywh")
            return ttnn_mod.to_layout(tensor, ttnn_mod.TILE_LAYOUT)
        return tensor
    shape = list(x_tt.shape)
    if len(shape) < 3 or shape[-1] != 4:
        raise ValueError("Expected last dim=4 for box conversion")
    x_tt = _ensure_tile(x_tt)
    B, L = shape[0], shape[1]
    x0 = _ensure_tile(ttnn_mod.slice(x_tt, [0, 0, 0], [B, L, 1]))
    y0 = _ensure_tile(ttnn_mod.slice(x_tt, [0, 0, 1], [B, L, 2]))
    x1 = _ensure_tile(ttnn_mod.slice(x_tt, [0, 0, 2], [B, L, 3]))
    y1 = _ensure_tile(ttnn_mod.slice(x_tt, [0, 0, 3], [B, L, 4]))
    half = ttnn_mod.full_like(x0, 0.5)
    cx = ttnn_mod.multiply(ttnn_mod.add(x0, x1), half)
    cy = ttnn_mod.multiply(ttnn_mod.add(y0, y1), half)
    w = ttnn_mod.subtract(x1, x0)
    h = ttnn_mod.subtract(y1, y0)
    return ttnn_mod.concat([cx, cy, w, h], dim=-1)


def _ttnn_distance2bbox(ttnn_mod, points_tt, distance_tt, reg_scale: float, trace_mode: bool = False):
    def _ensure_tile(tensor):
        if tensor.get_layout() != ttnn_mod.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _ttnn_distance2bbox")
            return ttnn_mod.to_layout(tensor, ttnn_mod.TILE_LAYOUT)
        return tensor
    reg_scale = abs(float(reg_scale))
    if points_tt.dtype != ttnn_mod.float32:
        points_tt = ttnn_mod.typecast(points_tt, ttnn_mod.float32)
    if distance_tt.dtype != ttnn_mod.float32:
        distance_tt = ttnn_mod.typecast(distance_tt, ttnn_mod.float32)
    points_tt = _ensure_tile(points_tt)
    distance_tt = _ensure_tile(distance_tt)
    shape = list(points_tt.shape)
    if len(shape) < 3 or shape[-1] != 4:
        raise ValueError("Expected points with last dim=4")
    B, L = shape[0], shape[1]
    x = _ensure_tile(ttnn_mod.slice(points_tt, [0, 0, 0], [B, L, 1]))
    y = _ensure_tile(ttnn_mod.slice(points_tt, [0, 0, 1], [B, L, 2]))
    w = _ensure_tile(ttnn_mod.slice(points_tt, [0, 0, 2], [B, L, 3]))
    h = _ensure_tile(ttnn_mod.slice(points_tt, [0, 0, 3], [B, L, 4]))
    l = _ensure_tile(ttnn_mod.slice(distance_tt, [0, 0, 0], [B, L, 1]))
    t = _ensure_tile(ttnn_mod.slice(distance_tt, [0, 0, 1], [B, L, 2]))
    r = _ensure_tile(ttnn_mod.slice(distance_tt, [0, 0, 2], [B, L, 3]))
    b = _ensure_tile(ttnn_mod.slice(distance_tt, [0, 0, 3], [B, L, 4]))

    reg_inv = 1.0 / reg_scale
    w_scale = ttnn_mod.multiply(w, reg_inv)
    h_scale = ttnn_mod.multiply(h, reg_inv)

    if os.environ.get("TTNN_DISTANCE2BBOX_DIRECT", "0") != "0":
        cx = ttnn_mod.add(x, ttnn_mod.multiply(ttnn_mod.subtract(r, l), ttnn_mod.multiply(w_scale, 0.5)))
        cy = ttnn_mod.add(y, ttnn_mod.multiply(ttnn_mod.subtract(b, t), ttnn_mod.multiply(h_scale, 0.5)))
        bw = ttnn_mod.multiply(ttnn_mod.add(ttnn_mod.add(l, r), reg_scale), w_scale)
        bh = ttnn_mod.multiply(ttnn_mod.add(ttnn_mod.add(t, b), reg_scale), h_scale)
        return ttnn_mod.concat([cx, cy, bw, bh], dim=-1)

    half = ttnn_mod.full_like(x, 0.5 * reg_scale)
    x1 = ttnn_mod.subtract(x, ttnn_mod.multiply(ttnn_mod.add(half, l), w_scale))
    y1 = ttnn_mod.subtract(y, ttnn_mod.multiply(ttnn_mod.add(half, t), h_scale))
    x2 = ttnn_mod.add(x, ttnn_mod.multiply(ttnn_mod.add(half, r), w_scale))
    y2 = ttnn_mod.add(y, ttnn_mod.multiply(ttnn_mod.add(half, b), h_scale))

    bboxes = ttnn_mod.concat([x1, y1, x2, y2], dim=-1)
    return _ttnn_box_xyxy_to_cxcywh(ttnn_mod, bboxes, trace_mode=trace_mode)


class TTNNLinearWrap(nn.Module):
    def __init__(
        self,
        linear_pt: nn.Linear,
        device,
        dtype,
        ttnn_mod,
        use_fp32: bool = False,
        weight_store=None,
        registry: ModuleKeyRegistry | None = None,
    ):
        super().__init__()
        self.ttnn = ttnn_mod
        self.device = device
        self.dtype = dtype
        self.use_fp32 = use_fp32
        self.out_features = int(linear_pt.out_features)

        # Use FP32 for weights and biases if requested (better precision)
        dtype_internal = ttnn_mod.float32 if use_fp32 else dtype
        if weight_store is None:
            raise RuntimeError("TTNN decoder requires a weight store (set TTNN_WEIGHT_DIR).")
        key = _module_key(registry, linear_pt)
        if key is None:
            raise RuntimeError("Missing weight store key for linear module.")
        if weight_store.mode == "load":
            self.W = _load_tensor_from_store(weight_store, f"{key}.weight_t", device, dtype_internal, ttnn_mod.TILE_LAYOUT)
            self.b = _load_tensor_from_store(weight_store, f"{key}.bias", device, dtype_internal, ttnn_mod.TILE_LAYOUT)
        elif weight_store.mode == "save":
            W = linear_pt.weight.detach().t().contiguous()
            b = linear_pt.bias.detach().reshape(1, 1, -1)
            if use_fp32:
                W = W.to(torch.float32)
                b = b.to(torch.float32)
            self.W = ttnn_mod.from_torch(W, device=device, dtype=dtype_internal, layout=ttnn_mod.TILE_LAYOUT)
            self.b = ttnn_mod.from_torch(b, device=device, dtype=dtype_internal, layout=ttnn_mod.TILE_LAYOUT)
            weight_store.save_tensor(f"{key}.weight_t", self.W)
            weight_store.save_tensor(f"{key}.bias", self.b)
        else:
            raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")

    def forward(self, x, return_ttnn: bool = True, keep_fp32: bool = False) -> "ttnn.Tensor":
        if not _is_ttnn_tensor(x):
            raise TypeError("TTNNLinearWrap expects TTNN tensor input")
        if self.use_fp32:
            x_tt = self.ttnn.to_layout(x, self.ttnn.TILE_LAYOUT)
            x_tt = self.ttnn.typecast(x_tt, self.ttnn.float32)
            y_tt = self.ttnn.linear(x_tt, self.W, bias=self.b)
        else:
            x_tt = self.ttnn.to_layout(x, self.ttnn.TILE_LAYOUT)
            if x_tt.dtype != self.dtype:
                x_tt = self.ttnn.typecast(x_tt, self.dtype)
            y_tt = self.ttnn.linear(x_tt, self.W, bias=self.b)
        return y_tt


class TTNNMLPWrap(nn.Module):
    def __init__(
        self,
        mlp_pt: nn.Module,
        device,
        dtype,
        ttnn_mod,
        use_fp32: bool = False,
        weight_store=None,
        registry: ModuleKeyRegistry | None = None,
    ):
        super().__init__()
        self.ttnn = ttnn_mod
        self.device = device
        self.dtype = dtype
        self.use_fp32 = use_fp32  # Enable FP32 precision for critical MLPs (bbox regression)
        self.out_features = int(mlp_pt.layers[-1].out_features)
        layers = []
        activations: List[Optional[str]] = []
        # MLP has attributes: num_layers, layers (ModuleList[Linear]), act
        for i, layer in enumerate(mlp_pt.layers):
            layers.append(
                TTNNLinearWrap(
                    layer,
                    device,
                    dtype,
                    ttnn_mod,
                    use_fp32=use_fp32,
                    weight_store=weight_store,
                    registry=registry,
                )
            )
            act_kind = "gelu" if isinstance(mlp_pt.act, nn.GELU) else ("relu" if i < mlp_pt.num_layers - 1 else None)
            activations.append(act_kind)
        activations[-1] = None
        self.layers = nn.ModuleList(layers)
        self.activations = activations

    def _apply_act(self, x_tt, kind: Optional[str]):
        if not kind:
            return x_tt
        if kind == "gelu":
            return self.ttnn.gelu(x_tt)
        return self.ttnn.relu(x_tt)

    def forward(self, x, return_ttnn: bool = True, keep_fp32: bool = False) -> "ttnn.Tensor":
        if not _is_ttnn_tensor(x):
            raise TypeError("TTNNMLPWrap expects TTNN tensor input")
        x_tt = self.ttnn.to_layout(x, self.ttnn.TILE_LAYOUT)
        if self.use_fp32:
            # FP32 precision path for critical MLPs (bbox regression)
            x_tt = self.ttnn.typecast(x_tt, self.ttnn.float32)
            for i, lin in enumerate(self.layers):
                y_tt = self.ttnn.linear(x_tt, lin.W, bias=lin.b)
                y_tt = self._apply_act(y_tt, self.activations[i])
                x_tt = y_tt
            # Convert back to BF16 for output unless requested to keep FP32
            if not keep_fp32:
                x_tt = self.ttnn.typecast(x_tt, self.dtype)
        else:
            # Original BF16 path
            if x_tt.dtype != self.dtype:
                x_tt = self.ttnn.typecast(x_tt, self.dtype)
            for i, lin in enumerate(self.layers):
                y_tt = self.ttnn.linear(x_tt, lin.W, bias=lin.b)
                y_tt = self._apply_act(y_tt, self.activations[i])
                x_tt = y_tt
        return x_tt


class DFINETransformerTTNN(nn.Module):
    def __init__(self, decoder_pt: nn.Module, device=None, device_id: int = 0, weight_store=None, registry: ModuleKeyRegistry | None = None):
        super().__init__()
        self.ttnn = ttnn
        self._owns_device = device is None
        if device is None:
            try:
                device = ttnn.open_device(device_id=device_id, l1_small_size=655360)
            except TypeError:
                device = ttnn.open_device(device_id=device_id)
        self.device = device
        self.dtype = ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT

        # Keep reference to PyTorch decoder for structure
        self.decoder_pt = decoder_pt
        self.weight_store = weight_store
        if weight_store is not None and registry is None:
            registry = ModuleKeyRegistry(decoder_pt)
        self.registry = registry

        # enc_output: Sequential([Linear, LayerNorm])
        enc_proj = self.decoder_pt.enc_output[0]
        enc_norm = self.decoder_pt.enc_output[1]
        enc_dtype = ttnn.float32
        self.W_enc_fp32, self.b_enc_fp32 = _load_linear_params(
            self.weight_store, self.registry, enc_proj, device, enc_dtype
        )
        self.ln_w_fp32, self.ln_b_fp32 = _load_norm_params(
            self.weight_store, self.registry, enc_norm, device, enc_dtype
        )
        self.W_enc = ttnn.typecast(self.W_enc_fp32, self.dtype)
        self.b_enc = ttnn.typecast(self.b_enc_fp32, self.dtype)
        self.ln_w = ttnn.typecast(self.ln_w_fp32, self.dtype)
        self.ln_b = ttnn.typecast(self.ln_b_fp32, self.dtype)

        # enc_score_head: Linear(hidden_dim -> num_classes or 1)
        self.enc_score_out_features = int(self.decoder_pt.enc_score_head.out_features)
        score_dtype = ttnn.float32
        self.W_enc_score_fp32, self.b_enc_score_fp32 = _load_linear_params(
            self.weight_store, self.registry, self.decoder_pt.enc_score_head, device, score_dtype
        )
        self.W_enc_score = ttnn.typecast(self.W_enc_score_fp32, self.dtype)
        self.b_enc_score = ttnn.typecast(self.b_enc_score_fp32, self.dtype)

        # Build TTNN decoder layers mapped from PyTorch
        self.layers: List[TTNNTransformerDecoderLayer] = []
        for layer_pt in self.decoder_pt.decoder.layers:
            self.layers.append(
                TTNNTransformerDecoderLayer(
                    layer_pt,
                    device=self.device,
                    dtype=self.dtype,
                    layout=self.layout,
                    weight_store=self.weight_store,
                    registry=self.registry,
                )
            )

        # TTNN Integral for distribution-to-distance conversion
        self.integral = TTNNIntegral(int(self.decoder_pt.reg_max), device=self.device, dtype=self.dtype, layout=self.layout)

        # Build TTNN versions of heads (enc/dec score & bbox, pre_bbox, query_pos)

        # Encoder heads
        # Enable FP32 for encoder score head (improves PCC from 0.982 to 0.99+)
        self.enc_score_head_tt = TTNNLinearWrap(
            self.decoder_pt.enc_score_head,
            device,
            self.dtype,
            self.ttnn,
            use_fp32=True,
            weight_store=self.weight_store,
            registry=self.registry,
        )
        # Enable FP32 for bbox regression MLPs (critical for box coordinate precision)
        self.enc_bbox_head_tt = TTNNMLPWrap(
            self.decoder_pt.enc_bbox_head,
            device,
            self.dtype,
            self.ttnn,
            use_fp32=True,
            weight_store=self.weight_store,
            registry=self.registry,
        )
        # Decoder heads
        # Enable FP32 for bbox regression MLPs (critical for box coordinate precision)
        self.pre_bbox_head_tt = TTNNMLPWrap(
            self.decoder_pt.pre_bbox_head,
            device,
            self.dtype,
            self.ttnn,
            use_fp32=True,
            weight_store=self.weight_store,
            registry=self.registry,
        )
        # Enable FP32 for decoder score heads (improves classification accuracy)
        self.dec_score_head_tt = nn.ModuleList([
            TTNNLinearWrap(
                m,
                device,
                self.dtype,
                self.ttnn,
                use_fp32=True,
                weight_store=self.weight_store,
                registry=self.registry,
            )
            for m in self.decoder_pt.dec_score_head
        ])
        # Enable FP32 for all decoder bbox heads (critical for box coordinate precision)
        self.dec_bbox_head_tt = nn.ModuleList([
            TTNNMLPWrap(
                m,
                device,
                self.dtype,
                self.ttnn,
                use_fp32=True,
                weight_store=self.weight_store,
                registry=self.registry,
            )
            for m in self.decoder_pt.dec_bbox_head
        ])
        # Query pos head
        self.query_pos_head_tt = TTNNMLPWrap(
            self.decoder_pt.query_pos_head,
            device,
            self.dtype,
            self.ttnn,
            weight_store=self.weight_store,
            registry=self.registry,
        )

        # LQE (location quality estimator) for eval layer
        self.eval_idx = int(
            self.decoder_pt.decoder.eval_idx
            if hasattr(self.decoder_pt.decoder, "eval_idx")
            else len(self.layers) - 1
        )
        self.lqe_layer_tt: Optional[TTNNLQE] = None
        lqe_layers = getattr(self.decoder_pt.decoder, "lqe_layers", None)
        if lqe_layers is not None and len(lqe_layers) > self.eval_idx:
            lqe_pt = lqe_layers[self.eval_idx]
            if not isinstance(lqe_pt, nn.Identity):
                self.lqe_layer_tt = TTNNLQE(
                    lqe_pt,
                    device,
                    self.dtype,
                    self.ttnn,
                    weight_store=self.weight_store,
                    registry=self.registry,
                )

        # Precompute weighting function projection for integral (device constant)
        self.reg_scale = float(self.decoder_pt.reg_scale)
        up = float(self.decoder_pt.up)
        project_key = "dfine_decoder.project"
        if self.weight_store is None:
            raise RuntimeError("TTNN decoder requires a weight store (set TTNN_WEIGHT_DIR).")
        if self.weight_store.mode == "load":
            self.project_tt = _load_tensor_from_store(
                self.weight_store, project_key, device, ttnn.float32, ttnn.TILE_LAYOUT
            )
        elif self.weight_store.mode == "save":
            project = weighting_function(int(self.decoder_pt.reg_max), torch.tensor([up]), self.reg_scale, deploy=True)
            self.project_tt = ttnn.from_torch(
                project.reshape(-1, 1).float(), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
            self.weight_store.save_tensor(project_key, self.project_tt)
        else:
            raise RuntimeError(f"Unsupported weight store mode: {self.weight_store.mode}")

        # Cache anchors/valid mask for eval size to keep encoder->decoder on device.
        self._anchors_cache = None
        self._valid_mask_cache = None
        self._spatial_shapes_cache = None
        self.anchors_tt = None
        self.valid_mask_tt = None
        self.trace_ctx: Optional[_DecoderTraceContext] = None
        self.trace_mode = False
        self._last_timing_entries = None
        self._topk_timing_entries = None
        if getattr(self.decoder_pt, "eval_spatial_size", None) is not None:
            eval_h, eval_w = self.decoder_pt.eval_spatial_size
            self._spatial_shapes_cache = [
                [int(eval_h / s), int(eval_w / s)] for s in self.decoder_pt.feat_strides
            ]
            self.anchors_tt, self.valid_mask_tt = self._generate_anchors_ttnn(self._spatial_shapes_cache)

    def _num_queries(self) -> int:
        base_num_queries = int(self.decoder_pt.num_queries)
        if os.environ.get("TTNN_DECODER_ALLOW_EXTRA_QUERIES", "0") != "0":
            max_queries = int(os.environ.get("TTNN_DECODER_MAX_QUERIES", base_num_queries))
        else:
            max_queries = base_num_queries
        level_quotas = os.environ.get("TTNN_TOPK_LEVEL_QUOTAS")
        if level_quotas:
            try:
                quota_total = sum(max(0, int(part.strip())) for part in level_quotas.split(",") if part.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid TTNN_TOPK_LEVEL_QUOTAS={level_quotas!r}") from exc
            if quota_total > 0:
                return min(quota_total, max_queries)
        num_queries = int(os.environ.get("TTNN_DECODER_NUM_QUERIES", base_num_queries))
        return max(1, min(num_queries, max_queries))

    def prepare_trace(self, batch: int = 1) -> None:
        if self._spatial_shapes_cache is None:
            raise RuntimeError("Trace requires eval_spatial_size to be set.")
        B = int(batch)
        K = self._num_queries()
        ttnn = self.ttnn

        # Softmax padding (regression bins)
        R = int(self.decoder_pt.reg_max) + 1
        pad = (32 - (R % 32)) % 32
        softmax_pad = None
        if pad:
            N = B * K * 4
            softmax_pad = ttnn.full((N, pad), -1.0e4, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        # LQE padding on leading dimension
        lqe_pad = None
        N = B * K * 4
        pad_n = (32 - (N % 32)) % 32
        if pad_n:
            lqe_pad = ttnn.zeros((pad_n, R), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        topk_score_pad = None
        topk_mask_rng = None
        topk_tie_break = None
        L = sum(int(h) * int(w) for h, w in self._spatial_shapes_cache)
        use_multipass_topk = os.environ.get("TTNN_TOPK_MULTIPASS64", "0") != "0" and K > 64
        if os.environ.get("TTNN_TOPK_PAD_POWER2", "0") != "0" or use_multipass_topk:
            padded_l = 1 << (L - 1).bit_length()
            pad_width = padded_l - L
            if pad_width > 0:
                topk_score_pad = ttnn.full(
                    (B, pad_width),
                    -1.0e9,
                    device=self.device,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                )
            if use_multipass_topk:
                pass_k = min(64, K)
                topk_mask_rng = ttnn.arange(0, padded_l, 1, device=self.device, dtype=ttnn.uint32)
                topk_mask_rng = ttnn.reshape(topk_mask_rng, (1, 1, 1, padded_l))
                topk_mask_rng = ttnn.repeat(topk_mask_rng, (1, B, pass_k, 1))
                if topk_mask_rng.get_layout() != ttnn.TILE_LAYOUT:
                    topk_mask_rng = ttnn.to_layout(topk_mask_rng, ttnn.TILE_LAYOUT)
        else:
            padded_l = L

        tie_break_eps = float(os.environ.get("TTNN_TOPK_TIE_BREAK_EPS", "0") or "0")
        if tie_break_eps != 0.0:
            topk_tie_break = ttnn.arange(0, padded_l, 1, device=self.device, dtype=ttnn.float32)
            topk_tie_break = ttnn.reshape(topk_tie_break, (1, padded_l))
            if B != 1:
                topk_tie_break = ttnn.repeat(topk_tie_break, (B, 1))
            topk_tie_break = ttnn.multiply(topk_tie_break, tie_break_eps / max(1, padded_l - 1))
            if topk_tie_break.dtype != self.dtype:
                topk_tie_break = ttnn.typecast(topk_tie_break, self.dtype)
            if topk_tie_break.get_layout() != ttnn.TILE_LAYOUT:
                topk_tie_break = ttnn.to_layout(topk_tie_break, ttnn.TILE_LAYOUT)

        self.trace_ctx = _DecoderTraceContext(
            softmax_pad=softmax_pad,
            lqe_pad=lqe_pad,
            topk_score_pad=topk_score_pad,
            topk_mask_rng=topk_mask_rng,
            topk_tie_break=topk_tie_break,
        )
        self.integral.trace_ctx = self.trace_ctx
        if self.lqe_layer_tt is not None:
            self.lqe_layer_tt.trace_ctx = self.trace_ctx

    def enable_trace_mode(self, enabled: bool = True, batch: int = 1, strict: bool = True) -> None:
        """Enable trace preparation. When strict=False, only precompute trace buffers without changing layout rules."""
        self.trace_mode = bool(enabled) if strict else False
        for layer in self.layers:
            layer.trace_mode = bool(enabled) if strict else False
            if hasattr(layer, "cross_attn"):
                setattr(layer.cross_attn, "trace_mode", bool(enabled) if strict else False)
        self.integral.trace_mode = bool(enabled) if strict else False
        if self.lqe_layer_tt is not None:
            self.lqe_layer_tt.trace_mode = bool(enabled) if strict else False
        if enabled:
            self.prepare_trace(batch=batch)

    #Utils
    def _get_encoder_input_ttnn(self, feats: Sequence[TTActivation]):
        """Flatten multi-scale TTActivations to [B, L, C] and return spatial shapes."""
        ttnn = self.ttnn
        feat_flatten: List["ttnn.Tensor"] = []
        spatial_shapes: List[List[int]] = []
        for act in feats:
            spatial_shapes.append([act.height, act.width])
            nhwc = _activation_to_nhwc(act)
            if self.trace_mode:
                if nhwc.get_layout() not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
                    raise RuntimeError("Decoder trace_mode expects TILE or ROW_MAJOR encoder features.")
            else:
                nhwc = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT)
            blc = ttnn.reshape(nhwc, (act.batch, act.height * act.width, act.channels))
            feat_flatten.append(blc)
        if not feat_flatten:
            raise RuntimeError("No encoder features provided")
        if len(feat_flatten) == 1:
            memory = feat_flatten[0]
        else:
            memory = ttnn.concat(feat_flatten, dim=1)
        return memory, spatial_shapes

    def _generate_anchors_ttnn(self, spatial_shapes: List[List[int]], grid_size: float = 0.05):
        """TTNN-only anchor generation. Returns anchors [1, L, 4] and valid_mask [1, L, 1]."""
        ttnn = self.ttnn
        anchors_list: List["ttnn.Tensor"] = []
        eps = float(self.decoder_pt.eps)
        for lvl, (h, w) in enumerate(spatial_shapes):
            h = int(h)
            w = int(w)
            grid_x = ttnn.arange(0, w, 1, device=self.device, dtype=ttnn.float32)
            grid_y = ttnn.arange(0, h, 1, device=self.device, dtype=ttnn.float32)

            grid_x = ttnn.reshape(grid_x, (1, w))
            grid_x = ttnn.repeat(grid_x, (h, 1))
            grid_y = ttnn.reshape(grid_y, (h, 1))
            grid_y = ttnn.repeat(grid_y, (1, w))

            grid_x = ttnn.reshape(grid_x, (h, w, 1))
            grid_y = ttnn.reshape(grid_y, (h, w, 1))
            grid_xy = ttnn.concat([grid_x, grid_y], dim=2)
            if grid_xy.get_layout() != ttnn.TILE_LAYOUT:
                grid_xy = ttnn.to_layout(grid_xy, ttnn.TILE_LAYOUT)
            grid_xy = ttnn.add(grid_xy, 0.5)

            denom_w = ttnn.full((h, w, 1), float(w), device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            denom_h = ttnn.full((h, w, 1), float(h), device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            denom = ttnn.concat([denom_w, denom_h], dim=2)
            grid_xy = ttnn.div(grid_xy, denom)

            wh_val = grid_size * (2.0 ** lvl)
            wh = ttnn.full((h, w, 2), float(wh_val), device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

            lvl_anchors = ttnn.concat([grid_xy, wh], dim=2)
            lvl_anchors = ttnn.reshape(lvl_anchors, (1, h * w, 4))
            anchors_list.append(lvl_anchors)

        anchors = ttnn.concat(anchors_list, dim=1) if len(anchors_list) > 1 else anchors_list[0]
        anchors_tile = ttnn.to_layout(anchors, ttnn.TILE_LAYOUT)

        gt = ttnn.gt(anchors_tile, eps)
        lt = ttnn.lt(anchors_tile, 1.0 - eps)
        mask = ttnn.logical_and(gt, lt)
        mask = ttnn.typecast(mask, ttnn.float32)
        valid = ttnn.min(mask, dim=-1)
        valid = ttnn.reshape(valid, (anchors.shape[0], anchors.shape[1], 1))

        one = ttnn.full_like(anchors_tile, 1.0)
        denom = ttnn.subtract(one, anchors_tile)
        ratio = ttnn.div(anchors_tile, denom)
        anchors_logit = ttnn.log(ratio)

        inf_tt = ttnn.full_like(anchors_logit, float("inf"))
        anchors_logit = ttnn.where(valid, anchors_logit, inf_tt)
        if valid.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects ROW_MAJOR valid mask for anchors")
            valid = ttnn.to_layout(valid, ttnn.ROW_MAJOR_LAYOUT)
        if anchors_logit.get_layout() != ttnn.TILE_LAYOUT:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects TILE anchors_logit")
            anchors_logit = ttnn.to_layout(anchors_logit, ttnn.TILE_LAYOUT)
        return anchors_logit, valid

    def _enc_output_ttnn(self, memory_blc: "ttnn.Tensor", return_ttnn: bool = True):
        """Apply TTNN Linear + LayerNorm to [B, L, C] memory (TTNN only)."""
        ttnn = self.ttnn
        if not _is_ttnn_tensor(memory_blc):
            raise TypeError("TTNN decoder expects TTNN memory tensor")
        use_fp32 = True
        if use_fp32:
            if memory_blc.get_layout() != self.layout:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects encoder memory in decoder layout (fp32)")
                x_tt = ttnn.to_layout(memory_blc, self.layout)
            else:
                x_tt = memory_blc
            x_tt = ttnn.typecast(x_tt, ttnn.float32)
            compute_cfg = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            y = ttnn.linear(x_tt, self.W_enc_fp32, bias=self.b_enc_fp32, compute_kernel_config=compute_cfg)
            y = ttnn.layer_norm(
                y, weight=self.ln_w_fp32, bias=self.ln_b_fp32, epsilon=1e-5, compute_kernel_config=compute_cfg
            )
        else:
            if memory_blc.get_layout() != self.layout:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects encoder memory in decoder layout")
                x_tt = ttnn.to_layout(memory_blc, self.layout)
            else:
                x_tt = memory_blc
            if x_tt.dtype != self.dtype:
                x_tt = ttnn.typecast(x_tt, self.dtype)
            y = ttnn.linear(x_tt, self.W_enc, bias=self.b_enc)
            # Use larger epsilon for BF16 stability (1e-4 instead of 1e-5)
            y = ttnn.layer_norm(y, weight=self.ln_w, bias=self.ln_b, epsilon=1e-4)
        return y

    def _enc_score_ttnn(self, memory_blc: "ttnn.Tensor", return_ttnn: bool = True):
        ttnn = self.ttnn
        if not _is_ttnn_tensor(memory_blc):
            raise TypeError("TTNN decoder expects TTNN memory tensor")
        use_fp32 = True
        if use_fp32:
            if memory_blc.get_layout() != self.layout:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects encoder memory in decoder layout (score fp32)")
                x_tt = ttnn.to_layout(memory_blc, self.layout)
            else:
                x_tt = memory_blc
            x_tt = ttnn.typecast(x_tt, ttnn.float32)
            compute_cfg = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            y = ttnn.linear(x_tt, self.W_enc_score_fp32, bias=self.b_enc_score_fp32, compute_kernel_config=compute_cfg)
        else:
            if memory_blc.get_layout() != self.layout:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects encoder memory in decoder layout (score)")
                x_tt = ttnn.to_layout(memory_blc, self.layout)
            else:
                x_tt = memory_blc
            if x_tt.dtype != self.dtype:
                x_tt = ttnn.typecast(x_tt, self.dtype)
            y = ttnn.linear(x_tt, self.W_enc_score, bias=self.b_enc_score)
        return y

    def _one_hot_from_indices_ttnn(
        self,
        idx_tt,
        length: int,
        batch: int,
        k: int,
        rng_tt=None,
        dtype=None,
        trace_mode: bool = False,
    ):
        """Create one-hot tensor [B, K, L] on device using TTNN ops."""
        ttnn = self.ttnn
        out_dtype = self.dtype if dtype is None else dtype
        if idx_tt.get_layout() != ttnn.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _one_hot_from_indices_ttnn(idx)")
            idx_tt = ttnn.to_layout(idx_tt, ttnn.TILE_LAYOUT)
        if idx_tt.dtype != ttnn.uint32:
            idx_tt = ttnn.typecast(idx_tt, ttnn.uint32)
        idx_4d = ttnn.reshape(idx_tt, (1, batch, k, 1))
        idx_4d = ttnn.repeat(idx_4d, (1, 1, 1, length))
        if idx_4d.get_layout() != ttnn.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _one_hot_from_indices_ttnn(idx_4d)")
            idx_4d = ttnn.to_layout(idx_4d, ttnn.TILE_LAYOUT)

        if rng_tt is None:
            rng_tt = ttnn.arange(0, length, 1, device=self.device, dtype=ttnn.uint32)
            rng_tt = ttnn.reshape(rng_tt, (1, 1, 1, length))
            rng_tt = ttnn.repeat(rng_tt, (1, batch, k, 1))
        if rng_tt.get_layout() != ttnn.TILE_LAYOUT:
            if trace_mode:
                raise RuntimeError("trace_mode expects TILE layout in _one_hot_from_indices_ttnn(rng)")
            rng_tt = ttnn.to_layout(rng_tt, ttnn.TILE_LAYOUT)

        one_hot = ttnn.eq(idx_4d, rng_tt)
        one_hot = ttnn.typecast(one_hot, out_dtype)
        return ttnn.reshape(one_hot, (batch, k, length))

    def _anchors_from_indices_ttnn(
        self,
        idx_tt,
        spatial_shapes: List[List[int]],
        batch: int,
        k: int,
    ) -> "ttnn.Tensor":
        """Recreate encoder anchors from top-k indices without a sequence gather."""
        ttnn = self.ttnn
        if idx_tt.get_layout() != ttnn.TILE_LAYOUT:
            idx_tt = ttnn.to_layout(idx_tt, ttnn.TILE_LAYOUT)
        idx_f = idx_tt if idx_tt.dtype == ttnn.float32 else ttnn.typecast(idx_tt, ttnn.float32)

        zero = ttnn.multiply(idx_f, 0.0)
        if os.environ.get("TTNN_TOPK_ANCHOR_FAST", "1") != "0":
            start_acc = zero
            w_acc = zero
            h_acc = zero
            wh_acc = zero
            start = 0
            grid_size = 0.05
            for lvl, (h, w) in enumerate(spatial_shapes):
                h = int(h)
                w = int(w)
                end = start + h * w
                in_level = ttnn.logical_and(ttnn.ge(idx_f, float(start)), ttnn.lt(idx_f, float(end)))
                start_acc = ttnn.where(in_level, ttnn.add(zero, float(start)), start_acc)
                w_acc = ttnn.where(in_level, ttnn.add(zero, float(w)), w_acc)
                h_acc = ttnn.where(in_level, ttnn.add(zero, float(h)), h_acc)
                wh_acc = ttnn.where(in_level, ttnn.add(zero, float(grid_size * (2.0 ** lvl))), wh_acc)
                start = end
            local = ttnn.subtract(idx_f, start_acc)
            row = ttnn.floor(ttnn.div(local, w_acc))
            col = ttnn.subtract(local, ttnn.multiply(row, w_acc))
            x_acc = ttnn.div(ttnn.add(col, 0.5), w_acc)
            y_acc = ttnn.div(ttnn.add(row, 0.5), h_acc)
            w_acc = wh_acc
            h_acc = wh_acc
        else:
            x_acc = zero
            y_acc = zero
            w_acc = zero
            h_acc = zero
            start = 0
            grid_size = 0.05

            for lvl, (h, w) in enumerate(spatial_shapes):
                h = int(h)
                w = int(w)
                end = start + h * w
                in_level = ttnn.logical_and(ttnn.ge(idx_f, float(start)), ttnn.lt(idx_f, float(end)))
                local = ttnn.subtract(idx_f, float(start))
                row = ttnn.floor(ttnn.div(local, float(w)))
                col = ttnn.subtract(local, ttnn.multiply(row, float(w)))
                x_norm = ttnn.div(ttnn.add(col, 0.5), float(w))
                y_norm = ttnn.div(ttnn.add(row, 0.5), float(h))
                wh_norm = ttnn.add(zero, float(grid_size * (2.0 ** lvl)))
                x_acc = ttnn.where(in_level, x_norm, x_acc)
                y_acc = ttnn.where(in_level, y_norm, y_acc)
                w_acc = ttnn.where(in_level, wh_norm, w_acc)
                h_acc = ttnn.where(in_level, wh_norm, h_acc)
                start = end

        anchors = ttnn.concat(
            [
                ttnn.reshape(x_acc, (batch, k, 1)),
                ttnn.reshape(y_acc, (batch, k, 1)),
                ttnn.reshape(w_acc, (batch, k, 1)),
                ttnn.reshape(h_acc, (batch, k, 1)),
            ],
            dim=2,
        )
        one = ttnn.full_like(anchors, 1.0)
        denom = ttnn.subtract(one, anchors)
        ratio = ttnn.div(anchors, denom)
        return ttnn.log(ratio)

    def _anchors_from_level_indices_ttnn(
        self,
        idx_tt,
        h: int,
        w: int,
        lvl: int,
        batch: int,
        k: int,
    ) -> "ttnn.Tensor":
        """Recreate one feature level's anchors from local top-k indices."""
        ttnn = self.ttnn
        if idx_tt.get_layout() != ttnn.TILE_LAYOUT:
            idx_tt = ttnn.to_layout(idx_tt, ttnn.TILE_LAYOUT)
        idx_f = idx_tt if idx_tt.dtype == ttnn.float32 else ttnn.typecast(idx_tt, ttnn.float32)
        zero = ttnn.multiply(idx_f, 0.0)
        row = ttnn.floor(ttnn.div(idx_f, float(w)))
        col = ttnn.subtract(idx_f, ttnn.multiply(row, float(w)))
        x_acc = ttnn.div(ttnn.add(col, 0.5), float(w))
        y_acc = ttnn.div(ttnn.add(row, 0.5), float(h))
        wh_acc = ttnn.add(zero, float(0.05 * (2.0 ** lvl)))
        anchors = ttnn.concat(
            [
                ttnn.reshape(x_acc, (batch, k, 1)),
                ttnn.reshape(y_acc, (batch, k, 1)),
                ttnn.reshape(wh_acc, (batch, k, 1)),
                ttnn.reshape(wh_acc, (batch, k, 1)),
            ],
            dim=2,
        )
        one = ttnn.full_like(anchors, 1.0)
        denom = ttnn.subtract(one, anchors)
        ratio = ttnn.div(anchors, denom)
        return ttnn.log(ratio)

    def _topk_gather_ttnn(
        self,
        enc_logits: "ttnn.Tensor",  # [B, L, C] or [B, L, 1]
        anchors: "ttnn.Tensor",     # [B, L, 4]
        memory: "ttnn.Tensor",      # [B, L, C_hidden]
        k: int,
        spatial_shapes: Optional[List[List[int]]] = None,
        return_ttnn: bool = False,
    ) -> Tuple["ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor"]:
        """Perform on-device top-k over sequence dim and gather anchors/memory accordingly.

        Returns: (topk_scores[B,K], topk_indices[B,K], topk_anchors[B,K,4], topk_memory[B,K,C_hidden])
        """
        ttnn = self.ttnn
        def _ensure_layout(tensor, layout):
            return tensor if tensor.get_layout() == layout else ttnn.to_layout(tensor, layout)
        def _time_topk(name, fn):
            timing_entries = self._topk_timing_entries
            if timing_entries is None or os.environ.get("TTNN_TOPK_TIMING", "0") != "1":
                return fn()
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            t1 = time.perf_counter()
            timing_entries.append((name, (t1 - t0) * 1000.0))
            return out

        if not (_is_ttnn_tensor(enc_logits) and _is_ttnn_tensor(anchors) and _is_ttnn_tensor(memory)):
            raise TypeError("TTNN topk expects TTNN logits/anchors/memory tensors")

        B, L, _ = enc_logits.shape
        k = min(int(k), int(L))

        # Compute scores on device (force FP32 max reduction for stability).
        if self.trace_mode:
            if enc_logits.get_layout() != self.layout:
                raise RuntimeError("Decoder trace_mode expects logits in decoder layout.")
            enc_logits_tt = enc_logits
        else:
            enc_logits_tt = ttnn.to_layout(enc_logits, self.layout)
        if enc_logits_tt.dtype != ttnn.float32:
            enc_logits_tt = ttnn.typecast(enc_logits_tt, ttnn.float32)
        if enc_logits.shape[-1] > 1:
            enc_logits_tt = ttnn.fill_implicit_tile_padding(enc_logits_tt, -1e9)
        if enc_logits.shape[-1] == 1:
            scores_tt = ttnn.squeeze(enc_logits_tt, dim=-1)
        else:
            scores_tt = _time_topk("topk.score_max", lambda: ttnn.max(enc_logits_tt, dim=-1))
            if list(scores_tt.shape) != [B, L]:
                scores_tt = ttnn.reshape(scores_tt, (B, L))
        if self.trace_mode:
            if scores_tt.get_layout() != ttnn.TILE_LAYOUT:
                raise RuntimeError("Decoder trace_mode expects scores in TILE layout.")
        else:
            scores_tt = ttnn.to_layout(scores_tt, ttnn.TILE_LAYOUT)
        scores_tt = ttnn.fill_implicit_tile_padding(scores_tt, -1.0e9)
        topk_score_dtype_env = os.environ.get("TTNN_TOPK_SCORE_DTYPE", "").lower()
        if topk_score_dtype_env in ("bfloat8_b", "bf8", "bfp8", "bfp8_b"):
            topk_score_dtype = ttnn.bfloat8_b
        else:
            topk_score_dtype = self.dtype
        scores_for_topk = (
            scores_tt if scores_tt.dtype == topk_score_dtype else ttnn.typecast(scores_tt, topk_score_dtype)
        )
        use_multipass_topk = os.environ.get("TTNN_TOPK_MULTIPASS64", "0") != "0" and k > 64
        if os.environ.get("TTNN_TOPK_PAD_POWER2", "0") != "0" or use_multipass_topk:
            padded_l = 1 << (int(L) - 1).bit_length()
            if padded_l != int(L):
                pad_width = padded_l - int(L)
                pad_tt = self.trace_ctx.topk_score_pad if self.trace_ctx is not None else None
                if pad_tt is None:
                    pad_tt = ttnn.full(
                        (B, pad_width),
                        -1.0e9,
                        device=self.device,
                        dtype=scores_for_topk.dtype,
                        layout=ttnn.TILE_LAYOUT,
                    )
                scores_for_topk = ttnn.concat([scores_for_topk, pad_tt], dim=1)
        tie_break_eps = float(os.environ.get("TTNN_TOPK_TIE_BREAK_EPS", "0") or "0")
        if tie_break_eps != 0.0:
            topk_len_for_tie = int(scores_for_topk.shape[1])
            tie_break = self.trace_ctx.topk_tie_break if self.trace_ctx is not None else None
            if tie_break is None:
                tie_break = ttnn.arange(0, topk_len_for_tie, 1, device=self.device, dtype=ttnn.float32)
                tie_break = ttnn.reshape(tie_break, (1, topk_len_for_tie))
                if int(B) != 1:
                    tie_break = ttnn.repeat(tie_break, (int(B), 1))
                tie_break = ttnn.multiply(tie_break, tie_break_eps / max(1, topk_len_for_tie - 1))
            if list(tie_break.shape) != [B, topk_len_for_tie]:
                tie_break = ttnn.reshape(tie_break, (B, topk_len_for_tie))
            if tie_break.dtype != scores_for_topk.dtype:
                tie_break = ttnn.typecast(tie_break, scores_for_topk.dtype)
            if tie_break.get_layout() != scores_for_topk.get_layout():
                tie_break = ttnn.to_layout(tie_break, scores_for_topk.get_layout())
            scores_for_topk = _time_topk(
                "topk.tie_break_add",
                lambda scores_for_topk=scores_for_topk, tie_break=tie_break: ttnn.add(scores_for_topk, tie_break),
            )
        topk_sorted = os.environ.get("TTNN_TOPK_SORTED", "1") != "0"
        level_quotas_env = os.environ.get("TTNN_TOPK_LEVEL_QUOTAS")
        if level_quotas_env:
            if spatial_shapes is None:
                raise RuntimeError("TTNN_TOPK_LEVEL_QUOTAS requires spatial_shapes.")
            level_quotas = [max(0, int(part.strip())) for part in level_quotas_env.split(",") if part.strip()]
            if len(level_quotas) != len(spatial_shapes):
                raise ValueError(
                    f"TTNN_TOPK_LEVEL_QUOTAS expected {len(spatial_shapes)} entries, got {len(level_quotas)}"
                )
            vals_parts = []
            idx_parts = []
            anchor_parts = []
            memory_parts = []
            start = 0
            for lvl, ((h, w), quota) in enumerate(zip(spatial_shapes, level_quotas)):
                h = int(h)
                w = int(w)
                level_len = h * w
                quota = min(int(quota), level_len)
                if quota <= 0:
                    start += level_len
                    continue
                scores_lvl = _time_topk(
                    f"topk.level{lvl}.scores_slice",
                    lambda start=start, level_len=level_len: ttnn.slice(
                        scores_for_topk,
                        [0, start],
                        [B, start + level_len],
                    ),
                )
                vals_part, idx_local = _time_topk(
                    f"topk.level{lvl}.score_topk",
                    lambda scores_lvl=scores_lvl, quota=quota: ttnn.topk(
                        scores_lvl,
                        k=quota,
                        dim=1,
                        largest=True,
                        sorted=topk_sorted,
                    ),
                )
                idx_local = ttnn.typecast(idx_local, ttnn.uint32)
                if list(idx_local.shape) != [B, quota]:
                    idx_local = ttnn.reshape(idx_local, (B, quota))
                idx_local = _ensure_layout(idx_local, ttnn.TILE_LAYOUT)
                vals_parts.append(vals_part)
                idx_parts.append(
                    _time_topk(
                        f"topk.level{lvl}.global_idx",
                        lambda idx_local=idx_local, start=start: ttnn.typecast(
                            ttnn.add(ttnn.typecast(idx_local, ttnn.float32), float(start)),
                            ttnn.uint32,
                        ),
                    )
                )
                anchor_parts.append(
                    _time_topk(
                        f"topk.level{lvl}.anchor_analytic",
                        lambda idx_local=idx_local, h=h, w=w, lvl=lvl, quota=quota: self._anchors_from_level_indices_ttnn(
                            idx_local,
                            h,
                            w,
                            lvl,
                            int(B),
                            quota,
                        ),
                    )
                )
                memory_weight_tt, embedding_idx_tt = _time_topk(
                    f"topk.level{lvl}.memory_embedding_prepare",
                    lambda start=start, level_len=level_len, idx_local=idx_local: (
                        ttnn.to_layout(
                            ttnn.reshape(
                                (
                                    ttnn.slice(memory, [0, start, 0], [B, start + level_len, int(memory.shape[-1])])
                                    if memory.dtype == self.dtype
                                    else ttnn.typecast(
                                        ttnn.slice(
                                            memory,
                                            [0, start, 0],
                                            [B, start + level_len, int(memory.shape[-1])],
                                        ),
                                        self.dtype,
                                    )
                                ),
                                (level_len, int(memory.shape[-1])),
                            ),
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        ttnn.to_layout(idx_local, ttnn.ROW_MAJOR_LAYOUT),
                    ),
                )
                memory_parts.append(
                    _time_topk(
                        f"topk.level{lvl}.memory_embedding",
                        lambda embedding_idx_tt=embedding_idx_tt, memory_weight_tt=memory_weight_tt: ttnn.embedding(
                            embedding_idx_tt,
                            memory_weight_tt,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=self.dtype,
                        ),
                    )
                )
                start += level_len
            if not vals_parts:
                raise ValueError(f"TTNN_TOPK_LEVEL_QUOTAS selects no queries: {level_quotas_env!r}")
            topk_vals_tt = ttnn.concat(vals_parts, dim=1)
            topk_idx_tt = ttnn.concat(idx_parts, dim=1)
            topk_anchors_tt = ttnn.concat(anchor_parts, dim=1)
            topk_memory_tt = ttnn.concat(memory_parts, dim=1)
            return topk_vals_tt, topk_idx_tt, topk_anchors_tt, topk_memory_tt
        use_sort_topk = os.environ.get("TTNN_TOPK_USE_SORT", "0") != "0"
        if use_sort_topk:
            sort_dtype_env = os.environ.get("TTNN_TOPK_SORT_DTYPE", "").lower()
            if sort_dtype_env in ("float32", "fp32", "f32"):
                scores_for_sort = scores_tt if scores_tt.dtype == ttnn.float32 else ttnn.typecast(scores_tt, ttnn.float32)
            else:
                scores_for_sort = scores_for_topk
            sorted_vals_tt, sorted_idx_tt = _time_topk(
                "topk.score_sort",
                lambda scores_for_sort=scores_for_sort: ttnn.sort(
                    scores_for_sort,
                    dim=1,
                    descending=True,
                ),
            )
            topk_vals_tt = _time_topk(
                "topk.sort_vals_slice",
                lambda sorted_vals_tt=sorted_vals_tt: ttnn.slice(sorted_vals_tt, [0, 0], [B, k]),
            )
            topk_idx_tt = _time_topk(
                "topk.sort_idx_slice",
                lambda sorted_idx_tt=sorted_idx_tt: ttnn.slice(sorted_idx_tt, [0, 0], [B, k]),
            )
        elif use_multipass_topk:
            pass_k = min(64, k)
            num_passes = (k + pass_k - 1) // pass_k
            topk_len = int(scores_for_topk.shape[1])
            scores_current = scores_for_topk
            vals_parts = []
            idx_parts = []
            rng_tt = self.trace_ctx.topk_mask_rng if self.trace_ctx is not None else None
            use_threshold_mask = os.environ.get("TTNN_TOPK_MULTIPASS64_THRESHOLD", "0") != "0"
            threshold_strict_gt = os.environ.get("TTNN_TOPK_MULTIPASS64_THRESHOLD_GT", "0") != "0"
            for pass_idx in range(num_passes):
                vals_part, idx_part = _time_topk(
                    f"topk.score_topk_pass{pass_idx}",
                    lambda scores_current=scores_current: ttnn.topk(
                        scores_current,
                        k=pass_k,
                        dim=1,
                        largest=True,
                        sorted=True,
                    ),
                )
                idx_part = ttnn.typecast(idx_part, ttnn.uint32)
                if list(idx_part.shape) != [B, pass_k]:
                    idx_part = ttnn.reshape(idx_part, (B, pass_k))
                vals_parts.append(vals_part)
                idx_parts.append(idx_part)
                if pass_idx + 1 < num_passes:
                    if use_threshold_mask:
                        kth_value = _time_topk(
                            f"topk.mask_threshold_pass{pass_idx}",
                            lambda vals_part=vals_part: ttnn.slice(vals_part, [0, pass_k - 1], [B, pass_k]),
                        )
                        if list(kth_value.shape) != [B, 1]:
                            kth_value = ttnn.reshape(kth_value, (B, 1))
                        kth_value = _time_topk(
                            f"topk.mask_threshold_repeat_pass{pass_idx}",
                            lambda kth_value=kth_value: ttnn.repeat(kth_value, (1, topk_len)),
                        )
                        mask_tt = _time_topk(
                            f"topk.mask_threshold_cmp_pass{pass_idx}",
                            lambda scores_current=scores_current, kth_value=kth_value: (
                                ttnn.gt(scores_current, kth_value)
                                if threshold_strict_gt
                                else ttnn.ge(scores_current, kth_value)
                            ),
                        )
                        scores_current = _time_topk(
                            f"topk.mask_threshold_apply_pass{pass_idx}",
                            lambda scores_current=scores_current, mask_tt=mask_tt: ttnn.where(
                                mask_tt,
                                ttnn.add(ttnn.multiply(scores_current, 0.0), -1.0e9),
                                scores_current,
                            ),
                        )
                    else:
                        one_hot = _time_topk(
                            f"topk.mask_one_hot_pass{pass_idx}",
                            lambda idx_part=idx_part: self._one_hot_from_indices_ttnn(
                                idx_part,
                                topk_len,
                                int(B),
                                pass_k,
                                rng_tt=rng_tt,
                                dtype=scores_current.dtype,
                                trace_mode=self.trace_mode,
                            ),
                        )
                        mask_tt = _time_topk(
                            f"topk.mask_reduce_pass{pass_idx}",
                            lambda one_hot=one_hot: ttnn.max(one_hot, dim=1),
                        )
                        if list(mask_tt.shape) != [B, topk_len]:
                            mask_tt = ttnn.reshape(mask_tt, (B, topk_len))
                        scores_current = _time_topk(
                            f"topk.mask_apply_pass{pass_idx}",
                            lambda scores_current=scores_current, mask_tt=mask_tt: ttnn.add(
                                scores_current,
                                ttnn.multiply(mask_tt, -1.0e9),
                            ),
                        )
            topk_vals_tt = ttnn.concat(vals_parts, dim=1)
            topk_idx_tt = ttnn.concat(idx_parts, dim=1)
            if int(topk_idx_tt.shape[1]) != k:
                topk_vals_tt = ttnn.slice(topk_vals_tt, [0, 0], [B, k])
                topk_idx_tt = ttnn.slice(topk_idx_tt, [0, 0], [B, k])
        else:
            topk_vals_tt, topk_idx_tt = _time_topk(
                "topk.score_topk",
                lambda: ttnn.topk(scores_for_topk, k=k, dim=1, largest=True, sorted=topk_sorted),
            )
        topk_idx_tt = ttnn.typecast(topk_idx_tt, ttnn.uint32)
        if list(topk_idx_tt.shape) != [B, k]:
            topk_idx_tt = ttnn.reshape(topk_idx_tt, (B, k))
        topk_idx_tt = _ensure_layout(topk_idx_tt, ttnn.TILE_LAYOUT)

        use_analytic_anchor = (
            os.environ.get("TTNN_TOPK_ANCHOR_ANALYTIC", "1") != "0"
            and int(B) == 1
            and spatial_shapes is not None
        )
        use_embedding_anchor = os.environ.get("TTNN_TOPK_ANCHOR_EMBEDDING", "0") != "0" and int(B) == 1
        if use_analytic_anchor:
            topk_anchors_tt = _time_topk(
                "topk.anchor_analytic",
                lambda: self._anchors_from_indices_ttnn(topk_idx_tt, spatial_shapes, int(B), int(k)),
            )
        elif use_embedding_anchor:
            anchor_embedding_dtype = self.dtype
            anchor_weight_tt, anchor_idx_tt = _time_topk(
                "topk.anchor_embedding_prepare",
                lambda: (
                    ttnn.to_layout(
                        ttnn.reshape(
                            anchors if anchors.dtype == anchor_embedding_dtype else ttnn.typecast(anchors, anchor_embedding_dtype),
                            (int(L), int(anchors.shape[-1])),
                        ),
                        ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    ttnn.to_layout(topk_idx_tt, ttnn.ROW_MAJOR_LAYOUT),
                ),
            )
            topk_anchors_tt = _time_topk(
                "topk.anchor_embedding",
                lambda: ttnn.embedding(
                    anchor_idx_tt,
                    anchor_weight_tt,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=anchor_embedding_dtype,
                ),
            )
        elif os.environ.get("TTNN_TOPK_ANCHOR_TOSA_GATHER", "1") != "0":
            topk_anchors_tt = _time_topk(
                "topk.anchor_tosa_gather",
                lambda: ttnn.tosa_gather(anchors, topk_idx_tt),
            )
        else:
            idx_anchor = ttnn.reshape(topk_idx_tt, (B, k, 1))
            idx_anchor = ttnn.repeat(idx_anchor, (1, 1, anchors.shape[-1]))
            topk_anchors_tt = _time_topk("topk.anchor_gather", lambda: ttnn.gather(anchors, dim=1, index=idx_anchor))

        use_embedding_memory = os.environ.get("TTNN_TOPK_MEMORY_EMBEDDING", "1") != "0" and int(B) == 1
        if use_embedding_memory:
            memory_weight_tt, embedding_idx_tt = _time_topk(
                "topk.memory_embedding_prepare",
                lambda: (
                    ttnn.to_layout(
                        ttnn.reshape(
                            memory if memory.dtype == self.dtype else ttnn.typecast(memory, self.dtype),
                            (int(L), int(memory.shape[-1])),
                        ),
                        ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    ttnn.to_layout(topk_idx_tt, ttnn.ROW_MAJOR_LAYOUT),
                ),
            )
            topk_memory_tt = _time_topk(
                "topk.memory_embedding",
                lambda: ttnn.embedding(
                    embedding_idx_tt,
                    memory_weight_tt,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.dtype,
                ),
            )
        else:
            idx_mem = ttnn.reshape(topk_idx_tt, (B, k, 1))
            idx_mem = ttnn.repeat(idx_mem, (1, 1, memory.shape[-1]))
            topk_memory_tt = _time_topk("topk.memory_gather", lambda: ttnn.gather(memory, dim=1, index=idx_mem))

        return topk_vals_tt, topk_idx_tt, topk_anchors_tt, topk_memory_tt

    def _prepare_msda_value_list_ttnn(
        self,
        value_list: Sequence["ttnn.Tensor"],
        spatial_shapes: Sequence[Sequence[int]],
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> List["ttnn.Tensor"]:
        """Prepare static MSDA value tensors once instead of once per decoder layer."""
        ttnn = self.ttnn
        prepared = []
        bh = int(batch_size) * int(num_heads)
        c_pad = (32 - (int(head_dim) % 32)) % 32
        for v, (h, w) in zip(value_list, spatial_shapes):
            v_tt = ttnn.reshape(v, (batch_size, num_heads, head_dim, int(h), int(w)))
            v_tt = ttnn.permute(v_tt, (0, 1, 3, 4, 2))
            v_tt = ttnn.reshape(v_tt, (bh, int(h), int(w), head_dim))
            if c_pad != 0:
                v_tt = ttnn.pad(v_tt, padding=[(0, 0), (0, 0), (0, 0), (0, c_pad)], value=0.0)
            if v_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                v_tt = ttnn.to_layout(v_tt, ttnn.ROW_MAJOR_LAYOUT)
            prepared.append(v_tt)
        return prepared

    #Full forward
    def forward(self, feats: Sequence[TTActivation]) -> Dict[str, "ttnn.Tensor"]:
        ttnn = self.ttnn
        timing_entries = [] if os.environ.get("TTNN_DECODER_TIMING", "0") == "1" else None
        def _time_block(name, fn):
            if timing_entries is None:
                return fn()
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            t1 = time.perf_counter()
            timing_entries.append((name, (t1 - t0) * 1000.0))
            return out
        if not feats or not _is_tt_activation(feats[0]):
            raise TypeError("TTNN decoder expects TTActivation features from TTNN encoder")

        memory_blc_tt, spatial_shapes = self._get_encoder_input_ttnn(feats)
        anchors_tt = self.anchors_tt
        valid_mask_tt = self.valid_mask_tt
        if anchors_tt is None or valid_mask_tt is None or (
            self._spatial_shapes_cache is not None and self._spatial_shapes_cache != spatial_shapes
        ):
            anchors_tt, valid_mask_tt = self._generate_anchors_ttnn(spatial_shapes)
        B = memory_blc_tt.shape[0]
        if anchors_tt.shape[0] != B:
            anchors_tt = ttnn.repeat(anchors_tt, (B, 1, 1))
            valid_mask_tt = ttnn.repeat(valid_mask_tt, (B, 1, 1))
        if self.trace_mode:
            if valid_mask_tt.get_layout() != ttnn.TILE_LAYOUT:
                valid_mask_tt = ttnn.to_layout(valid_mask_tt, ttnn.TILE_LAYOUT)
            mask_tt = valid_mask_tt
        else:
            mask_tt = ttnn.to_layout(valid_mask_tt, ttnn.TILE_LAYOUT)
        if mask_tt.dtype != memory_blc_tt.dtype:
            mask_tt = ttnn.typecast(mask_tt, memory_blc_tt.dtype)
        if self.trace_mode:
            if memory_blc_tt.get_layout() != ttnn.TILE_LAYOUT:
                raise RuntimeError("trace_mode expects TILE encoder memory for masking")
            memory_tile = memory_blc_tt
        else:
            memory_tile = ttnn.to_layout(memory_blc_tt, ttnn.TILE_LAYOUT)
        memory_masked = ttnn.multiply(memory_tile, mask_tt)
        if not self.trace_mode:
            memory_masked = ttnn.to_layout(memory_masked, ttnn.ROW_MAJOR_LAYOUT)

        enc_out_tt = _time_block("enc_output", lambda: self._enc_output_ttnn(memory_masked, return_ttnn=True))
        enc_logits_tt = _time_block("enc_score_head", lambda: self.enc_score_head_tt(enc_out_tt, return_ttnn=True))

        self._topk_timing_entries = timing_entries
        try:
            _, _, topk_anchors_tt, topk_memory_tt = _time_block(
                "topk_gather",
                lambda: self._topk_gather_ttnn(
                    enc_logits_tt,
                    anchors_tt,
                    enc_out_tt,
                    self._num_queries(),
                    spatial_shapes=spatial_shapes,
                    return_ttnn=True,
                ),
            )
        finally:
            self._topk_timing_entries = None
        if os.environ.get("TTNN_DECODER_TOPK_FP32", "1") != "0":
            topk_memory_tt = ttnn.typecast(topk_memory_tt, ttnn.float32)
            topk_anchors_tt = ttnn.typecast(topk_anchors_tt, ttnn.float32)

        enc_topk_bbox_unact = _time_block(
            "enc_bbox_head",
            lambda: ttnn.add(
                self.enc_bbox_head_tt(topk_memory_tt, return_ttnn=True, keep_fp32=True),
                topk_anchors_tt,
            ),
        )

        output_tt = topk_memory_tt
        ref_points_detach = _time_block("enc_bbox_sigmoid", lambda: ttnn.sigmoid(enc_topk_bbox_unact))

        B, L, C = enc_out_tt.shape
        num_head = int(self.decoder_pt.nhead)
        head_dim = C // num_head
        split_shape = [h * w for h, w in spatial_shapes]
        value_tt = ttnn.reshape(memory_blc_tt, (B, L, num_head, head_dim))
        value_tt = ttnn.permute(value_tt, (0, 2, 3, 1))
        value_list = ttnn.split(value_tt, split_shape, dim=3)
        value_grid_list = _time_block(
            "value_grid_prepare",
            lambda: self._prepare_msda_value_list_ttnn(value_list, spatial_shapes, int(B), num_head, head_dim),
        )

        eval_idx = int(
            self.decoder_pt.decoder.eval_idx if hasattr(self.decoder_pt.decoder, "eval_idx") else len(self.layers) - 1
        )
        eval_idx_env = os.environ.get("TTNN_DECODER_EVAL_IDX")
        if eval_idx_env is not None:
            eval_idx = max(0, min(int(eval_idx_env), len(self.layers) - 1))

        dec_out_bboxes_tt = None
        dec_out_logits_tt = None
        output_detach = None
        prev_pred_corners = None

        for i, layer in enumerate(self.layers):
            layer._timing_entries = timing_entries
            layer._timing_prefix = f"layer{i}"
            Bq = output_tt.shape[0]
            Kq = output_tt.shape[1]
            ref_points_input = ttnn.reshape(ref_points_detach, (Bq, Kq, 1, 4))
            query_pos_embed = _time_block(
                f"layer{i}.query_pos_head",
                lambda: self.query_pos_head_tt(ref_points_detach, return_ttnn=True),
            )
            query_pos_embed = _time_block(
                f"layer{i}.query_pos_clip",
                lambda: ttnn.clip(query_pos_embed, min=-10.0, max=10.0),
            )

            output_tt = _time_block(
                f"layer{i}.decoder_layer",
                lambda: layer(
                    output_tt,
                    ref_points_input,
                    value_grid_list,
                    spatial_shapes,
                    attn_mask=None,
                    query_pos_embed=query_pos_embed,
                    return_ttnn=True,
                ),
            )

            if i == 0:
                pre_bbox_tt = _time_block(
                    "pre_bbox_head",
                    lambda: self.pre_bbox_head_tt(output_tt, return_ttnn=True, keep_fp32=True),
                )
                pre_bboxes_tt = _time_block(
                    "pre_bbox_sigmoid",
                    lambda: ttnn.sigmoid(
                        ttnn.add(
                            pre_bbox_tt,
                            _ttnn_inverse_sigmoid(ttnn, ref_points_detach, trace_mode=self.trace_mode),
                        )
                    ),
                )
                ref_points_initial = pre_bboxes_tt

            output_in = output_tt if output_detach is None else ttnn.add(output_tt, output_detach)
            pred_corners_tt = _time_block(
                f"layer{i}.dec_bbox_head",
                lambda: self.dec_bbox_head_tt[i](output_in, return_ttnn=True, keep_fp32=True),
            )
            if prev_pred_corners is not None:
                pred_corners_tt = ttnn.add(pred_corners_tt, prev_pred_corners)
            pred_corners_tt = ttnn.typecast(pred_corners_tt, ttnn.float32)

            distance_tt = _time_block(
                f"layer{i}.integral",
                lambda: self.integral(pred_corners_tt, self.project_tt, return_ttnn=True),
            )
            inter_ref_bbox = _time_block(
                f"layer{i}.distance2bbox",
                lambda: _ttnn_distance2bbox(
                    ttnn,
                    ref_points_initial,
                    distance_tt,
                    self.reg_scale,
                    trace_mode=self.trace_mode,
                ),
            )

            if i == eval_idx:
                scores_tt = _time_block(
                    f"layer{i}.dec_score_head",
                    lambda: self.dec_score_head_tt[i](output_tt, return_ttnn=True),
                )
                if scores_tt.dtype != ttnn.float32:
                    scores_tt = ttnn.typecast(scores_tt, ttnn.float32)
                if self.lqe_layer_tt is not None:
                    scores_tt = _time_block(
                        f"layer{i}.lqe",
                        lambda: self.lqe_layer_tt(scores_tt, pred_corners_tt, return_ttnn=True),
                    )
                dec_out_logits_tt = scores_tt
                dec_out_bboxes_tt = inter_ref_bbox
                break

            prev_pred_corners = pred_corners_tt
            ref_points_detach = inter_ref_bbox
            output_detach = output_tt

        if dec_out_logits_tt is None or dec_out_bboxes_tt is None:
            raise RuntimeError("TTNN decoder produced no outputs")

        if timing_entries is not None:
            self._last_timing_entries = timing_entries
        return {
            "pred_logits": dec_out_logits_tt,
            "pred_boxes": dec_out_bboxes_tt,
        }

    def close(self):
        if self._owns_device:
            self.ttnn.close_device(self.device)


class TTNNMSDeformableAttention(nn.Module):
    """TTNN port of MSDeformableAttention using ttnn.grid_sample.

    This module consumes the parameters from a PyTorch MSDeformableAttention instance
    and reproduces its forward pass using TTNN primitives where appropriate.
    """

    def __init__(self, msda_pt: nn.Module, device, dtype=None, layout=None, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        dtype_default = dtype if dtype is not None else ttnn.bfloat16
        self.dtype = dtype_default
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self.trace_mode = False

        self.embed_dim = int(msda_pt.embed_dim)
        self.num_heads = int(msda_pt.num_heads)
        self.num_levels = int(msda_pt.num_levels)
        self.num_points_list: List[int] = list(msda_pt.num_points_list)
        # total_points in the PyTorch module already includes num_heads
        self.total_points = int(msda_pt.total_points)
        if self.total_points % self.num_heads != 0:
            raise ValueError("total_points must be divisible by num_heads")
        self.points_per_head = int(self.total_points // self.num_heads)
        self.method = msda_pt.method
        self.offset_scale = float(getattr(msda_pt, 'offset_scale', 0.5))

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        # Port sampling_offsets, attention_weights linears
        if weight_store is None:
            raise RuntimeError("TTNN MSDA requires a weight store (set TTNN_WEIGHT_DIR).")
        off_key = _module_key(registry, msda_pt.sampling_offsets)
        attn_key = _module_key(registry, msda_pt.attention_weights)
        if off_key is None or attn_key is None:
            raise RuntimeError("Missing weight store key for MSDA module.")
        if weight_store.mode == "load":
            self.W_off = _load_tensor_from_store(weight_store, f"{off_key}.weight_t", device, self.dtype, ttnn.TILE_LAYOUT)
            self.b_off = _load_tensor_from_store(weight_store, f"{off_key}.bias", device, self.dtype, ttnn.TILE_LAYOUT)
            self.W_attn = _load_tensor_from_store(weight_store, f"{attn_key}.weight_t", device, self.dtype, ttnn.TILE_LAYOUT)
            self.b_attn = _load_tensor_from_store(weight_store, f"{attn_key}.bias", device, self.dtype, ttnn.TILE_LAYOUT)
        elif weight_store.mode == "save":
            self.W_off = ttnn.from_torch(
                msda_pt.sampling_offsets.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
            )
            self.b_off = ttnn.from_torch(
                msda_pt.sampling_offsets.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
            )
            self.W_attn = ttnn.from_torch(
                msda_pt.attention_weights.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
            )
            self.b_attn = ttnn.from_torch(
                msda_pt.attention_weights.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
            )
            weight_store.save_tensor(f"{off_key}.weight_t", self.W_off)
            weight_store.save_tensor(f"{off_key}.bias", self.b_off)
            weight_store.save_tensor(f"{attn_key}.weight_t", self.W_attn)
            weight_store.save_tensor(f"{attn_key}.bias", self.b_attn)
        else:
            raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")
        self._offset_norm_cache = {}
        self._timing_entries = None
        self._timing_prefix = None

    def _get_offset_norm_levels(self, value_spatial_shapes: List[List[int]]):
        key = tuple(tuple(int(x) for x in pair) for pair in value_spatial_shapes)
        cached = self._offset_norm_cache.get(key)
        if cached is not None:
            return cached
        levels = []
        for h, w in value_spatial_shapes:
            w_tt = ttnn.full((1, 1, 1, 1, 1), float(w), device=self.device, dtype=ttnn.float32, layout=self.layout)
            h_tt = ttnn.full((1, 1, 1, 1, 1), float(h), device=self.device, dtype=ttnn.float32, layout=self.layout)
            levels.append(ttnn.concat([w_tt, h_tt], dim=-1))
        self._offset_norm_cache[key] = levels
        return levels

    def forward(
        self,
        query: "ttnn.Tensor",                     # [B, Lq, C] TTNN
        reference_points: "ttnn.Tensor",          # [B, Lq, num_levels, 2 or 4] TTNN
        value: List["ttnn.Tensor"],               # list of [B, H, C_head, H_lv, W_lv] TTNN
        value_spatial_shapes: List[List[int]],   # [[H1,W1], [H2,W2], ...]
        return_ttnn: bool = False,
    ) -> "ttnn.Tensor":
        ttnn = self.ttnn
        timing_entries = self._timing_entries
        timing_prefix = self._timing_prefix

        def _time_block(name, fn):
            if timing_entries is None or timing_prefix is None:
                return fn()
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            t1 = time.perf_counter()
            timing_entries.append((f"{timing_prefix}.{name}", (t1 - t0) * 1000.0))
            return out

        if not _is_ttnn_tensor(query):
            raise TypeError("TTNN MSDA expects TTNN query tensor")
        B, Lq, C = query.shape

        # Offsets and attention weights via TTNN linear
        q_tt = query
        off = _time_block("offset_linear", lambda: ttnn.linear(q_tt, self.W_off, bias=self.b_off))
        attn = _time_block("attn_linear", lambda: ttnn.linear(q_tt, self.W_attn, bias=self.b_attn))
        if off.get_layout() != ttnn.TILE_LAYOUT:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects TILE layout for MSDA offsets")
            off = ttnn.to_layout(off, ttnn.TILE_LAYOUT)
        if attn.get_layout() != ttnn.TILE_LAYOUT:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects TILE layout for MSDA attn weights")
            attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT)
        off = ttnn.slice(off, [0, 0, 0], [B, Lq, self.total_points * 2])
        attn = ttnn.slice(attn, [0, 0, 0], [B, Lq, self.total_points])
        off = ttnn.reshape(off, (B, Lq, self.num_heads, self.points_per_head, 2))
        # Softmax must be per-head, matching Torch (B, Lq, num_heads, num_points)
        attn = ttnn.reshape(attn, (B, Lq, self.num_heads, self.points_per_head))
        attn = _time_block(
            "attn_softmax",
            lambda: _softmax_lastdim_ttnn(ttnn, attn, dim=-1, trace_mode=self.trace_mode),
        )

        # Compute sampling locations in [0,1] per level
        if not _is_ttnn_tensor(reference_points):
            raise TypeError("TTNN MSDA expects TTNN reference_points")
        if reference_points.get_layout() != self.layout:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects reference_points in MSDA layout")
            ref_tt = ttnn.to_layout(reference_points, self.layout)
        else:
            ref_tt = reference_points
        if ref_tt.dtype != ttnn.float32:
            ref_tt = ttnn.typecast(ref_tt, ttnn.float32)
        direct_grid = False
        if reference_points.shape[-1] == 2:
            sampling_locations_per_level = []
            off_splits = ttnn.split(off, self.num_points_list, dim=3)
            norm_levels = self._get_offset_norm_levels(value_spatial_shapes)
            for lvl in range(self.num_levels):
                ref_lvl = ttnn.slice(ref_tt, [0, 0, lvl, 0], [B, Lq, lvl + 1, 2])
                ref_lvl = ttnn.reshape(ref_lvl, (B, Lq, 1, 1, 2))
                ref_lvl = ttnn.repeat(ref_lvl, (1, 1, self.num_heads, self.num_points_list[lvl], 1))

                norm_lvl = norm_levels[lvl]
                norm_lvl = ttnn.repeat(norm_lvl, (B, Lq, self.num_heads, self.num_points_list[lvl], 1))

                loc = _time_block(
                    f"level{lvl}.sampling_locations",
                    lambda ref_lvl=ref_lvl, off_lvl=off_splits[lvl], norm_lvl=norm_lvl: ttnn.add(
                        ref_lvl,
                        ttnn.div(off_lvl, norm_lvl),
                    ),
                )
                sampling_locations_per_level.append(loc)
        elif reference_points.shape[-1] == 4:
            direct_grid_env = os.environ.get("TTNN_MSDA_DIRECT_GRID")
            if direct_grid_env is None:
                auto_max_queries = int(os.environ.get("TTNN_MSDA_DIRECT_GRID_AUTO_MAX_QUERIES", "96"))
                direct_grid = int(Lq) <= auto_max_queries
            else:
                direct_grid = direct_grid_env != "0"
            if reference_points.shape[2] == 1 and self.num_levels > 1:
                ref_tt = ttnn.repeat(ref_tt, (1, 1, self.num_levels, 1))
            off_splits = ttnn.split(off, self.num_points_list, dim=3)
            sampling_locations_per_level = []
            for lvl in range(self.num_levels):
                ref_lvl = ttnn.slice(ref_tt, [0, 0, lvl, 0], [B, Lq, lvl + 1, 4])
                ref_lvl = ttnn.reshape(ref_lvl, (B, Lq, 1, 1, 4))
                ref_xy = ttnn.slice(ref_lvl, [0, 0, 0, 0, 0], [B, Lq, 1, 1, 2])
                ref_wh = ttnn.slice(ref_lvl, [0, 0, 0, 0, 2], [B, Lq, 1, 1, 4])

                scale = float(self.offset_scale) / float(self.num_points_list[lvl])
                if direct_grid:
                    ref_xy = _time_block(f"level{lvl}.ref_xy_grid_scale", lambda ref_xy=ref_xy: ttnn.multiply(ref_xy, 2.0))
                    ref_xy = _time_block(f"level{lvl}.ref_xy_grid_shift", lambda ref_xy=ref_xy: ttnn.subtract(ref_xy, 1.0))
                    scale *= 2.0
                    if scale != 1.0:
                        ref_wh = _time_block(
                            f"level{lvl}.ref_wh_scale",
                            lambda ref_wh=ref_wh, scale=scale: ttnn.multiply(ref_wh, scale),
                        )
                ref_xy = ttnn.repeat(ref_xy, (1, 1, self.num_heads, self.num_points_list[lvl], 1))
                ref_wh = ttnn.repeat(ref_wh, (1, 1, self.num_heads, self.num_points_list[lvl], 1))

                off_lvl = ttnn.multiply(off_splits[lvl], ref_wh)
                if (not direct_grid) and scale != 1.0:
                    off_lvl = ttnn.multiply(off_lvl, scale)
                loc = _time_block(
                    f"level{lvl}.sampling_locations",
                    lambda ref_xy=ref_xy, off_lvl=off_lvl: ttnn.add(ref_xy, off_lvl),
                )
                sampling_locations_per_level.append(loc)
        else:
            raise ValueError("reference_points last dim must be 2 or 4")
        grid_already_normalized = direct_grid and reference_points.shape[-1] == 4

        # Prepare attention_weights per level
        attn_splits = ttnn.split(attn, self.num_points_list, dim=3)

        # Accumulate per-level sampled values
        sampled_sum = None  # [B*H, C_head, Lq]
        for lvl, (h, w) in enumerate(value_spatial_shapes):
            # value[lvl]: [B, H, C_head, H_lv*W_lv] or [B,H,C_head,H_lv,W_lv]
            v = value[lvl]
            if not _is_ttnn_tensor(v):
                raise TypeError("TTNN MSDA expects TTNN value tensors")
            v_shape = list(v.shape)
            Bh = B * self.num_heads
            prepared_nhwc = (
                len(v_shape) == 4
                and int(v_shape[0]) == int(Bh)
                and int(v_shape[1]) == int(h)
                and int(v_shape[2]) == int(w)
                and int(v_shape[3]) >= int(self.head_dim)
            )
            if prepared_nhwc:
                v_tt = v
            elif len(v_shape) == 4:
                v_tt = ttnn.reshape(v, (B, self.num_heads, self.head_dim, h, w))
            else:
                v_tt = v
            # NHWC input for TTNN grid_sample
            c_pad = (32 - (self.head_dim % 32)) % 32
            if not prepared_nhwc:
                v_tt = ttnn.permute(v_tt, (0, 1, 3, 4, 2))
                v_tt = ttnn.reshape(v_tt, (Bh, h, w, self.head_dim))
                if c_pad != 0:
                    v_tt = ttnn.pad(v_tt, padding=[(0, 0), (0, 0), (0, 0), (0, c_pad)], value=0.0)
            if v_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                v_tt = ttnn.to_layout(v_tt, ttnn.ROW_MAJOR_LAYOUT)

            # Grid for this level: [B, Lq, H, num_points_lvl, 2] -> reshape to [Bh, Lq, num_points_lvl, 2]
            loc = sampling_locations_per_level[lvl]  # [B,Lq,H,num_pts,2]
            grid_tt = _time_block(f"level{lvl}.grid_permute", lambda loc=loc: ttnn.permute(loc, (0, 2, 1, 3, 4)))
            num_points = self.num_points_list[lvl]
            grid_tt = ttnn.reshape(grid_tt, (Bh, Lq, num_points, 2))
            # Normalize to [-1, 1] and cast to FP32
            if not grid_already_normalized:
                if os.environ.get("TTNN_MSDA_GRID_MAC", "1") != "0":
                    grid_tt = _time_block(
                        f"level{lvl}.grid_mac",
                        lambda grid_tt=grid_tt: ttnn.mac(grid_tt, 2.0, -1.0),
                    )
                else:
                    grid_tt = _time_block(f"level{lvl}.grid_scale", lambda grid_tt=grid_tt: ttnn.multiply(grid_tt, 2.0))
                    grid_tt = _time_block(f"level{lvl}.grid_shift", lambda grid_tt=grid_tt: ttnn.subtract(grid_tt, 1.0))
            if os.environ.get("TTNN_GRID_SAMPLE_GRID_FP32", "0") != "0":
                grid_tt = _time_block(f"level{lvl}.grid_fp32", lambda grid_tt=grid_tt: ttnn.typecast(grid_tt, ttnn.float32))
            if grid_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                grid_tt = _time_block(
                    f"level{lvl}.grid_to_row_major",
                    lambda grid_tt=grid_tt: ttnn.to_layout(grid_tt, ttnn.ROW_MAJOR_LAYOUT),
                )
            pack_points = os.environ.get("TTNN_GRID_SAMPLE_PACK_POINTS", "0") != "0" and num_points > 1
            batch_output_channels = (
                pack_points and os.environ.get("TTNN_GRID_SAMPLE_BATCH_OUTPUT_CHANNELS", "0") != "0"
            )
            if pack_points:
                grid_tt = ttnn.reshape(grid_tt, (Bh, Lq, 1, 2 * num_points))
            shard_grid = os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID", "0") != "0"
            if shard_grid:
                grid_size = self.device.compute_with_storage_grid_size()
                grid_core_x = int(os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID_X", grid_size.x))
                grid_core_y = int(os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID_Y", grid_size.y))
                grid_core_x = max(1, min(grid_core_x, int(grid_size.x)))
                grid_core_y = max(1, min(grid_core_y, int(grid_size.y)))
                grid_mem_config = ttnn.create_sharded_memory_config_(
                    grid_tt.shape,
                    ttnn.CoreGrid(x=grid_core_x, y=grid_core_y),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                grid_tt = ttnn.to_memory_config(grid_tt, grid_mem_config)

            grid_sample_kwargs = {
                "mode": "bilinear",
                "padding_mode": "zeros",
                "use_precomputed_grid": False,
                "batch_output_channels": batch_output_channels,
            }
            if os.environ.get("TTNN_GRID_SAMPLE_OUTPUT_L1", "1") != "0":
                grid_sample_kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
            out = _time_block(
                f"level{lvl}.grid_sample",
                lambda v_tt=v_tt, grid_tt=grid_tt, grid_sample_kwargs=grid_sample_kwargs: ttnn.grid_sample(
                    v_tt,
                    grid_tt,
                    **grid_sample_kwargs,
                ),
            )
            if shard_grid:
                out = _time_block(
                    f"level{lvl}.grid_sample_to_l1",
                    lambda out=out: ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG),
                )
            if batch_output_channels:
                out = ttnn.reshape(out, (Bh, Lq, num_points, int(v_tt.shape[-1])))
                out_chw = _time_block(
                    f"level{lvl}.out_permute",
                    lambda out=out: ttnn.permute(out, (0, 3, 1, 2)),
                )
            else:
                out_chw = _time_block(
                    f"level{lvl}.out_permute",
                    lambda out=out: ttnn.permute(out, (0, 3, 1, 2)),
                )
            if c_pad != 0:
                out_chw = _time_block(
                    f"level{lvl}.out_unpad",
                    lambda out_chw=out_chw: ttnn.slice(out_chw, [0, 0, 0, 0], [Bh, self.head_dim, Lq, num_points]),
                )
            # Binary ops require TILE layout; make the boundary explicit.
            if out_chw.get_layout() != ttnn.TILE_LAYOUT:
                out_chw = _time_block(
                    f"level{lvl}.out_to_tile",
                    lambda out_chw=out_chw: ttnn.to_layout(out_chw, ttnn.TILE_LAYOUT),
                )
            # Apply attention weights for this level: [B, Lq, H, P] -> [Bh, 1, Lq, P]
            aw = attn_splits[lvl]  # [B,Lq,H,P]
            aw_bh = _time_block(f"level{lvl}.attn_permute", lambda aw=aw: ttnn.permute(aw, (0, 2, 1, 3)))
            aw_bh = ttnn.reshape(aw_bh, (Bh, 1, Lq, num_points))
            if aw_bh.get_layout() != ttnn.TILE_LAYOUT:
                aw_bh = _time_block(
                    f"level{lvl}.attn_to_tile",
                    lambda aw_bh=aw_bh: ttnn.to_layout(aw_bh, ttnn.TILE_LAYOUT),
                )
            weighted = _time_block(
                f"level{lvl}.weighted",
                lambda out_chw=out_chw, aw_bh=aw_bh: ttnn.multiply(out_chw, aw_bh),
            )
            # Sum over points
            wsum = _time_block(f"level{lvl}.sum_points", lambda weighted=weighted: ttnn.sum(weighted, dim=3))

            sampled_sum = wsum if sampled_sum is None else _time_block(
                f"level{lvl}.accumulate",
                lambda sampled_sum=sampled_sum, wsum=wsum: ttnn.add(sampled_sum, wsum),
            )

        # Reshape back: (Bh, C, Lq) -> (B, H, C_head, Lq)
        out = ttnn.reshape(sampled_sum, (B, self.num_heads, self.head_dim, Lq))
        out = ttnn.permute(out, (0, 3, 1, 2))
        out = ttnn.reshape(out, (B, Lq, self.embed_dim))
        return out


class TTNNTransformerDecoderLayer(nn.Module):
    def __init__(self, layer_pt: nn.Module, device, dtype=None, layout=None, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        dtype_default = dtype if dtype is not None else ttnn.bfloat16
        self.dtype = dtype_default
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self.trace_mode = False

        self.d_model = int(layer_pt.self_attn.embed_dim)
        self.n_head = int(layer_pt.self_attn.num_heads)

        # Self-attention
        from .mha_ttnn import TTNNMHA
        self.self_attn = TTNNMHA(
            layer_pt.self_attn,
            device,
            dtype=self.dtype,
            layout=self.layout,
            weight_store=weight_store,
            registry=registry,
        )
        # Norms
        self.ln1_w, self.ln1_b = _load_norm_params(weight_store, registry, layer_pt.norm1, device, self.dtype)
        self.ln3_w, self.ln3_b = _load_norm_params(weight_store, registry, layer_pt.norm3, device, self.dtype)

        # Cross-attention via MSDeformableAttention
        self.cross_attn = TTNNMSDeformableAttention(
            layer_pt.cross_attn,
            device,
            dtype=self.dtype,
            layout=self.layout,
            weight_store=weight_store,
            registry=registry,
        )

        # Gate
        # gate is linear on concat(target, cross): 2*d_model -> 2*d_model
        self.W_gate, self.b_gate = _load_linear_params(weight_store, registry, layer_pt.gateway.gate, device, self.dtype)
        self.ln_g_w, self.ln_g_b = _load_norm_params(weight_store, registry, layer_pt.gateway.norm, device, self.dtype)

        # FFN
        self.W1, self.b1 = _load_linear_params(weight_store, registry, layer_pt.linear1, device, self.dtype)
        self.W2, self.b2 = _load_linear_params(weight_store, registry, layer_pt.linear2, device, self.dtype)
        # Activation
        self._act = ttnn.gelu if isinstance(layer_pt.activation, nn.GELU) else ttnn.relu
        self._timing_entries = None
        self._timing_prefix = None

    def _ln(self, x_tt, w, b):
        # Use larger epsilon for BF16 stability (1e-4 instead of 1e-5)
        # Use FP32 accumulation for better precision (unless low-precision forced)
        math_fidelity = self.ttnn.MathFidelity.HiFi4
        fp32_ok = True
        compute_cfg = self.ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_ok,
            packer_l1_acc=True,
        )
        return self.ttnn.layer_norm(x_tt, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=compute_cfg)

    def _linear_fp32(self, x_tt, W, b):
        """Linear with FP32 accumulation for better precision."""
        math_fidelity = self.ttnn.MathFidelity.HiFi4
        fp32_ok = True
        compute_cfg = self.ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_ok,
            packer_l1_acc=True,
        )
        return self.ttnn.linear(x_tt, W, bias=b, compute_kernel_config=compute_cfg)

    def forward(
        self,
        target: "ttnn.Tensor",
        reference_points: "ttnn.Tensor",
        value_list: List["ttnn.Tensor"],
        spatial_shapes: List[List[int]],
        attn_mask=None,
        query_pos_embed=None,
        return_ttnn: bool = False,
    ) -> "ttnn.Tensor":
        ttnn = self.ttnn
        timing_entries = self._timing_entries
        timing_prefix = self._timing_prefix
        def _time_block(name, fn):
            if timing_entries is None or timing_prefix is None:
                return fn()
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            t1 = time.perf_counter()
            timing_entries.append((f"{timing_prefix}.{name}", (t1 - t0) * 1000.0))
            return out
        def _ensure_tile(tensor):
            if tensor.get_layout() != ttnn.TILE_LAYOUT:
                if self.trace_mode:
                    raise RuntimeError("trace_mode expects TILE layout in decoder layer")
                return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
            return tensor
        x_tt = target
        if x_tt.get_layout() != self.layout:
            if self.trace_mode:
                raise RuntimeError("trace_mode expects decoder inputs in target layout")
            x_tt = ttnn.to_layout(x_tt, self.layout)
        if query_pos_embed is not None:
            q_tt = ttnn.add(_ensure_tile(x_tt), _ensure_tile(query_pos_embed))
        else:
            q_tt = _ensure_tile(x_tt)
        sa_tt = _time_block("self_attn", lambda: self.self_attn(q_tt, x_k=q_tt, x_v=x_tt, return_ttnn=True))
        x_tt = ttnn.add(_ensure_tile(x_tt), _ensure_tile(sa_tt))
        x_tt = _time_block("ln1", lambda: self._ln(x_tt, self.ln1_w, self.ln1_b))

        qpos_tt = (
            ttnn.add(_ensure_tile(x_tt), _ensure_tile(query_pos_embed))
            if query_pos_embed is not None
            else _ensure_tile(x_tt)
        )
        msda_timing = os.environ.get("TTNN_MSDA_TIMING", "0") == "1"
        if msda_timing and timing_entries is not None and timing_prefix is not None:
            self.cross_attn._timing_entries = timing_entries
            self.cross_attn._timing_prefix = f"{timing_prefix}.cross_attn"
        else:
            self.cross_attn._timing_entries = None
            self.cross_attn._timing_prefix = None
        try:
            ca_tt = _time_block(
                "cross_attn",
                lambda: self.cross_attn(qpos_tt, reference_points, value_list, spatial_shapes, return_ttnn=True),
            )
        finally:
            self.cross_attn._timing_entries = None
            self.cross_attn._timing_prefix = None
        if ca_tt.dtype != x_tt.dtype:
            ca_tt = ttnn.typecast(ca_tt, x_tt.dtype)

        gate_in_tt = ttnn.concat([x_tt, ca_tt], dim=-1)
        gates_tt = _time_block("gate_linear", lambda: self._linear_fp32(gate_in_tt, self.W_gate, self.b_gate))
        gates_tt = _time_block("gate_sigmoid", lambda: ttnn.sigmoid(gates_tt))

        B, L = x_tt.shape[0], x_tt.shape[1]
        D = self.d_model
        g1_tt = ttnn.slice(gates_tt, [0, 0, 0], [B, L, D])
        g2_tt = ttnn.slice(gates_tt, [0, 0, D], [B, L, 2 * D])
        gx_tt = _time_block(
            "gate_mix",
            lambda: ttnn.add(
                ttnn.multiply(_ensure_tile(g1_tt), _ensure_tile(x_tt)),
                ttnn.multiply(_ensure_tile(g2_tt), _ensure_tile(ca_tt)),
            ),
        )
        gx_tt = _time_block("gate_norm", lambda: self._ln(gx_tt, self.ln_g_w, self.ln_g_b))

        y_tt = _time_block("ffn_linear1", lambda: self._linear_fp32(gx_tt, self.W1, self.b1))
        y_tt = _time_block("ffn_act", lambda: self._act(y_tt))
        y_tt = _time_block("ffn_linear2", lambda: self._linear_fp32(y_tt, self.W2, self.b2))
        x_tt = ttnn.add(_ensure_tile(gx_tt), _ensure_tile(y_tt))
        x_tt = _time_block("ln3", lambda: self._ln(x_tt, self.ln3_w, self.ln3_b))
        return x_tt


class TTNNIntegral(nn.Module):
    def __init__(self, reg_max: int, device, dtype=None, layout=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        dtype_default = dtype if dtype is not None else ttnn.bfloat16
        self.dtype = dtype_default
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self.reg_max = int(reg_max)
        self.trace_ctx: Optional[_DecoderTraceContext] = None
        self.trace_mode = False

    def forward(self, x: "ttnn.Tensor", project: "ttnn.Tensor", return_ttnn: bool = True) -> "ttnn.Tensor":
        # x: [B, L, 4*(reg_max+1)]
        ttnn = self.ttnn
        if not _is_ttnn_tensor(x):
            raise TypeError("TTNNIntegral expects TTNN tensor input")
        x_tt = ttnn.to_layout(x, self.layout)
        B, L, D = x_tt.shape
        R = self.reg_max + 1
        x_tt = ttnn.reshape(x_tt, (B * L * 4, R))
        x_tt = ttnn.typecast(x_tt, ttnn.float32)
        x_tt = ttnn.fill_implicit_tile_padding(x_tt, -1e9)
        pad_tensor = None
        if self.trace_ctx is not None and self.trace_ctx.softmax_pad is not None:
            pad_tensor = self.trace_ctx.softmax_pad
        probs = _softmax_lastdim_ttnn(ttnn, x_tt, dim=-1, pad_tensor=pad_tensor, trace_mode=self.trace_mode)
        if not _is_ttnn_tensor(project):
            raise TypeError("TTNNIntegral expects TTNN project tensor")
        proj_tt = project
        compute_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        y = ttnn.matmul(probs, proj_tt, compute_kernel_config=compute_cfg)
        y = ttnn.reshape(y, (B, L, 4))
        return y


class TTNNLQE(nn.Module):
    """TTNN port of the LQE module used in DFINE decoder."""

    def __init__(self, lqe_pt: nn.Module, device, dtype, ttnn_mod, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn_mod
        self.device = device
        self.dtype = dtype
        self.k = int(getattr(lqe_pt, "k"))
        self.reg_max = int(getattr(lqe_pt, "reg_max"))
        self.reg_conf_tt = TTNNMLPWrap(
            lqe_pt.reg_conf,
            device,
            dtype,
            ttnn_mod,
            use_fp32=True,
            weight_store=weight_store,
            registry=registry,
        )
        self.trace_ctx: Optional[_DecoderTraceContext] = None
        self.trace_mode = False

    def forward(self, scores: "ttnn.Tensor", pred_corners: "ttnn.Tensor", return_ttnn: bool = True) -> "ttnn.Tensor":
        ttnn = self.ttnn
        if not _is_ttnn_tensor(pred_corners):
            raise TypeError("TTNNLQE expects TTNN pred_corners")
        pred_tt = pred_corners
        B, L, _ = pred_tt.shape
        R = self.reg_max + 1
        pred_flat = ttnn.reshape(pred_tt, (B * L * 4, R))

        # Softmax over regression bins (prefer FP32 unless low-precision forced).
        prob_tt = pred_flat
        prob_tt = ttnn.typecast(prob_tt, ttnn.float32)
        prob_tt = ttnn.fill_implicit_tile_padding(prob_tt, -1e9)
        prob_tt = _softmax_lastdim_ttnn(ttnn, prob_tt, dim=-1, trace_mode=self.trace_mode)

        # topk expects BF16; pad leading dim to satisfy N*C*H multiple-of-32 constraint.
        prob_bf16 = ttnn.typecast(prob_tt, ttnn.bfloat16)
        N = int(B * L * 4)
        pad_n = (32 - (N % 32)) % 32
        if pad_n:
            if self.trace_ctx is not None and self.trace_ctx.lqe_pad is not None:
                prob_bf16 = ttnn.concat([prob_bf16, self.trace_ctx.lqe_pad], dim=0)
            else:
                prob_bf16 = ttnn.pad(prob_bf16, [(0, pad_n), (0, 0)], value=0.0)

        topk_vals_tt, _ = ttnn.topk(prob_bf16, k=self.k, dim=-1)
        mean_tt = ttnn.mean(topk_vals_tt, dim=-1, keepdim=True)

        if pad_n:
            topk_vals_tt = ttnn.slice(topk_vals_tt, [0, 0], [N, self.k])
            mean_tt = ttnn.slice(mean_tt, [0, 0], [N, 1])

        stat_tt = ttnn.concat([topk_vals_tt, mean_tt], dim=-1)
        stat_tt = ttnn.reshape(stat_tt, (B, L, 4 * (self.k + 1)))
        quality_tt = self.reg_conf_tt(stat_tt, return_ttnn=True, keep_fp32=True)

        if not _is_ttnn_tensor(scores):
            raise TypeError("TTNNLQE expects TTNN scores")
        scores_tt = scores
        out_tt = ttnn.add(scores_tt, quality_tt)
        return out_tt
