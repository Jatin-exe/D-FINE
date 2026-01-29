
"""TTNN reimplementation of the HGNetv2 backbone used by D-FINE."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import ttnn

def _env_int(name: str, default: int, *, min_value: Optional[int] = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"Expected {name} to be an integer, got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"Expected {name} >= {min_value}, got {parsed}")
    return parsed


def _env_optional_int(name: str, *, min_value: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Expected {name} to be an integer, got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"Expected {name} >= {min_value}, got {parsed}")
    return parsed


# Production defaults: fast runtime, model cache, and strict fallback handling.
ttnn.CONFIG.enable_fast_runtime_mode = True
ttnn.CONFIG.enable_model_cache = True
ttnn.CONFIG.throw_exception_on_fallback = True


@dataclass
class TTActivation:
    tensor: "ttnn.Tensor"
    batch: int
    height: int
    width: int
    channels: int

# on host or TT ? 
def fold_bn_to_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fused (weight, bias) for conv2d + batchnorm."""
    W = conv.weight.detach().clone()  # [Cout, Cin/groups, Kh, Kw]
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(W.shape[0], dtype=W.dtype)

    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()
    eps = bn.eps

    denom = torch.sqrt(var + eps)
    scale = (gamma / denom).reshape(-1, 1, 1, 1)

    W_fused = W * scale
    b_fused = beta + (b - mean) * gamma / denom
    return W_fused, b_fused


def _torch_to_tt_activation(
    x: torch.Tensor, device, dtype, layout
) -> TTActivation:
    n, c, h, w = x.shape
    nhwc = x.permute(0, 2, 3, 1).contiguous()
    tt_tensor = ttnn.from_torch(
        nhwc.detach(),
        dtype=dtype,
        layout=layout,
        device=device,
    )
    return TTActivation(tt_tensor, n, h, w, c)


def _activation_to_nhwc(act: TTActivation):
    return act.tensor


def _activation_from_nhwc(tensor, batch: int, height: int, width: int, channels: int) -> TTActivation:
    return TTActivation(tensor, batch, height, width, channels)


def _activation_to_nchw(act: TTActivation):
    nhwc = _activation_to_nhwc(act)
    return ttnn.permute(nhwc, (0, 3, 1, 2))


def _activation_from_nchw(tensor, batch: int, channels: int, height: int, width: int) -> TTActivation:
    nhwc = ttnn.permute(tensor, (0, 2, 3, 1))
    return _activation_from_nhwc(nhwc, batch, height, width, channels)


def _pad_activation(act: TTActivation, pads: Tuple[int, int, int, int], value: float = 0.0) -> TTActivation:
    left, right, top, bottom = pads
    nhwc = _activation_to_nhwc(act)
    padding = [(0, 0), (top, bottom), (left, right), (0, 0)]
    padded = ttnn.pad(nhwc, padding=padding, value=value)
    return _activation_from_nhwc(padded, act.batch, act.height + top + bottom, act.width + left + right, act.channels)


def _is_ttnn_tensor(x) -> bool:
    return isinstance(x, ttnn.Tensor)


def _concat_activations(acts: Sequence[TTActivation], trace_mode: bool = False) -> TTActivation:
    if not acts:
        raise ValueError("concat requires non-empty sequence")
    base = acts[0]
    base_dtype = base.tensor.dtype
    base_layout = base.tensor.get_layout()
    base_mem_cfg = ttnn.get_memory_config(base.tensor) if trace_mode else ttnn.DRAM_MEMORY_CONFIG
    nhwc_tensors = []
    for act in acts:
        nhwc = _activation_to_nhwc(act)
        if nhwc.dtype != base_dtype:
            if trace_mode:
                raise RuntimeError("trace_mode expects matching dtypes for concat")
            if nhwc.get_layout() != ttnn.TILE_LAYOUT:
                nhwc = ttnn.to_layout(nhwc, ttnn.TILE_LAYOUT)
            nhwc = ttnn.typecast(nhwc, base_dtype)
            if base_layout != ttnn.TILE_LAYOUT:
                nhwc = ttnn.to_layout(nhwc, base_layout)
        if nhwc.get_layout() != base_layout:
            if trace_mode:
                raise RuntimeError("trace_mode expects matching layouts for concat")
            nhwc = ttnn.to_layout(nhwc, base_layout)
        if ttnn.get_memory_config(nhwc) != base_mem_cfg:
            if trace_mode:
                raise RuntimeError("trace_mode expects matching memory configs for concat")
            nhwc = ttnn.to_memory_config(nhwc, base_mem_cfg)
        nhwc_tensors.append(nhwc)
    concatenated_nhwc = ttnn.concat(nhwc_tensors, dim=-1)
    channels = sum(act.channels for act in acts)
    return _activation_from_nhwc(concatenated_nhwc, base.batch, base.height, base.width, channels)


def _cast_activation_dtype(act: TTActivation, dtype, trace_mode: bool = False) -> TTActivation:
    if act.tensor.dtype == dtype:
        return act
    tensor = act.tensor
    if tensor.get_layout() != ttnn.TILE_LAYOUT:
        if not trace_mode:
            tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    tensor = ttnn.typecast(tensor, dtype)
    if act.tensor.get_layout() != ttnn.TILE_LAYOUT:
        if trace_mode:
            raise RuntimeError("trace_mode expects dtype cast to preserve layout")
        tensor = ttnn.to_layout(tensor, act.tensor.get_layout())
    return TTActivation(tensor, act.batch, act.height, act.width, act.channels)


def _broadcast_spatial(act: TTActivation, target_height: int, target_width: int) -> TTActivation:
    if act.height == target_height and act.width == target_width:
        return act
    nhwc = _activation_to_nhwc(act)
    repeats = (1, target_height // act.height, target_width // act.width, 1)
    expanded = ttnn.repeat(nhwc, repeats)
    return _activation_from_nhwc(expanded, act.batch, target_height, target_width, act.channels)


def _pool_output_dim(
    input_size: int, kernel: int, stride: int, padding: int, dilation: int = 1, ceil_mode: bool = False
) -> int:
    numerator = input_size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil_mode:
        numerator += stride - 1
    return numerator // stride + 1


def _conv_output_dim(input_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    numerator = input_size + 2 * padding - dilation * (kernel - 1) - 1
    return numerator // stride + 1


def _normalize_padding(padding) -> Tuple[int, ...]:
    if isinstance(padding, tuple):
        return tuple(int(p) for p in padding)
    if isinstance(padding, int):
        return (int(padding), int(padding))
    raise ValueError(f"Unsupported padding type: {padding!r}")


def _split_conv_module(conv_module: nn.Module) -> Tuple[nn.Conv2d, Optional[Tuple[int, int, int, int]]]:
    if isinstance(conv_module, nn.Conv2d):
        return conv_module, None
    if isinstance(conv_module, nn.Sequential):
        conv = None
        pad: Optional[Tuple[int, int, int, int]] = None
        for mod in conv_module:
            if isinstance(mod, nn.Conv2d):
                conv = mod
            elif isinstance(mod, nn.ZeroPad2d):
                pad = tuple(mod.padding)  # (left, right, top, bottom)
        if conv is None:
            raise ValueError("Sequential conv module missing Conv2d")
        return conv, pad
    raise TypeError(f"Unsupported conv container: {type(conv_module)}")


class TTNNConv2d(nn.Module):
    """TTNN Conv wrapper with pre-fused weights and bias (tile layout)."""

    def __init__(
        self,
        device,
        weight: torch.Tensor | "ttnn.Tensor" | None,
        bias: torch.Tensor | "ttnn.Tensor" | None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, ...] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        dtype=None,
        layout=None,
        activation: Optional[str] = None,
        use_fp32: bool = False,
        weight_store=None,
        weight_key: Optional[str] = None,
    ):
        super().__init__()
        ttnn_mod = ttnn
        self.ttnn = ttnn_mod
        self.device = device
        self.stride = tuple(int(s) for s in stride)
        self.padding = tuple(int(p) for p in padding)
        self.dilation = tuple(int(d) for d in dilation)
        self.groups = int(groups)
        self.use_fp32 = use_fp32
        dtype_default = ttnn.float32 if use_fp32 else (dtype if dtype is not None else ttnn.bfloat16)
        self.dtype = dtype_default
        # Prefer ROW_MAJOR layout for conv outputs to avoid tile sharding constraints
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self._default_layout = self.layout
        self.activation = activation
        self.dump_dir: Optional[Path] = None
        self.dump_name: Optional[str] = None
        self._dumped = False
        # Tile-only input path is unstable for depthwise stride-2 today; keep ROW_MAJOR for conv inputs by default.
        self.force_tile_input = False
        self._default_force_tile_input = self.force_tile_input
        self.use_linear = False
        self.linear_weight = None
        self.linear_bias = None
        self.linear_compute_config = None

        self.weight_store = weight_store
        self.weight_key = weight_key
        if weight_store is None:
            raise ValueError("TTNNConv2d requires a weight store (set TTNN_WEIGHT_DIR).")
        if weight_key is None:
            raise ValueError("TTNNConv2d requires a weight key when using weight store.")
        if weight_store.mode == "load":
            # Conv weights must stay in host storage for prepare_conv_weights.
            weight = weight_store.load_tensor(f"{weight_key}.weight", device=None)
            bias_key = f"{weight_key}.bias"
            try:
                bias = weight_store.load_tensor(bias_key, device=None)
            except KeyError:
                bias = None
        elif weight_store.mode == "save":
            if weight is None:
                raise ValueError("weight_store=save requires torch weights for export")
        else:
            raise ValueError(f"Unsupported weight_store mode: {weight_store.mode}")
        if weight is None:
            raise ValueError("TTNNConv2d requires weights; provide weight_store or torch weights")
        if not _is_ttnn_tensor(weight):
            weight = weight.detach().clone()
            bias = bias.detach().clone() if bias is not None else None
        # Reshape bias to 4D NHWC [1,1,1,Cout] as expected by TTNN conv2d
        if _is_ttnn_tensor(weight):
            weight_shape = list(weight.shape)
            self.in_channels = int(weight_shape[1]) * self.groups
            self.out_channels = int(weight_shape[0])
            self.kernel_size = (int(weight_shape[-2]), int(weight_shape[-1]))
        else:
            self.in_channels = weight.shape[1] * self.groups
            self.out_channels = weight.shape[0]
            self.kernel_size = tuple(weight.shape[-2:])
        if bias is not None and not _is_ttnn_tensor(bias):
            bias = bias.reshape(1, 1, 1, -1)

        if len(self.padding) == 2:
            self.pad_hw = (self.padding[0], self.padding[1])
        elif len(self.padding) == 4:
            self.pad_hw = (self.padding[0], self.padding[2])
        else:
            raise ValueError(f"Unsupported padding spec {self.padding}")

        # Keep a host copy in ROW_MAJOR; device-prepared weights are cached later
        if _is_ttnn_tensor(weight):
            self.weight_host = ttnn_mod.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        else:
            self.weight_host = ttnn_mod.from_torch(weight, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if bias is not None:
            if _is_ttnn_tensor(bias):
                self.bias_host = ttnn_mod.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
            else:
                self.bias_host = ttnn_mod.from_torch(bias, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            self.bias_host = None
        self.weight = self.weight_host
        self.bias = self.bias_host
        if self.weight_store is not None and self.weight_key is not None:
            if self.weight_store.mode == "save":
                self.weight_store.save_tensor(f"{self.weight_key}.weight", self.weight_host)
                if self.bias_host is not None:
                    self.weight_store.save_tensor(f"{self.weight_key}.bias", self.bias_host)

        conv_activation = None
        if activation == "relu":
            conv_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        # Configure shard layout and conv settings based on conv type
        is_depthwise = (self.groups == self.in_channels) and self.groups > 1
        is_depthwise_strided = is_depthwise and any(s > 1 for s in self.stride)
        self.is_depthwise_strided = is_depthwise_strided  # Store for forward pass
        self.pointwise_chunk_size = 0

        # Default shard layout - HEIGHT_SHARDED is most stable for depthwise.
        if is_depthwise_strided:
            # HEIGHT_SHARDED provides better numerical accuracy for depthwise stride>1
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif is_depthwise:
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif self.groups == 1 and self.out_channels >= 512 and self.in_channels >= 512:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

        # Mesh devices rely on halo exchange; keep sharded layouts and configure fabric instead.
        self._default_shard_layout = shard_layout
        self._active_shard_layout = shard_layout

        # Chunked conv paths rely on host-side slicing; enable only if pre-chunked weights are available.
        self.disable_chunking = False
        if os.environ.get("TTNN_DISABLE_CHUNKED_CONV") == "1":
            self.disable_chunking = True

        # Depthwise chunk metadata (optional, used for weight-store load)
        self.dw_chunk_size = None
        self.dw_weight_chunks = None
        self.dw_bias_chunks = None
        self.dw_weight_chunks_prepared = None
        self.dw_bias_chunks_prepared = None
        self.pw_weight_chunks_prepared = None
        self.pw_bias_chunks_prepared = None

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight.dtype,
            activation=conv_activation,
            output_layout=self.layout,
            shard_layout=shard_layout,
        )
        # Store conv config tensors in DRAM to reduce L1_SMALL pressure on large inputs.
        self.conv_config.config_tensors_in_dram = True

        # Depthwise stride>1: disable kernel stride folding and tune act block size
        if is_depthwise_strided:
            self.conv_config.enable_kernel_stride_folding = False
            # Use smaller block size to reduce L1 pressure for depthwise stride>1
            self.conv_config.act_block_h_override = 32
            # Keep ROW_MAJOR input for depthwise strided
            self.force_tile_input = False

        # Configure FP32 accumulation: enable more aggressively for precision
        fp32_dest_ok = False
        packer_l1_ok = False
        if is_depthwise:
            # Enable FP32 for ALL depthwise convolutions for better precision
            fp32_dest_ok = True
            packer_l1_ok = True
        elif self.groups == 1:
            # Enable FP32 for pointwise convs to improve accuracy on large channels
            if self.kernel_size == (1, 1):
                fp32_dest_ok = True
            else:
                # Enable FP32 for smaller plain convolutions
                fp32_dest_ok = self.out_channels <= 512 and self.in_channels <= 512
        math_fidelity = ttnn.MathFidelity.HiFi4
        if not hasattr(ttnn, "init_device_compute_kernel_config"):
            raise RuntimeError("TTNN build is missing init_device_compute_kernel_config; update TTNN.")
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_ok,
            packer_l1_acc=packer_l1_ok,  # Changed from hardcoded False
        )
        # Prefer precise math by default; allow override via TTNN_MATH_APPROX=1
        if self.compute_config is not None and hasattr(self.compute_config, "math_approx_mode"):
            self.compute_config.math_approx_mode = False
        # Output memory config (can be switched to L1 for speed where safe)
        self.memory_config = self.ttnn.DRAM_MEMORY_CONFIG
        # Track whether weights have been prepared on device to avoid reprocessing
        self._weights_prepared = False
        self.trace_mode = False

        # Pointwise chunking and linear fallbacks are intentionally disabled in production.
        self.pointwise_chunk_size = 0
        self.use_linear = False

        # Pre-slice depthwise weights for chunked path when using weight store.
        if self.weight_store is not None and self.weight_key is not None:
            self._init_depthwise_chunks(weight, bias)

    def set_debug_dump(self, name: str, dump_dir: Optional[Path]):
        if dump_dir is None:
            return
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.dump_name = name

    def use_l1_output(self):
        """Enable L1 output memory for this conv (faster, riskier)."""
        self.memory_config = self.ttnn.L1_MEMORY_CONFIG

    def _select_output_memory_config(self, out_height: int, out_width: int, batch: int) -> object:
        """Select output memory config based on size heuristics (prefers L1 when safe)."""
        # Default: respect configured memory config.
        output_mem_cfg = self.memory_config
        if os.environ.get("TTNN_AUTO_L1_OUTPUT", "1") != "1":
            return output_mem_cfg
        # Skip L1 auto-selection for depthwise stride>1 (more L1 pressure).
        if self.is_depthwise_strided:
            return output_mem_cfg
        # Conservative size threshold in bytes (BF16 = 2 bytes).
        max_bytes = _env_int("TTNN_AUTO_L1_MAX_BYTES", 256 * 1024, min_value=1)
        out_bytes = int(batch) * int(out_height) * int(out_width) * int(self.out_channels) * 2
        if out_bytes <= max_bytes:
            output_mem_cfg = self.ttnn.L1_MEMORY_CONFIG
        return output_mem_cfg

    def _init_depthwise_chunks(self, weight, bias) -> None:
        if not (self.is_depthwise_strided and self.groups == self.in_channels):
            return
        if self.weight_store is None or self.weight_key is None:
            return
        meta_key = f"{self.weight_key}.dw_chunk_meta"
        chunk_size = None
        if self.weight_store.mode == "load":
            meta = self.weight_store.get_meta(meta_key)
            if meta is None:
                # Stored weights have no chunk metadata; keep chunking disabled.
                self.disable_chunking = True
                return
            chunk_size_raw = meta.get("chunk_size", 0)
            try:
                chunk_size = int(chunk_size_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid chunk_size in {meta_key}: {chunk_size_raw}") from exc
            if chunk_size <= 0:
                self.disable_chunking = True
                return
            num_chunks_raw = meta.get("num_chunks", 0)
            try:
                num_chunks = int(num_chunks_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid num_chunks in {meta_key}: {num_chunks_raw}") from exc
            if num_chunks <= 0:
                self.disable_chunking = True
                return
            self.dw_chunk_size = chunk_size
            self.dw_weight_chunks = []
            self.dw_bias_chunks = []
            for idx in range(num_chunks):
                w_key = f"{self.weight_key}.dw_chunk.{idx}.weight"
                b_key = f"{self.weight_key}.dw_chunk.{idx}.bias"
                w_tt = self.weight_store.load_tensor(w_key, device=None)
                if w_tt.get_layout() != self.ttnn.ROW_MAJOR_LAYOUT:
                    w_tt = self.ttnn.to_layout(w_tt, self.ttnn.ROW_MAJOR_LAYOUT)
                self.dw_weight_chunks.append(w_tt)
                try:
                    b_tt = self.weight_store.load_tensor(b_key, device=None)
                    if b_tt.get_layout() != self.ttnn.ROW_MAJOR_LAYOUT:
                        b_tt = self.ttnn.to_layout(b_tt, self.ttnn.ROW_MAJOR_LAYOUT)
                except KeyError:
                    b_tt = None
                self.dw_bias_chunks.append(b_tt)
            self.disable_chunking = False
            return

        # Save mode: precompute and store chunk weights.
        chunk_size_raw = os.environ.get("TTNN_DEPTHWISE_CHUNK_SIZE", "256")
        try:
            chunk_size = int(chunk_size_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid TTNN_DEPTHWISE_CHUNK_SIZE={chunk_size_raw}") from exc
        if chunk_size <= 0 or self.in_channels <= chunk_size:
            return
        weight_src = weight if isinstance(weight, torch.Tensor) else None
        bias_src = bias if isinstance(bias, torch.Tensor) else None
        if bias_src is not None and hasattr(bias_src, "dim") and bias_src.dim() > 1:
            bias_src = bias_src.reshape(-1)
        if weight_src is None:
            raise RuntimeError("Depthwise chunk export requires torch weights in save mode.")
        num_chunks = int((self.in_channels + chunk_size - 1) // chunk_size)
        for idx in range(num_chunks):
            c_start = idx * chunk_size
            c_end = min(self.in_channels, c_start + chunk_size)
            w_chunk = weight_src[c_start:c_end, :, :, :]
            w_tt = self.ttnn.from_torch(w_chunk, dtype=self.dtype, layout=self.ttnn.ROW_MAJOR_LAYOUT)
            self.weight_store.save_tensor(f"{self.weight_key}.dw_chunk.{idx}.weight", w_tt)
            if bias_src is not None:
                b_chunk = bias_src[c_start:c_end]
                b_tt = self.ttnn.from_torch(
                    b_chunk.reshape(1, 1, 1, -1), dtype=self.dtype, layout=self.ttnn.ROW_MAJOR_LAYOUT
                )
                self.weight_store.save_tensor(f"{self.weight_key}.dw_chunk.{idx}.bias", b_tt)
        self.weight_store.save_meta(meta_key, {"chunk_size": chunk_size, "num_chunks": num_chunks})

    def _prepare_weights_once(self, act: TTActivation, x_nhwc):
        """Prepare weights/bias for the current spatial shape to avoid host-side reprocessing."""
        input_mem_cfg = self.ttnn.get_memory_config(x_nhwc)
        input_layout = x_nhwc.get_layout()
        prepared = self.ttnn.prepare_conv_weights(
            # prepare on host, then move to device in one shot
            weight_tensor=self.weight_host,
            input_memory_config=input_mem_cfg,
            input_layout=input_layout,
            weights_format="OIHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=act.batch,
            input_height=act.height,
            input_width=act.width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad_hw,
            dilation=self.dilation,
            has_bias=self.bias_host is not None,
            groups=self.groups,
            device=self.device,
            input_dtype=self.dtype,
            output_dtype=self.dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        if isinstance(prepared, tuple):
            # Some runtimes return (weight, bias); bias may be None if unchanged
            self.weight = prepared[0]
            if len(prepared) > 1 and prepared[1] is not None:
                self.bias = prepared[1]
        else:
            self.weight = prepared

        if self.bias_host is not None and (not hasattr(self.bias, "storage_type") or self.bias is self.bias_host):
            self.bias = self.ttnn.prepare_conv_bias(
                bias_tensor=self.bias_host,
                input_memory_config=input_mem_cfg,
                input_layout=input_layout,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=act.batch,
                input_height=act.height,
                input_width=act.width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.pad_hw,
                dilation=self.dilation,
                device=self.device,
                input_dtype=self.dtype,
                output_dtype=self.dtype,
                groups=self.groups,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
            )
        self._weights_prepared = True

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        if self.trace_mode:
            self.force_tile_input = False
            self.layout = self.ttnn.ROW_MAJOR_LAYOUT
        else:
            self.force_tile_input = self._default_force_tile_input
            self.layout = self._default_layout

    def forward(self, act: TTActivation) -> TTActivation:
        # Keep inputs TILE in the pipeline, but convert to ROW_MAJOR at the conv call unless explicitly overridden
        x = act
        x_nhwc = _activation_to_nhwc(x)
        desired_layout = self.ttnn.TILE_LAYOUT if self.force_tile_input else self.ttnn.ROW_MAJOR_LAYOUT
        if x_nhwc.get_layout() != desired_layout:
            if self.trace_mode:
                if x_nhwc.get_layout() not in (self.ttnn.TILE_LAYOUT, self.ttnn.ROW_MAJOR_LAYOUT):
                    raise RuntimeError("TTNNConv2d trace_mode expects inputs pre-laid-out for conv")
            else:
                x_nhwc = self.ttnn.to_layout(x_nhwc, desired_layout)
        # Note: FP32 conv paths rely on internal accumulation; optionally cast inputs to FP32 for accuracy.
        if (not self.trace_mode) and self.use_fp32 and os.environ.get("TTNN_FP32_INPUT", "0") == "1":
            min_ch = int(os.environ.get("TTNN_FP32_INPUT_MIN_CHANNELS", "0"))
            if (not min_ch) or max(self.in_channels, self.out_channels) >= min_ch:
                if x_nhwc.get_layout() != self.ttnn.TILE_LAYOUT:
                    x_tile = self.ttnn.to_layout(x_nhwc, self.ttnn.TILE_LAYOUT)
                else:
                    x_tile = x_nhwc
                x_tile = self.ttnn.typecast(x_tile, self.ttnn.float32)
                if x_nhwc.get_layout() != self.ttnn.TILE_LAYOUT:
                    x_nhwc = self.ttnn.to_layout(x_tile, self.ttnn.ROW_MAJOR_LAYOUT)
                else:
                    x_nhwc = x_tile

        # No mesh-specific padding; operate on single-device tensors only.
        orig_height = act.height
        orig_width = act.width
        padded_height = orig_height
        padded_width = orig_width
        pad_extra_h = 0
        pad_extra_w = 0
        conv_act = act
        # Precompute original output dims for cropping padded mesh outputs
        out_height_orig = _conv_output_dim(
            orig_height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
        )
        out_width_orig = _conv_output_dim(
            orig_width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
        )

        # Depthwise stride>1 with large channels can exceed L1. Chunk channels and run conv per chunk.
        if (not self.disable_chunking) and self.is_depthwise_strided and self.groups == self.in_channels:
            if self.dw_chunk_size is not None:
                chunk_size = int(self.dw_chunk_size)
            else:
                chunk_env = os.environ.get("TTNN_DEPTHWISE_CHUNK_SIZE")
                chunk_size = int(chunk_env) if chunk_env else 256
                if os.environ.get("TTNN_DEPTHWISE_CHUNK_DYNAMIC", "1") == "1":
                    if max(act.height, act.width) >= 80 and chunk_size >= 256:
                        chunk_size = 128
            if self.in_channels > chunk_size:
                out = self._forward_depthwise_chunked(conv_act, x_nhwc, chunk_size)
                if pad_extra_h or pad_extra_w:
                    out_tensor = self.ttnn.slice(
                        out.tensor,
                        slice_start=(0, 0, 0, 0),
                        slice_end=(out.batch, out_height_orig, out_width_orig, out.channels),
                    )
                    return TTActivation(out_tensor, out.batch, out_height_orig, out_width_orig, out.channels)
                return out

        if (not self.disable_chunking) and self.pointwise_chunk_size and self.in_channels > self.pointwise_chunk_size:
            out = self._forward_pointwise_chunked(conv_act, x_nhwc, self.pointwise_chunk_size)
            if pad_extra_h or pad_extra_w:
                out_tensor = self.ttnn.slice(
                    out.tensor,
                    slice_start=(0, 0, 0, 0),
                    slice_end=(out.batch, out_height_orig, out_width_orig, out.channels),
                )
                return TTActivation(out_tensor, out.batch, out_height_orig, out_width_orig, out.channels)
            return out
        # Optional linear path for pointwise convs (accuracy-first)
        output_mem_cfg = self._select_output_memory_config(out_height_orig, out_width_orig, act.batch)
        if self.use_linear:
            if self.trace_mode:
                if x_nhwc.get_layout() != self.ttnn.TILE_LAYOUT:
                    raise RuntimeError("TTNNConv2d trace_mode expects TILE layout for linear path.")
                linear_in = x_nhwc
            else:
                linear_in = self.ttnn.to_layout(x_nhwc, self.ttnn.TILE_LAYOUT)
            if self.dtype == self.ttnn.float32:
                linear_in = self.ttnn.typecast(linear_in, self.ttnn.float32)
            linear_in = self.ttnn.reshape(linear_in, (act.batch, padded_height * padded_width, self.in_channels))
            if self.linear_compute_config is not None:
                linear_out = self.ttnn.linear(
                    linear_in, self.linear_weight, bias=self.linear_bias, compute_kernel_config=self.linear_compute_config
                )
            else:
                linear_out = self.ttnn.linear(linear_in, self.linear_weight, bias=self.linear_bias)
            output_tensor = self.ttnn.reshape(
                linear_out, (act.batch, padded_height, padded_width, self.out_channels)
            )
            if self.layout != output_tensor.get_layout():
                if self.trace_mode:
                    raise RuntimeError("TTNNConv2d trace_mode expects linear output in conv layout")
                output_tensor = self.ttnn.to_layout(output_tensor, self.layout)
            output_tensor = self.ttnn.to_memory_config(output_tensor, output_mem_cfg)
            out_height = _conv_output_dim(
                padded_height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
            )
            out_width = _conv_output_dim(
                padded_width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
            )
            return TTActivation(output_tensor, act.batch, out_height, out_width, self.out_channels)

        if not self._weights_prepared:
            self._prepare_weights_once(conv_act, x_nhwc)
        pad_for_conv = self.pad_hw

        try:
            result = self.ttnn.conv2d(
                input_tensor=x_nhwc,
                weight_tensor=self.weight,
                bias_tensor=self.bias,  # apply folded bias during convolution
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=x.batch,
                input_height=padded_height,
                input_width=padded_width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=pad_for_conv,
                dilation=self.dilation,
                groups=self.groups,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=output_mem_cfg,
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=self.dtype,
            )
            # result can be tensor or (tensor, (weights,bias)); keep tensor as NHWC
            if isinstance(result, tuple) and len(result) == 2:
                output_tensor, meta = result
                if (
                    isinstance(meta, tuple)
                    and len(meta) == 2
                    and hasattr(meta[0], "get_layout")
                ):
                    w_new, b_new = meta
                    if w_new is not None:
                        self.weight = w_new
                        self._weights_prepared = True
                    if b_new is not None:
                        self.bias = b_new
            else:
                output_tensor = result

        except Exception as e:
            raise RuntimeError(
                (
                    "TTNNConv2d failed with shapes: "
                    f"in=[N={act.batch},H={act.height},W={act.width},C={act.channels}], "
                    f"weight=[OC={self.out_channels},IC_per_group={self.in_channels // max(self.groups,1)},KH={self.kernel_size[0]},KW={self.kernel_size[1]}], "
                    f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}"
                )
            ) from e
        out_height = _conv_output_dim(
            padded_height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
        )
        out_width = _conv_output_dim(
            padded_width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
        )
        # Ensure NHWC shape
        output_tensor = self.ttnn.reshape(output_tensor, (act.batch, out_height, out_width, self.out_channels))
        # Crop back to original output size if we padded input
        if pad_extra_h or pad_extra_w:
            output_tensor = self.ttnn.slice(
                output_tensor,
                slice_start=(0, 0, 0, 0),
                slice_end=(act.batch, out_height_orig, out_width_orig, self.out_channels),
            )
            out_height = out_height_orig
            out_width = out_width_orig

        return TTActivation(output_tensor, act.batch, out_height, out_width, self.out_channels)

    def _forward_depthwise_chunked(self, act: TTActivation, x_nhwc, chunk_size: int) -> TTActivation:
        """Run depthwise conv in channel chunks to reduce per-core L1 usage."""
        chunks = self.ttnn.split(x_nhwc, chunk_size, dim=3)
        outputs = []
        c_start = 0
        out_height = _conv_output_dim(
            act.height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
        )
        out_width = _conv_output_dim(
            act.width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
        )
        for idx, x_chunk in enumerate(chunks):
            c_end = c_start + int(x_chunk.shape[3])
            if self.dw_weight_chunks is not None:
                weight_host = self.dw_weight_chunks[idx]
                bias_host = self.dw_bias_chunks[idx] if self.dw_bias_chunks is not None else None
            else:
                weight_host = self.ttnn.slice(
                    self.weight_host,
                    [c_start, 0, 0, 0],
                    [c_end, 1, self.kernel_size[0], self.kernel_size[1]],
                )
                if self.bias_host is not None:
                    bias_host = self.ttnn.slice(
                        self.bias_host,
                        [0, 0, 0, c_start],
                        [1, 1, 1, c_end],
                    )
                else:
                    bias_host = None
            input_mem_cfg = self.ttnn.get_memory_config(x_chunk)
            input_layout = x_chunk.get_layout()
            if self.dw_weight_chunks_prepared is None or len(self.dw_weight_chunks_prepared) != len(chunks):
                self.dw_weight_chunks_prepared = [None] * len(chunks)
                self.dw_bias_chunks_prepared = [None] * len(chunks)
            weight_prepared = self.dw_weight_chunks_prepared[idx]
            bias_prepared = self.dw_bias_chunks_prepared[idx]
            if weight_prepared is None:
                weight_prepared = self.ttnn.prepare_conv_weights(
                    weight_tensor=weight_host,
                    input_memory_config=input_mem_cfg,
                    input_layout=input_layout,
                    weights_format="OIHW",
                    in_channels=int(x_chunk.shape[3]),
                    out_channels=int(x_chunk.shape[3]),
                    batch_size=act.batch,
                    input_height=act.height,
                    input_width=act.width,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.pad_hw,
                    dilation=self.dilation,
                    has_bias=bias_host is not None,
                    groups=int(x_chunk.shape[3]),
                    device=self.device,
                    input_dtype=self.dtype,
                    output_dtype=self.dtype,
                    conv_config=self.conv_config,
                    compute_config=self.compute_config,
                )
                bias_prepared = None
                if bias_host is not None:
                    bias_prepared = self.ttnn.prepare_conv_bias(
                        bias_tensor=bias_host,
                        input_memory_config=input_mem_cfg,
                        input_layout=input_layout,
                        in_channels=int(x_chunk.shape[3]),
                        out_channels=int(x_chunk.shape[3]),
                        batch_size=act.batch,
                        input_height=act.height,
                        input_width=act.width,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.pad_hw,
                        dilation=self.dilation,
                        device=self.device,
                        input_dtype=self.dtype,
                        output_dtype=self.dtype,
                        groups=int(x_chunk.shape[3]),
                        conv_config=self.conv_config,
                        compute_config=self.compute_config,
                    )
                self.dw_weight_chunks_prepared[idx] = weight_prepared
                self.dw_bias_chunks_prepared[idx] = bias_prepared
            result = self.ttnn.conv2d(
                input_tensor=x_chunk,
                weight_tensor=weight_prepared,
                bias_tensor=bias_prepared,
                device=self.device,
                in_channels=int(x_chunk.shape[3]),
                out_channels=int(x_chunk.shape[3]),
                batch_size=act.batch,
                input_height=act.height,
                input_width=act.width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.pad_hw,
                dilation=self.dilation,
                groups=int(x_chunk.shape[3]),
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=self.memory_config,
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=self.dtype,
            )
            if isinstance(result, tuple) and len(result) == 2:
                output_tensor = result[0]
            else:
                output_tensor = result
            output_tensor = self.ttnn.reshape(output_tensor, (act.batch, out_height, out_width, int(x_chunk.shape[3])))
            outputs.append(TTActivation(output_tensor, act.batch, out_height, out_width, int(x_chunk.shape[3])))
            c_start = c_end
        return _concat_activations(outputs, trace_mode=self.trace_mode)

    def _forward_pointwise_chunked(self, act: TTActivation, x_nhwc, chunk_size: int) -> TTActivation:
        """Run pointwise conv by splitting input channels and accumulating outputs."""
        chunks = self.ttnn.split(x_nhwc, chunk_size, dim=3)
        out_height = _conv_output_dim(
            act.height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
        )
        out_width = _conv_output_dim(
            act.width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
        )
        accum = None
        c_start = 0
        for idx, x_chunk in enumerate(chunks):
            c_end = c_start + int(x_chunk.shape[3])
            weight_host = self.ttnn.slice(
                self.weight_host,
                [0, c_start, 0, 0],
                [self.out_channels, c_end, self.kernel_size[0], self.kernel_size[1]],
            )
            if idx == 0 and self.bias_host is not None:
                bias_host = self.bias_host
            else:
                bias_host = None
            input_mem_cfg = self.ttnn.get_memory_config(x_chunk)
            input_layout = x_chunk.get_layout()
            if self.pw_weight_chunks_prepared is None or len(self.pw_weight_chunks_prepared) != len(chunks):
                self.pw_weight_chunks_prepared = [None] * len(chunks)
                self.pw_bias_chunks_prepared = [None] * len(chunks)
            weight_prepared = self.pw_weight_chunks_prepared[idx]
            bias_prepared = self.pw_bias_chunks_prepared[idx]
            if weight_prepared is None:
                weight_prepared = self.ttnn.prepare_conv_weights(
                    weight_tensor=weight_host,
                    input_memory_config=input_mem_cfg,
                    input_layout=input_layout,
                    weights_format="OIHW",
                    in_channels=int(x_chunk.shape[3]),
                    out_channels=self.out_channels,
                    batch_size=act.batch,
                    input_height=act.height,
                    input_width=act.width,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.pad_hw,
                    dilation=self.dilation,
                    has_bias=bias_host is not None,
                    groups=self.groups,
                    device=self.device,
                    input_dtype=self.dtype,
                    output_dtype=self.dtype,
                    conv_config=self.conv_config,
                    compute_config=self.compute_config,
                )
                bias_prepared = None
                if bias_host is not None:
                    bias_prepared = self.ttnn.prepare_conv_bias(
                        bias_tensor=bias_host,
                        input_memory_config=input_mem_cfg,
                        input_layout=input_layout,
                        in_channels=int(x_chunk.shape[3]),
                        out_channels=self.out_channels,
                        batch_size=act.batch,
                        input_height=act.height,
                        input_width=act.width,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.pad_hw,
                        dilation=self.dilation,
                        device=self.device,
                        input_dtype=self.dtype,
                        output_dtype=self.dtype,
                        groups=self.groups,
                        conv_config=self.conv_config,
                        compute_config=self.compute_config,
                    )
                self.pw_weight_chunks_prepared[idx] = weight_prepared
                self.pw_bias_chunks_prepared[idx] = bias_prepared
            result = self.ttnn.conv2d(
                input_tensor=x_chunk,
                weight_tensor=weight_prepared,
                bias_tensor=bias_prepared,
                device=self.device,
                in_channels=int(x_chunk.shape[3]),
                out_channels=self.out_channels,
                batch_size=act.batch,
                input_height=act.height,
                input_width=act.width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.pad_hw,
                dilation=self.dilation,
                groups=self.groups,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=self.memory_config,
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=self.dtype,
            )
            if isinstance(result, tuple) and len(result) == 2:
                out_tensor = result[0]
            else:
                out_tensor = result
            out_tensor = self.ttnn.reshape(out_tensor, (act.batch, out_height, out_width, self.out_channels))
            if accum is None:
                accum = out_tensor
            else:
                if self.trace_mode:
                    if accum.get_layout() != out_tensor.get_layout():
                        raise RuntimeError("trace_mode expects matching layouts for pointwise chunk accumulation")
                    accum = self.ttnn.add(accum, out_tensor)
                else:
                    if accum.get_layout() != self.ttnn.TILE_LAYOUT:
                        accum_tile = self.ttnn.to_layout(accum, self.ttnn.TILE_LAYOUT)
                    else:
                        accum_tile = accum
                    if out_tensor.get_layout() != self.ttnn.TILE_LAYOUT:
                        out_tile = self.ttnn.to_layout(out_tensor, self.ttnn.TILE_LAYOUT)
                    else:
                        out_tile = out_tensor
                    summed = self.ttnn.add(accum_tile, out_tile)
                    # Preserve original layout for downstream ops
                    if accum.get_layout() != self.ttnn.TILE_LAYOUT:
                        accum = self.ttnn.to_layout(summed, accum.get_layout())
                    else:
                        accum = summed
            c_start = c_end
        if accum is None:
            raise RuntimeError("pointwise chunked conv produced no output")
        return TTActivation(accum, act.batch, out_height, out_width, self.out_channels)


class TTNNLearnableAffine(nn.Module):
    def __init__(self, lab_pt: nn.Module, weight_store=None, registry=None):
        super().__init__()
        self.trace_mode = False
        key = None
        if registry is not None:
            name = registry.name_of(lab_pt)
            if name:
                key = f"{name}.lab"
        if weight_store is not None and weight_store.mode == "load" and key is None:
            raise RuntimeError("Missing weight store key for LearnableAffine in load mode.")
        if weight_store is not None and key is not None and weight_store.mode == "load":
            meta = weight_store.get_meta(key)
            if meta is None:
                raise KeyError(f"Missing lab meta in weight store: {key}")
            self.scale = float(meta["scale"])
            self.bias = float(meta["bias"])
        else:
            self.scale = float(lab_pt.scale.detach().item())
            self.bias = float(lab_pt.bias.detach().item())
            if weight_store is not None and key is not None and weight_store.mode == "save":
                weight_store.save_meta(key, {"scale": self.scale, "bias": self.bias})

    def forward(self, act: TTActivation) -> TTActivation:
        tens = act.tensor
        if (not self.trace_mode) and tens.get_layout() != ttnn.TILE_LAYOUT:
            tens = ttnn.to_layout(tens, ttnn.TILE_LAYOUT)
        scaled = ttnn.multiply(tens, self.scale)
        shifted = ttnn.add(scaled, self.bias)
        if (not self.trace_mode) and shifted.get_layout() != act.tensor.get_layout():
            shifted = ttnn.to_layout(shifted, act.tensor.get_layout())
        return TTActivation(shifted, act.batch, act.height, act.width, act.channels)

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)


class TTNNConvBNAct(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn
        self.trace_mode = False
        conv_pt, explicit_pad = _split_conv_module(module_pt.conv)
        disable_bn_fusion = os.environ.get("TTNN_DISABLE_BN_FUSION") == "1"
        self.use_bn = disable_bn_fusion
        self.bn_scale = None
        self.bn_shift = None
        self.bn_scale_rm = None
        self.bn_shift_rm = None
        weight_key = None
        if registry is not None:
            conv_name = registry.name_of(conv_pt)
            if conv_name:
                weight_key = f"{conv_name}.fused"
        weight = None
        bias = None
        if weight_store is not None and weight_key is not None and weight_store.mode == "load":
            if disable_bn_fusion:
                raise RuntimeError("BN fusion disabled but weight store provided; export BN scale/shift if needed.")
        else:
            if disable_bn_fusion:
                weight = conv_pt.weight.detach().clone()
                bias = conv_pt.bias.detach().clone() if conv_pt.bias is not None else None
                bn = module_pt.bn
                gamma = bn.weight.detach().clone()
                beta = bn.bias.detach().clone()
                mean = bn.running_mean.detach().clone()
                var = bn.running_var.detach().clone()
                eps = bn.eps
                denom = torch.sqrt(var + eps)
                scale = (gamma / denom).reshape(1, 1, 1, -1)
                shift = (beta - mean * gamma / denom).reshape(1, 1, 1, -1)
                # Store BN params as tile tensors for elementwise ops
                self.bn_scale = self.ttnn.from_torch(scale, dtype=dtype, layout=self.ttnn.TILE_LAYOUT, device=device)
                self.bn_shift = self.ttnn.from_torch(shift, dtype=dtype, layout=self.ttnn.TILE_LAYOUT, device=device)
                # Row-major versions for trace-mode path
                self.bn_scale_rm = self.ttnn.from_torch(scale, dtype=dtype, layout=self.ttnn.ROW_MAJOR_LAYOUT, device=device)
                self.bn_shift_rm = self.ttnn.from_torch(shift, dtype=dtype, layout=self.ttnn.ROW_MAJOR_LAYOUT, device=device)
            else:
                weight, bias = fold_bn_to_conv(conv_pt, module_pt.bn)
        padding = _normalize_padding(conv_pt.padding)
        use_fp32 = False
        conv_layout = layout
        force_fp32 = os.environ.get("TTNN_FORCE_FP32") == "1"
        use_lab_flag = getattr(module_pt, "use_lab", False)
        if conv_pt.kernel_size == (1, 1) and conv_pt.in_channels >= 512 and conv_pt.out_channels >= 512:
            use_fp32 = os.environ.get("TTNN_DISABLE_FP32_POINTWISE") != "1"
            # Prefer row-major outputs for large pointwise convs to stabilize accuracy
            if os.environ.get("TTNN_POINTWISE_ROW_MAJOR") != "0":
                conv_layout = ttnn.ROW_MAJOR_LAYOUT
        if force_fp32 or (use_lab_flag and os.environ.get("TTNN_FP32_FOR_LAB") == "1"):
            use_fp32 = True
        self.conv = TTNNConv2d(
            device,
            weight,
            bias,
            stride=tuple(conv_pt.stride),
            padding=padding,
            dilation=tuple(conv_pt.dilation),
            groups=conv_pt.groups,
            dtype=dtype,
            layout=conv_layout,
            use_fp32=use_fp32,
            weight_store=weight_store,
            weight_key=weight_key,
        )
        self.pre_pad = explicit_pad
        self.use_act = module_pt.use_act
        self.use_lab = module_pt.use_act and getattr(module_pt, "use_lab", False)
        self.lab = TTNNLearnableAffine(module_pt.lab, weight_store=weight_store, registry=registry) if self.use_lab else None

    def enable_l1(self):
        # Opt-in to faster L1 outputs when safe
        if hasattr(self, "conv") and hasattr(self.conv, "use_l1_output"):
            self.conv.use_l1_output()

    def forward(self, act: TTActivation) -> TTActivation:
        x = act
        if self.pre_pad is not None:
            x = _pad_activation(x, self.pre_pad)
        x = self.conv(x)
        if self.use_bn and self.bn_scale is not None and self.bn_shift is not None:
            if self.trace_mode:
                layout = x.tensor.get_layout()
                if layout == self.ttnn.TILE_LAYOUT:
                    scale = self.bn_scale
                    shift = self.bn_shift
                else:
                    scale = self.bn_scale_rm
                    shift = self.bn_shift_rm
                    if scale is None or shift is None:
                        raise RuntimeError("trace_mode expects BN params pre-laid-out for row-major")
                y = self.ttnn.multiply(x.tensor, scale)
                y = self.ttnn.add(y, shift)
            else:
                x_tensor_tile = self.ttnn.to_layout(x.tensor, self.ttnn.TILE_LAYOUT)
                y = self.ttnn.multiply(x_tensor_tile, self.bn_scale)
                y = self.ttnn.add(y, self.bn_shift)
                y = self.ttnn.to_layout(y, x.tensor.get_layout())
            x = TTActivation(y, x.batch, x.height, x.width, x.channels)
        if self.use_act:
            # Apply ReLU in the current layout when supported to avoid precision loss
            force_tile_relu = os.environ.get("TTNN_FORCE_TILE_RELU") == "1"
            if self.trace_mode:
                if force_tile_relu and x.tensor.get_layout() != self.ttnn.TILE_LAYOUT:
                    raise RuntimeError("trace_mode expects TILE layout for forced tile ReLU")
                activated = self.ttnn.relu(x.tensor)
            else:
                if x.tensor.get_layout() == self.ttnn.ROW_MAJOR_LAYOUT and not force_tile_relu:
                    activated = self.ttnn.relu(x.tensor)
                else:
                    x_tensor_tile = self.ttnn.to_layout(x.tensor, self.ttnn.TILE_LAYOUT)
                    activated = self.ttnn.relu(x_tensor_tile)
                    activated = self.ttnn.to_layout(activated, x.tensor.get_layout())
            x = TTActivation(activated, x.batch, x.height, x.width, x.channels)
        if self.lab is not None:
            x = self.lab(x)
        return x

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        if hasattr(self, "conv") and hasattr(self.conv, "enable_trace_mode"):
            self.conv.enable_trace_mode(enabled)
        if self.lab is not None and hasattr(self.lab, "enable_trace_mode"):
            self.lab.enable_trace_mode(enabled)


class TTNNLightConvBNAct(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout, weight_store=None, registry=None):
        super().__init__()
        self.conv1 = TTNNConvBNAct(module_pt.conv1, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.conv2 = TTNNConvBNAct(module_pt.conv2, device, dtype, layout, weight_store=weight_store, registry=registry)

    def forward(self, act: TTActivation) -> TTActivation:
        x = self.conv1(act)
        x = self.conv2(x)
        return x


class TTNNEseModule(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout, weight_store=None, registry=None):
        super().__init__()
        self.trace_mode = False
        conv_pt = module_pt.conv
        weight = None
        bias = None
        weight_key = None
        if registry is not None:
            conv_name = registry.name_of(conv_pt)
            if conv_name:
                weight_key = conv_name
        if weight_store is None or weight_key is None or weight_store.mode != "load":
            weight = conv_pt.weight.detach().clone()
            bias = conv_pt.bias.detach().clone()
        self.conv = TTNNConv2d(
            device,
            weight,
            bias,
            stride=tuple(conv_pt.stride),
            padding=_normalize_padding(conv_pt.padding),
            dilation=tuple(conv_pt.dilation),
            groups=conv_pt.groups,
            dtype=dtype,
            layout=layout,
            weight_store=weight_store,
            weight_key=weight_key,
        )

    def forward(self, act: TTActivation) -> TTActivation:
        identity = act
        nchw = _activation_to_nchw(act)
        pooled = ttnn.global_avg_pool2d(nchw)
        pooled_act = _activation_from_nchw(pooled, act.batch, act.channels, 1, 1)
        gating = self.conv(pooled_act)
        if self.trace_mode:
            gate_tensor = ttnn.sigmoid(gating.tensor)
        else:
            gate_tensor = ttnn.sigmoid(ttnn.to_layout(gating.tensor, ttnn.TILE_LAYOUT))
            gate_tensor = ttnn.to_layout(gate_tensor, gating.tensor.get_layout())
        gating = TTActivation(gate_tensor, gating.batch, gating.height, gating.width, gating.channels)
        gating = _broadcast_spatial(gating, identity.height, identity.width)
        if self.trace_mode:
            if identity.tensor.get_layout() != gating.tensor.get_layout():
                raise RuntimeError("trace_mode expects ESE gating/identity layouts to match")
            scaled = ttnn.multiply(identity.tensor, gating.tensor)
        else:
            identity_layout = identity.tensor.get_layout()
            id_tile = identity.tensor if identity_layout == ttnn.TILE_LAYOUT else ttnn.to_layout(identity.tensor, ttnn.TILE_LAYOUT)
            gate_tile = gating.tensor if gating.tensor.get_layout() == ttnn.TILE_LAYOUT else ttnn.to_layout(gating.tensor, ttnn.TILE_LAYOUT)
            scaled = ttnn.multiply(id_tile, gate_tile)
            if identity_layout != ttnn.TILE_LAYOUT:
                scaled = ttnn.to_layout(scaled, identity_layout)
        return TTActivation(scaled, identity.batch, identity.height, identity.width, identity.channels)

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        if hasattr(self, "conv") and hasattr(self.conv, "enable_trace_mode"):
            self.conv.enable_trace_mode(enabled)


class TTNNHGBlock(nn.Module):
    def __init__(self, block_pt: nn.Module, device, dtype, layout, weight_store=None, registry=None):
        super().__init__()
        self.trace_mode = False
        self.layers = nn.ModuleList()
        for layer_pt in block_pt.layers:
            if hasattr(layer_pt, "conv1") and hasattr(layer_pt, "conv2"):
                self.layers.append(TTNNLightConvBNAct(layer_pt, device, dtype, layout, weight_store=weight_store, registry=registry))
            else:
                self.layers.append(TTNNConvBNAct(layer_pt, device, dtype, layout, weight_store=weight_store, registry=registry))

        self.aggregation = nn.ModuleList()
        for agg_mod in block_pt.aggregation:
            if isinstance(agg_mod, nn.Identity):
                continue
            if hasattr(agg_mod, "conv") and hasattr(agg_mod, "bn"):
                self.aggregation.append(TTNNConvBNAct(agg_mod, device, dtype, layout, weight_store=weight_store, registry=registry))
            else:
                self.aggregation.append(TTNNEseModule(agg_mod, device, dtype, layout, weight_store=weight_store, registry=registry))

        self.residual = block_pt.residual

    def forward(self, act: TTActivation) -> TTActivation:
        identity = act
        outputs = [act]
        x = act
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = _concat_activations(outputs, trace_mode=self.trace_mode)
        for module in self.aggregation:
            x = module(x)
        if self.residual:
            if self.trace_mode:
                if identity.tensor.get_layout() != x.tensor.get_layout():
                    raise RuntimeError("trace_mode expects residual layouts to match")
                combined = ttnn.add(identity.tensor, x.tensor)
            else:
                identity_layout = identity.tensor.get_layout()
                id_tile = identity.tensor if identity_layout == ttnn.TILE_LAYOUT else ttnn.to_layout(identity.tensor, ttnn.TILE_LAYOUT)
                x_tile = x.tensor if x.tensor.get_layout() == ttnn.TILE_LAYOUT else ttnn.to_layout(x.tensor, ttnn.TILE_LAYOUT)
                combined = ttnn.add(id_tile, x_tile)
                if identity_layout != ttnn.TILE_LAYOUT:
                    combined = ttnn.to_layout(combined, identity_layout)
            x = TTActivation(combined, x.batch, x.height, x.width, x.channels)
        return x

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        for layer in self.layers:
            if hasattr(layer, "enable_trace_mode"):
                layer.enable_trace_mode(enabled)
        for module in self.aggregation:
            if hasattr(module, "enable_trace_mode"):
                module.enable_trace_mode(enabled)


class TTNNHGStage(nn.Module):
    def __init__(
        self,
        stage_pt: nn.Module,
        device,
        dtype,
        layout,
        stage_idx: int,
        debug_dump_dir: Optional[Path] = None,
        downsample_dtype=None,
        block_dtype=None,
        weight_store=None,
        registry=None,
    ):
        super().__init__()
        self.trace_mode = False
        if downsample_dtype is None:
            downsample_dtype = dtype
        if block_dtype is None:
            block_dtype = dtype
        self.downsample_dtype = downsample_dtype
        self.block_dtype = block_dtype
        self.downsample_op: Optional[nn.Module]
        if isinstance(stage_pt.downsample, nn.Identity):
            self.downsample_op = None
        else:
            self.downsample_op = TTNNConvBNAct(
                stage_pt.downsample,
                device,
                self.downsample_dtype,
                layout,
                weight_store=weight_store,
                registry=registry,
            )
            self.downsample_op.conv.set_debug_dump(f"stage{stage_idx}_down", debug_dump_dir)
        self.blocks = nn.ModuleList(
            [
                TTNNHGBlock(
                    block_pt,
                    device,
                    self.block_dtype,
                    layout,
                    weight_store=weight_store,
                    registry=registry,
                )
                for block_pt in stage_pt.blocks
            ]
        )

    def forward(self, act: TTActivation) -> TTActivation:
        x = act
        if self.downsample_op is not None:
            if x.tensor.dtype != self.downsample_dtype:
                x = _cast_activation_dtype(x, self.downsample_dtype, trace_mode=self.trace_mode)
            x = self.downsample_op(x)
        if x.tensor.dtype != self.block_dtype:
            x = _cast_activation_dtype(x, self.block_dtype, trace_mode=self.trace_mode)
        for block in self.blocks:
            x = block(x)
        return x

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        if self.downsample_op is not None and hasattr(self.downsample_op, "enable_trace_mode"):
            self.downsample_op.enable_trace_mode(enabled)
        for block in self.blocks:
            if hasattr(block, "enable_trace_mode"):
                block.enable_trace_mode(enabled)


class StemTTNN(nn.Module):
    """TTNN implementation of StemBlock in HGNetv2."""

    def __init__(self, stem_pt: nn.Module, device, dtype, layout, weight_store=None, registry=None):
        super().__init__()
        self.trace_mode = False
        self.stem1 = TTNNConvBNAct(stem_pt.stem1, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.stem2a = TTNNConvBNAct(stem_pt.stem2a, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.stem2b = TTNNConvBNAct(stem_pt.stem2b, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.stem3 = TTNNConvBNAct(stem_pt.stem3, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.stem4 = TTNNConvBNAct(stem_pt.stem4, device, dtype, layout, weight_store=weight_store, registry=registry)
        self.pool_kernel = (2, 2)
        self.pool_stride = (1, 1)
        self.pool_padding = (0, 0)
        # Enable L1 outputs for internal stem convolutions to improve performance
        self.stem2a.enable_l1()
        self.stem2b.enable_l1()
        self.stem3.enable_l1()

    def forward(self, act: TTActivation) -> TTActivation:
        x = self.stem1(act)

        y_pad = _pad_activation(x, (0, 1, 0, 1))
        x2 = self.stem2a(y_pad)
        x2 = _pad_activation(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)

        # Pool the padded tensor to mirror PyTorch StemBlock behavior
        x1 = self._max_pool(y_pad)
        if self.trace_mode:
            x1_layout = x1.tensor.get_layout()
            x2_layout = x2.tensor.get_layout()
            if x1_layout != x2_layout:
                # Align layouts for concat during trace capture.
                x2_tensor = ttnn.to_layout(x2.tensor, x1_layout)
                x2 = TTActivation(x2_tensor, x2.batch, x2.height, x2.width, x2.channels)
            # Keep existing memory config when both branches match; fall back to DRAM only if needed.
            if ttnn.get_memory_config(x1.tensor) != ttnn.get_memory_config(x2.tensor):
                x1_tensor = ttnn.to_memory_config(x1.tensor, ttnn.DRAM_MEMORY_CONFIG)
                x2_tensor = ttnn.to_memory_config(x2.tensor, ttnn.DRAM_MEMORY_CONFIG)
                x1 = TTActivation(x1_tensor, x1.batch, x1.height, x1.width, x1.channels)
                x2 = TTActivation(x2_tensor, x2.batch, x2.height, x2.width, x2.channels)
        x = _concat_activations([x1, x2], trace_mode=self.trace_mode)

        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def _max_pool(self, act: TTActivation) -> TTActivation:
        kernel = self.pool_kernel
        stride = self.pool_stride
        padding = self.pool_padding
        # MaxPool expects row-major input; convert to avoid tile-layout hangs.
        if self.trace_mode:
            input_tensor = act.tensor
            if input_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                if input_tensor.get_layout() == ttnn.TILE_LAYOUT:
                    # Allow trace-safe on-device untilize for max_pool inputs.
                    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
                else:
                    raise RuntimeError("trace_mode expects ROW_MAJOR/TILE input for max_pool")
            if ttnn.get_memory_config(input_tensor) != ttnn.DRAM_MEMORY_CONFIG:
                raise RuntimeError("trace_mode expects DRAM memory config for max_pool")
        else:
            input_tensor = ttnn.to_layout(act.tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        pooled = ttnn.max_pool2d(
            input_tensor=input_tensor,
            batch_size=act.batch,
            input_h=act.height,
            input_w=act.width,
            channels=act.channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            dilation=[1, 1],
            ceil_mode=True,
        )
        out_h = _pool_output_dim(act.height, kernel[0], stride[0], padding[0], ceil_mode=True)
        out_w = _pool_output_dim(act.width, kernel[1], stride[1], padding[1], ceil_mode=True)
        pooled_nhwc = ttnn.reshape(pooled, (act.batch, out_h, out_w, act.channels))
        return TTActivation(pooled_nhwc, act.batch, out_h, out_w, act.channels)

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        for module in (self.stem1, self.stem2a, self.stem2b, self.stem3, self.stem4):
            if hasattr(module, "enable_trace_mode"):
                module.enable_trace_mode(enabled)
            if enabled and hasattr(module, "conv") and hasattr(module.conv, "memory_config"):
                # Preserve existing L1 configs during trace unless explicitly forced.
                if os.environ.get("TTNN_TRACE_FORCE_DRAM", "0") == "1":
                    module.conv.memory_config = ttnn.DRAM_MEMORY_CONFIG


class HGNetv2TTNNManual(nn.Module):
    """TTNN implementation of the HGNetv2 backbone."""

    def __init__(
        self,
        backbone_pt: nn.Module,
        device_id: int = 0,
        debug_dump_dir: Optional[Path] = None,
        weight_store=None,
        registry=None,
    ):
        super().__init__()
        ttnn_mod = ttnn
        self.ttnn = ttnn_mod
        self.trace_mode = False
        if weight_store is None:
            raise RuntimeError("TTNN backbone requires a weight store (set TTNN_WEIGHT_DIR).")
        if weight_store is not None and registry is None:
            from .weight_store import ModuleKeyRegistry
            registry = ModuleKeyRegistry(backbone_pt)
        self.registry = registry
        # Allow tuning L1 small partition size; smaller values free more L1 for conv CBs.
        l1_small_size = _env_int("TTNN_L1_SMALL_SIZE", 32768, min_value=0)
        # Open device with the configured L1 small buffer partition size.
        available_ids = list(ttnn_mod.get_device_ids())
        print(f"[TTNN] Available device ids: {available_ids}", flush=True)
        num_cqs = _env_int("TTNN_NUM_COMMAND_QUEUES", 1, min_value=1)
        trace_region_size = _env_optional_int("TTNN_TRACE_REGION_SIZE", min_value=1)

        device_kwargs = {"l1_small_size": l1_small_size}
        if num_cqs > 1:
            device_kwargs["num_command_queues"] = num_cqs
        if trace_region_size is not None and trace_region_size > 0:
            device_kwargs["trace_region_size"] = trace_region_size

        if available_ids and device_id not in available_ids:
            device_id = available_ids[0]
        print(f"[TTNN] Opening Single Device ID: {device_id}", flush=True)
        self.device = ttnn_mod.open_device(device_id=device_id, **device_kwargs)
        base_dtype = ttnn_mod.bfloat16
        self.stem_dtype = base_dtype
        stage_count = len(backbone_pt.stages)
        stage_fp32_set = set()
        stage_fp32_env = os.environ.get("TTNN_STAGE_FP32")
        if stage_fp32_env:
            for token in stage_fp32_env.replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    stage_fp32_set.add(int(token))
                except ValueError:
                    pass
            if stage_fp32_set:
                print(f"[TTNN] Stage FP32 override enabled for stages: {sorted(stage_fp32_set)}", flush=True)
        if os.environ.get("TTNN_BACKBONE_FP32") == "1" and not stage_fp32_set:
            stage_fp32_set = set(range(stage_count))
        self.stage_dtypes = [
            (ttnn_mod.float32 if idx in stage_fp32_set else base_dtype)
            for idx in range(stage_count)
        ]
        skip_downsample_fp32 = os.environ.get("TTNN_FP32_SKIP_DOWNSAMPLE") == "1"
        if skip_downsample_fp32:
            print("[TTNN] FP32 downsample disabled; using BF16 for downsample ops.", flush=True)
        self.stage_downsample_dtypes = []
        for idx, stage_dtype in enumerate(self.stage_dtypes):
            ds_dtype = stage_dtype
            if skip_downsample_fp32 and stage_dtype == ttnn_mod.float32:
                ds_dtype = base_dtype
            self.stage_downsample_dtypes.append(ds_dtype)
        self.dtype = self.stem_dtype
        # Tile-only path
        self.input_layout = ttnn_mod.TILE_LAYOUT
        self.conv_layout = ttnn_mod.TILE_LAYOUT

        self.stem = StemTTNN(
            backbone_pt.stem,
            self.device,
            self.stem_dtype,
            self.conv_layout,
            weight_store=weight_store,
            registry=self.registry,
        )
        self.stages = nn.ModuleList(
            [
                TTNNHGStage(
                    stage_pt,
                    self.device,
                    self.stage_dtypes[idx],
                    self.conv_layout,
                    idx,
                    debug_dump_dir,
                    downsample_dtype=self.stage_downsample_dtypes[idx],
                    block_dtype=self.stage_dtypes[idx],
                    weight_store=weight_store,
                    registry=self.registry,
                )
                for idx, stage_pt in enumerate(backbone_pt.stages)
            ]
        )
        self.return_idx = tuple(backbone_pt.return_idx)

    def _cast_activation(self, act: TTActivation, dtype) -> TTActivation:
        return _cast_activation_dtype(act, dtype, trace_mode=self.trace_mode)

    def _to_ttnn(self, x: torch.Tensor) -> TTActivation:
        return _torch_to_tt_activation(x, self.device, self.dtype, self.input_layout)

    def enable_trace_mode(self, enabled: bool = True) -> None:
        self.trace_mode = bool(enabled)
        if enabled:
            self.input_layout = ttnn.ROW_MAJOR_LAYOUT
        else:
            self.input_layout = ttnn.TILE_LAYOUT
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "enable_trace_mode"):
                module.enable_trace_mode(enabled)

    def forward(self, x) -> List[TTActivation]:
        if isinstance(x, TTActivation):
            act = x
        else:
            act = self._to_ttnn(x)
        outputs: List[TTActivation] = []
        act = self.stem(act)
        for idx, stage in enumerate(self.stages):
            stage_dtype = self.stage_downsample_dtypes[idx]
            if act.tensor.dtype != stage_dtype:
                act = self._cast_activation(act, stage_dtype)
            act = stage(act)
            if idx in self.return_idx:
                outputs.append(act)
        return outputs

    def close(self):
        self.ttnn.close_device(self.device)
