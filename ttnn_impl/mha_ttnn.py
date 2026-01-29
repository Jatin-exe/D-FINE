from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import ttnn


class TTNNMHA(nn.Module):
    """TTNN-backed Multi-Head Attention using SDPA + linear layers.

    Mirrors torch.nn.MultiheadAttention (batch_first=True, no dropout) for inference.
    """

    def __init__(self, mha_pt: nn.MultiheadAttention, device, dtype=None, layout=None, weight_store=None, registry=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.embed_dim = int(mha_pt.embed_dim)
        self.num_heads = int(mha_pt.num_heads)
        assert getattr(mha_pt, "batch_first", False), "Expected batch_first=True"
        assert (self.embed_dim % self.num_heads) == 0
        self.head_dim = self.embed_dim // self.num_heads
        dtype_default = dtype if dtype is not None else ttnn.bfloat16
        self.dtype = dtype_default
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT

        if weight_store is None:
            raise RuntimeError("TTNNMHA requires a weight store (set TTNN_WEIGHT_DIR).")
        base_key = None
        if registry is not None:
            name = registry.name_of(mha_pt)
            if name:
                base_key = name
        if base_key is None:
            raise RuntimeError("Missing weight store key for MHA module.")

        def _load_tt(key: str, dtype=None):
            tt = weight_store.load_tensor(key, device=device)
            if tt.get_layout() != ttnn.TILE_LAYOUT:
                tt = ttnn.to_layout(tt, ttnn.TILE_LAYOUT)
            if dtype is not None and tt.dtype != dtype:
                tt = ttnn.typecast(tt, dtype)
            return tt

        def _save_tt(key: str, tt_tensor):
            if weight_store is not None and base_key is not None and weight_store.mode == "save":
                weight_store.save_tensor(key, tt_tensor)

        if weight_store.mode == "load":
            self.W_q_tt = _load_tt(f"{base_key}.q.weight_t", dtype=self.dtype)
            self.W_k_tt = _load_tt(f"{base_key}.k.weight_t", dtype=self.dtype)
            self.W_v_tt = _load_tt(f"{base_key}.v.weight_t", dtype=self.dtype)
            self.b_q_tt = _load_tt(f"{base_key}.q.bias", dtype=self.dtype)
            self.b_k_tt = _load_tt(f"{base_key}.k.bias", dtype=self.dtype)
            self.b_v_tt = _load_tt(f"{base_key}.v.bias", dtype=self.dtype)
            self.W_o_tt = _load_tt(f"{base_key}.o.weight_t", dtype=self.dtype)
            self.b_o_tt = _load_tt(f"{base_key}.o.bias", dtype=self.dtype)
        elif weight_store.mode == "save":
            # Extract QKV and out_proj from torch (export path)
            W_qkv = mha_pt.in_proj_weight.detach().clone()  # [3*E, E]
            b_qkv = mha_pt.in_proj_bias.detach().clone()    # [3*E]
            W_q, W_k, W_v = torch.chunk(W_qkv, 3, dim=0)
            b_q, b_k, b_v = torch.chunk(b_qkv, 3, dim=0)

            W_o = mha_pt.out_proj.weight.detach().clone()  # [E, E]
            b_o = mha_pt.out_proj.bias.detach().clone()    # [E]

            def to_tt(t):
                return ttnn.from_torch(t, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

            self.W_q_tt = to_tt(W_q.t().contiguous())
            self.W_k_tt = to_tt(W_k.t().contiguous())
            self.W_v_tt = to_tt(W_v.t().contiguous())
            self.b_q_tt = to_tt(b_q.reshape(1, 1, -1))
            self.b_k_tt = to_tt(b_k.reshape(1, 1, -1))
            self.b_v_tt = to_tt(b_v.reshape(1, 1, -1))
            self.W_o_tt = to_tt(W_o.t().contiguous())
            self.b_o_tt = to_tt(b_o.reshape(1, 1, -1))

            if weight_store is not None and base_key is not None and weight_store.mode == "save":
                _save_tt(f"{base_key}.q.weight_t", self.W_q_tt)
                _save_tt(f"{base_key}.k.weight_t", self.W_k_tt)
                _save_tt(f"{base_key}.v.weight_t", self.W_v_tt)
                _save_tt(f"{base_key}.q.bias", self.b_q_tt)
                _save_tt(f"{base_key}.k.bias", self.b_k_tt)
                _save_tt(f"{base_key}.v.bias", self.b_v_tt)
                _save_tt(f"{base_key}.o.weight_t", self.W_o_tt)
                _save_tt(f"{base_key}.o.bias", self.b_o_tt)
        else:
            raise RuntimeError(f"Unsupported weight store mode: {weight_store.mode}")

        # Choose SDPA fast-path only when head_dim >= 32 to avoid unsupported padding
        self._use_sdpa = (self.head_dim >= 32)

        # Optimized fused-QKV path (optional). Disable by default to avoid TTNN transformer
        # shape constraints (head_dim multiple of 32) and K/V length mismatches.
        self._use_fused_qkv = False

    def _linear(self, x_tt, W_tt, b_tt):
        # x_tt: [B, S, E], W_tt: [E_out, E_in] matching torch linear semantics
        ttnn = self.ttnn
        # Use FP32 accumulation for better precision (unless low-precision is forced)
        math_fidelity = ttnn.MathFidelity.HiFi4
        fp32_ok = True
        compute_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_ok,
            packer_l1_acc=True,
        )
        return ttnn.linear(x_tt, W_tt, bias=b_tt, compute_kernel_config=compute_cfg)

    def _reshape_to_qkv(self, x_tt, b: int, s: int):
        # x_tt: [B, S, E] => [B, H, S, Dh]
        ttnn = self.ttnn
        y = ttnn.reshape(x_tt, (b, s, self.num_heads, self.head_dim))
        y = ttnn.permute(y, (0, 2, 1, 3))
        return y

    def _merge_heads(self, x_tt, b: int, s: int):
        # x_tt: [B, H, S, Dh] => [B, S, E]
        ttnn = self.ttnn
        y = ttnn.permute(x_tt, (0, 2, 1, 3))
        y = ttnn.reshape(y, (b, s, self.num_heads * self.head_dim))
        return y

    def forward(self, x_q, x_k=None, x_v=None, return_ttnn: bool = True):
        """Forward pass for TTNN tensors only."""
        ttnn = self.ttnn
        if isinstance(x_q, torch.Tensor):
            raise TypeError("TTNNMHA expects TTNN tensor inputs")
        xq_tt = x_q
        xk_tt = x_k if x_k is not None else x_q
        xv_tt = x_v if x_v is not None else x_q
        shape = xq_tt.shape
        b, s = shape[0], shape[1]

        # QKV projections (always in TTNN)
        if self._use_fused_qkv and (xk_tt is xq_tt) and (xv_tt is xq_tt):
            # Fused QKV single linear
            fused = ttnn.linear(xq_tt, self.W_qkv_tt, bias=self.b_qkv_tt)
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                fused, num_heads=self.num_heads
            )
        else:
            # Separate projections
            q = self._linear(xq_tt, self.W_q_tt, self.b_q_tt)
            k = self._linear(xk_tt, self.W_k_tt, self.b_k_tt)
            v = self._linear(xv_tt, self.W_v_tt, self.b_v_tt)
            # Reshape to [B, H, S, Dh]
            q = self._reshape_to_qkv(q, b, s)
            k = self._reshape_to_qkv(k, b, s)
            v = self._reshape_to_qkv(v, b, s)

        # Attention (always in TTNN)
        scale = 1.0 / (self.head_dim ** 0.5)
        if self._use_sdpa:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale
            )
        else:
            # Manual attention via BMM to support small head_dim
            # Flatten heads into batch: [B*H, S, Dh]
            bh = ttnn.reshape(q, (b * self.num_heads, s, self.head_dim))
            bk = ttnn.reshape(k, (b * self.num_heads, s, self.head_dim))
            bv = ttnn.reshape(v, (b * self.num_heads, s, self.head_dim))
            # q @ k^T -> [B*H, S, S] with FP32 accumulation
            bkt = ttnn.permute(bk, (0, 2, 1))
            math_fidelity = ttnn.MathFidelity.HiFi4
            fp32_ok = True
            compute_cfg = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_ok,
                packer_l1_acc=True,
            )
            scores = ttnn.matmul(bh, bkt, compute_kernel_config=compute_cfg)
            scores = ttnn.multiply(scores, scale)
            probs = ttnn.softmax(scores)
            ctx = ttnn.matmul(probs, bv, compute_kernel_config=compute_cfg)
            # Back to [B, H, S, Dh]
            attn_out = ttnn.reshape(ctx, (b, self.num_heads, s, self.head_dim))

        # Merge heads and out_proj (always in TTNN)
        y = self._merge_heads(attn_out, b, s)
        y = self._linear(y, self.W_o_tt, self.b_o_tt)

        return y
