"""Helpers for single-device TTNN tensor conversion."""

from __future__ import annotations

import ttnn


def to_torch_tensor(tensor: "ttnn.Tensor", expected_shape=None):
    """Convert a TTNN tensor to torch (single-device only)."""
    out = ttnn.to_torch(tensor)
    if expected_shape is None:
        return out
    slices = []
    for size in expected_shape:
        if size is None:
            slices.append(slice(None))
        else:
            slices.append(slice(0, int(size)))
    return out[tuple(slices)]
