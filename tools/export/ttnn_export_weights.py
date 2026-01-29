#!/usr/bin/env python3
"""Export TTNN-ready weights into a TTNNWeightStore manifest."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TTNN weights to a weight store.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.eval_size is not None:
        overrides["eval_spatial_size"] = [int(args.eval_size), int(args.eval_size)]
    if args.num_classes is not None:
        overrides["num_classes"] = int(args.num_classes)

    cfg = YAMLConfig(args.config, **overrides)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model_state = cfg.model.state_dict()
    filtered = {}
    dropped = []
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"[export] Dropping {len(dropped)} state_dict keys with mismatched shapes.")
    cfg.model.load_state_dict(filtered, strict=False)
    cfg.model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TTNN_WEIGHT_DIR"] = str(out_dir)
    os.environ["TTNN_WEIGHT_STORE_MODE"] = "save"
    os.environ["TTNN_WARMUP"] = "0"

    model_tt = DFINE_TTNN(cfg.model, device_id=args.device_id, weight_store_dir=out_dir)
    model_tt.close()

    print(f"[export] TTNN weight store written to: {out_dir}")


if __name__ == "__main__":
    main()
