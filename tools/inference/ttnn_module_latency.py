#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig  # noqa: E402
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN  # noqa: E402


def _infer_num_classes(ann_path: Path) -> int:
    import json
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(len(data.get("categories", [])))


def _filter_state_by_shape(state, model_state):
    filtered = {}
    dropped = []
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            filtered[key] = value
        else:
            dropped.append(key)
    return filtered, dropped


def parse_args():
    ap = argparse.ArgumentParser(description="TTNN per-module latency measurement (backbone/encoder/decoder).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--decoder-timing", action="store_true", help="Print decoder internal timing breakdown.")
    return ap.parse_args()


def _p50(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[len(values) // 2]


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_folder = data_root / "images" / "val"
    ann_path = data_root / "annotations" / "instances_val.json"
    overrides = {
        "eval_spatial_size": [int(args.eval_size), int(args.eval_size)],
        "val_dataloader": {
            "dataset": {
                "img_folder": str(img_folder),
                "ann_file": str(ann_path),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [int(args.eval_size), int(args.eval_size)]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                },
            },
            "total_batch_size": int(args.batch),
            "num_workers": 0,
        },
    }
    overrides["num_classes"] = _infer_num_classes(ann_path)
    cfg = YAMLConfig(args.config, **overrides)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    filtered, dropped = _filter_state_by_shape(state, cfg.model.state_dict())
    if dropped:
        print(f"Skipping {len(dropped)} incompatible parameters: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    cfg.model.load_state_dict(filtered, strict=False)
    cfg.model.eval()

    model_tt = DFINE_TTNN(cfg.model, device_id=args.device_id, return_ttnn=True)
    if args.decoder_timing:
        os.environ["TTNN_DECODER_TIMING"] = "1"

    loader = cfg.val_dataloader
    samples, _ = next(iter(loader))
    tt_input = model_tt.backbone_tt._to_ttnn(samples)
    device = model_tt.backbone_tt.device

    # Warmup
    for _ in range(args.warmup):
        _ = model_tt.backbone_tt(tt_input)
        ttnn.synchronize_device(device)
        enc = model_tt.encoder_tt(model_tt.backbone_tt(tt_input))
        ttnn.synchronize_device(device)
        _ = model_tt.decoder_tt(enc)
        ttnn.synchronize_device(device)

    timings: Dict[str, List[float]] = {
        "backbone_ms": [],
        "encoder_ms": [],
        "decoder_ms": [],
        "end_to_end_ms": [],
    }

    iters = 1 if args.decoder_timing else args.iters
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        bb = model_tt.backbone_tt(tt_input)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        enc = model_tt.encoder_tt(bb)
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        _ = model_tt.decoder_tt(enc)
        ttnn.synchronize_device(device)
        t3 = time.perf_counter()

        timings["backbone_ms"].append(_ms(t1 - t0))
        timings["encoder_ms"].append(_ms(t2 - t1))
        timings["decoder_ms"].append(_ms(t3 - t2))

        # End-to-end
        ttnn.synchronize_device(device)
        t4 = time.perf_counter()
        _ = model_tt.forward(tt_input, return_ttnn=True)
        ttnn.synchronize_device(device)
        t5 = time.perf_counter()
        timings["end_to_end_ms"].append(_ms(t5 - t4))

    print("Latency summary (ms):")
    for key, vals in timings.items():
        print(f"  {key}: mean={_mean(vals):.3f}  p50={_p50(vals):.3f}  n={len(vals)}")

    if args.decoder_timing and getattr(model_tt.decoder_tt, "_last_timing_entries", None):
        print("Decoder timing breakdown (ms):")
        for name, ms in model_tt.decoder_tt._last_timing_entries:
            print(f"  {name}: {ms:.3f}")

    model_tt.close()


if __name__ == "__main__":
    main()
