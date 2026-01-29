#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


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
    ap = argparse.ArgumentParser(description="Trace-mode smoke test for TTNN DFINE.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    trace_bytes = int(os.environ.get("TTNN_TRACE_REGION_SIZE", str(64 * 1024 * 1024)))
    os.environ.setdefault("TTNN_TRACE_REGION_SIZE", str(trace_bytes))
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

    model_tt = DFINE_TTNN(cfg.model, device_id=args.device_id)
    model_tt.enable_trace_mode(True, batch=args.batch)

    loader = cfg.val_dataloader
    samples, _ = next(iter(loader))

    # Prepare input once to keep buffer addresses stable.
    tt_input = model_tt.backbone_tt._to_ttnn(samples)

    # Warmup (compile) outside trace capture.
    _ = model_tt.forward(tt_input, return_ttnn=True)
    ttnn.synchronize_device(model_tt.backbone_tt.device)

    trace_id = ttnn.begin_trace_capture(model_tt.backbone_tt.device, cq_id=0)
    _ = model_tt.forward(tt_input, return_ttnn=True)
    ttnn.end_trace_capture(model_tt.backbone_tt.device, trace_id, cq_id=0)

    # Execute the trace once to validate.
    ttnn.execute_trace(model_tt.backbone_tt.device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(model_tt.backbone_tt.device)
    print(f"Trace capture/execute OK (trace_id={trace_id})")

    model_tt.close()


if __name__ == "__main__":
    main()
