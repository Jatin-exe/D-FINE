#!/usr/bin/env python3
"""Run TTNN-accelerated D-FINE evaluation on a COCO-style dataset."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.solver.det_engine import evaluate
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APsmall",
    "APmedium",
    "APlarge",
    "AR1",
    "AR10",
    "AR100",
    "ARsmall",
    "ARmedium",
    "ARlarge",
]


def _filter_state_by_shape(
    state: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    filtered: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            filtered[key] = value
        else:
            dropped.append(key)
    return filtered, dropped


def _infer_num_classes(ann_path: Path) -> int:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(len(data.get("categories", [])))


def _dataset_overrides(
    data_root: Optional[str],
    split: str,
    num_classes: Optional[int],
    remap_mscoco_category: Optional[bool],
    val_batch_size: Optional[int],
    eval_size: Optional[int],
    num_workers: Optional[int],
) -> Tuple[Dict[str, object], Optional[int], Optional[Path]]:
    overrides: Dict[str, object] = {}
    ann_path: Optional[Path] = None

    if data_root:
        root = Path(data_root)
        img_folder = root / "images" / split
        ann_path = root / "annotations" / f"instances_{split}.json"
        overrides["val_dataloader"] = {
            "dataset": {
                "img_folder": str(img_folder),
                "ann_file": str(ann_path),
            }
        }
        if num_classes is None:
            num_classes = _infer_num_classes(ann_path)

    if num_classes is not None:
        overrides["num_classes"] = int(num_classes)
    if remap_mscoco_category is not None:
        overrides["remap_mscoco_category"] = bool(remap_mscoco_category)
    if val_batch_size is not None:
        overrides.setdefault("val_dataloader", {})
        overrides["val_dataloader"]["total_batch_size"] = int(val_batch_size)
    if num_workers is not None:
        overrides.setdefault("val_dataloader", {})
        overrides["val_dataloader"]["num_workers"] = int(num_workers)
    if eval_size is not None:
        overrides["eval_spatial_size"] = [int(eval_size), int(eval_size)]
        overrides.setdefault("val_dataloader", {}).setdefault("dataset", {}).setdefault("transforms", {})
        overrides["val_dataloader"]["dataset"]["transforms"]["ops"] = [
            {"type": "Resize", "size": [int(eval_size), int(eval_size)]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ]

    return overrides, num_classes, ann_path


def _metric_map(coco_stats: Optional[List[float]]) -> Dict[str, float]:
    if not coco_stats:
        return {}
    return {name: float(coco_stats[i]) for i, name in enumerate(COCO_METRIC_NAMES) if i < len(coco_stats)}


def _write_results(
    output_path: Path,
    payload: Dict[str, object],
    metrics: Dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    lines = [
        "# TTNN Evaluation Results",
        "",
        f"- Timestamp: {payload.get('timestamp')}",
        f"- Backend: `{payload.get('backend')}`",
        f"- Config: `{payload.get('config')}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Split: `{payload.get('split')}`",
        f"- Data root: `{payload.get('data_root')}`",
        f"- Num classes: `{payload.get('num_classes')}`",
        f"- TTNN device id: `{payload.get('device_id')}`",
        f"- Num images: `{payload.get('num_images')}`",
        f"- Eval seconds: `{payload.get('eval_seconds')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in COCO_METRIC_NAMES:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFINE using TTNN backend.")
    parser.add_argument("--config", required=True, help="Path to the D-FINE YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--data-root", help="Dataset root containing images/ and annotations/.")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"], help="Dataset split.")
    parser.add_argument("--num-classes", type=int, help="Override number of classes.")
    parser.add_argument("--remap-mscoco-category", action="store_true", default=None)
    parser.add_argument("--no-remap-mscoco-category", action="store_false", dest="remap_mscoco_category")
    parser.add_argument("--val-batch-size", type=int, help="Override validation total batch size.")
    parser.add_argument("--eval-size", type=int, help="Override evaluation spatial size (square).")
    parser.add_argument("--num-workers", type=int, help="Override validation dataloader workers.")
    parser.add_argument("--fast-runtime", action="store_true", default=None, help="Enable TTNN fast runtime mode.")
    parser.add_argument("--no-fast-runtime", action="store_false", dest="fast_runtime")
    parser.add_argument("--enable-model-cache", action="store_true", default=None, help="Enable TTNN model cache.")
    parser.add_argument("--disable-model-cache", action="store_false", dest="enable_model_cache")
    parser.add_argument("--max-samples", type=int, help="Limit evaluation to the first N samples.")
    parser.add_argument("--backend", choices=["ttnn", "torch"], default="ttnn", help="Select inference backend.")
    parser.add_argument("--device-id", type=int, default=0, help="TTNN device id.")
    parser.add_argument("--output", default="results/ttnn_eval.json", help="Output JSON path.")
    args = parser.parse_args()

    import ttnn

    if args.fast_runtime is not None:
        ttnn.CONFIG.enable_fast_runtime_mode = bool(args.fast_runtime)
    if args.enable_model_cache is not None:
        ttnn.CONFIG.enable_model_cache = bool(args.enable_model_cache)

    overrides, inferred_classes, ann_path = _dataset_overrides(
        args.data_root,
        args.split,
        args.num_classes,
        args.remap_mscoco_category,
        args.val_batch_size,
        args.eval_size,
        args.num_workers,
    )
    cfg = YAMLConfig(args.config, **overrides)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model_state = cfg.model.state_dict()
    filtered_state, dropped = _filter_state_by_shape(state, model_state)
    if dropped:
        print(f"Skipping {len(dropped)} incompatible parameters: {dropped}")
    cfg.model.load_state_dict(filtered_state, strict=False)
    model_pt = cfg.model.eval()

    model_tt = None
    if args.backend == "ttnn":
        model_tt = DFINE_TTNN(model_pt, device_id=args.device_id)

    val_loader = cfg.val_dataloader
    evaluator = cfg.evaluator
    if args.max_samples:
        from torch.utils.data import DataLoader
        from src.data import CocoEvaluator
        from src.data.dataset.coco_utils import convert_to_coco_api

        class _SubsetWithLoad:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

            def load_item(self, idx):
                return self.dataset.load_item(self.indices[idx])

        indices = list(range(int(args.max_samples)))
        subset = _SubsetWithLoad(val_loader.dataset, indices)
        val_loader = DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            collate_fn=val_loader.collate_fn,
            drop_last=False,
        )
        base_ds = convert_to_coco_api(subset)
        evaluator_cfg = cfg.yaml_cfg.get("evaluator", {})
        iou_types = evaluator_cfg.get("iou_types", ["bbox"])
        evaluator = CocoEvaluator(base_ds, iou_types=iou_types)

    stats = {}
    start_time = time.perf_counter()
    try:
        device = torch.device("cpu")
        model_eval = model_pt if args.backend == "torch" else model_tt
        stats, _ = evaluate(
            model_eval,
            cfg.criterion,
            cfg.postprocessor,
            val_loader,
            evaluator,
            device,
            epoch=0,
            use_wandb=False,
        )
    finally:
        if model_tt is not None:
            model_tt.close()
    elapsed = time.perf_counter() - start_time

    num_images = int(len(val_loader.dataset))

    coco_stats = stats.get("coco_eval_bbox", [])
    metrics = _metric_map(coco_stats)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "backend": args.backend,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "data_root": str(Path(args.data_root)) if args.data_root else None,
        "split": args.split,
        "num_classes": inferred_classes if inferred_classes is not None else args.num_classes,
        "device_id": args.device_id,
        "num_images": num_images,
        "eval_seconds": round(elapsed, 4),
        "ann_file": str(ann_path) if ann_path else None,
        "coco_eval_bbox": coco_stats,
        "metrics": metrics,
    }

    _write_results(Path(args.output), payload, metrics)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
