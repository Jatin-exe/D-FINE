#!/usr/bin/env python3
"""Compare Torch vs TTNN detections and save side-by-side collages."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.data.dataset import mscoco_category2label
from src.solver.validator import Validator, scale_boxes
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN
from ttnn_impl.hgnetv2_ttnn_manual import TTActivation
from ttnn_impl.ttnn_utils import to_torch_tensor


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
) -> Tuple[Dict[str, object], Optional[int]]:
    overrides: Dict[str, object] = {}

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

    return overrides, num_classes


def _prepare_text_block(lines: List[str]) -> str:
    return "\n".join(lines)


def _draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    color: Tuple[int, int, int],
    max_dets: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if boxes.numel() == 0:
        return image
    count = min(int(boxes.shape[0]), max_dets)
    for idx in range(count):
        box = boxes[idx].tolist()
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = int(labels[idx].item()) if labels is not None else -1
        score = float(scores[idx].item()) if scores is not None else 0.0
        text = f"{label}:{score:.2f}"
        draw.text((x1 + 2, y1 + 2), text, fill=color, font=font)
    return image


def _compute_image_counts(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float,
    conf_thresh: float,
) -> Dict[str, int]:
    if pred_scores is not None:
        keep = pred_scores >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

    tps = fps = fns = 0
    all_labels = torch.unique(torch.cat([pred_labels, gt_labels])) if pred_labels.numel() or gt_labels.numel() else []
    for label in all_labels:
        pred_mask = pred_labels == label
        gt_mask = gt_labels == label
        pb = pred_boxes[pred_mask]
        gb = gt_boxes[gt_mask]
        if pb.numel() == 0 and gb.numel() == 0:
            continue
        if pb.numel() == 0:
            fns += int(gb.shape[0])
            continue
        if gb.numel() == 0:
            fps += int(pb.shape[0])
            continue
        ious = box_iou(pb, gb)
        matched_pred = set()
        matched_gt = set()
        pred_indices, gt_indices = torch.nonzero(ious >= iou_thresh, as_tuple=True)
        if pred_indices.numel() == 0:
            fps += int(pb.shape[0])
            fns += int(gb.shape[0])
            continue
        iou_values = ious[pred_indices, gt_indices]
        order = torch.argsort(-iou_values)
        pred_indices = pred_indices[order]
        gt_indices = gt_indices[order]
        for p_idx, g_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
            if p_idx in matched_pred or g_idx in matched_gt:
                continue
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            tps += 1
        fps += int(pb.shape[0]) - len(matched_pred)
        fns += int(gb.shape[0]) - len(matched_gt)
    return {"TPs": tps, "FPs": fps, "FNs": fns}


def _make_collage(
    left: Image.Image,
    right: Image.Image,
    left_title: str,
    right_title: str,
    left_metrics: List[str],
    right_metrics: List[str],
    output_path: Path,
) -> None:
    font = ImageFont.load_default()
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    metrics_lines = max(len(left_metrics), len(right_metrics)) + 1
    metrics_height = 10 + metrics_lines * (line_height + 2)

    width = left.width + right.width
    height = max(left.height, right.height)
    collage = Image.new("RGB", (width, height + metrics_height), color=(20, 20, 20))
    collage.paste(left, (0, 0))
    collage.paste(right, (left.width, 0))

    draw = ImageDraw.Draw(collage)
    draw.text((10, 5), left_title, fill=(255, 255, 255), font=font)
    draw.text((left.width + 10, 5), right_title, fill=(255, 255, 255), font=font)

    base_y = height + 5
    draw.text((10, base_y), _prepare_text_block(left_metrics), fill=(220, 220, 220), font=font)
    draw.text(
        (left.width + 10, base_y),
        _prepare_text_block(right_metrics),
        fill=(220, 220, 220),
        font=font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path)


def _ensure_torch_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; pass --torch-device cpu to override.")
        return torch.device("cuda")
    return torch.device(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Torch vs TTNN detections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--remap-mscoco-category", action="store_true", default=None)
    parser.add_argument("--no-remap-mscoco-category", action="store_false", dest="remap_mscoco_category")
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=None, help="Limit to first N images.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index within the dataset.")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument(
        "--score-sweep",
        default=None,
        help="Optional sweep over score thresholds. Use 'start,end,step' or a comma list.",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=50)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--output-dir", default="results/ttnn_torch_compare")
    parser.add_argument("--fast-runtime", action="store_true", default=None)
    parser.add_argument("--no-fast-runtime", action="store_false", dest="fast_runtime")
    parser.add_argument("--enable-model-cache", action="store_true", default=None)
    parser.add_argument("--disable-model-cache", action="store_false", dest="enable_model_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    import ttnn

    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.CONFIG.root_report_path = str(output_dir / "ttnn_reports")
    if args.fast_runtime is not None:
        ttnn.CONFIG.enable_fast_runtime_mode = bool(args.fast_runtime)
    if args.enable_model_cache is not None:
        ttnn.CONFIG.enable_model_cache = bool(args.enable_model_cache)

    torch_device = _ensure_torch_device(args.torch_device)

    overrides, _ = _dataset_overrides(
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

    model_pt_cpu = cfg.model.eval()
    postprocessor_cpu = cfg.postprocessor.eval()

    if torch_device.type == "cpu":
        model_torch = model_pt_cpu
        postprocessor_torch = postprocessor_cpu
    else:
        model_torch = copy.deepcopy(model_pt_cpu).to(torch_device)
        postprocessor_torch = copy.deepcopy(postprocessor_cpu).to(torch_device)

    model_tt = DFINE_TTNN(model_pt_cpu, device_id=args.device_id)
    use_trace = os.environ.get("TTNN_USE_TRACE", "0") == "1"
    trace_id = None
    trace_outputs = None
    trace_input_act = None
    trace_cq_id = int(os.environ.get("TTNN_TRACE_CQ", "0"))

    val_loader = cfg.val_dataloader
    if args.val_batch_size != 1:
        print("Warning: best visual quality is with --val-batch-size 1.")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gt_all: List[Dict[str, torch.Tensor]] = []
    preds_torch_all: List[Dict[str, torch.Tensor]] = []
    preds_ttnn_all: List[Dict[str, torch.Tensor]] = []
    per_image: List[Dict[str, object]] = []

    processed = 0
    seen = 0
    torch_times_ms: List[float] = []
    ttnn_times_ms: List[float] = []
    start_time = time.perf_counter()
    trace_debug = os.environ.get("TTNN_TRACE_DEBUG", "0") == "1"
    trace_debug_prints = 0
    prev_trace_logits: Optional[torch.Tensor] = None

    def _sync_device(device):
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
            return
        if hasattr(device, "synchronize"):
            device.synchronize()
            return
        raise RuntimeError("TTNN device does not support synchronize().")

    for batch_idx, (samples, targets) in enumerate(val_loader):
        if args.num_images is not None and processed >= args.num_images:
            break
        if not isinstance(targets, list):
            targets = list(targets)

        samples_cpu = samples
        samples_torch = samples_cpu.to(torch_device)
        orig_target_sizes_cpu = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_target_sizes_torch = orig_target_sizes_cpu.to(torch_device)

        with torch.no_grad():
            torch_start = time.perf_counter()
            outputs_torch = model_torch(samples_torch)
            results_torch = postprocessor_torch(outputs_torch, orig_target_sizes_torch)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            torch_elapsed_ms = (time.perf_counter() - torch_start) * 1000.0

            ttnn_start = time.perf_counter()
            outputs_ttnn = None
            if use_trace:
                try:
                    if trace_id is None:
                        trace_strict = os.environ.get("TTNN_TRACE_STRICT", "0") == "1"
                        if os.environ.get("TTNN_TRACE_MODE", "1") == "1":
                            model_tt.enable_trace_mode(True, batch=int(samples_cpu.shape[0]), strict=trace_strict)
                        batch_size = int(samples_cpu.shape[0])
                        _, channels, height, width = samples_cpu.shape
                        input_layout = getattr(model_tt.backbone_tt, "input_layout", ttnn.ROW_MAJOR_LAYOUT)
                        input_dtype = getattr(model_tt.backbone_tt, "dtype", ttnn.bfloat16)
                        nhwc = samples_cpu.permute(0, 2, 3, 1).contiguous()
                        input_tensor = ttnn.from_torch(
                            nhwc,
                            dtype=input_dtype,
                            layout=input_layout,
                            device=model_tt.backbone_tt.device,
                        )
                        trace_input_act = TTActivation(input_tensor, batch_size, height, width, channels)
                        if trace_debug and hasattr(ttnn, "get_memory_config"):
                            print(f"[trace_debug] input_mem={ttnn.get_memory_config(trace_input_act.tensor)} ")
                        # Warmup outside capture to avoid weight prep during trace.
                        _ = model_tt.forward_ttnn(trace_input_act)
                        _sync_device(model_tt.backbone_tt.device)
                        trace_id = ttnn.begin_trace_capture(model_tt.backbone_tt.device, cq_id=trace_cq_id)
                        trace_outputs = model_tt.forward_ttnn(trace_input_act)
                        ttnn.end_trace_capture(model_tt.backbone_tt.device, trace_id, cq_id=trace_cq_id)

                    if trace_input_act is None or trace_outputs is None:
                        raise RuntimeError("Trace state not initialized.")

                    nhwc = samples_cpu.permute(0, 2, 3, 1).contiguous()
                    host_tt = ttnn.from_torch(
                        nhwc,
                        dtype=trace_input_act.tensor.dtype,
                        layout=trace_input_act.tensor.get_layout(),
                    )
                    ttnn.copy_host_to_device_tensor(host_tt, trace_input_act.tensor, cq_id=trace_cq_id)
                    if trace_debug and trace_debug_prints < 3:
                        input_dbg = to_torch_tensor(
                            trace_input_act.tensor,
                            expected_shape=trace_input_act.tensor.shape,
                        )
                        print(f"[trace_debug] idx={batch_idx} input_sum={float(input_dbg.sum()):.4f}")
                    _sync_device(model_tt.backbone_tt.device)

                    ttnn.execute_trace(model_tt.backbone_tt.device, trace_id, cq_id=trace_cq_id, blocking=True)
                    _sync_device(model_tt.backbone_tt.device)
                    if trace_debug and os.environ.get("TTNN_TRACE_COMPARE_DIRECT", "0") == "1":
                        try:
                            direct_out = model_tt.forward_ttnn(trace_input_act)
                            direct_logits = to_torch_tensor(
                                direct_out["pred_logits"], expected_shape=direct_out["pred_logits"].shape
                            )
                            print(
                                f"[trace_debug] direct_logits_sum={float(direct_logits.sum()):.4f} "
                                f"direct_logits_max={float(direct_logits.max()):.4f}"
                            )
                        except Exception as exc:
                            print(f"[trace_debug] direct compare failed: {exc}")
                    outputs_ttnn = trace_outputs
                except Exception as exc:
                    print(f"[trace] Disable tracing (capture failed): {exc}")
                    use_trace = False
                    outputs_ttnn = model_tt(samples_cpu)
            else:
                outputs_ttnn = model_tt(samples_cpu)

            if outputs_ttnn is None:
                raise RuntimeError("TTNN outputs missing.")

            if not isinstance(outputs_ttnn["pred_logits"], torch.Tensor):
                outputs_ttnn = {
                    "pred_logits": to_torch_tensor(outputs_ttnn["pred_logits"], expected_shape=outputs_ttnn["pred_logits"].shape),
                    "pred_boxes": to_torch_tensor(outputs_ttnn["pred_boxes"], expected_shape=outputs_ttnn["pred_boxes"].shape),
                }
            if trace_debug and trace_debug_prints < 3:
                trace_debug_prints += 1
                logits = outputs_ttnn["pred_logits"]
                print(
                    f"[trace_debug] idx={batch_idx} logits_sum={float(logits.sum()):.4f} "
                    f"logits_max={float(logits.max()):.4f}"
                )
                if prev_trace_logits is not None:
                    diff = (logits - prev_trace_logits).abs()
                    print(
                        f"[trace_debug] logits_diff max={float(diff.max()):.6f} mean={float(diff.mean()):.6f}"
                    )
                prev_trace_logits = logits.detach().clone()
            results_ttnn = postprocessor_cpu(outputs_ttnn, orig_target_sizes_cpu)
            _sync_device(model_tt.backbone_tt.device)
            ttnn_elapsed_ms = (time.perf_counter() - ttnn_start) * 1000.0

        batch_size = int(samples_cpu.shape[0])
        per_image_torch_ms = torch_elapsed_ms / max(batch_size, 1)
        per_image_ttnn_ms = ttnn_elapsed_ms / max(batch_size, 1)

        for idx, (target, result_torch, result_ttnn) in enumerate(
            zip(targets, results_torch, results_ttnn)
        ):
            if seen < args.start_index:
                seen += 1
                continue
            if args.num_images is not None and processed >= args.num_images:
                break
            seen += 1
            image_path = target.get("image_path")
            if image_path is None:
                continue

            image = Image.open(image_path).convert("RGB")
            image_right = image.copy()
            image_left = image.copy()

            torch_boxes = result_torch["boxes"].detach().cpu()
            torch_labels = result_torch["labels"].detach().cpu()
            torch_scores = result_torch["scores"].detach().cpu()

            ttnn_boxes = result_ttnn["boxes"].detach().cpu()
            ttnn_labels = result_ttnn["labels"].detach().cpu()
            ttnn_scores = result_ttnn["scores"].detach().cpu()

            image_left = _draw_boxes(
                image_left, torch_boxes, torch_labels, torch_scores, (0, 200, 0), args.max_dets
            )
            image_right = _draw_boxes(
                image_right, ttnn_boxes, ttnn_labels, ttnn_scores, (200, 0, 0), args.max_dets
            )

            gt_boxes = target["boxes"].clone().detach().cpu()
            gt_labels = target["labels"].clone().detach().cpu()
            gt_boxes = scale_boxes(
                gt_boxes,
                (target["orig_size"][1], target["orig_size"][0]),
                (samples_cpu[idx].shape[-1], samples_cpu[idx].shape[-2]),
            )

            label_map = None
            if postprocessor_cpu.remap_mscoco_category:
                label_map = mscoco_category2label

            def _map_labels(labels: torch.Tensor) -> torch.Tensor:
                if label_map is None:
                    return labels
                mapped = [label_map[int(x.item())] for x in labels.flatten()]
                return torch.tensor(mapped, dtype=labels.dtype).reshape(labels.shape)

            torch_labels_eval = _map_labels(torch_labels)
            ttnn_labels_eval = _map_labels(ttnn_labels)

            preds_torch_all.append(
                {"boxes": torch_boxes, "labels": torch_labels_eval, "scores": torch_scores}
            )
            preds_ttnn_all.append(
                {"boxes": ttnn_boxes, "labels": ttnn_labels_eval, "scores": ttnn_scores}
            )
            gt_all.append({"boxes": gt_boxes, "labels": gt_labels})

            torch_counts = _compute_image_counts(
                torch_boxes,
                torch_labels_eval,
                torch_scores,
                gt_boxes,
                gt_labels,
                args.iou_thresh,
                args.score_thresh,
            )
            ttnn_counts = _compute_image_counts(
                ttnn_boxes,
                ttnn_labels_eval,
                ttnn_scores,
                gt_boxes,
                gt_labels,
                args.iou_thresh,
                args.score_thresh,
            )

            torch_keep = torch_scores >= args.score_thresh
            ttnn_keep = ttnn_scores >= args.score_thresh
            torch_count = int(torch_keep.sum().item())
            ttnn_count = int(ttnn_keep.sum().item())
            torch_mean_score = float(torch_scores[torch_keep].mean().item()) if torch_count else 0.0
            ttnn_mean_score = float(ttnn_scores[ttnn_keep].mean().item()) if ttnn_count else 0.0

            left_metrics = [
                f"GT: {int(gt_boxes.shape[0])} boxes",
                f"Det: {torch_count} boxes, mean {torch_mean_score:.2f}",
                f"TP {torch_counts['TPs']} | FP {torch_counts['FPs']} | FN {torch_counts['FNs']}",
                f"thr={args.score_thresh:.2f}, IoU={args.iou_thresh:.2f}",
                f"time={per_image_torch_ms:.2f} ms",
            ]
            right_metrics = [
                f"GT: {int(gt_boxes.shape[0])} boxes",
                f"Det: {ttnn_count} boxes, mean {ttnn_mean_score:.2f}",
                f"TP {ttnn_counts['TPs']} | FP {ttnn_counts['FPs']} | FN {ttnn_counts['FNs']}",
                f"thr={args.score_thresh:.2f}, IoU={args.iou_thresh:.2f}",
                f"time={per_image_ttnn_ms:.2f} ms",
            ]

            out_name = Path(image_path).stem + f"_{processed:04d}.jpg"
            out_path = images_dir / out_name
            _make_collage(
                image_left,
                image_right,
                "Torch",
                "TTNN",
                left_metrics,
                right_metrics,
                out_path,
            )

            per_image.append(
                {
                    "image_path": image_path,
                    "output_path": str(out_path),
                    "torch_time_ms": round(per_image_torch_ms, 4),
                    "ttnn_time_ms": round(per_image_ttnn_ms, 4),
                    "torch": {
                        "num_boxes": torch_count,
                        "mean_score": torch_mean_score,
                        **torch_counts,
                    },
                    "ttnn": {
                        "num_boxes": ttnn_count,
                        "mean_score": ttnn_mean_score,
                        **ttnn_counts,
                    },
                }
            )

            processed += 1
            torch_times_ms.append(per_image_torch_ms)
            ttnn_times_ms.append(per_image_ttnn_ms)

    elapsed = time.perf_counter() - start_time

    torch_metrics = Validator(gt_all, preds_torch_all, conf_thresh=args.score_thresh, iou_thresh=args.iou_thresh).compute_metrics()
    ttnn_metrics = Validator(gt_all, preds_ttnn_all, conf_thresh=args.score_thresh, iou_thresh=args.iou_thresh).compute_metrics()

    sweep_results = None
    best_thresholds = None
    if args.score_sweep:
        raw = [item.strip() for item in args.score_sweep.split(",") if item.strip()]
        values = [float(item) for item in raw]
        if len(values) == 3:
            start, end, step = values
            thresholds = []
            cur = start
            while cur <= end + 1e-6:
                thresholds.append(round(cur, 4))
                cur += step
        else:
            thresholds = values

        sweep_results = []
        best_thresholds = {
            "torch": {"threshold": None, "f1": -1.0},
            "ttnn": {"threshold": None, "f1": -1.0},
        }
        for thr in thresholds:
            torch_metrics_thr = Validator(
                gt_all,
                copy.deepcopy(preds_torch_all),
                conf_thresh=thr,
                iou_thresh=args.iou_thresh,
            ).compute_metrics()
            ttnn_metrics_thr = Validator(
                gt_all,
                copy.deepcopy(preds_ttnn_all),
                conf_thresh=thr,
                iou_thresh=args.iou_thresh,
            ).compute_metrics()
            sweep_results.append(
                {
                    "threshold": thr,
                    "torch_f1": torch_metrics_thr.get("f1", 0.0),
                    "ttnn_f1": ttnn_metrics_thr.get("f1", 0.0),
                }
            )
            if torch_metrics_thr.get("f1", 0.0) >= best_thresholds["torch"]["f1"]:
                best_thresholds["torch"] = {"threshold": thr, "f1": torch_metrics_thr.get("f1", 0.0)}
            if ttnn_metrics_thr.get("f1", 0.0) >= best_thresholds["ttnn"]["f1"]:
                best_thresholds["ttnn"] = {"threshold": thr, "f1": ttnn_metrics_thr.get("f1", 0.0)}

    def _time_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
        vals = sorted(values)
        n = len(vals)
        def _pct(p: float) -> float:
            if n == 1:
                return vals[0]
            idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return vals[idx]
        return {
            "mean_ms": round(sum(vals) / n, 4),
            "p50_ms": round(_pct(50.0), 4),
            "p90_ms": round(_pct(90.0), 4),
            "max_ms": round(max(vals), 4),
        }

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "split": args.split,
        "num_images": processed,
        "eval_size": args.eval_size,
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "torch_device": str(torch_device),
        "ttnn_device_id": args.device_id,
        "elapsed_seconds": round(elapsed, 4),
        "torch_metrics": torch_metrics,
        "ttnn_metrics": ttnn_metrics,
        "torch_time_ms": _time_stats(torch_times_ms),
        "ttnn_time_ms": _time_stats(ttnn_times_ms),
        "accuracy_percent": {
            "torch_f1": round(torch_metrics.get("f1", 0) * 100.0, 2),
            "ttnn_f1": round(ttnn_metrics.get("f1", 0) * 100.0, 2),
        },
        "score_sweep": sweep_results,
        "best_thresholds": best_thresholds,
        "things_right": {
            "torch_TPs": int(torch_metrics.get("TPs", 0)),
            "ttnn_TPs": int(ttnn_metrics.get("TPs", 0)),
        },
        "per_image": per_image,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Torch vs TTNN Comparison",
        "",
        f"- Timestamp: {summary['timestamp']}",
        f"- Config: `{summary['config']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Split: `{summary['split']}`",
        f"- Num images: `{summary['num_images']}`",
        f"- Torch device: `{summary['torch_device']}`",
        f"- TTNN device id: `{summary['ttnn_device_id']}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']}`",
        f"- Torch time ms: mean {summary['torch_time_ms']['mean_ms']}, p50 {summary['torch_time_ms']['p50_ms']}, p90 {summary['torch_time_ms']['p90_ms']}, max {summary['torch_time_ms']['max_ms']}",
        f"- TTNN time ms: mean {summary['ttnn_time_ms']['mean_ms']}, p50 {summary['ttnn_time_ms']['p50_ms']}, p90 {summary['ttnn_time_ms']['p90_ms']}, max {summary['ttnn_time_ms']['max_ms']}",
        "",
        "## Metrics",
        "",
        "| Backend | F1 (%) | Precision | Recall | IoU | TPs | FPs | FNs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Torch | {summary['accuracy_percent']['torch_f1']:.2f} | {torch_metrics.get('precision', 0):.4f} | {torch_metrics.get('recall', 0):.4f} | {torch_metrics.get('iou', 0):.4f} | {torch_metrics.get('TPs', 0)} | {torch_metrics.get('FPs', 0)} | {torch_metrics.get('FNs', 0)} |",
        f"| TTNN | {summary['accuracy_percent']['ttnn_f1']:.2f} | {ttnn_metrics.get('precision', 0):.4f} | {ttnn_metrics.get('recall', 0):.4f} | {ttnn_metrics.get('iou', 0):.4f} | {ttnn_metrics.get('TPs', 0)} | {ttnn_metrics.get('FPs', 0)} | {ttnn_metrics.get('FNs', 0)} |",
        "",
        "## Output",
        "",
        f"- Collage images: `{images_dir}`",
        f"- Summary JSON: `{summary_path}`",
    ]
    if best_thresholds:
        md_lines[md_lines.index("## Output") : md_lines.index("## Output")] = [
            "## Best Thresholds (F1)",
            "",
            f"- Torch: threshold {best_thresholds['torch']['threshold']}, F1 {best_thresholds['torch']['f1']:.4f}",
            f"- TTNN: threshold {best_thresholds['ttnn']['threshold']}, F1 {best_thresholds['ttnn']['f1']:.4f}",
            "",
        ]
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved {processed} collages to {images_dir}")
    print(f"Wrote summary to {summary_path}")

    model_tt.close()


if __name__ == "__main__":
    main()
