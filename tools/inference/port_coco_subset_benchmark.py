#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torchvision.transforms as T
import ttnn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.data.dataset import mscoco_label2category
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN
from ttnn_impl.hgnetv2_ttnn_manual import TTActivation
from ttnn_impl.ttnn_utils import to_torch_tensor


COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"

CHECKPOINTS = {
    "n_coco": (
        "D-FINE-N COCO",
        "configs/dfine/dfine_hgnetv2_n_coco.yml",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
    ),
    "s_coco": (
        "D-FINE-S COCO",
        "configs/dfine/dfine_hgnetv2_s_coco.yml",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
    ),
    "s_obj2coco": (
        "D-FINE-S Objects365+COCO",
        "configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth",
    ),
}

DEFAULT_WEIGHT_DIRS = {
    "n_coco": "weight/ttnn_store_n_coco",
    "s_coco": "weight/ttnn_store_s_coco",
    "s_obj2coco": "weight/ttnn_store_s_obj2coco",
}


def git_metadata() -> tuple[str | None, str | None]:
    def _git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return _git(["branch", "--show-current"]), _git(["rev-parse", "HEAD"])


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}")
    urlretrieve(url, path)


def ensure_coco_subset(root: Path, max_images: int) -> tuple[Path, Path]:
    ann_file = root / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        zip_path = root / "annotations_trainval2017.zip"
        download(COCO_ANN_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("annotations/instances_val2017.json", root)

    coco = COCO(str(ann_file))
    image_ids = sorted(coco.getImgIds())[:max_images]
    images_dir = root / "val2017"
    for img in coco.loadImgs(image_ids):
        download(COCO_IMAGE_URL.format(file_name=img["file_name"]), images_dir / img["file_name"])

    keep = set(image_ids)
    full = json.loads(ann_file.read_text())
    subset = {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "categories": full["categories"],
        "images": [img for img in full["images"] if img["id"] in keep],
        "annotations": [ann for ann in full["annotations"] if ann["image_id"] in keep],
    }
    subset_file = root / "annotations" / f"instances_val2017_first{max_images}.json"
    subset_file.write_text(json.dumps(subset))
    return images_dir, subset_file


def load_cfg(config_path: Path, checkpoint_path: Path, eval_size: int) -> YAMLConfig:
    cfg = YAMLConfig(
        str(config_path),
        resume=str(checkpoint_path),
        eval_spatial_size=[int(eval_size), int(eval_size)],
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model_state = cfg.model.state_dict()
    filtered_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    cfg.model.load_state_dict(filtered_state, strict=False)
    cfg.model.eval()
    cfg.postprocessor.deploy()
    return cfg


class TorchWrapper:
    def __init__(self, cfg: YAMLConfig, eval_size: int):
        self.model = cfg.model
        self.postprocessor = cfg.postprocessor
        self.transform = T.Compose([T.Resize((eval_size, eval_size)), T.ToTensor()])
        self.last_timing = {}

    @torch.no_grad()
    def __call__(self, image_path: Path):
        timing = {}
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        timing["image_load_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        x = self.transform(image).unsqueeze(0)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        raw = self.model(x)
        timing["model_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        labels, boxes, scores = self.postprocessor(raw, torch.tensor([[w, h]], dtype=torch.float32))
        timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_timing = timing
        return {"labels": labels[0], "boxes": boxes[0], "scores": scores[0]}

    def close(self):
        pass


class TTNNWrapper:
    def __init__(self, cfg: YAMLConfig, device_id: int, weight_dir: Path, use_trace: bool = False, trace_strict: bool = False):
        os.environ.setdefault("TTNN_WEIGHT_DIR", str(weight_dir))
        if not (weight_dir / "manifest.json").exists():
            os.environ["TTNN_WEIGHT_STORE_MODE"] = "save"
        else:
            os.environ.setdefault("TTNN_WEIGHT_STORE_MODE", "load")
        self.model = DFINE_TTNN(cfg.model, device_id=device_id)
        trace_input_layout = os.environ.get("TTNN_TRACE_INPUT_LAYOUT", "row_major" if use_trace else "").lower()
        self.trace_device_tile_input = trace_input_layout in (
            "device_tile",
            "device-tile",
            "row_to_tile",
            "row-to-tile",
            "row_major_to_tile",
            "row-major-to-tile",
        )
        if trace_input_layout in ("row_major", "row-major", "rm"):
            self.model.backbone_tt.input_layout = ttnn.ROW_MAJOR_LAYOUT
        elif trace_input_layout in ("tile", "tiled") or self.trace_device_tile_input:
            self.model.backbone_tt.input_layout = ttnn.TILE_LAYOUT
        self.postprocessor = cfg.postprocessor
        self.transform = T.Compose([T.Resize(tuple(self.model.input_size)), T.ToTensor()])
        self.use_trace = bool(use_trace)
        self.trace_strict = bool(trace_strict)
        self.trace_uint8_input = bool(use_trace) and os.environ.get("TTNN_TRACE_INPUT_DTYPE", "").lower() in (
            "uint8",
            "u8",
        )
        self.trace_id = None
        self.trace_input_act = None
        self.trace_outputs = None
        self._resident_trace_timing = {}
        self.last_timing = {}
        self._last_trace_timing = {}

    def _input_activation(self, x: torch.Tensor, device: bool, layout=None, dtype=None) -> TTActivation:
        batch_size = int(x.shape[0])
        _, channels, height, width = x.shape
        nhwc = x.permute(0, 2, 3, 1).contiguous()
        dtype = dtype or getattr(self.model.backbone_tt, "dtype", ttnn.bfloat16)
        if dtype == ttnn.uint8:
            nhwc = torch.clamp(torch.round(nhwc * 255.0), 0, 255).to(torch.uint8)
        kwargs = {
            "dtype": dtype,
            "layout": layout or getattr(self.model.backbone_tt, "input_layout", ttnn.TILE_LAYOUT),
        }
        if device:
            kwargs["device"] = self.model.backbone_tt.device
        if device:
            tensor = ttnn.from_torch(nhwc, **kwargs)
        elif os.environ.get("TTNN_HOST_INPUT_FROM_TORCH", "0") == "1":
            tensor = ttnn.from_torch(nhwc, **kwargs)
        else:
            tensor = ttnn.as_tensor(nhwc, **kwargs)
        return TTActivation(tensor, batch_size, height, width, channels)

    def _trace_model_input(self) -> TTActivation:
        if self.trace_uint8_input:
            tensor = ttnn.typecast(self.trace_input_act.tensor, getattr(self.model.backbone_tt, "dtype", ttnn.bfloat16))
            tensor = ttnn.multiply(tensor, 1.0 / 255.0)
            target_layout = getattr(self.model.backbone_tt, "input_layout", ttnn.ROW_MAJOR_LAYOUT)
            if tensor.get_layout() != target_layout:
                tensor = ttnn.to_layout(tensor, target_layout)
        elif self.trace_device_tile_input:
            tensor = ttnn.to_layout(self.trace_input_act.tensor, ttnn.TILE_LAYOUT)
        else:
            return self.trace_input_act
        return TTActivation(
            tensor,
            self.trace_input_act.batch,
            self.trace_input_act.height,
            self.trace_input_act.width,
            self.trace_input_act.channels,
        )

    def _init_trace(self, x: torch.Tensor) -> None:
        self.model.enable_trace_mode(True, batch=int(x.shape[0]), strict=self.trace_strict)
        trace_input_layout = ttnn.ROW_MAJOR_LAYOUT if self.trace_device_tile_input else None
        trace_input_dtype = ttnn.uint8 if self.trace_uint8_input else None
        self.trace_input_act = self._input_activation(
            x,
            device=True,
            layout=trace_input_layout,
            dtype=trace_input_dtype,
        )
        _ = self.model.forward_ttnn(self._trace_model_input())
        ttnn.synchronize_device(self.model.backbone_tt.device)
        self.trace_id = ttnn.begin_trace_capture(self.model.backbone_tt.device, cq_id=0)
        self.trace_outputs = self.model.forward_ttnn(self._trace_model_input())
        ttnn.end_trace_capture(self.model.backbone_tt.device, self.trace_id, cq_id=0)
        self._measure_resident_trace()

    def _measure_resident_trace(self) -> None:
        iters = int(os.environ.get("TTNN_BENCH_RESIDENT_TRACE_ITERS", "0") or "0")
        if iters <= 0 or self.trace_id is None:
            return
        device = self.model.backbone_tt.device
        times = []
        ttnn.synchronize_device(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, self.trace_id, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) * 1000.0)
        self._resident_trace_timing = {
            "resident_trace_execute_ms": sum(times) / max(1, len(times)),
            "resident_trace_execute_min_ms": min(times) if times else 0.0,
            "resident_trace_execute_max_ms": max(times) if times else 0.0,
        }

    def _run_trace(self, x: torch.Tensor) -> dict:
        if self.trace_id is None:
            self._init_trace(x)
        timing = {}
        device = self.model.backbone_tt.device
        phase_sync = os.environ.get("TTNN_BENCH_PHASE_SYNC", "0") == "1"
        t0 = time.perf_counter()
        host_layout = ttnn.ROW_MAJOR_LAYOUT if self.trace_device_tile_input else None
        host_dtype = ttnn.uint8 if self.trace_uint8_input else None
        host_act = self._input_activation(x, device=False, layout=host_layout, dtype=host_dtype)
        timing["tt_input_host_tensor_ms"] = (time.perf_counter() - t0) * 1000.0
        if phase_sync:
            ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(host_act.tensor, self.trace_input_act.tensor, cq_id=0)
        if phase_sync:
            ttnn.synchronize_device(device)
        timing["h2d_copy_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        ttnn.execute_trace(device, self.trace_id, cq_id=0, blocking=True)
        timing["trace_execute_ms"] = (time.perf_counter() - t0) * 1000.0
        timing.update(self._resident_trace_timing)
        t0 = time.perf_counter()
        pred_logits = to_torch_tensor(
            self.trace_outputs["pred_logits"], expected_shape=self.trace_outputs["pred_logits"].shape
        )
        timing["d2h_logits_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        pred_boxes = to_torch_tensor(
            self.trace_outputs["pred_boxes"], expected_shape=self.trace_outputs["pred_boxes"].shape
        )
        timing["d2h_boxes_ms"] = (time.perf_counter() - t0) * 1000.0
        raw = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }
        logit_scale = self.model._env_float("TTNN_LOGIT_SCALE", 1.0)
        if logit_scale != 1.0:
            raw["pred_logits"] = raw["pred_logits"] * logit_scale
        self._last_trace_timing = timing
        return raw

    @torch.no_grad()
    def __call__(self, image_path: Path):
        timing = {}
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        timing["image_load_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        x = self.transform(image).unsqueeze(0)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        raw = self._run_trace(x) if self.use_trace else self.model(x)
        timing["model_ms"] = (time.perf_counter() - t0) * 1000.0
        timing.update(self._last_trace_timing if self.use_trace else {})
        t0 = time.perf_counter()
        labels, boxes, scores = self.postprocessor(raw, torch.tensor([[w, h]], dtype=torch.float32))
        timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_timing = timing
        return {"labels": labels[0], "boxes": boxes[0], "scores": scores[0]}

    def close(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.model.backbone_tt.device, self.trace_id)
        self.model.close()


def evaluate(
    model,
    images_dir: Path,
    annotations: Path,
    max_images: int,
    warmup: int,
    output_json: Path,
) -> dict:
    coco = COCO(str(annotations))
    image_infos = coco.loadImgs(sorted(coco.getImgIds())[:max_images])
    predictions = []
    times = []
    phase_times = {}

    for idx, img in enumerate(image_infos):
        image_path = images_dir / img["file_name"]
        if idx < warmup:
            _ = model(image_path)

        start = time.perf_counter()
        out = model(image_path)
        times.append(time.perf_counter() - start)
        for key, value in getattr(model, "last_timing", {}).items():
            phase_times.setdefault(key, []).append(float(value))

        labels = out["labels"].to(torch.int64).tolist()
        boxes = out["boxes"].to(torch.float32).tolist()
        scores = out["scores"].to(torch.float32).tolist()
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            predictions.append(
                {
                    "image_id": img["id"],
                    "category_id": int(mscoco_label2category[int(label)]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions))

    coco_dt = coco.loadRes(str(output_json)) if predictions else None
    if coco_dt is None:
        stats = [0.0] * 12
    else:
        evaluator = COCOeval(coco, coco_dt, "bbox")
        evaluator.params.imgIds = [img["id"] for img in image_infos]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats.tolist()

    result = {
        "images": len(image_infos),
        "detections": len(predictions),
        "latency_ms_mean": 1000.0 * sum(times) / max(len(times), 1),
        "latency_ms_min": 1000.0 * min(times) if times else 0.0,
        "latency_ms_max": 1000.0 * max(times) if times else 0.0,
        "coco_ap": stats[0],
        "coco_ap50": stats[1],
        "coco_ap75": stats[2],
        "coco_ar100": stats[8],
    }
    if os.environ.get("TTNN_BENCH_PHASE_TIMING", "1") == "1":
        result["phase_latency_ms_mean"] = {
            key: sum(values) / max(len(values), 1) for key, values in sorted(phase_times.items())
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "ttnn"], required=True)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="s_coco")
    parser.add_argument("--data-root", default="data/coco2017_subset")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results/port_coco_subset")
    parser.add_argument(
        "--weight-dir",
        default=None,
        help="TTNN serialized weight store. Defaults to a model-specific store under weight/.",
    )
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--eval-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--use-trace", action="store_true", help="Use TTNN trace capture/execute for TTNN backend.")
    parser.add_argument("--trace-strict", action="store_true", help="Enable strict trace-mode layout checks.")
    parser.add_argument("--min-ap", type=float, default=None, help="Fail if COCO AP is below this value.")
    parser.add_argument("--max-latency-ms", type=float, default=None, help="Fail if mean latency is above this value.")
    args = parser.parse_args()

    model_name, rel_config, checkpoint_url = CHECKPOINTS[args.model]
    weight_dir = Path(args.weight_dir or DEFAULT_WEIGHT_DIRS[args.model])
    images_dir, ann_file = ensure_coco_subset(Path(args.data_root), args.max_images)
    ckpt_path = Path(args.checkpoint_dir) / Path(checkpoint_url).name
    download(checkpoint_url, ckpt_path)

    cfg = load_cfg(ROOT / rel_config, ckpt_path, args.eval_size)
    if args.backend == "torch":
        model = TorchWrapper(cfg, args.eval_size)
    else:
        model = TTNNWrapper(cfg, args.device_id, weight_dir, use_trace=args.use_trace, trace_strict=args.trace_strict)

    try:
        pred_path = Path(args.output_dir) / f"{args.backend}_{args.model}_predictions.json"
        metrics = evaluate(model, images_dir, ann_file, args.max_images, args.warmup, pred_path)
    finally:
        model.close()

    branch, commit = git_metadata()
    result = {
        "backend": args.backend,
        "model": model_name,
        "branch": branch,
        "commit": commit,
        "config": str(ROOT / rel_config),
        "checkpoint": str(ckpt_path),
        "weight_dir": str(weight_dir) if args.backend == "ttnn" else None,
        "annotations": str(ann_file),
        "eval_size": args.eval_size,
        "device_id": args.device_id if args.backend == "ttnn" else None,
        "ttnn_allow_torch_conv_fallback": os.environ.get("TTNN_ALLOW_TORCH_CONV_FALLBACK") if args.backend == "ttnn" else None,
        "ttnn_depthwise_chunk_size": os.environ.get("TTNN_DEPTHWISE_CHUNK_SIZE") if args.backend == "ttnn" else None,
        "ttnn_conv_fp32_acc": os.environ.get("TTNN_CONV_FP32_ACC") if args.backend == "ttnn" else None,
        "ttnn_conv_weights_dtype": (
            os.environ.get("TTNN_CONV_WEIGHTS_DTYPE") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_shard_layout": (
            os.environ.get("TTNN_CONV_SHARD_LAYOUT") if args.backend == "ttnn" else None
        ),
        "ttnn_math_fidelity": os.environ.get("TTNN_MATH_FIDELITY", "HiFi4") if args.backend == "ttnn" else None,
        "ttnn_math_approx": os.environ.get("TTNN_MATH_APPROX", "0") if args.backend == "ttnn" else None,
        "ttnn_pointwise_linear": os.environ.get("TTNN_POINTWISE_LINEAR", "0") if args.backend == "ttnn" else None,
        "ttnn_pointwise_linear_min_channels": (
            os.environ.get("TTNN_POINTWISE_LINEAR_MIN_CHANNELS") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_config_tensors_in_dram": (
            os.environ.get("TTNN_CONV_CONFIG_TENSORS_IN_DRAM") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_deallocate_activation": (
            os.environ.get("TTNN_CONV_DEALLOCATE_ACTIVATION") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_reallocate_halo_output": (
            os.environ.get("TTNN_CONV_REALLOCATE_HALO_OUTPUT") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_reshard_if_not_optimal": (
            os.environ.get("TTNN_CONV_RESHARD_IF_NOT_OPTIMAL") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_enable_act_double_buffer": (
            os.environ.get("TTNN_CONV_ENABLE_ACT_DOUBLE_BUFFER") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_enable_weights_double_buffer": (
            os.environ.get("TTNN_CONV_ENABLE_WEIGHTS_DOUBLE_BUFFER") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_enable_activation_reuse": (
            os.environ.get("TTNN_CONV_ENABLE_ACTIVATION_REUSE") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_act_block_h_override": (
            os.environ.get("TTNN_CONV_ACT_BLOCK_H_OVERRIDE") if args.backend == "ttnn" else None
        ),
        "ttnn_conv_act_block_w_div": os.environ.get("TTNN_CONV_ACT_BLOCK_W_DIV") if args.backend == "ttnn" else None,
        "ttnn_auto_l1_output": os.environ.get("TTNN_AUTO_L1_OUTPUT", "1") if args.backend == "ttnn" else None,
        "ttnn_auto_l1_max_bytes": os.environ.get("TTNN_AUTO_L1_MAX_BYTES") if args.backend == "ttnn" else None,
        "ttnn_topk_memory_embedding": os.environ.get("TTNN_TOPK_MEMORY_EMBEDDING", "1") if args.backend == "ttnn" else None,
        "ttnn_topk_anchor_analytic": os.environ.get("TTNN_TOPK_ANCHOR_ANALYTIC", "1") if args.backend == "ttnn" else None,
        "ttnn_topk_anchor_fast": os.environ.get("TTNN_TOPK_ANCHOR_FAST", "1") if args.backend == "ttnn" else None,
        "ttnn_topk_anchor_embedding": os.environ.get("TTNN_TOPK_ANCHOR_EMBEDDING", "0") if args.backend == "ttnn" else None,
        "ttnn_topk_anchor_tosa_gather": os.environ.get("TTNN_TOPK_ANCHOR_TOSA_GATHER", "1") if args.backend == "ttnn" else None,
        "ttnn_topk_multipass64": os.environ.get("TTNN_TOPK_MULTIPASS64", "0") if args.backend == "ttnn" else None,
        "ttnn_topk_multipass64_threshold": (
            os.environ.get("TTNN_TOPK_MULTIPASS64_THRESHOLD", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_topk_multipass64_threshold_gt": (
            os.environ.get("TTNN_TOPK_MULTIPASS64_THRESHOLD_GT", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_topk_level_quotas": os.environ.get("TTNN_TOPK_LEVEL_QUOTAS") if args.backend == "ttnn" else None,
        "ttnn_topk_sorted": os.environ.get("TTNN_TOPK_SORTED", "1") if args.backend == "ttnn" else None,
        "ttnn_topk_pad_power2": os.environ.get("TTNN_TOPK_PAD_POWER2", "0") if args.backend == "ttnn" else None,
        "ttnn_topk_tie_break_eps": os.environ.get("TTNN_TOPK_TIE_BREAK_EPS", "0") if args.backend == "ttnn" else None,
        "ttnn_topk_score_dtype": os.environ.get("TTNN_TOPK_SCORE_DTYPE") if args.backend == "ttnn" else None,
        "ttnn_topk_use_sort": os.environ.get("TTNN_TOPK_USE_SORT", "0") if args.backend == "ttnn" else None,
        "ttnn_topk_sort_dtype": os.environ.get("TTNN_TOPK_SORT_DTYPE") if args.backend == "ttnn" else None,
        "ttnn_decoder_num_queries": os.environ.get("TTNN_DECODER_NUM_QUERIES") if args.backend == "ttnn" else None,
        "ttnn_decoder_allow_extra_queries": (
            os.environ.get("TTNN_DECODER_ALLOW_EXTRA_QUERIES", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_decoder_max_queries": os.environ.get("TTNN_DECODER_MAX_QUERIES") if args.backend == "ttnn" else None,
        "ttnn_decoder_eval_idx": os.environ.get("TTNN_DECODER_EVAL_IDX") if args.backend == "ttnn" else None,
        "ttnn_decoder_topk_fp32": os.environ.get("TTNN_DECODER_TOPK_FP32", "1") if args.backend == "ttnn" else None,
        "ttnn_distance2bbox_direct": os.environ.get("TTNN_DISTANCE2BBOX_DIRECT", "0") if args.backend == "ttnn" else None,
        "ttnn_trace_concat_align": os.environ.get("TTNN_TRACE_CONCAT_ALIGN", "0") if args.backend == "ttnn" else None,
        "ttnn_concat_preserve_memory": (
            os.environ.get("TTNN_CONCAT_PRESERVE_MEMORY", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_trace_conv_align": os.environ.get("TTNN_TRACE_CONV_ALIGN", "0") if args.backend == "ttnn" else None,
        "ttnn_trace_conv_input_layout": (
            os.environ.get("TTNN_TRACE_CONV_INPUT_LAYOUT", "row_major") if args.backend == "ttnn" else None
        ),
        "ttnn_trace_conv_output_layout": (
            os.environ.get("TTNN_TRACE_CONV_OUTPUT_LAYOUT", "row_major") if args.backend == "ttnn" else None
        ),
        "ttnn_trace_input_dtype": os.environ.get("TTNN_TRACE_INPUT_DTYPE") if args.backend == "ttnn" else None,
        "ttnn_host_input_from_torch": os.environ.get("TTNN_HOST_INPUT_FROM_TORCH", "0") if args.backend == "ttnn" else None,
        "ttnn_grid_sample_pack_points": os.environ.get("TTNN_GRID_SAMPLE_PACK_POINTS", "0") if args.backend == "ttnn" else None,
        "ttnn_grid_sample_batch_output_channels": (
            os.environ.get("TTNN_GRID_SAMPLE_BATCH_OUTPUT_CHANNELS", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_grid_sample_output_l1": os.environ.get("TTNN_GRID_SAMPLE_OUTPUT_L1", "1") if args.backend == "ttnn" else None,
        "ttnn_grid_sample_grid_fp32": os.environ.get("TTNN_GRID_SAMPLE_GRID_FP32", "0") if args.backend == "ttnn" else None,
        "ttnn_grid_sample_shard_grid": (
            os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_grid_sample_shard_grid_x": (
            os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID_X") if args.backend == "ttnn" else None
        ),
        "ttnn_grid_sample_shard_grid_y": (
            os.environ.get("TTNN_GRID_SAMPLE_SHARD_GRID_Y") if args.backend == "ttnn" else None
        ),
        "ttnn_msda_direct_grid": (
            os.environ.get(
                "TTNN_MSDA_DIRECT_GRID",
                f"auto<={os.environ.get('TTNN_MSDA_DIRECT_GRID_AUTO_MAX_QUERIES', '96')}",
            )
            if args.backend == "ttnn"
            else None
        ),
        "ttnn_msda_direct_grid_auto_max_queries": (
            os.environ.get("TTNN_MSDA_DIRECT_GRID_AUTO_MAX_QUERIES", "96") if args.backend == "ttnn" else None
        ),
        "ttnn_msda_grid_mac": os.environ.get("TTNN_MSDA_GRID_MAC", "1") if args.backend == "ttnn" else None,
        "ttnn_trace_input_layout": (
            os.environ.get("TTNN_TRACE_INPUT_LAYOUT") or ("row_major" if args.use_trace else None)
        ) if args.backend == "ttnn" else None,
        "ttnn_bench_resident_trace_iters": (
            os.environ.get("TTNN_BENCH_RESIDENT_TRACE_ITERS", "0") if args.backend == "ttnn" else None
        ),
        "ttnn_use_trace": args.use_trace if args.backend == "ttnn" else None,
        "ttnn_trace_strict": args.trace_strict if args.backend == "ttnn" else None,
        **metrics,
    }
    result_path = Path(args.output_dir) / f"{args.backend}_{args.model}_metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    failures = []
    if args.min_ap is not None and result["coco_ap"] < args.min_ap:
        failures.append(f"AP {result['coco_ap']:.6f} < min {args.min_ap:.6f}")
    if args.max_latency_ms is not None and result["latency_ms_mean"] > args.max_latency_ms:
        failures.append(f"latency {result['latency_ms_mean']:.3f} ms > max {args.max_latency_ms:.3f} ms")
    if failures:
        raise SystemExit("Benchmark gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
