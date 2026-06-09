#!/usr/bin/env python3
import argparse
import gc
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
from ttnn_impl.hgnetv2_ttnn_manual import _concat_activations  # noqa: E402


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
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--img-folder", default=None)
    ap.add_argument("--ann-file", default=None)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--input-layout", choices=["default", "tile", "row_major"], default="default")
    ap.add_argument("--backbone-timing", action="store_true", help="Print stem/stage timing breakdown.")
    ap.add_argument("--encoder-timing", action="store_true", help="Print encoder projection/transformer/FPN/PAN timing breakdown.")
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
    if args.img_folder and args.ann_file:
        img_folder = Path(args.img_folder)
        ann_path = Path(args.ann_file)
    elif args.data_root:
        data_root = Path(args.data_root)
        if (data_root / "val2017").exists():
            img_folder = data_root / "val2017"
            ann_path = data_root / "annotations" / "instances_val2017_first10.json"
            if not ann_path.exists():
                ann_path = data_root / "annotations" / "instances_val2017.json"
        else:
            img_folder = data_root / "images" / "val"
            ann_path = data_root / "annotations" / "instances_val.json"
    else:
        raise SystemExit("Pass either --data-root or both --img-folder and --ann-file.")
    if not img_folder.exists():
        raise SystemExit(f"Image folder does not exist: {img_folder}")
    if not ann_path.exists():
        raise SystemExit(f"Annotation file does not exist: {ann_path}")
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
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    filtered, dropped = _filter_state_by_shape(state, cfg.model.state_dict())
    if dropped:
        print(f"Skipping {len(dropped)} incompatible parameters: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    cfg.model.load_state_dict(filtered, strict=False)
    cfg.model.eval()

    model_tt = DFINE_TTNN(cfg.model, device_id=args.device_id, return_ttnn=True)
    input_layout = args.input_layout
    if input_layout == "default":
        input_layout = os.environ.get("TTNN_TRACE_INPUT_LAYOUT", "").lower() or "default"
    if input_layout in ("row_major", "row-major", "rm"):
        model_tt.backbone_tt.input_layout = ttnn.ROW_MAJOR_LAYOUT
    elif input_layout in ("tile", "tiled"):
        model_tt.backbone_tt.input_layout = ttnn.TILE_LAYOUT
    if args.decoder_timing:
        os.environ["TTNN_DECODER_TIMING"] = "1"

    loader = cfg.val_dataloader
    samples, _ = next(iter(loader))
    device = model_tt.backbone_tt.device

    def _sync_gc():
        gc.collect()
        ttnn.synchronize_device(device)

    # Warmup
    for _ in range(args.warmup):
        bb = model_tt.backbone_tt(samples)
        ttnn.synchronize_device(device)
        enc = model_tt.encoder_tt(bb)
        ttnn.synchronize_device(device)
        dec = model_tt.decoder_tt(enc)
        ttnn.synchronize_device(device)
        del dec, enc, bb
        _sync_gc()

    timings: Dict[str, List[float]] = {
        "backbone_ms": [],
        "encoder_ms": [],
        "decoder_ms": [],
        "end_to_end_ms": [],
    }
    backbone_timings: Dict[str, List[float]] = {}
    encoder_timings: Dict[str, List[float]] = {}

    def _record(name: str, seconds: float) -> None:
        backbone_timings.setdefault(name, []).append(_ms(seconds))

    def _record_encoder(name: str, seconds: float) -> None:
        encoder_timings.setdefault(name, []).append(_ms(seconds))

    def _run_backbone_timed(x: torch.Tensor):
        backbone = model_tt.backbone_tt
        outputs = []
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        act = backbone._to_ttnn(x)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        _record("backbone_input_ms", t1 - t0)

        act = backbone.stem(act)
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        _record("backbone_stem_ms", t2 - t1)

        prev = t2
        for idx, stage in enumerate(backbone.stages):
            stage_dtype = backbone.stage_downsample_dtypes[idx]
            if act.tensor.dtype != stage_dtype:
                act = backbone._cast_activation(act, stage_dtype)
            act = stage(act)
            ttnn.synchronize_device(device)
            now = time.perf_counter()
            _record(f"backbone_stage{idx}_ms", now - prev)
            prev = now
            if idx in backbone.return_idx:
                outputs.append(act)
        return outputs

    def _run_encoder_timed(feats):
        encoder = model_tt.encoder_tt

        def _timed(name: str, fn):
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            value = fn()
            ttnn.synchronize_device(device)
            _record_encoder(name, time.perf_counter() - start)
            return value

        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        proj_acts = encoder._project_inputs_ttnn(feats)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        _record_encoder("encoder_project_ms", t1 - t0)

        if encoder._encoder_layers:
            encoded_acts = encoder._run_encoder_layers_ttnn(proj_acts)
        else:
            encoded_acts = proj_acts
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        _record_encoder("encoder_transformer_ms", t2 - t1)

        inner = [encoded_acts[-1]]
        for idx in range(len(encoder.in_channels) - 1, 0, -1):
            fpn_id = len(encoder.in_channels) - 1 - idx
            hi = inner[0]
            lo = encoded_acts[idx - 1]
            lateral = encoder.lateral_convs_tt[fpn_id]
            hi = _timed(f"encoder_fpn{fpn_id}.lateral_ms", lambda hi=hi, lateral=lateral: lateral(hi))
            inner[0] = hi
            upsampled = _timed(f"encoder_fpn{fpn_id}.upsample_ms", lambda hi=hi: encoder._upsample2x_ttnn_act(hi))
            if upsampled.height != lo.height or upsampled.width != lo.width:
                upsampled = _timed(
                    f"encoder_fpn{fpn_id}.slice_ms",
                    lambda upsampled=upsampled, lo=lo: encoder._slice_activation_spatial(upsampled, lo.height, lo.width),
                )
            fused = _timed(f"encoder_fpn{fpn_id}.concat_ms", lambda upsampled=upsampled, lo=lo: _concat_activations([upsampled, lo]))
            block = encoder.fpn_blocks_tt_native[fpn_id]
            inner.insert(0, _timed(f"encoder_fpn{fpn_id}.block_ms", lambda fused=fused, block=block: block(fused)))
        fpn_acts = inner
        ttnn.synchronize_device(device)
        t3 = time.perf_counter()
        _record_encoder("encoder_fpn_ms", t3 - t2)

        outs = [fpn_acts[0]]
        for idx in range(len(encoder.in_channels) - 1):
            low = outs[-1]
            high = fpn_acts[idx + 1]
            down = _timed(f"encoder_pan{idx}.downsample_ms", lambda low=low, idx=idx: encoder.downsample_convs_tt_native[idx](low))
            if down.height != high.height or down.width != high.width:
                down = _timed(
                    f"encoder_pan{idx}.slice_ms",
                    lambda down=down, high=high: encoder._slice_activation_spatial(down, high.height, high.width),
                )
            fused = _timed(f"encoder_pan{idx}.concat_ms", lambda down=down, high=high: _concat_activations([down, high]))
            block = encoder.pan_blocks_tt_native[idx]
            outs.append(_timed(f"encoder_pan{idx}.block_ms", lambda fused=fused, block=block: block(fused)))
        pan_acts = outs
        ttnn.synchronize_device(device)
        t4 = time.perf_counter()
        _record_encoder("encoder_pan_ms", t4 - t3)
        return pan_acts

    iters = 1 if args.decoder_timing else args.iters
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        bb = _run_backbone_timed(samples) if args.backbone_timing else model_tt.backbone_tt(samples)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        enc = _run_encoder_timed(bb) if args.encoder_timing else model_tt.encoder_tt(bb)
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        dec = model_tt.decoder_tt(enc)
        ttnn.synchronize_device(device)
        t3 = time.perf_counter()

        timings["backbone_ms"].append(_ms(t1 - t0))
        timings["encoder_ms"].append(_ms(t2 - t1))
        timings["decoder_ms"].append(_ms(t3 - t2))
        del dec, enc, bb
        _sync_gc()

        # End-to-end
        ttnn.synchronize_device(device)
        t4 = time.perf_counter()
        dec = model_tt.forward(samples, return_ttnn=True)
        ttnn.synchronize_device(device)
        t5 = time.perf_counter()
        timings["end_to_end_ms"].append(_ms(t5 - t4))
        del dec
        _sync_gc()

    print("Latency summary (ms):")
    for key, vals in timings.items():
        print(f"  {key}: mean={_mean(vals):.3f}  p50={_p50(vals):.3f}  n={len(vals)}")
    if args.backbone_timing:
        print("Backbone timing breakdown (ms):")
        for key, vals in sorted(backbone_timings.items()):
            print(f"  {key}: mean={_mean(vals):.3f}  p50={_p50(vals):.3f}  n={len(vals)}")
    if args.encoder_timing:
        print("Encoder timing breakdown (ms):")
        for key, vals in sorted(encoder_timings.items()):
            print(f"  {key}: mean={_mean(vals):.3f}  p50={_p50(vals):.3f}  n={len(vals)}")

    if args.decoder_timing and getattr(model_tt.decoder_tt, "_last_timing_entries", None):
        print("Decoder timing breakdown (ms):")
        for name, ms in model_tt.decoder_tt._last_timing_entries:
            print(f"  {name}: {ms:.3f}")

    model_tt.close()


if __name__ == "__main__":
    main()
