#!/usr/bin/env python3
import argparse
import json
import os
import time
import threading
from pathlib import Path
import sys
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN
from ttnn_impl.hgnetv2_ttnn_manual import TTActivation, _concat_activations, _pad_activation


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
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--img-folder", default=None)
    ap.add_argument("--ann-file", default=None)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--strict", action="store_true", help="Enable strict trace-mode layout checks for all modules.")
    ap.add_argument("--rt-profile", action="store_true", help="Collect real-time profiler records during timed loops.")
    ap.add_argument("--graph-report", default=None, help="Optionally save one warmed TTNN graph capture JSON for the scope.")
    ap.add_argument("--graph-top-n", type=int, default=16, help="Number of graph-duration rows to print.")
    ap.add_argument(
        "--scope",
        choices=[
            "full",
            "backbone",
            "backbone_encoder",
            "decoder",
            "stem",
            "stem1",
            "stem2a",
            "stem2b",
            "stem_pool",
            "stem_concat",
            "stem3",
            "stem4",
            "stage0",
            "stage1",
            "stage2",
            "stage3",
        ],
        default="full",
        help="Which portion of the TTNN graph to capture and time.",
    )
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    trace_bytes = int(os.environ.get("TTNN_TRACE_REGION_SIZE", str(64 * 1024 * 1024)))
    os.environ.setdefault("TTNN_TRACE_REGION_SIZE", str(trace_bytes))
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

    model_tt = DFINE_TTNN(cfg.model, device_id=args.device_id)
    model_tt.enable_trace_mode(True, batch=args.batch, strict=args.strict)
    trace_input_layout = os.environ.get("TTNN_TRACE_INPUT_LAYOUT", "").lower()
    if trace_input_layout in ("row_major", "row-major", "rm"):
        model_tt.backbone_tt.input_layout = ttnn.ROW_MAJOR_LAYOUT
    elif trace_input_layout in ("tile", "tiled"):
        model_tt.backbone_tt.input_layout = ttnn.TILE_LAYOUT

    loader = cfg.val_dataloader
    samples, _ = next(iter(loader))

    # Prepare input once to keep buffer addresses stable.
    tt_input = model_tt.backbone_tt._to_ttnn(samples)

    decoder_inputs = None
    stem_inputs = {}
    stage_inputs = {}
    stage_scope = None
    if args.scope.startswith("stage"):
        stage_scope = int(args.scope.replace("stage", ""))
        if stage_scope < 0 or stage_scope >= len(model_tt.backbone_tt.stages):
            raise SystemExit(f"Unsupported stage scope {args.scope}; model has {len(model_tt.backbone_tt.stages)} stages")

    def _run_backbone_stage(stage_idx: int, act):
        stage_dtype = model_tt.backbone_tt.stage_downsample_dtypes[stage_idx]
        if act.tensor.dtype != stage_dtype:
            act = model_tt.backbone_tt._cast_activation(act, stage_dtype)
        return model_tt.backbone_tt.stages[stage_idx](act)

    if stage_scope is not None:
        act = model_tt.backbone_tt.stem(tt_input)
        for idx in range(stage_scope):
            act = _run_backbone_stage(idx, act)
        stage_inputs[stage_scope] = act
        ttnn.synchronize_device(model_tt.backbone_tt.device)

    def _align_stem_branches(x1: TTActivation, x2: TTActivation):
        stem = model_tt.backbone_tt.stem
        if stem.trace_mode:
            x1_layout = x1.tensor.get_layout()
            x2_layout = x2.tensor.get_layout()
            if x1_layout != x2_layout:
                x2_tensor = ttnn.to_layout(x2.tensor, x1_layout)
                x2 = TTActivation(x2_tensor, x2.batch, x2.height, x2.width, x2.channels)
            if ttnn.get_memory_config(x1.tensor) != ttnn.get_memory_config(x2.tensor):
                x1_tensor = ttnn.to_memory_config(x1.tensor, ttnn.DRAM_MEMORY_CONFIG)
                x2_tensor = ttnn.to_memory_config(x2.tensor, ttnn.DRAM_MEMORY_CONFIG)
                x1 = TTActivation(x1_tensor, x1.batch, x1.height, x1.width, x1.channels)
                x2 = TTActivation(x2_tensor, x2.batch, x2.height, x2.width, x2.channels)
        return x1, x2

    def _run_stem_concat(x1: TTActivation, x2: TTActivation):
        x1, x2 = _align_stem_branches(x1, x2)
        return _concat_activations([x1, x2], trace_mode=model_tt.backbone_tt.stem.trace_mode)

    def _prepare_stem_scope(scope: str) -> None:
        stem = model_tt.backbone_tt.stem
        if scope == "stem1":
            stem_inputs["stem1"] = tt_input
            return
        x = stem.stem1(tt_input)
        y_pad = _pad_activation(x, (0, 1, 0, 1))
        if scope == "stem2a":
            stem_inputs["stem2a"] = y_pad
            return
        if scope == "stem_pool":
            stem_inputs["stem_pool"] = y_pad
            return
        x2 = stem.stem2a(y_pad)
        x2_pad = _pad_activation(x2, (0, 1, 0, 1))
        if scope == "stem2b":
            stem_inputs["stem2b"] = x2_pad
            return
        x2 = stem.stem2b(x2_pad)
        x1 = stem._max_pool(y_pad)
        if scope == "stem_concat":
            stem_inputs["stem_concat"] = (x1, x2)
            return
        x = _run_stem_concat(x1, x2)
        if scope == "stem3":
            stem_inputs["stem3"] = x
            return
        x = stem.stem3(x)
        if scope == "stem4":
            stem_inputs["stem4"] = x
            return
        raise RuntimeError(f"Unsupported stem scope: {scope}")

    if args.scope.startswith("stem") and args.scope != "stem":
        _prepare_stem_scope(args.scope)
        ttnn.synchronize_device(model_tt.backbone_tt.device)

    if args.scope == "decoder":
        bb_feats = model_tt.backbone_tt(tt_input)
        decoder_inputs = model_tt.encoder_tt(bb_feats)
        ttnn.synchronize_device(model_tt.backbone_tt.device)

    def _run_scope():
        if args.scope == "full":
            return model_tt.forward_ttnn(tt_input)
        if args.scope == "backbone":
            return model_tt.backbone_tt(tt_input)
        if args.scope == "backbone_encoder":
            bb_feats = model_tt.backbone_tt(tt_input)
            return model_tt.encoder_tt(bb_feats)
        if args.scope == "stem":
            return model_tt.backbone_tt.stem(tt_input)
        if args.scope == "stem1":
            return model_tt.backbone_tt.stem.stem1(stem_inputs["stem1"])
        if args.scope == "stem2a":
            return model_tt.backbone_tt.stem.stem2a(stem_inputs["stem2a"])
        if args.scope == "stem2b":
            return model_tt.backbone_tt.stem.stem2b(stem_inputs["stem2b"])
        if args.scope == "stem_pool":
            return model_tt.backbone_tt.stem._max_pool(stem_inputs["stem_pool"])
        if args.scope == "stem_concat":
            x1, x2 = stem_inputs["stem_concat"]
            return _run_stem_concat(x1, x2)
        if args.scope == "stem3":
            return model_tt.backbone_tt.stem.stem3(stem_inputs["stem3"])
        if args.scope == "stem4":
            return model_tt.backbone_tt.stem.stem4(stem_inputs["stem4"])
        if stage_scope is not None:
            return _run_backbone_stage(stage_scope, stage_inputs[stage_scope])
        if args.scope == "decoder":
            return model_tt.decoder_tt(decoder_inputs)
        raise RuntimeError(f"Unsupported trace scope: {args.scope}")

    # Warmup (compile) outside trace capture.
    _ = _run_scope()
    ttnn.synchronize_device(model_tt.backbone_tt.device)

    records = []
    records_lock = threading.Lock()
    handle = None

    def _collect(record):
        row = {
            "program_id": int(record.program_id),
            "chip_id": int(record.chip_id),
            "start_timestamp": int(record.start_timestamp),
            "end_timestamp": int(record.end_timestamp),
            "frequency": float(record.frequency),
            "kernel_sources": list(record.kernel_sources),
        }
        with records_lock:
            records.append(row)

    def _snapshot_records():
        with records_lock:
            snapshot = list(records)
            records.clear()
        return snapshot

    def _summarize_records(name, snapshot, top_n: int = 8):
        rows = []
        for row in snapshot:
            delta = row["end_timestamp"] - row["start_timestamp"]
            freq = row["frequency"]
            if row["program_id"] != 0 and delta > 0 and freq > 0:
                rows.append((delta / freq / 1.0e6, row))
        durations = [duration for duration, _ in rows]
        if not durations:
            print(f"{name} RT records: count={len(snapshot)} valid=0")
            return
        durations_sorted = sorted(durations)
        print(
            f"{name} RT records: count={len(snapshot)} valid={len(durations)} "
            f"sum_device_ms={sum(durations):.3f} max_program_ms={max(durations):.3f} "
            f"p50_program_ms={durations_sorted[len(durations_sorted) // 2]:.3f}"
        )
        for rank, (duration, row) in enumerate(sorted(rows, key=lambda item: item[0], reverse=True)[:top_n], start=1):
            kernels = row["kernel_sources"][:3]
            kernels_str = "; ".join(kernels) if kernels else "<no kernel sources>"
            print(
                f"  {rank}. program_id={row['program_id']} chip={row['chip_id']} "
                f"duration_ms={duration:.3f} kernels={kernels_str}"
            )
        grouped = {}
        for duration, row in rows:
            kernels = tuple(row["kernel_sources"][:3])
            key = kernels if kernels else ("<no kernel sources>",)
            entry = grouped.setdefault(key, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
            entry["count"] += 1
            entry["sum_ms"] += duration
            entry["max_ms"] = max(entry["max_ms"], duration)
        print(f"{name} RT grouped totals:")
        for rank, (kernels, entry) in enumerate(
            sorted(grouped.items(), key=lambda item: item[1]["sum_ms"], reverse=True)[:top_n],
            start=1,
        ):
            kernels_str = "; ".join(kernels)
            mean_ms = entry["sum_ms"] / max(1, entry["count"])
            print(
                f"  {rank}. count={entry['count']} sum_ms={entry['sum_ms']:.3f} "
                f"mean_ms={mean_ms:.3f} max_ms={entry['max_ms']:.3f} kernels={kernels_str}"
            )

    def _summarize_graph(graph, top_n: int = 16):
        rows = []
        stack = []
        for node in graph:
            if not isinstance(node, dict):
                continue
            node_type = node.get("node_type")
            if node_type == "function_start":
                stack.append(node.get("params", {}).get("name", "unknown"))
                continue
            if node_type != "function_end":
                continue
            name = node.get("params", {}).get("name", "unknown")
            duration_ms = float(node.get("duration_ns", 0) or 0) / 1.0e6
            level = len(stack)
            rows.append((duration_ms, level, name))
            if stack:
                stack.pop()
        if not rows:
            print("Graph capture summary: no function_end duration rows found")
            return
        skip_names = {"ttnn::to_string", "Tensor::cpu"}
        aggregate = {}
        for duration_ms, _level, name in rows:
            if name in skip_names:
                continue
            if not (
                name.startswith("ttnn.")
                or name.startswith("ttnn::")
                or name.endswith("DeviceOperation")
                or name.startswith("Tensor::")
            ):
                continue
            entry = aggregate.setdefault(name, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
            entry["count"] += 1
            entry["sum_ms"] += duration_ms
            entry["max_ms"] = max(entry["max_ms"], duration_ms)
        total_ms = 0.0
        try:
            total_ms = float(ttnn.graph.extract_total_duration_from_graph(graph)) * 1000.0
        except Exception:
            total_ms = 0.0
        print(f"Graph capture summary: total_capture_ms={total_ms:.3f} function_rows={len(rows)}")
        print("Graph aggregate durations:")
        for rank, (name, entry) in enumerate(
            sorted(aggregate.items(), key=lambda item: item[1]["sum_ms"], reverse=True)[:top_n], start=1
        ):
            mean_ms = entry["sum_ms"] / max(1, entry["count"])
            print(
                f"  {rank}. sum={entry['sum_ms']:.3f} ms  count={entry['count']} "
                f"mean={mean_ms:.3f} ms  max={entry['max_ms']:.3f} ms  {name}"
            )
        print("Graph top-level durations:")
        top_level_rows = [row for row in rows if row[1] == 1]
        for rank, (duration_ms, _level, name) in enumerate(
            sorted(top_level_rows, key=lambda item: item[0], reverse=True)[:top_n], start=1
        ):
            print(f"  {rank}. {duration_ms:.3f} ms  {name}")
        print("Graph all-op durations:")
        for rank, (duration_ms, level, name) in enumerate(
            sorted(rows, key=lambda item: item[0], reverse=True)[:top_n], start=1
        ):
            print(f"  {rank}. {duration_ms:.3f} ms  level={level}  {name}")

    if args.rt_profile:
        handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(_collect)

    if args.graph_report:
        graph_path = Path(args.graph_report)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            _ = _run_scope()
            ttnn.synchronize_device(model_tt.backbone_tt.device)
            graph = ttnn.graph.end_graph_capture()
        except Exception:
            if ttnn.graph.is_graph_capture_active():
                try:
                    ttnn.graph.end_graph_capture()
                except Exception:
                    pass
            raise
        graph_path.write_text(json.dumps(graph, indent=2, default=str), encoding="utf-8")
        print(f"Graph capture saved: {graph_path}")
        _summarize_graph(graph, top_n=max(1, int(args.graph_top_n)))

    direct_times = []
    direct_records = []
    for _ in range(args.iters):
        if args.rt_profile:
            _snapshot_records()
        start = time.perf_counter()
        _ = _run_scope()
        ttnn.synchronize_device(model_tt.backbone_tt.device)
        direct_times.append((time.perf_counter() - start) * 1000.0)
        if args.rt_profile:
            direct_records.extend(_snapshot_records())

    trace_id = ttnn.begin_trace_capture(model_tt.backbone_tt.device, cq_id=0)
    _ = _run_scope()
    ttnn.end_trace_capture(model_tt.backbone_tt.device, trace_id, cq_id=0)

    try:
        trace_times = []
        trace_records = []
        for _ in range(args.iters):
            if args.rt_profile:
                _snapshot_records()
            start = time.perf_counter()
            ttnn.execute_trace(model_tt.backbone_tt.device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(model_tt.backbone_tt.device)
            trace_times.append((time.perf_counter() - start) * 1000.0)
            if args.rt_profile:
                trace_records.extend(_snapshot_records())

        def _mean(values):
            return sum(values) / max(1, len(values))

        print(f"Trace capture/execute OK (scope={args.scope}, trace_id={trace_id})")
        print(f"Direct forward mean: {_mean(direct_times):.3f} ms over {len(direct_times)} iters")
        print(f"Trace execute mean: {_mean(trace_times):.3f} ms over {len(trace_times)} iters")
        if args.rt_profile:
            _summarize_records("Direct forward", direct_records)
            _summarize_records("Trace execute", trace_records)
    finally:
        if handle is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
        model_tt.close()


if __name__ == "__main__":
    main()
