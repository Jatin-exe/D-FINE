#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUTPUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SIZE="${EVAL_SIZE:-256}"
FULL_DATASET="${FULL_DATASET:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

DATA_ROOT="${LASER_DATA_ROOT:-$ROOT_DIR/dataset}"
CKPT_PATH="${LASER_CHECKPOINT:-$ROOT_DIR/output/dfine_hgnetv2_n_custom/best_stg1.pth}"
CONFIG_PATH="${LASER_CONFIG:-$ROOT_DIR/configs/dfine/custom/dfine_hgnetv2_n_custom.yml}"
NUM_CLASSES="${LASER_NUM_CLASSES:-3}"

export TT_METAL_DEVICE_IDS=0
export PYTHONPATH=/opt/venv/lib/python3.10/site-packages
export TTNN_WEIGHT_DIR="${TTNN_WEIGHT_DIR:-$ROOT_DIR/weight/ttnn_store_single}"
export TTNN_TRACE_REGION_SIZE=67108864
export TTNN_USE_TRACE=1
export TTNN_TRACE_STRICT=0
export TTNN_TRACE_MODE=1

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
then
  PYTHON_BIN="/opt/venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

TTNN_ONLY_ARGS=()
if [[ "$FULL_DATASET" != "1" ]]; then
  TTNN_ONLY_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" "$ROOT_DIR/tools/inference/ttnn_eval.py" \
  --backend ttnn \
  --config "$CONFIG_PATH" \
  --checkpoint "$CKPT_PATH" \
  --data-root "$DATA_ROOT" \
  --split val \
  --val-batch-size 1 \
  --num-workers 0 \
  --eval-size "$EVAL_SIZE" \
  --num-classes "$NUM_CLASSES" \
  "${TTNN_ONLY_ARGS[@]}" \
  --output "$OUTPUT_DIR/laser_ttnn_only_${EVAL_SIZE}_${STAMP}.json"

echo "Saved results to $OUTPUT_DIR/laser_ttnn_only_${EVAL_SIZE}_${STAMP}.json"
