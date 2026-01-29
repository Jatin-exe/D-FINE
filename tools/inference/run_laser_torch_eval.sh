#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUTPUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SIZE="${EVAL_SIZE:-256}"
OUTPUT_JSON="$OUTPUT_DIR/torch_eval_laser_${STAMP}.json"

DATA_ROOT="${LASER_DATA_ROOT:-$ROOT_DIR/dataset}"
CKPT_PATH="${LASER_CHECKPOINT:-$ROOT_DIR/output/dfine_hgnetv2_n_custom/best_stg1.pth}"
CONFIG_PATH="${LASER_CONFIG:-$ROOT_DIR/configs/dfine/custom/dfine_hgnetv2_n_custom.yml}"
NUM_CLASSES="${LASER_NUM_CLASSES:-3}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/tools/inference/ttnn_eval.py" \
  --backend torch \
  --config "$CONFIG_PATH" \
  --checkpoint "$CKPT_PATH" \
  --data-root "$DATA_ROOT" \
  --split val \
  --val-batch-size 1 \
  --num-workers 0 \
  --eval-size "$EVAL_SIZE" \
  --num-classes "$NUM_CLASSES" \
  --output "$OUTPUT_JSON"

echo "Saved results to $OUTPUT_JSON"
