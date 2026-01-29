#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUTPUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
TTNN_JSON="$OUTPUT_DIR/ttnn_eval_laser_${STAMP}.json"
TORCH_JSON="$OUTPUT_DIR/torch_eval_laser_${STAMP}.json"
WATCHER_LOG="$OUTPUT_DIR/ttnn_watcher_${STAMP}.log"

CONFIG="$ROOT_DIR/configs/dfine/custom/dfine_hgnetv2_n_custom.yml"
CKPT="/mnt/laser/LASER/software/alpha/alpha_training/output/dfine_hgnetv2_n_custom/best_stg1.pth"
DATA_ROOT="/mnt/laser/LASER/software/alpha/alpha_training/dataset"
TTNN_TIMEOUT_SECONDS="${TTNN_TIMEOUT_SECONDS:-600}"

set +e
export TT_METAL_DEVICE_IDS=0
export TTNN_FORCE_SINGLE_DEVICE=1

TT_METAL_WATCHER=10 timeout "$TTNN_TIMEOUT_SECONDS" python "$ROOT_DIR/tools/inference/ttnn_eval.py" \
  --backend ttnn \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --data-root "$DATA_ROOT" \
  --split val \
  --val-batch-size 1 \
  --num-workers 0 \
  --eval-size 640 \
  --output "$TTNN_JSON"
TTNN_STATUS=$?

if [[ -f "$ROOT_DIR/generated/watcher/watcher.log" ]]; then
  cp "$ROOT_DIR/generated/watcher/watcher.log" "$WATCHER_LOG"
fi
set -e

if [[ $TTNN_STATUS -ne 0 ]]; then
  echo "TTNN eval failed or timed out (status=$TTNN_STATUS). Falling back to torch."
  python "$ROOT_DIR/tools/inference/ttnn_eval.py" \
    --backend torch \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --val-batch-size 1 \
    --num-workers 0 \
    --eval-size 640 \
    --output "$TORCH_JSON"
  echo "Saved torch results to $TORCH_JSON"
else
  echo "Saved TTNN results to $TTNN_JSON"
fi

if [[ -f "$WATCHER_LOG" ]]; then
  echo "Saved watcher log to $WATCHER_LOG"
fi
