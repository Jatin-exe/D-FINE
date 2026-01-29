#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUTPUT_DIR"

EVAL_SIZE="${EVAL_SIZE:-256}"

export TT_METAL_DEVICE_IDS=0
export TT_METAL_HOME=/mnt/vol/loc/tt-metal
export PYTHONPATH=/mnt/vol/loc/tt-metal/ttnn:/mnt/vol/loc/tt-metal/tools
export TTNN_WEIGHT_DIR="$ROOT_DIR/weight/ttnn_store_single"
export TTNN_TRACE_REGION_SIZE=67108864
export TTNN_USE_TRACE=1
export TTNN_TRACE_STRICT=0
export TTNN_TRACE_MODE=1

PYTHON_BIN="/mnt/vol/loc/tt-metal/python_env/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "TT-Metal python not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/tools/inference/ttnn_trace_smoke.py" \
  --config "$ROOT_DIR/configs/dfine/custom/dfine_hgnetv2_n_custom.yml" \
  --checkpoint "/mnt/laser/LASER/software/alpha/alpha_training/output/dfine_hgnetv2_n_custom/best_stg1.pth" \
  --data-root "/mnt/laser/LASER/software/alpha/alpha_training/dataset" \
  --eval-size "$EVAL_SIZE" \
  --batch 1 \
  --device-id 0
