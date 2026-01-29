#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

DATASET_REPO="${LASER_DATASET_REPO:-Laudando-Associates-LLC/pucks}"
MODEL_REPO_N="${LASER_MODEL_REPO_N:-Laudando-Associates-LLC/d-fine-nano}"
MODEL_REPO_S="${LASER_MODEL_REPO_S:-Laudando-Associates-LLC/d-fine-small}"
MODEL_REPO_M="${LASER_MODEL_REPO_M:-Laudando-Associates-LLC/d-fine-medium}"
MODEL_REPO_L="${LASER_MODEL_REPO_L:-Laudando-Associates-LLC/d-fine-large}"
MODEL_REPO_X="${LASER_MODEL_REPO_X:-Laudando-Associates-LLC/d-fine-xlarge}"

DATA_ROOT="${LASER_DATA_ROOT:-$ROOT_DIR/dataset}"
OUTPUT_ROOT="${LASER_OUTPUT_ROOT:-$ROOT_DIR/output}"
WEIGHT_DIR="${TTNN_WEIGHT_DIR:-$ROOT_DIR/weight/ttnn_store_single}"

echo "[setup] Using python: $PYTHON_BIN"
echo "[setup] Installing Python deps from requirements.txt"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install --upgrade huggingface_hub[hf_xet]

if [[ ! -d /opt/venv/lib/python3.10/site-packages ]]; then
  echo "[setup] WARNING: /opt/venv/lib/python3.10/site-packages not found." >&2
  echo "[setup] TTNN must be available via PYTHONPATH or site-packages." >&2
else
  echo "[setup] TTNN path detected at /opt/venv/lib/python3.10/site-packages"
fi

if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[setup] Using HUGGINGFACE_HUB_TOKEN for HF downloads"
fi

echo "[setup] Downloading dataset to: $DATA_ROOT"
"$PYTHON_BIN" - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "${DATASET_REPO}"
local_dir = "${DATA_ROOT}"
Path(local_dir).mkdir(parents=True, exist_ok=True)
print(f"Downloading dataset {repo_id} to {local_dir} ...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["images/*", "annotations/*"],
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print("Dataset download complete.")
PY

echo "[setup] Downloading model checkpoints to: $OUTPUT_ROOT"
"$PYTHON_BIN" - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

models = [
    ("${MODEL_REPO_N}", "dfine_hgnetv2_n_custom"),
    ("${MODEL_REPO_S}", "dfine_hgnetv2_s_custom"),
    ("${MODEL_REPO_M}", "dfine_hgnetv2_m_custom"),
    ("${MODEL_REPO_L}", "dfine_hgnetv2_l_custom"),
    ("${MODEL_REPO_X}", "dfine_hgnetv2_x_custom"),
]

output_root = Path("${OUTPUT_ROOT}")
temp_root = output_root / "hf_temp_download"
temp_root.mkdir(parents=True, exist_ok=True)

for repo_id, out_dir in models:
    print(f"Downloading {repo_id}...")
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["pytorch_model.bin"],
        local_dir=temp_root,
        local_dir_use_symlinks=False,
    )
    out_path = output_root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    src = Path(downloaded_dir) / "pytorch_model.bin"
    dst = out_path / "best_stg1.pth"
    shutil.copy(src, dst)
    print(f"Saved {dst}")

if temp_root.exists():
    shutil.rmtree(temp_root)
PY

if [[ ! -f "$WEIGHT_DIR/manifest.json" ]]; then
  echo "[setup] Exporting TTNN weights to: $WEIGHT_DIR"
  export TTNN_WEIGHT_DIR="$WEIGHT_DIR"
  export PYTHONPATH="/opt/venv/lib/python3.10/site-packages:$ROOT_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/tools/export/ttnn_export_weights.py" \
    --config "$ROOT_DIR/configs/dfine/custom/dfine_hgnetv2_n_custom.yml" \
    --checkpoint "$OUTPUT_ROOT/dfine_hgnetv2_n_custom/best_stg1.pth"
else
  echo "[setup] TTNN weights already present at $WEIGHT_DIR"
fi

echo "[setup] Done."
