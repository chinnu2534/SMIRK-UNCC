#!/bin/bash
set -e

# ── 1. Download mediapipe face landmarker ─────────────────────────────────────
if [ ! -f "assets/face_landmarker.task" ]; then
    echo "Downloading mediapipe face landmarker..."
    mkdir -p assets
    wget -q "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
        -O assets/face_landmarker.task
    echo "Done."
fi

# ── 2. Download SMIRK pretrained checkpoint ───────────────────────────────────
if [ ! -f "pretrained_models/SMIRK_em1.pt" ]; then
    echo "Downloading SMIRK checkpoint..."
    mkdir -p pretrained_models
    gdown --id 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O pretrained_models/SMIRK_em1.pt
    echo "Done."
fi

# ── 3. Download FLAME 2020 from private HF dataset ───────────────────────────
FLAME_PKL="assets/FLAME2020/generic_model.pkl"
if [ ! -f "$FLAME_PKL" ]; then
    echo "Downloading FLAME 2020 from HF dataset..."
    mkdir -p assets/FLAME2020
    python3 -c "
import os, sys, shutil
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id='abhinavvelaga/smirk-assets',
        filename='FLAME2020.zip',
        repo_type='dataset',
        token=os.environ.get('HF_TOKEN'),
        local_dir='/app',
    )
    print('Downloaded to:', path)
except Exception as e:
    print('ERROR downloading FLAME:', e, file=sys.stderr)
    sys.exit(1)
"
    if [ -f "/app/FLAME2020.zip" ] && [ -s "/app/FLAME2020.zip" ]; then
        # zip contains FLAME2020/ subfolder, so unzip to assets/ → assets/FLAME2020/generic_model.pkl
        unzip -q /app/FLAME2020.zip -d assets/
        rm /app/FLAME2020.zip
        echo "FLAME extracted successfully."
    else
        echo "ERROR: FLAME zip not found after download."
    fi
else
    echo "FLAME model already present."
fi

# ── 4. Launch Gradio app ──────────────────────────────────────────────────────
exec python /app/app.py
