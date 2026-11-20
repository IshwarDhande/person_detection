#!/usr/bin/env bash
# Download pre-trained model weights for Vision-Based First Contact
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

echo "=============================================================================="
echo " Checking & Downloading Model Weights into ${MODELS_DIR}"
echo "=============================================================================="

# 1. YOLOv8n Person Detector weights
if [ ! -f "${MODELS_DIR}/yolov8n.pt" ]; then
    echo "[1/2] Downloading yolov8n.pt (6.3 MB)..."
    curl -L --progress-bar -o "${MODELS_DIR}/yolov8n.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
else
    echo "[1/2] yolov8n.pt already present. Skipping download."
fi

# 2. MediaPipe Face Landmarker model
if [ ! -f "${MODELS_DIR}/face_landmarker.task" ]; then
    echo "[2/2] Downloading face_landmarker.task (3.6 MB)..."
    curl -L --progress-bar -o "${MODELS_DIR}/face_landmarker.task" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
else
    echo "[2/2] face_landmarker.task already present. Skipping download."
fi

echo "------------------------------------------------------------------------------"
echo "All model weights are present in ${MODELS_DIR}:"
ls -lh "${MODELS_DIR}"
echo "=============================================================================="

