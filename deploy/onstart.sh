#!/usr/bin/env bash
# Startup script executed by vast.ai when the instance boots.
# Paste this into the vast.ai "On-start Script" field, or it is
# automatically used if you set --onstart in create_instance.sh.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
OUTPUT_DIR="/workspace/outputs"
MLFLOW_URI="/workspace/mlruns"
DATA_DIR="/workspace/data"

mkdir -p "$OUTPUT_DIR" "$MLFLOW_URI" "$DATA_DIR"

cd /workspace

echo "==> Starting optimizer comparison run at $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "==> GPUs: $NUM_GPUS" | tee -a "$OUTPUT_DIR/run.log"

python main.py \
    --task-type all \
    --gpu-num "$NUM_GPUS" \
    --num-samples 40 \
    --output-dir "$OUTPUT_DIR" \
    --mlflow-uri "$MLFLOW_URI" \
    --data-dir "$DATA_DIR" \
    --no-ngrok \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo "==> Run finished at $(date)" | tee -a "$OUTPUT_DIR/run.log"
