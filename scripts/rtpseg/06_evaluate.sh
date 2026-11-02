#!/usr/bin/env bash
# Evaluate the best checkpoint on the RTPSeg val sequences -> results/<run>/ + W&B eval run
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

CONFIG="config/RTPSegNet_RTPSeg_rgb.yaml"
RUN_NAME="rtpseg_rgb_dinov2"

python evaluate.py \
  --config-path "$CONFIG" \
  --checkpoint "checkpoints/$RUN_NAME/best.pt" \
  --split val \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 1 \
  --num-vote 1 \
  --out-dir "results/$RUN_NAME" \
  --save-pred \
  --num-vis 8 \
  --wandb-project rtpsegnet \
  --wandb-mode online \
  --run-name "${RUN_NAME}_eval"
