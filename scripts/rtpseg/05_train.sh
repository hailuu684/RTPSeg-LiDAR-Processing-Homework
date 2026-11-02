#!/usr/bin/env bash
# Train RTPSegNet (point cloud + RGB camera, no TIR, DINOv2 2D branch) on RTPSeg; logs to W&B
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

CONFIG="config/RTPSegNet_RTPSeg_rgb.yaml"
RUN_NAME="rtpseg_rgb_dinov2"

python train.py \
  --config-path "$CONFIG" \
  --device cuda:0 \
  --seed 0 \
  --epochs 200 \
  --batch-size 2 \
  --val-batch-size 8 \
  --num-workers 1 \
  --ckpt-dir "checkpoints/$RUN_NAME" \
  --resume \
  --val-every 1 \
  --log-every 20 \
  --num-vis 4 \
  --wandb-project rtpsegnet \
  --wandb-mode online \
  --run-name "$RUN_NAME"
