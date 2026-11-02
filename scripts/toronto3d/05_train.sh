#!/usr/bin/env bash
# Train RTPSegNet (point cloud + virtual RGB, DINOv2 2D branch): L001/L003/L004 -> val L002; logs to W&B
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

CONFIG="config/RTPSegNet_Toronto3D.yaml"
RUN_NAME="toronto3d_rgb_dinov2"

python train.py \
  --config-path "$CONFIG" \
  --device cuda:0 \
  --seed 0 \
  --epochs 200 \
  --batch-size 2 \
  --val-batch-size 2 \
  --num-workers 1 \
  --ckpt-dir "checkpoints/$RUN_NAME" \
  --resume \
  --val-every 1 \
  --log-every 20 \
  --num-vis 4 \
  --wandb-project rtpsegnet \
  --wandb-mode online \
  --run-name "$RUN_NAME"
