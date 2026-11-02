#!/usr/bin/env bash
# Authors' released RTPSegNet (RGB + TIR training, LiDAR-only inference): convert torchsparse -> spconv weights,
# then evaluate on the RTPSeg val sequences exactly like 06_evaluate.sh -> results/<run>/ + W&B eval run
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

CONFIG="config/RTPSegNet_RTPSeg_author.yaml"
AUTHOR_CKPT="/home/luutunghai@gmail.com/projects/course_ecnu_lidar_processing/RTPSegNet_best.ckpt"
RUN_NAME="rtpseg_author_rgb_tir"

PYTHONPATH=. python tools/convert_author_ckpt.py \
  --src "$AUTHOR_CKPT" \
  --config-path "$CONFIG" \
  --out "checkpoints/$RUN_NAME/best.pt"

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
