#!/usr/bin/env bash
# Input inspection before training -> RTPSegNet/vis_inputs/rtpseg/*.png + stats.txt
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

CONFIG="config/RTPSegNet_RTPSeg_rgb.yaml"

PYTHONPATH=. python tools/vis_inputs.py \
  --config-path "$CONFIG" \
  --out-dir vis_inputs/rtpseg \
  --num-frames 4 \
  --num-timing 10 \
  --seed 0
