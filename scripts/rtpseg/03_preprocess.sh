#!/usr/bin/env bash
# Link sequences to <DATA_ROOT>/sequences, precompute IMU-rectified LiDAR -> RGB projections -> <DATA_ROOT>/processed_rgb_proj
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

DATA_ROOT="/data/user/luutunghai@gmail.com/dataset/point_clouds/RTPSeg"

python tools/prepare_rtpseg.py \
  --data-root "$DATA_ROOT" \
  --json-dir json \
  --min-depth 1.0
