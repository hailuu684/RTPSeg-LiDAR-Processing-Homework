#!/usr/bin/env bash
# Subsample tiles (0.05 m), estimate the scan trajectory -> <DATA_ROOT>/processed_grid0.05/L00x.npz
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

DATA_ROOT="/data/user/luutunghai@gmail.com/dataset/point_clouds/Toronto3D"

python tools/prepare_toronto3d.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$DATA_ROOT/processed_grid0.05" \
  --grid-size 0.05 \
  --pose-spacing 1.0 \
  --bin-dt 0.2 \
  --ground-band 0.7 \
  --min-bin-points 200 \
  --smooth-bins 9 \
  --seed 0
