#!/usr/bin/env bash
# Toronto-3D archive from the official OneDrive share (https://github.com/WeikaiTan/Toronto-3D); skipped if present
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

DATA_ROOT="/data/user/luutunghai@gmail.com/dataset/point_clouds/Toronto3D"

python tools/download.py \
  --url 'https://1drv.ms/u/s!Amlc6yZnF87psX6hKS8VOQllVvj4?e=yWhrYX' \
  --source onedrive \
  --output "$DATA_ROOT/Toronto_3D.zip"
