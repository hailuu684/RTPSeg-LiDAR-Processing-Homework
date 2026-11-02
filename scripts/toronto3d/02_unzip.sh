#!/usr/bin/env bash
# Extract the archive into <DATA_ROOT>/raw (the zip itself is kept)
set -euo pipefail

DATA_ROOT="/data/user/luutunghai@gmail.com/dataset/point_clouds/Toronto3D"

if [ -d "$DATA_ROOT/raw" ]; then
  echo "$DATA_ROOT/raw exists, skip unzip"
  exit 0
fi
python -m zipfile \
  -e "$DATA_ROOT/Toronto_3D.zip" \
  "$DATA_ROOT/raw"
