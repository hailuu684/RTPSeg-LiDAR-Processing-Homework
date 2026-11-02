#!/usr/bin/env bash
# RTPSeg archive from the authors' Google Drive link (README of sssssyf/RTPSeg); resumable, skipped if present
set -euo pipefail
cd "$(dirname "$0")/../../RTPSegNet"

DATA_ROOT="/data/user/luutunghai@gmail.com/dataset/point_clouds/RTPSeg"

mkdir -p "$DATA_ROOT"
python tools/download.py \
  --url 'https://drive.google.com/file/d/1F0SVT87ynoa_3JaOE602tBzMojjLVPWR/view?usp=sharing' \
  --source gdrive \
  --output "$DATA_ROOT/RTPSeg.zip"
