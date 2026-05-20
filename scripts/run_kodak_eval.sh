#!/usr/bin/env bash
set -e

source .venv/bin/activate

python eval.py \
  --input-dir data/kodak \
  --output-dir reports/kodak_eval \
  --mode ratio_first \
  --target-ratio 16 \
  --min-psnr 35 \
  --codecs jpeg,jxl,avif,jpeg2000 \
  --recursive false
