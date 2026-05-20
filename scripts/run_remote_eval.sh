#!/usr/bin/env bash
set -e

source .venv/bin/activate

python eval.py \
  --input-dir data/remote/mass_roads \
  --output-dir reports/remote_eval \
  --mode remote_sensing \
  --target-ratio 16 \
  --min-psnr 35 \
  --codecs jxl,avif,jpeg2000 \
  --recursive true
