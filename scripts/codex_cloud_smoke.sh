#!/usr/bin/env bash
set -euo pipefail

SMOKE_ARGS=(
  train.py
  --epochs 1
  --batch-size 32
  --device cpu
  --no-aug
  --limit-train-samples 256
  --limit-val-samples 64
  --limit-test-samples 64
  --run-name codex_cloud_smoke
  --save-path ./checkpoints/codex_cloud_smoke.pt
)

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "dl_mnist"; then
  conda run -n dl_mnist python "${SMOKE_ARGS[@]}"
else
  python "${SMOKE_ARGS[@]}"
fi
