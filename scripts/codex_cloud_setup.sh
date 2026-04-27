#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "dl_mnist"; then
    conda env update -n dl_mnist -f environment.yml --prune
  else
    conda env create -f environment.yml
  fi
else
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi
