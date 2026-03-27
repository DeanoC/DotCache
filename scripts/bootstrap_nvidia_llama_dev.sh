#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.8}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-1}"

VENV_ARGS=()
if [[ "$VENV_SYSTEM_SITE_PACKAGES" == "1" ]]; then
  VENV_ARGS+=("--system-site-packages")
fi

"$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

if ! "$VENV_DIR/bin/python" - <<'PY'
import sys

try:
    import torch
except Exception:
    raise SystemExit(1)

if not torch.cuda.is_available():
    raise SystemExit(1)

print(f"Reusing existing torch={torch.__version__}")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY
then
  "$VENV_DIR/bin/pip" install "$TORCH_SPEC"
fi

"$VENV_DIR/bin/pip" install -e ".[dev,hf]"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import pytest
import numpy

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"pytest={pytest.__version__}")
print(f"numpy={numpy.__version__}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the project virtualenv")

device_name = torch.cuda.get_device_name(0)
print(f"cuda_device={device_name}")
PY
