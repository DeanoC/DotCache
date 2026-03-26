#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install "$TORCH_SPEC"
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
