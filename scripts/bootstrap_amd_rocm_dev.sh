#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_rocm.sh"

DOTCACHE_ROCM_VENV="${DOTCACHE_ROCM_VENV:-${HOME}/venvs/torch-rocm}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  _dotcache_python_bin="${PYTHON_BIN}"
else
  _dotcache_python_bin=""
  for _dotcache_candidate in python3 python3.14 python3.13 python3.12 python3.11; do
    if command -v "${_dotcache_candidate}" >/dev/null 2>&1; then
      _dotcache_python_bin="${_dotcache_candidate}"
      break
    fi
  done
fi

if [[ -z "${_dotcache_python_bin}" ]]; then
  echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.9}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-1}"

if [[ -x "${DOTCACHE_ROCM_VENV}/bin/python" ]]; then
  if "${DOTCACHE_ROCM_VENV}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(1)

if getattr(torch.version, "hip", None) is None:
    raise SystemExit(1)
PY
  then
    if [[ -e "${VENV_DIR}" && ! -L "${VENV_DIR}" ]]; then
      mv "${VENV_DIR}" "${VENV_DIR}.bak.$(date +%Y%m%d%H%M%S)"
    fi
    ln -sfn "${DOTCACHE_ROCM_VENV}" "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install -e ".[dev,hf]"
    "${VENV_DIR}/bin/python" - <<'PY'
import torch

print(f"Reusing external ROCm venv at {torch.__file__}")
print(f"torch={torch.__version__}")
print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"rocm_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
PY
    exit 0
  fi
fi

VENV_ARGS=()
if [[ "${VENV_SYSTEM_SITE_PACKAGES}" == "1" ]]; then
  VENV_ARGS+=("--system-site-packages")
fi

"${_dotcache_python_bin}" -m venv "${VENV_ARGS[@]}" "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

if ! "${VENV_DIR}/bin/python" - <<'PY'
import sys

try:
    import torch
except Exception:
    raise SystemExit(1)

if not torch.cuda.is_available():
    raise SystemExit(1)

if getattr(torch.version, "hip", None) is None:
    raise SystemExit(1)

print(f"Reusing existing torch={torch.__version__}")
print(f"hip_runtime={torch.version.hip}")
print(f"rocm_device={torch.cuda.get_device_name(0)}")
PY
then
  "${VENV_DIR}/bin/pip" install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
fi

"${VENV_DIR}/bin/pip" install -e ".[dev,hf]"

"${VENV_DIR}/bin/python" - <<'PY'
import numpy
import pytest
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"transformers={transformers.__version__}")
print(f"pytest={pytest.__version__}")
print(f"numpy={numpy.__version__}")

if not torch.cuda.is_available():
    raise SystemExit("ROCm-backed torch is not available inside the project virtualenv")

if getattr(torch.version, "hip", None) is None:
    raise SystemExit("Installed torch does not appear to be a ROCm build")

print(f"rocm_device={torch.cuda.get_device_name(0)}")
PY

unset _dotcache_candidate
unset _dotcache_python_bin
