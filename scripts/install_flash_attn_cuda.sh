#!/usr/bin/env bash
# Install flash-attn for the Qwen3.5 fast path, restricted to this pod's SM.
#
# flash-attn 2.8.3's setup.py honors FLASH_ATTN_CUDA_ARCHS (semicolon-separated
# list like "120"), so unlike causal-conv1d we don't have to patch source — we
# just pass the right env var. If upstream drops that knob, fall back to a
# source patch equivalent to install_causal_conv1d_patched.sh.
#
# Env knobs:
#   VENV_DIR  - virtualenv to install into (default: .venv)
#   VERSION   - flash-attn version to install (default: 2.8.3)
#   SM        - override the detected SM (e.g. "120"). If unset, read from torch.
#   MAX_JOBS  - nvcc parallel jobs (default: 4; flash-attn compile is memory-hungry)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
VERSION="${VERSION:-2.8.3}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: $VENV_DIR/bin/python not found — run scripts/bootstrap_nvidia_llama_dev.sh first" >&2
  exit 1
fi

if [[ -z "${SM:-}" ]]; then
  SM="$("$VENV_DIR/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in venv")
p = torch.cuda.get_device_properties(0)
print(f"{p.major}{p.minor}")
PY
)"
fi

echo "Target SM: sm_${SM} (FLASH_ATTN_CUDA_ARCHS=${SM})"

# Keep TMPDIR on the same filesystem as pip's wheel cache. On runpod pods the
# default /tmp is an overlay FS while ~/.cache / /workspace/.cache is MooseFS,
# and pip's post-build rename of the downloaded prebuilt wheel fails with
# "Invalid cross-device link" unless both sides live on the same mount.
WORKSPACE_TMP="${TMPDIR:-$ROOT_DIR/tmp}"
mkdir -p "$WORKSPACE_TMP"

TMPDIR="$WORKSPACE_TMP" FLASH_ATTN_CUDA_ARCHS="$SM" MAX_JOBS="$MAX_JOBS" \
  "$VENV_DIR/bin/pip" install "flash-attn==$VERSION" --no-build-isolation

"$VENV_DIR/bin/python" - <<'PY'
import flash_attn
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = flash_attn_func(q, k, v, causal=True)
print(f"flash_attn={flash_attn.__version__} smoke_fwd_ok shape={tuple(out.shape)} dtype={out.dtype}")
PY
