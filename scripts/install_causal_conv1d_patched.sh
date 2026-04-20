#!/usr/bin/env bash
# Build and install causal-conv1d for the Qwen3.5 fast path on a single CUDA SM.
#
# causal-conv1d 1.6.1's setup.py hard-codes `-gencode arch=compute_XX,code=sm_XX`
# for every supported architecture, and the usual `TORCH_CUDA_ARCH_LIST` env var
# does NOT override that list — so a fresh install tries to compile for every SM
# and takes an unreasonable amount of time. This script clones the release tag,
# patches setup.py to emit only the GPU's actual SM (detected from torch), and
# installs from source. Run after the NVIDIA bootstrap.
#
# Env knobs:
#   VENV_DIR     - virtualenv to install into (default: .venv)
#   VERSION      - causal-conv1d tag to clone (default: v1.6.1)
#   SM           - override the detected SM (e.g. "120"). If unset, read from torch.
#   WORK_DIR     - where to clone (default: /tmp/build/causal-conv1d)
#   MAX_JOBS     - nvcc parallel jobs (default: 8)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
VERSION="${VERSION:-v1.6.1}"
WORK_DIR="${WORK_DIR:-/tmp/build/causal-conv1d}"
MAX_JOBS="${MAX_JOBS:-8}"

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

echo "Target SM: sm_${SM} (arch=compute_${SM})"

rm -rf "$WORK_DIR"
mkdir -p "$(dirname "$WORK_DIR")"
git clone --depth 1 --branch "$VERSION" https://github.com/Dao-AILab/causal-conv1d.git "$WORK_DIR"

# Replace the hard-coded gencode block with a single -gencode for this pod's SM.
# The block we target spans the `if bare_metal_version <= ...` ladder in the
# non-HIP branch; we swap the whole thing with a one-liner that only emits the
# arch we actually need.
"$VENV_DIR/bin/python" - "$WORK_DIR/setup.py" "$SM" <<'PY'
import re
import sys
from pathlib import Path

setup_path = Path(sys.argv[1])
sm = sys.argv[2]
text = setup_path.read_text()

pattern = re.compile(
    r"(        )if bare_metal_version <= Version\(\"12\.9\"\):\n"
    r"(?:.*?\n)*?"
    r"            cc_flag\.append\(\"arch=compute_121,code=sm_121\"\)\n",
    re.MULTILINE,
)

replacement = (
    "        # DotCache patch: restrict to the pod's actual SM to avoid compiling every arch.\n"
    "        cc_flag.append(\"-gencode\")\n"
    f"        cc_flag.append(\"arch=compute_{sm},code=sm_{sm}\")\n"
)

new_text, count = pattern.subn(replacement, text)
if count != 1:
    raise SystemExit(
        f"failed to patch {setup_path} (matched {count} times); the upstream "
        "setup.py layout has changed — update install_causal_conv1d_patched.sh"
    )
setup_path.write_text(new_text)
print(f"patched {setup_path} for sm_{sm}")
PY

cd "$WORK_DIR"
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS" \
  "$VENV_DIR/bin/pip" install . --no-build-isolation

"$VENV_DIR/bin/python" - <<'PY'
import causal_conv1d
import torch
from causal_conv1d import causal_conv1d_fn

x = torch.randn(2, 8, 16, device="cuda", dtype=torch.float16)
w = torch.randn(8, 4, device="cuda", dtype=torch.float16)
y = causal_conv1d_fn(x, w)
print(f"causal_conv1d={causal_conv1d.__version__} smoke_fwd_ok shape={tuple(y.shape)} dtype={y.dtype}")
PY
