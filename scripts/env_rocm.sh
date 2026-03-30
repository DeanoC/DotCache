#!/usr/bin/env bash
# shellcheck shell=bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/env_rocm.sh"
  exit 1
fi

_dotcache_root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DOTCACHE_HF_HOME="${DOTCACHE_HF_HOME:-${HOME}/.cache/dotcache/huggingface}"
export DOTCACHE_TRANSFORMERS_CACHE="${DOTCACHE_TRANSFORMERS_CACHE:-${DOTCACHE_HF_HOME}/transformers}"
export DOTCACHE_GGUF_MODELS_DIR="${DOTCACHE_GGUF_MODELS_DIR:-${HOME}/.cache/dotcache/models/gguf}"

# Reuse the shared Hugging Face and llama.cpp environment normalization.
# ROCm still runs through PyTorch's cuda namespace, so the rest of the repo
# can keep using the same runtime flags once the wheel and loader paths exist.
# shellcheck disable=SC1091
source "${_dotcache_root_dir}/scripts/env_cuda.sh"

_dotcache_rocm_candidates=()
if [[ -n "${ROCM_HOME:-}" ]]; then
  _dotcache_rocm_candidates+=("${ROCM_HOME}")
fi
if [[ -n "${ROCM_PATH:-}" ]]; then
  _dotcache_rocm_candidates+=("${ROCM_PATH}")
fi
if [[ -n "${HIP_PATH:-}" ]]; then
  _dotcache_rocm_candidates+=("${HIP_PATH}")
fi
_dotcache_rocm_candidates+=(
  /usr/lib64/rocm
  /opt/rocm
  /usr/lib/rocm
)

for _dotcache_candidate in "${_dotcache_rocm_candidates[@]}"; do
  if [[ -n "${_dotcache_candidate}" && -d "${_dotcache_candidate}" ]]; then
    export ROCM_HOME="${_dotcache_candidate}"
    export ROCM_PATH="${_dotcache_candidate}"
    export HIP_PATH="${_dotcache_candidate}"
    break
  fi
done

if [[ -n "${ROCM_HOME:-}" && -d "${ROCM_HOME}/bin" ]]; then
  case ":${PATH}:" in
    *":${ROCM_HOME}/bin:"*) ;;
    *) export PATH="${ROCM_HOME}/bin:${PATH}" ;;
  esac
fi

if [[ -n "${ROCM_HOME:-}" && -d "${ROCM_HOME}/llvm/bin" ]]; then
  case ":${PATH}:" in
    *":${ROCM_HOME}/llvm/bin:"*) ;;
    *) export PATH="${ROCM_HOME}/llvm/bin:${PATH}" ;;
  esac
fi

for _dotcache_rocm_libdir in "${ROCM_HOME:-}/lib" "${ROCM_HOME:-}/lib64"; do
  if [[ -n "${ROCM_HOME:-}" && -d "${_dotcache_rocm_libdir}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${_dotcache_rocm_libdir}:"*) ;;
      *) export LD_LIBRARY_PATH="${_dotcache_rocm_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
done

unset _dotcache_root_dir
unset _dotcache_candidate
unset _dotcache_rocm_candidates
unset _dotcache_rocm_libdir
