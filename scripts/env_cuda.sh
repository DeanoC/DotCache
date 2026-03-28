#!/usr/bin/env bash
# shellcheck shell=bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/env_cuda.sh"
  exit 1
fi

_dotcache_root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_dotcache_hf_env_file="${DOTCACHE_HF_ENV_FILE:-}"
_dotcache_hf_env_candidates=()

if [[ -n "${_dotcache_hf_env_file}" ]]; then
  _dotcache_hf_env_candidates+=("${_dotcache_hf_env_file}")
else
  _dotcache_hf_env_candidates+=(
    /workspace/.secrets/dotcache/huggingface.env
    "${_dotcache_root_dir}/.secrets/huggingface.env"
  )
fi

for _dotcache_hf_env_candidate in "${_dotcache_hf_env_candidates[@]}"; do
  if [[ -f "${_dotcache_hf_env_candidate}" ]]; then
    # shellcheck disable=SC1090
    source "${_dotcache_hf_env_candidate}"
    _dotcache_hf_env_file="${_dotcache_hf_env_candidate}"
    break
  fi
done

_dotcache_cuda_candidates=()
if [[ -n "${CUDA_HOME:-}" ]]; then
  _dotcache_cuda_candidates+=("${CUDA_HOME}")
fi
if [[ -n "${CUDA_PATH:-}" ]]; then
  _dotcache_cuda_candidates+=("${CUDA_PATH}")
fi
_dotcache_cuda_candidates+=(
  /usr/local/cuda
  /usr/local/cuda-12.8
  /usr/local/cuda-12.7
  /usr/local/cuda-12.6
)

for _dotcache_candidate in "${_dotcache_cuda_candidates[@]}"; do
  if [[ -n "$_dotcache_candidate" && -d "$_dotcache_candidate" ]]; then
    export CUDA_HOME="$_dotcache_candidate"
    export CUDA_PATH="$_dotcache_candidate"
    break
  fi
done

if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
  case ":$PATH:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
  esac
fi

if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${CUDA_HOME}/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

export HF_HOME="${DOTCACHE_HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${DOTCACHE_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

unset _dotcache_root_dir
unset _dotcache_hf_env_file
unset _dotcache_hf_env_candidate
unset _dotcache_hf_env_candidates
unset _dotcache_candidate
unset _dotcache_cuda_candidates
