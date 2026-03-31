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

_dotcache_workspace_root="${DOTCACHE_WORKSPACE_ROOT:-/workspace}"
_dotcache_default_hf_home=""
_dotcache_default_gguf_models_dir=""

if [[ -d "${_dotcache_workspace_root}" && -w "${_dotcache_workspace_root}" ]]; then
  _dotcache_default_hf_home="${_dotcache_workspace_root}/.cache/huggingface"
  _dotcache_default_gguf_models_dir="${_dotcache_workspace_root}/models/gguf"
else
  _dotcache_default_hf_home="${HOME}/.cache/huggingface"
  _dotcache_default_gguf_models_dir="${HOME}/.cache/dotcache/gguf"
fi

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

export HF_HOME="${DOTCACHE_HF_HOME:-${_dotcache_default_hf_home}}"
export TRANSFORMERS_CACHE="${DOTCACHE_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export GGUF_MODELS_DIR="${DOTCACHE_GGUF_MODELS_DIR:-${_dotcache_default_gguf_models_dir}}"
export LLAMA_CPP_N_GPU_LAYERS="${DOTCACHE_LLAMA_CPP_N_GPU_LAYERS:-999}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${GGUF_MODELS_DIR}"

_dotcache_llama_cpp_root="${DOTCACHE_LLAMA_CPP_ROOT:-/workspace/llama.cpp}"
_dotcache_llama_cpp_bin="${_dotcache_llama_cpp_root}/build/bin"

if [[ -d "${_dotcache_llama_cpp_bin}" ]]; then
  case ":$PATH:" in
    *":${_dotcache_llama_cpp_bin}:"*) ;;
    *) export PATH="${_dotcache_llama_cpp_bin}:${PATH}" ;;
  esac
fi

if [[ -z "${LLAMA_CPP_CLI:-}" && -x "${_dotcache_llama_cpp_bin}/llama-cli" ]]; then
  export LLAMA_CPP_CLI="${_dotcache_llama_cpp_bin}/llama-cli"
fi

unset _dotcache_root_dir
unset _dotcache_workspace_root
unset _dotcache_default_hf_home
unset _dotcache_default_gguf_models_dir
unset _dotcache_hf_env_file
unset _dotcache_hf_env_candidate
unset _dotcache_hf_env_candidates
unset _dotcache_llama_cpp_root
unset _dotcache_llama_cpp_bin
unset _dotcache_candidate
unset _dotcache_cuda_candidates
