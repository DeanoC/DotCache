from __future__ import annotations

import os
from importlib import util as importlib_util
from importlib.metadata import PackageNotFoundError, version as package_version

VLLM_V1_MULTIPROCESSING_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


def vllm_available() -> bool:
    return importlib_util.find_spec("vllm") is not None


def get_vllm_version() -> str | None:
    try:
        return package_version("vllm")
    except PackageNotFoundError:
        return None


def configure_vllm_inprocess_runtime(*, overwrite: bool = False) -> str:
    current = os.environ.get(VLLM_V1_MULTIPROCESSING_ENV)
    if overwrite or current is None:
        os.environ[VLLM_V1_MULTIPROCESSING_ENV] = "0"
        return "0"
    if current != "0":
        raise RuntimeError(
            "DotCache's vLLM 0.18.x adapter currently requires the in-process runtime; "
            f"set {VLLM_V1_MULTIPROCESSING_ENV}=0 before constructing vllm.LLM"
        )
    return current


def require_supported_vllm_version(*, supported_minor: str = "0.18") -> str:
    version = get_vllm_version()
    if version is None:
        raise RuntimeError("vLLM is not installed; install the optional vllm extra on the CUDA machine")
    if version != supported_minor and not version.startswith(f"{supported_minor}."):
        raise RuntimeError(
            f"Unsupported vLLM version {version!r}; this Phase 6 adapter targets the pinned {supported_minor}.x line"
        )
    return version
