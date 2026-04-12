from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


_EXTENSION = None


def _native_enabled() -> bool:
    return os.environ.get("DOTCACHE_ENABLE_NATIVE_DIRECT_M0", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_final_mix_enabled() -> bool:
    return os.environ.get("DOTCACHE_ENABLE_NATIVE_DIRECT_M0_FINAL_MIX", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def native_direct_m0_available() -> bool:
    if not _native_enabled():
        return False
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    return bool(torch.cuda.is_available() and CUDA_HOME)


def native_direct_m0_final_mix_available() -> bool:
    if not _native_final_mix_enabled():
        return False
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    return bool(torch.cuda.is_available() and CUDA_HOME)


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    import torch
    from torch.utils.cpp_extension import load

    base = Path(__file__).resolve().parent / "cuda_kernels"
    cpp = base / "native_direct_m0.cpp"
    cu = base / "native_direct_m0_kernel.cu"
    digest = hashlib.sha1((cpp.read_text() + cu.read_text()).encode("utf-8")).hexdigest()[:12]
    build_dir = base / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name=f"dotcache_native_direct_m0_{digest}",
        sources=[str(cpp), str(cu)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    return _EXTENSION


def fused_selected_blocks_context_cuda(
    *,
    payload_words: Any,
    scales: Any,
    bias: Any,
    selected_block_ids: Any,
    valid_mask: Any,
    queries: Any,
    query_group_sums: Any,
    values: Any,
    query_scale: float,
) -> Any:
    ext = _load_extension()
    return ext.fused_selected_blocks_context_cuda(
        payload_words,
        scales,
        bias,
        selected_block_ids,
        valid_mask,
        queries,
        query_group_sums,
        values,
        float(query_scale),
    )


def fused_selected_blocks_stream_stats_cuda(
    *,
    payload_words: Any,
    scales: Any,
    bias: Any,
    selected_block_ids: Any,
    valid_mask: Any,
    queries: Any,
    query_group_sums: Any,
    values: Any,
    query_scale: float,
):
    ext = _load_extension()
    return ext.fused_selected_blocks_stream_stats_cuda(
        payload_words,
        scales,
        bias,
        selected_block_ids,
        valid_mask,
        queries,
        query_group_sums,
        values,
        float(query_scale),
    )


def softmax_value_context_cuda(
    *,
    logits: Any,
    values: Any,
    query_scale: float,
) -> Any:
    ext = _load_extension()
    return ext.softmax_value_context_cuda(
        logits,
        values,
        float(query_scale),
    )
