from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


_EXTENSION = None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def certified_blackwell_available() -> bool:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    if not (torch.cuda.is_available() and CUDA_HOME):
        return False
    if torch.cuda.get_device_capability()[0] < 12:
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    import torch
    from torch.utils.cpp_extension import load

    base = Path(__file__).resolve().parent / "cuda_kernels"
    cpp = base / "certified_blackwell.cpp"
    cu = base / "certified_blackwell_kernel.cu"
    digest = hashlib.sha1((cpp.read_text() + cu.read_text()).encode("utf-8")).hexdigest()[:12]
    build_dir = base / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name=f"dotcache_certified_blackwell_{digest}",
        sources=[str(cpp), str(cu)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        build_directory=str(build_dir),
        verbose=_env_flag("DOTCACHE_NATIVE_VERBOSE", default=False),
    )
    return _EXTENSION


def hybrid_mixedv_split_k_cuda(
    *,
    keys_int8: Any,
    keys_scale: Any,
    keys_zero_points: Any,
    keys_fp16: Any,
    topk_mask: Any,
    values_int4_packed: Any,
    values_int4_scales: Any,
    values_int4_zeros: Any,
    values_fp16_scratch: Any,
    value_fp16_mask: Any,
    value_block_slots: Any,
    q_all: Any,
    skip_mask_i32: Any,
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    num_splits: int | None = None,
) -> Any:
    ext = _load_extension()
    if num_splits is None:
        num_blocks = int(keys_int8.shape[1]) // int(block_size)
        target = int(os.environ.get("DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT", "128"))
        ns = max(1, (num_blocks + target - 1) // target)
        num_splits = 1
        while num_splits < ns:
            num_splits *= 2
        num_splits = min(num_splits, num_blocks)
    lbv = int(block_size if last_block_valid is None else last_block_valid)
    return ext.hybrid_mixedv_split_k_cuda(
        keys_int8.contiguous(),
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        keys_fp16.contiguous(),
        topk_mask.contiguous(),
        values_int4_packed.contiguous(),
        values_int4_scales.contiguous(),
        values_int4_zeros.contiguous(),
        values_fp16_scratch.contiguous(),
        value_fp16_mask.contiguous(),
        value_block_slots.contiguous(),
        q_all.contiguous(),
        skip_mask_i32.contiguous(),
        int(gqa_group),
        int(block_size),
        int(group_size),
        float(q_scale),
        int(lbv),
        int(num_splits),
    )
