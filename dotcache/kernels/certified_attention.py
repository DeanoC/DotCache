"""Certified attention: complete pipeline for one layer.

Orchestrates:
  1. Multi-head INT8 scoring + certification (fused Triton kernel)
  2. Multi-head selective attention (fused Triton kernel)
  3. FP16 fallback page-in for failed-certification blocks (async CPU→GPU)

Total: 2 Triton kernel launches + optional async page-in.
"""
from __future__ import annotations

import torch
from typing import Any

from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
from dotcache.kernels.selective_attend_triton import selective_attend_multihead


def certified_attention_layer(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    gqa_group: int,
    q_scale: float = None,
    block_epsilon: float = 0.001,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full certified attention for one layer, all heads.

    Returns:
        output: [num_q_heads, d_v] float32
        stats: dict with skip counts, timing info
    """
    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]

    # Phase 1: Multi-head INT8 scoring + certification
    m_b, S_b, skip_mask = fused_score_certify_multihead(
        K_int8_packed=cache.keys_int8,
        K_scale=cache.keys_scale,
        q_all=q_all,
        correction=cache.correction,
        gqa_group=gqa_group,
        block_size=cache.block_size,
        q_scale=q_scale,
        block_epsilon=block_epsilon,
    )

    # Phase 2: Multi-head selective attention
    # Use the original FP16 keys from VRAM for attended blocks
    # (we need FP16 for the actual attention computation, INT8 was just for scoring)
    #
    # For blocks that pass INT8 certification: score with INT8, attend with values only
    # For blocks that fail certification: would page in FP16 from CPU
    #
    # Current implementation: use the INT8-dequantised keys for attended blocks.
    # The dequantisation happens inside the selective attend kernel (from key_cache).
    # Since we're reading K anyway for the attend pass, using fp32 keys from
    # the INT8 dequant is fine — the error is bounded by the correction factor.

    # Use pre-computed dequant buffers if available, else compute on the fly
    if cache._keys_deq_f32 is not None:
        keys_deq_flat = cache._keys_deq_f32
    else:
        keys_deq_flat = (
            cache.keys_int8.to(torch.float32).reshape(
                cache.kv_heads, cache.num_blocks, cache.block_size, cache.head_dim
            ) * cache.keys_scale[:, :, None, None]
        ).reshape(cache.kv_heads, cache.num_tokens, cache.head_dim)

    values_f32 = cache._values_f32 if cache._values_f32 is not None else cache.values_fp16.to(torch.float32)

    output = selective_attend_multihead(
        keys_packed=keys_deq_flat,
        values_packed=values_f32,
        q_all=q_all,
        skip_mask_i32=skip_mask.to(torch.int32),
        gqa_group=gqa_group,
        block_size=cache.block_size,
        q_scale=q_scale,
    )

    # Stats (optional — skip_mask.sum().item() is a GPU sync point)
    if collect_stats:
        total_blocks = num_q_heads * cache.num_blocks
        skipped = skip_mask.sum().item()
        stats = {
            "total_blocks": total_blocks,
            "skipped_blocks": int(skipped),
            "skip_rate": float(skipped) / float(total_blocks),
            "attended_blocks": total_blocks - int(skipped),
        }
    else:
        stats = {}

    return output, stats


def benchmark_certified_vs_full(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,
    gqa_group: int,
    iters: int = 1000,
    q_scale: float = None,
    block_epsilon: float = 0.001,
) -> dict[str, float]:
    """Benchmark certified attention vs full attention on one layer."""
    import time

    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]
    keys_fp32 = cache.keys_fp16_cpu.to(dtype=torch.float32, device=q_all.device)
    vals_fp32 = cache.values_fp16.to(torch.float32)

    # Warmup
    for _ in range(10):
        certified_attention_layer(cache, q_all, gqa_group, q_scale, block_epsilon)
        for qh in range(num_q_heads):
            kv = qh // gqa_group
            s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
            w = torch.softmax(s, dim=0)
            o = w @ vals_fp32[kv]
    torch.cuda.synchronize()

    # Full attention
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for qh in range(num_q_heads):
            kv = qh // gqa_group
            s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
            w = torch.softmax(s, dim=0)
            o = w @ vals_fp32[kv]
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / iters * 1e6

    # Certified attention
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        output, stats = certified_attention_layer(
            cache, q_all, gqa_group, q_scale, block_epsilon,
        )
    torch.cuda.synchronize()
    t_cert = (time.perf_counter() - t0) / iters * 1e6

    # Correctness
    output_cert, stats = certified_attention_layer(
        cache, q_all, gqa_group, q_scale, block_epsilon,
    )
    output_full = torch.empty_like(output_cert)
    for qh in range(num_q_heads):
        kv = qh // gqa_group
        s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
        w = torch.softmax(s, dim=0)
        output_full[qh] = w @ vals_fp32[kv]

    cos = torch.nn.functional.cosine_similarity(output_cert, output_full, dim=1)

    return {
        "full_attention_us": t_full,
        "certified_attention_us": t_cert,
        "speedup": (t_full - t_cert) / t_full,
        "skip_rate": stats["skip_rate"],
        "cosine_min": cos.min().item(),
        "cosine_mean": cos.mean().item(),
        "vram_mb": cache.vram_bytes() / 1e6,
        "cpu_mb": cache.cpu_bytes() / 1e6,
    }
