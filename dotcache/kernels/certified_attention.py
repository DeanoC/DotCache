"""Certified attention: complete pipeline for one layer.

Orchestrates:
  1. Multi-head INT8 scoring + certification (fused Triton kernel)
  2. Runtime V-format decision based on mass partition (ρ check)
  3. Multi-head selective attention (fused Triton kernel)

The V-format decision uses Phase 1 outputs at zero additional cost:
  - Compute tier-2 residual mass ρ from m_b, S_b, skip_mask
  - If η₄ · ρ < tolerance → INT4 values (45% less VRAM)
  - Else → page in FP16 values from CPU
"""
from __future__ import annotations

import torch
from typing import Any

from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
from dotcache.kernels.selective_attend_triton import (
    selective_attend_multihead,
    selective_attend_multihead_int8,
    selective_attend_multihead_int8k_int4v,
    selective_attend_multihead_hybrid,
)


# Default tolerance for INT4 value error (η₄ · ρ must be below this)
DEFAULT_V_TOLERANCE = 0.5

# Number of top-K blocks whose mass counts toward α_K (not tier-2)
TOP_K_BLOCKS = 4


def compute_tier2_residual_mass(
    m_b: torch.Tensor,       # [num_q_heads, num_blocks] block maxima
    S_b: torch.Tensor,       # [num_q_heads, num_blocks] block sums
    skip_mask: torch.Tensor,  # [num_q_heads, num_blocks] bool (True=skip)
    top_k: int = TOP_K_BLOCKS,
) -> torch.Tensor:
    """Compute per-head tier-2 residual mass ρ from Phase 1 outputs.

    ρ = 1 - α_K - β, where:
      α_K = fraction of mass on top-K blocks (by m_b)
      β = fraction of mass on skipped blocks

    Returns: [num_q_heads] float32, the worst-case ρ per head.
    """
    num_q_heads, num_blocks = m_b.shape

    # Unnormalised mass per block: S_b * exp(m_b - m_global)
    m_global = m_b.amax(dim=1, keepdim=True)  # [num_q_heads, 1]
    log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
    mass = torch.exp(log_mass)  # [num_q_heads, num_blocks]
    total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)

    # Normalised mass fractions
    mass_frac = mass / total_mass  # [num_q_heads, num_blocks]

    # α_K: mass on top-K blocks (by m_b score, per head)
    k = min(top_k, num_blocks)
    _, topk_idx = m_b.topk(k, dim=1)  # [num_q_heads, k]
    alpha_K = mass_frac.gather(1, topk_idx).sum(dim=1)  # [num_q_heads]

    # β: mass on skipped blocks
    beta = (mass_frac * skip_mask.float()).sum(dim=1)  # [num_q_heads]

    # ρ = 1 - α_K - β (clamped to [0, 1])
    rho = (1.0 - alpha_K - beta).clamp(min=0.0, max=1.0)
    return rho


def decide_v_format(
    rho: torch.Tensor,       # [num_q_heads] tier-2 residual mass
    eta_int4: float,          # worst-case INT4 error bound for this layer
    tolerance: float = DEFAULT_V_TOLERANCE,
) -> str:
    """Decide INT4 vs FP16 values based on mass-weighted error bound.

    Uses worst-case ρ across all heads (conservative, per-layer decision).
    Returns 'int4' or 'fp16'.
    """
    rho_worst = rho.max().item()
    int4_error = eta_int4 * rho_worst
    return "int4" if int4_error < tolerance else "fp16"


def certified_attention_layer(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    gqa_group: int,
    q_scale: float = None,
    block_epsilon: float = 0.001,
    collect_stats: bool = True,
    v_tolerance: float = DEFAULT_V_TOLERANCE,
    top_k_fp16_keys: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full certified attention for one layer, all heads.

    When the cache has INT4 values, Phase 1 scoring outputs are used to
    decide at runtime whether INT4 is safe (η₄ · ρ < tolerance) or
    FP16 values should be paged in from CPU.

    Returns:
        output: [num_q_heads, d_v] float32
        stats: dict with skip counts, v_format decision
    """
    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]

    # Phase 1: Multi-head INT8 scoring + certification
    # Use active slices (not full pre-allocated buffers)
    m_b, S_b, skip_mask = fused_score_certify_multihead(
        K_int8_packed=cache.keys_int8_active(),
        K_scale=cache.keys_scale_active(),
        q_all=q_all,
        correction=cache.correction_active(),
        gqa_group=gqa_group,
        block_size=cache.block_size,
        q_scale=q_scale,
        block_epsilon=block_epsilon,
    )

    # Top-K FP16 key fallback: replace INT8 keys with FP16 originals for the
    # highest-scoring blocks. This eliminates the INT8 key quantisation error
    # on the blocks that contribute most attention mass (the "needle" blocks).
    # Cost: one CPU→GPU page-in per KV head for K blocks (~3μs at K=4).
    #
    # Only operate on block-aligned tokens (after append, num_tokens may
    # exceed num_blocks * block_size by a partial block).
    top_k_fp16 = top_k_fp16_keys
    aligned_tokens = cache.aligned_tokens
    use_mixed_keys = False

    num_active_blocks = cache.active_blocks
    use_hybrid = top_k_fp16 > 0 and cache.keys_fp16_cpu is not None and num_active_blocks > 0

    if use_hybrid:
        # Build per-head top-K mask
        topk_mask = torch.zeros(num_q_heads, num_active_blocks, dtype=torch.int32,
                                device=cache.keys_int8.device)
        all_topk = set()
        for qh in range(num_q_heads):
            topk_idx = m_b[qh].topk(min(top_k_fp16, num_active_blocks)).indices
            topk_mask[qh, topk_idx] = 1
            all_topk.update(topk_idx.tolist())

        # Hot-page cache: keep FP16 blocks in VRAM. Only page-in blocks
        # that aren't already cached. Top-K blocks tend to be stable across
        # decode steps (the "needle" blocks don't move).
        if not hasattr(cache, '_fp16_hot_cache'):
            cache._fp16_hot_cache = torch.zeros(
                cache.kv_heads, cache.keys_int8.shape[1], cache.head_dim,
                dtype=torch.float16, device=cache.keys_int8.device,
            )
            cache._fp16_hot_set = set()
        # Page in only new blocks not already in hot cache
        new_blocks = all_topk - cache._fp16_hot_set
        for bid in new_blocks:
            start = bid * cache.block_size
            end = start + cache.block_size
            cache._fp16_hot_cache[:, start:end, :] = cache.keys_fp16_cpu[:, start:end, :].to(
                device=cache.keys_int8.device, non_blocking=True,
            )
        cache._fp16_hot_set.update(new_blocks)
        keys_fp16_sparse = cache._fp16_hot_cache[:, :aligned_tokens, :]

    # Phase 2: Select V format and run attention
    v_format = "fp16"  # default

    if use_hybrid:
        # Hybrid kernel: INT8 keys + sparse FP16 for top-K, one Triton launch
        aligned = cache.aligned_tokens
        values_for_attend = (cache.values_fp16[:, :aligned, :]
                             if cache.values_fp16 is not None
                             else cache.values_fp16_cpu[:, :aligned, :].to(device=cache.keys_int8.device))
        output = selective_attend_multihead_hybrid(
            keys_int8=cache.keys_int8_active(),
            keys_scale=cache.keys_scale_active(),
            keys_fp16=keys_fp16_sparse,
            topk_mask=topk_mask,
            values_fp16=values_for_attend,
            q_all=q_all,
            skip_mask_i32=skip_mask.to(torch.int32),
            gqa_group=gqa_group,
            block_size=cache.block_size,
            q_scale=q_scale,
        )
        del topk_mask  # keys_fp16_sparse is the hot cache — keep it
    elif cache.values_int4_packed is not None:
        # Runtime V-format decision based on mass partition
        if collect_stats:
            rho = compute_tier2_residual_mass(m_b, S_b, skip_mask)
            eta_int4 = cache.values_int4_errors.max().item()
            v_format = decide_v_format(rho, eta_int4, v_tolerance)
        else:
            # When stats disabled (timed runs), use INT4 unconditionally
            # (the caller should set v_tolerance=inf to force INT4, or
            # pre-decide format and use the appropriate cache)
            v_format = "int4"

        if v_format == "int4":
            output = selective_attend_multihead_int8k_int4v(
                keys_int8=cache.keys_int8_active(),
                keys_scale=cache.keys_scale_active(),
                values_int4_packed=cache.values_int4_packed,
                values_int4_scales=cache.values_int4_scales,
                values_int4_zeros=cache.values_int4_zeros,
                q_all=q_all,
                skip_mask_i32=skip_mask.to(torch.int32),
                gqa_group=gqa_group,
                block_size=cache.block_size,
                group_size=cache.values_int4_group_size,
                q_scale=q_scale,
            )
        else:
            # Fallback: page in FP16 values from CPU
            if cache.values_fp16_cpu is not None:
                values_fp16 = cache.values_fp16_cpu.to(
                    device=cache.keys_int8.device, non_blocking=True,
                )
            elif cache.values_fp16 is not None:
                values_fp16 = cache.values_fp16
            else:
                raise ValueError("INT4 unsafe and no FP16 fallback available")
            output = selective_attend_multihead_int8(
                keys_int8=cache.keys_int8_active(),
                keys_scale=cache.keys_scale_active(),
                values_fp16=values_fp16,
                q_all=q_all,
                skip_mask_i32=skip_mask.to(torch.int32),
                gqa_group=gqa_group,
                block_size=cache.block_size,
                q_scale=q_scale,
            )
    elif cache.values_fp16 is not None:
        # Keys and values must have matching N (block-aligned)
        aligned = cache.aligned_tokens
        output = selective_attend_multihead_int8(
            keys_int8=cache.keys_int8_active(),
            keys_scale=cache.keys_scale_active(),
            values_fp16=cache.values_fp16[:, :aligned, :],
            q_all=q_all,
            skip_mask_i32=skip_mask.to(torch.int32),
            gqa_group=gqa_group,
            block_size=cache.block_size,
            q_scale=q_scale,
        )
    else:
        raise ValueError("No values available in cache")

    # Stats
    if collect_stats:
        total_blocks = num_q_heads * cache.num_blocks
        skipped = skip_mask.sum().item()
        stats = {
            "total_blocks": total_blocks,
            "skipped_blocks": int(skipped),
            "skip_rate": float(skipped) / float(total_blocks),
            "attended_blocks": total_blocks - int(skipped),
            "v_format": v_format,
        }
        if cache.values_int4_packed is not None:
            stats["rho_max"] = rho.max().item()
            stats["rho_mean"] = rho.mean().item()
            stats["eta_int4"] = eta_int4
            stats["int4_error_bound"] = eta_int4 * rho.max().item()
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
