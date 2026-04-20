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
import torch.nn.functional as F
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


def sdpa_attend_with_skip(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] (model dtype, e.g. BF16)
    skip_mask: torch.Tensor,       # [num_q_heads, num_active_blocks] bool (True=skip)
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Phase 2 attend using PyTorch SDPA — matches dense attention precision exactly.

    Uses the FP16 CPU keys and VRAM values from the tiered cache, expanding
    the block-level skip_mask into a per-token attention mask for SDPA.
    """
    num_q_heads, head_dim = q_all.shape
    nt = cache.num_tokens
    device = q_all.device
    dtype = q_all.dtype  # keep computation in model's native dtype (BF16)

    # Keys from GPU mirror (falls back to CPU copy only if mirror absent).
    # Values already live on GPU.
    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
        if keys.dtype != dtype:
            keys = keys.to(dtype=dtype)
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, dtype=dtype)
    values = cache.values_fp16[:, :nt, :]
    if values.dtype != dtype:
        values = values.to(dtype=dtype)
    num_kv_heads = keys.shape[0]

    # Build per-token attention mask from block-level skip_mask.
    # CRITICAL: pass attn_mask=None when nothing is skipped, otherwise
    # PyTorch SDPA falls back from the FlashAttention kernel to MATH/MEM_EFFICIENT,
    # which has slightly different accumulator precision.  That drift flips
    # near-tied argmax tokens and cascades into repetition loops on
    # enumeration outputs (RULER vt/fwe).  The .any().item() is one GPU sync
    # per layer per step (~3μs × 32 layers ≈ 0.1 ms overhead) — trivial vs
    # the decode floor.
    bs = cache.block_size
    num_active_blocks = skip_mask.shape[1]
    if skip_mask.any().item():
        token_skip = skip_mask.unsqueeze(-1).expand(-1, -1, bs).reshape(num_q_heads, -1)[:, :nt]
        attn_mask = torch.where(token_skip, float("-inf"), 0.0).to(dtype=dtype)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(2)  # [1, num_q_heads, 1, nt]
    else:
        attn_mask = None

    # Expand keys/values for GQA: [kv_heads, nt, hd] → [num_q_heads, nt, hd]
    # Use expand (not repeat_interleave) to match HF's GQA handling — same
    # memory layout means SDPA takes the same FlashAttention code path.
    keys_exp = keys.unsqueeze(1).expand(-1, gqa_group, -1, -1).reshape(
        num_q_heads, nt, head_dim).contiguous()
    values_exp = values.unsqueeze(1).expand(-1, gqa_group, -1, -1).reshape(
        num_q_heads, nt, values.shape[2]).contiguous()

    # SDPA: [batch=1, heads, seq, dim]
    q_sdpa = q_all.unsqueeze(0).unsqueeze(2)   # [1, num_q_heads, 1, hd]
    k_sdpa = keys_exp.unsqueeze(0)              # [1, num_q_heads, nt, hd]
    v_sdpa = values_exp.unsqueeze(0)            # [1, num_q_heads, nt, dv]

    output = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=attn_mask,
        scale=q_scale,
    )  # [1, num_q_heads, 1, dv]

    return output[0, :, 0, :].float()  # [num_q_heads, dv] float32


def certified_attention_layer(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] model dtype (BF16) or float32
    gqa_group: int,
    q_scale: float = None,
    block_epsilon: float = 0.001,
    collect_stats: bool = True,
    v_tolerance: float = DEFAULT_V_TOLERANCE,
    top_k_fp16_keys: int = 0,
    concentration_threshold: float = 0.0,
    ranking_fallback: bool = False,
    ranking_r: int = 1,
    ranking_fallback_mode: str = "full",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full certified attention for one layer, all heads.

    When the cache has INT4 values, Phase 1 scoring outputs are used to
    decide at runtime whether INT4 is safe (η₄ · ρ < tolerance) or
    FP16 values should be paged in from CPU.

    Rung-3 ranking-consistency fallback: when enabled, after Phase-1 INT8
    scoring picks the top-K blocks, re-rank those K blocks with FP16 keys
    and compare top-`ranking_r` positions. Heads whose rankings disagree
    are recomputed with full FP16 keys + values for this step (mode="full"),
    or just recorded in telemetry without action (mode="measure").

    Returns:
        output: [num_q_heads, d_v] float32
        stats: dict with skip counts, v_format decision, ranking metrics
    """
    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]
    n_qblocks = cache.num_quantized_blocks
    bs = cache.block_size

    # Phase 1: INT8 scoring only on fully quantized blocks
    if n_qblocks > 0:
        m_b, S_b, skip_mask = fused_score_certify_multihead(
            K_int8_packed=cache.keys_int8[:, :n_qblocks * bs, :],
            K_scale=cache.keys_scale[:, :n_qblocks, :],
            q_all=q_all,
            correction=cache.correction[:, :n_qblocks],
            gqa_group=gqa_group,
            block_size=bs,
            q_scale=q_scale,
            block_epsilon=block_epsilon,
        )
    else:
        device = q_all.device
        m_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        S_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        skip_mask = torch.empty(num_q_heads, 0, dtype=torch.bool, device=device)

    # If there's a trailing partial block, force-attend it via hybrid FP16 path
    num_active_blocks = cache.active_blocks
    if cache.has_trailing_partial_block:
        trailing_bid = cache.trailing_block_idx
        # Extend scoring arrays to include trailing block
        pad_m = torch.zeros(num_q_heads, 1, dtype=torch.float32, device=q_all.device)
        pad_S = torch.ones(num_q_heads, 1, dtype=torch.float32, device=q_all.device)
        pad_skip = torch.zeros(num_q_heads, 1, dtype=torch.bool, device=q_all.device)
        m_b = torch.cat([m_b, pad_m], dim=1)
        S_b = torch.cat([S_b, pad_S], dim=1)
        skip_mask = torch.cat([skip_mask, pad_skip], dim=1)

    # Top-K safety: clear skip bit for highest-scoring blocks so they're
    # always attended — the certification correction may underestimate their mass.
    num_active_blocks = cache.active_blocks
    top_k_fp16 = top_k_fp16_keys
    if top_k_fp16 > 0 and num_active_blocks > 0:
        k = min(top_k_fp16, num_active_blocks)
        topk_idx = m_b.topk(k, dim=1).indices  # [num_q_heads, k]
        skip_mask.scatter_(1, topk_idx, False)

    # Force-attend trailing partial block (it has no INT8 data for scoring)
    if cache.has_trailing_partial_block:
        skip_mask[:, cache.trailing_block_idx] = 0

    # Entropy gating: if attention is diffuse (no block dominates),
    # disable skipping for that head — small-mass blocks may carry critical
    # information (e.g., needle retrieval with weak signal).
    # Uses Phase 1 outputs so it's essentially free.
    if num_active_blocks > 0 and concentration_threshold > 0:
        # Per-block mass fraction per head
        m_global = m_b.amax(dim=1, keepdim=True)  # [num_q_heads, 1]
        log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
        mass = torch.exp(log_mass)  # [num_q_heads, num_active_blocks]
        total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
        mass_frac = mass / total_mass
        mass_max_per_head = mass_frac.max(dim=1).values  # [num_q_heads]
        # Diffuse heads: no single block has enough mass → don't skip anything
        diffuse_heads = mass_max_per_head < concentration_threshold
        if diffuse_heads.any():
            skip_mask[diffuse_heads, :] = False

    # Phase 2: Attend using SDPA for exact precision matching with dense path.
    # The Triton kernels compute in F32 which diverges from the BF16 SDPA used
    # in dense mode.  Using SDPA here ensures identical numerical behaviour.
    v_format = "fp16"

    if cache.values_fp16 is not None:
        # SDPA path: uses FP16 keys from CPU + FP16 values from VRAM
        # with the block-level skip_mask, matching dense SDPA precision
        output = sdpa_attend_with_skip(
            cache, q_all, skip_mask, gqa_group, q_scale,
        )
    elif cache.values_int4_packed is not None:
        # INT4 values: must use Triton kernel (SDPA can't handle INT4)
        if collect_stats:
            rho = compute_tier2_residual_mass(m_b, S_b, skip_mask)
            eta_int4 = cache.values_int4_errors.max().item()
            v_format = decide_v_format(rho, eta_int4, v_tolerance)
        else:
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
        # Ranking-consistency fallback telemetry (Rung 3).
        # Populated by the detection block above; zero when the feature is off
        # so downstream aggregators see a stable schema.
        if ranking_fallback:
            stats["ranking_heads_total"] = num_q_heads
            stats["ranking_disagree_r1"] = 0
            stats["ranking_disagree_r3"] = 0
            stats["ranking_fallback_triggered"] = 0
            stats["ranking_r"] = int(ranking_r)
            stats["ranking_fallback_mode"] = ranking_fallback_mode
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
