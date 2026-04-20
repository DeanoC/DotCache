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

import math
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

# Adaptive top-K* defaults (paper §3.3).
DEFAULT_TAU_COV = 0.995
DEFAULT_K_MIN = 2
# None = no upper clamp; the selector lets tau_cov fully dictate K* per head.
DEFAULT_K_MAX: int | None = None

# Rung-1 fallback defaults (paper §3.4). When the adaptive selector's tail
# mass exceeds DEFAULT_RUNG1_THRESHOLD (k_max hit, τ_cov not reached), expand
# the top-K set by multiplying K* by DEFAULT_RUNG1_MULTIPLIER.
DEFAULT_RUNG1_THRESHOLD = 0.02
DEFAULT_RUNG1_MULTIPLIER = 2.0


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


def compute_adaptive_topk_mask(
    m_b: torch.Tensor,       # [num_q_heads, num_blocks] Phase-1 block max (INT8 estimate)
    S_b: torch.Tensor,       # [num_q_heads, num_blocks] Phase-1 block sum
    tau_cov: float = DEFAULT_TAU_COV,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = DEFAULT_K_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper §3.3 adaptive top-K* selector (cumulative-mass threshold).

    Per head: sort blocks by estimated mass, find smallest K such that
    cumulative mass ≥ `tau_cov`, clamp to [k_min, k_max]. Returns:

    - topk_mask [H, B] bool: True = block is in the top-K* for that head.
    - k_star    [H] int32: actual K* selected per head (post-clamp).
    - tail_mass [H] float32: 1 − Σ mass on top-K* (INT8-estimated).
    - tau_cov_actual [H] float32: actual cumulative mass captured at K*.

    All computation stays on device; this function has zero CPU syncs.
    """
    num_q_heads, num_blocks = m_b.shape
    device = m_b.device
    if num_blocks == 0:
        empty_bool = torch.zeros(num_q_heads, 0, dtype=torch.bool, device=device)
        zeros_int = torch.zeros(num_q_heads, dtype=torch.int32, device=device)
        zeros_f32 = torch.zeros(num_q_heads, dtype=torch.float32, device=device)
        return empty_bool, zeros_int, zeros_f32, zeros_f32

    # Per-head normalised mass, stable via log-sum-exp.
    m_global = m_b.amax(dim=1, keepdim=True)
    log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
    mass = torch.exp(log_mass)
    total = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
    mass_frac = mass / total                                       # [H, B]

    # Sort descending per head; cumulative mass in sorted order.
    sorted_mass, sorted_idx = mass_frac.sort(dim=1, descending=True)
    cumsum = sorted_mass.cumsum(dim=1)                             # [H, B]

    # K*[h] = smallest k such that cumsum[h, k-1] ≥ tau_cov.
    # searchsorted on each row returns the insertion index of tau_cov;
    # since cumsum is non-decreasing in [0, 1], that index = (K* - 1).
    tau_vec = torch.full((num_q_heads, 1), float(tau_cov), device=device, dtype=cumsum.dtype)
    k_star = torch.searchsorted(cumsum, tau_vec).squeeze(1) + 1    # [H]
    # Clamp to [k_min, min(k_max, num_blocks)]. k_max=None means no cap
    # beyond num_blocks — let tau_cov alone dictate K* per head.
    hi = num_blocks if k_max is None else min(int(k_max), num_blocks)
    lo = min(int(k_min), hi)
    k_star = k_star.clamp(min=lo, max=hi).to(torch.int32)

    # Build [H, B] top-K mask: position < k_star[h] in the sorted order.
    pos = torch.arange(num_blocks, device=device).unsqueeze(0)     # [1, B]
    keep_sorted = pos < k_star.unsqueeze(1).to(pos.dtype)           # [H, B] bool
    topk_mask = torch.zeros_like(mass_frac, dtype=torch.bool)
    topk_mask.scatter_(1, sorted_idx, keep_sorted)                 # [H, B]

    # Tail mass + actual coverage using cumsum at (K*-1).
    k_idx = (k_star.long() - 1).clamp(min=0, max=num_blocks - 1).unsqueeze(1)
    tau_actual = cumsum.gather(1, k_idx).squeeze(1).float()
    tail_mass = (1.0 - tau_actual).clamp(min=0.0)
    return topk_mask, k_star, tail_mass, tau_actual


def compute_fp16_block_scores(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim]
    block_indices: torch.Tensor,   # [num_q_heads, K] int64 block ids to score
    num_scoring_blocks: int,       # upper bound on valid block id (fully-quantized blocks)
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Compute per-head per-block FP16 max-logit for the given block set.

    Mirrors Phase-1's m_b (the per-block max pre-softmax logit) but uses the
    FP16 keys from the tiered cache's GPU mirror (or CPU if no mirror). Only
    blocks in [0, num_scoring_blocks) are valid; others receive -inf.

    Returns: [num_q_heads, K] float32 block scores suitable for ranking.
    """
    num_q_heads, head_dim = q_all.shape
    _, K = block_indices.shape
    bs = cache.block_size
    device = q_all.device

    # Total tokens covered by the fully-quantized block range.
    nt = num_scoring_blocks * bs

    neg_inf = torch.full((num_q_heads, K), float("-inf"), dtype=torch.float32, device=device)
    if nt == 0 or K == 0:
        return neg_inf

    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, non_blocking=True)
    if keys.dtype != q_all.dtype:
        keys = keys.to(dtype=q_all.dtype)

    # [num_q_heads, K, bs, head_dim] gather: for each (h, k) pick tokens
    # [block*bs, block*bs + bs) from keys[kv_h].
    kv_per_h = torch.arange(num_q_heads, device=device) // gqa_group          # [H]
    kv_per_hk = kv_per_h.unsqueeze(1).expand(-1, K)                            # [H, K]
    starts = block_indices.to(torch.long) * bs                                 # [H, K]
    token_offsets = torch.arange(bs, device=device)                            # [bs]
    token_idx = starts.unsqueeze(-1) + token_offsets                           # [H, K, bs]
    valid = (token_idx < nt) & (starts.unsqueeze(-1) >= 0)                     # [H, K, bs]
    # Clamp out-of-range indices so the gather is always valid; masked later.
    token_idx_clamped = token_idx.clamp(min=0, max=max(nt - 1, 0))

    # keys[kv, t]: fancy indexing with [H, K, bs] index tensors.
    kv_idx = kv_per_hk.unsqueeze(-1).expand(-1, -1, bs)                        # [H, K, bs]
    k_gathered = keys[kv_idx, token_idx_clamped]                               # [H, K, bs, head_dim]

    # Dot with q_h: q_all [H, head_dim] → [H, 1, 1, head_dim]
    q_expanded = q_all.unsqueeze(1).unsqueeze(1)
    logits = (k_gathered.float() * q_expanded.float()).sum(dim=-1) * q_scale   # [H, K, bs]
    neg_inf_tok = torch.full_like(logits, float("-inf"))
    logits = torch.where(valid, logits, neg_inf_tok)
    scores = logits.amax(dim=-1)                                               # [H, K]
    return scores


def recompute_heads_dense_fp16(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,               # [num_q_heads, head_dim]
    output: torch.Tensor,              # [num_q_heads, d_v] to be patched in-place
    head_indices: torch.Tensor,        # [num_to_recompute] int64 q-head ids
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Rung-3 recompute: for each listed head, replace output[h] with a full
    FP16 dense attention using the cache's FP16 keys + FP16 values (dequantised
    from INT4 if that's the value tier).

    The recompute only touches listed heads — non-disagreeing heads keep their
    Phase-2 output unchanged. This is intentional: the spec calls for per-head
    granularity so only the heads that paid the detection are corrected.
    """
    if head_indices.numel() == 0:
        return output
    nt = cache.num_tokens
    device = q_all.device
    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, non_blocking=True)
    values_f32 = cache.get_values_f32()[:, :nt, :]  # FP32 from VRAM (either tier)
    keys_f32 = keys.to(device=device, dtype=torch.float32)

    # Loop-free per-head recompute: pull the rows we need and vectorise the
    # dot-products. head_indices is typically small (≤ num_q_heads).
    heads = head_indices.to(device=device, dtype=torch.long)
    kv_ids = heads // gqa_group                                        # [M]
    q_sel = q_all.index_select(0, heads).float()                        # [M, head_dim]
    k_sel = keys_f32.index_select(0, kv_ids)                            # [M, nt, head_dim]
    v_sel = values_f32.index_select(0, kv_ids)                          # [M, nt, d_v]
    logits = torch.einsum("mnd,md->mn", k_sel, q_sel) * q_scale        # [M, nt]
    weights = torch.softmax(logits, dim=1)                              # [M, nt]
    head_out = torch.einsum("mn,mnd->md", weights, v_sel)              # [M, d_v]
    output.index_copy_(0, heads, head_out.to(output.dtype))
    return output


def augment_mask_with_exploration(
    topk_mask: torch.Tensor,         # [H, B] bool — top-K* mask from adaptive selector
    exploration_rate: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Paper §6 exploration budget: randomly promote `exploration_rate` of
    the non-promoted blocks per head to FP16 for monitoring purposes.

    Returns (augmented_mask, exploration_mask, total_explored).

    Uses a rejection-free per-element Bernoulli so the number of exploration
    picks is a random variable around `rate · non_promoted`. Fully on device,
    no sync. Pass a `generator` to keep the exploration reproducible.
    """
    if exploration_rate <= 0.0 or topk_mask.numel() == 0:
        empty = torch.zeros_like(topk_mask)
        return topk_mask, empty, 0
    # Per-element Bernoulli(exploration_rate) on the non-promoted blocks only.
    non_promoted = ~topk_mask
    # Draw one uniform per block per head; no CPU sync.
    rand = torch.rand(topk_mask.shape, device=topk_mask.device, generator=generator)
    draw = rand < float(exploration_rate)
    exploration_mask = non_promoted & draw
    augmented = topk_mask | exploration_mask
    # Running total is on device until the caller decides to item() it.
    return augmented, exploration_mask, int(exploration_mask.sum().item())


def compute_delta_bound(
    q_all: torch.Tensor,        # [num_q_heads, head_dim]
    key_scales: torch.Tensor,    # [num_kv_heads, num_blocks, head_dim] float32
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Per-head tight Δ bound (paper Eq. 4, runtime form).

    Δ[h] = (1 / (2·√d)) · Σ_c |q[h,c]| · s_c · q_scale, where s_c is
    conservatively the per-channel max scale over all blocks in that head's
    KV head. Returns a [num_q_heads] float32 tensor on device.

    The q_scale (= 1/√d) factor is folded in so the resulting Δ compares
    apples-to-apples with the Phase-1 `m_b` (which is post-q_scale).
    """
    num_q_heads, head_dim = q_all.shape
    if key_scales.numel() == 0:
        return torch.zeros(num_q_heads, dtype=torch.float32, device=q_all.device)
    # Worst-case per-channel scale over blocks, per KV head: [num_kv_heads, head_dim].
    per_channel_scale = key_scales.amax(dim=1)
    # Expand to per Q head via GQA mapping.
    kv_per_h = torch.arange(num_q_heads, device=q_all.device) // gqa_group
    s_per_h = per_channel_scale.index_select(0, kv_per_h)  # [H, head_dim]
    # Δ = (1 / (2·√d)) · Σ_c |q_c| · s_c, and the paper's bound is on the
    # post-q_scale logit, so multiply once more by q_scale (= 1/√d).
    delta = (q_all.abs().float() * s_per_h.float()).sum(dim=1) / (2.0 * math.sqrt(head_dim))
    return delta * float(q_scale)


def score_consistency_violations(
    int8_scores: torch.Tensor,     # [H, K] INT8 block scores on the re-ranked set
    fp16_scores: torch.Tensor,     # [H, K] FP16 block scores on the same set
    delta_per_head: torch.Tensor,  # [H] Δ bound (paper Eq. 4)
    eps_guard: float = 0.01,
) -> torch.Tensor:
    """Per-head score-consistency (paper §6).

    Returns a [H] bool tensor: True when any block's |FP16 - INT8| score
    exceeds Δ + eps_guard. A non-zero count here indicates the Theorem-2
    bound is empirically broken on this step — a correctness red flag
    (stale quant metadata, cache corruption, etc.), not a quality knob.
    """
    if int8_scores.numel() == 0:
        return torch.zeros(int8_scores.shape[0], dtype=torch.bool, device=int8_scores.device)
    diff = (fp16_scores - int8_scores).abs().float()
    threshold = (delta_per_head + float(eps_guard)).unsqueeze(1)  # [H, 1]
    return (diff > threshold).any(dim=1)


def detect_ranking_disagreement(
    int8_scores: torch.Tensor,     # [num_q_heads, K]
    fp16_scores: torch.Tensor,     # [num_q_heads, K]
    r: int,
) -> torch.Tensor:
    """Per-head: does the top-r INT8 ranking match the top-r FP16 ranking?

    Returns a [num_q_heads] bool tensor; True = rankings disagree on at least
    one of the top-r positions. Uses argsort over the scoring set rather than
    global block ids so the two rankings share a vocabulary.
    """
    if int8_scores.numel() == 0 or r <= 0:
        return torch.zeros(int8_scores.shape[0], dtype=torch.bool, device=int8_scores.device)
    k = int8_scores.shape[1]
    r_eff = min(r, k)
    rank_int8 = int8_scores.argsort(dim=1, descending=True)[:, :r_eff]
    rank_fp16 = fp16_scores.argsort(dim=1, descending=True)[:, :r_eff]
    # Ordered top-r must match position-by-position (rank_int8[i] == rank_fp16[i])
    return (rank_int8 != rank_fp16).any(dim=1)


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
    tau_cov: float | None = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = DEFAULT_K_MAX,
    rung1_threshold: float = DEFAULT_RUNG1_THRESHOLD,
    rung1_multiplier: float = DEFAULT_RUNG1_MULTIPLIER,
    score_consistency_check: bool = False,
    eps_guard: float = 0.01,
    exploration_rate: float = 0.0,
    exploration_generator: torch.Generator | None = None,
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

    # Paper §3.3 adaptive K*: when tau_cov is supplied, replace the skip mask
    # with the per-head top-K* selection so that blocks whose cumulative
    # INT8-estimated mass reaches tau_cov are attended and the rest are
    # skipped. Block-epsilon certification still runs above this point; the
    # adaptive selection supersedes it when enabled (the paper's bound
    # E_key ≤ 2·V_max·(1-tau_cov) is tighter than the epsilon-only bound at
    # the default tau_cov=0.995).
    adaptive_topk_mask = None
    k_star: torch.Tensor | None = None
    tail_mass_est: torch.Tensor | None = None
    tau_cov_actual: torch.Tensor | None = None
    rung1_triggered_heads = 0
    explored_blocks_count = 0
    if tau_cov is not None and tau_cov > 0 and n_qblocks > 0:
        # Restrict adaptive selection to the fully-quantised block range —
        # the trailing partial block has no INT8 score and is force-attended
        # below. Build the [H, n_blocks] mask by padding the fully-quantised
        # selection with the trailing block forced in.
        m_b_cert = m_b[:, :n_qblocks]
        S_b_cert = S_b[:, :n_qblocks]
        topk_mask_cert, k_star, tail_mass_est, tau_cov_actual = compute_adaptive_topk_mask(
            m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=k_max,
        )

        # Rung-1 (paper §3.4): if any head's tail mass exceeded the configured
        # threshold — typically because k_max capped the selection before
        # tau_cov was reached on a diffuse head — expand the budget and re-pick.
        # The expansion uses a larger k_max = min(k_max * multiplier, n_qblocks);
        # heads whose selection was already good at the original k_max will just
        # land on the same (or a smaller) K* because the tau_cov threshold is
        # unchanged. Accounting: count heads that triggered the expansion.
        # k_max=None means no upper cap so the adaptive selector already hit
        # tau_cov fully → no expansion to do, skip the whole check (also the
        # cheap path sync-wise).
        if (
            rung1_threshold is not None and rung1_threshold >= 0
            and k_max is not None
        ):
            rung1_trigger_mask = tail_mass_est > rung1_threshold  # [H] bool
            # One .sum().item() sync covers both "any?" and "how many?" — no
            # need for a separate .any().item() gate.
            rung1_triggered_heads = int(rung1_trigger_mask.sum().item())
            if rung1_triggered_heads > 0:
                expanded_k_max = min(int(math.ceil(k_max * float(rung1_multiplier))), n_qblocks)
                topk_mask_cert2, k_star2, tail_mass_est2, tau_cov_actual2 = compute_adaptive_topk_mask(
                    m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=expanded_k_max,
                )
                # Only apply the expanded selection to triggered heads so
                # non-triggered heads keep their original K* (avoiding
                # unnecessary bandwidth). The selector is deterministic on the
                # same m_b/S_b so the original top-K entries are a subset of
                # the expanded top-K entries for triggered heads.
                trig = rung1_trigger_mask.unsqueeze(1)
                topk_mask_cert = torch.where(trig, topk_mask_cert2, topk_mask_cert)
                k_star = torch.where(rung1_trigger_mask, k_star2, k_star)
                tail_mass_est = torch.where(rung1_trigger_mask, tail_mass_est2, tail_mass_est)
                tau_cov_actual = torch.where(rung1_trigger_mask, tau_cov_actual2, tau_cov_actual)

        # Paper §6 exploration budget: randomly promote a small fraction of
        # the non-top-K* blocks so their FP16 scores can be cross-checked
        # against the INT8 estimates. Defence-in-depth only — does not
        # affect the paper's certified bounds because the explored blocks
        # are *added* to the attended set (never demote a top-K* block).
        exploration_mask_cert: torch.Tensor | None = None
        explored_blocks_count = 0
        if exploration_rate > 0.0:
            topk_mask_cert, exploration_mask_cert, explored_blocks_count = (
                augment_mask_with_exploration(
                    topk_mask_cert, exploration_rate, exploration_generator,
                )
            )

        # Skip = NOT top-K*; false for trailing partial block (force-attended).
        skip_cert = ~topk_mask_cert
        if cache.has_trailing_partial_block:
            trailing = torch.zeros(num_q_heads, 1, dtype=torch.bool, device=q_all.device)
            skip_mask = torch.cat([skip_cert, trailing], dim=1)
            adaptive_topk_mask = torch.cat(
                [topk_mask_cert, torch.ones(num_q_heads, 1, dtype=torch.bool, device=q_all.device)],
                dim=1,
            )
        else:
            skip_mask = skip_cert
            adaptive_topk_mask = topk_mask_cert

    # Force-attend trailing partial block (it has no INT8 data for scoring)
    if cache.has_trailing_partial_block:
        skip_mask[:, cache.trailing_block_idx] = 0

    # Rung-3 ranking-consistency detection. Runs over the fully-quantized
    # block range (excludes the trailing partial block, which has no INT8
    # score). Populates the telemetry counters below; commit 3 will add the
    # per-head fallback action that sets skip_mask[h, :] = False on disagree.
    ranking_disagree_r1_heads = 0
    ranking_disagree_r3_heads = 0
    ranking_disagree_mask: torch.Tensor | None = None
    fp16_block_scores: torch.Tensor | None = None
    top_block_indices: torch.Tensor | None = None
    ranking_k = 0
    score_consistency_violation_heads = 0
    score_consistency_violation_mask: torch.Tensor | None = None
    delta_bound_mean = 0.0
    # The FP16 block re-scoring is needed for either Rung-3 ranking check or
    # the score-consistency check. Compute it once when either is enabled.
    need_fp16_scores = (ranking_fallback or score_consistency_check) and n_qblocks > 0
    if need_fp16_scores:
        ranking_k = min(max(ranking_r, top_k_fp16_keys, 4), n_qblocks)
        int8_scores = m_b[:, :n_qblocks]
        top_block_indices = int8_scores.topk(ranking_k, dim=1).indices  # [H, K]
        top_int8_scores = int8_scores.gather(1, top_block_indices)       # [H, K]
        fp16_block_scores = compute_fp16_block_scores(
            cache, q_all, top_block_indices, n_qblocks, gqa_group, q_scale,
        )
        if ranking_fallback:
            # Single pair of argsorts covers r=1, r=3, and r=ranking_r — no
            # need to call detect_ranking_disagreement three times (each call
            # was redoing the same sort).
            k_for_rank = top_int8_scores.shape[1]
            if k_for_rank > 0 and ranking_r > 0:
                rank_int8 = top_int8_scores.argsort(dim=1, descending=True)
                rank_fp16 = fp16_block_scores.argsort(dim=1, descending=True)
                rank_diff = rank_int8 != rank_fp16  # [H, K]
                r_main = min(int(ranking_r), k_for_rank)
                r1 = min(1, k_for_rank)
                r3 = min(3, k_for_rank)
                ranking_disagree_mask = rank_diff[:, :r_main].any(dim=1)
                ranking_disagree_r1_heads = int(rank_diff[:, :r1].any(dim=1).sum().item())
                ranking_disagree_r3_heads = int(rank_diff[:, :r3].any(dim=1).sum().item())
            else:
                ranking_disagree_mask = torch.zeros(
                    num_q_heads, dtype=torch.bool, device=q_all.device,
                )
        if score_consistency_check:
            # Paper §6 instability-detection: |FP16 - INT8| per block bounded
            # by Δ + eps_guard. Any violation means Theorem 2 was empirically
            # broken on this step — a canary for stale metadata / cache
            # corruption, expected 0-count on well-behaved runs.
            delta_per_head = compute_delta_bound(
                q_all, cache.keys_scale[:, :n_qblocks, :], gqa_group, q_scale,
            )
            delta_bound_mean = float(delta_per_head.mean().item())
            score_consistency_violation_mask = score_consistency_violations(
                top_int8_scores, fp16_block_scores, delta_per_head, eps_guard,
            )
            score_consistency_violation_heads = int(
                score_consistency_violation_mask.sum().item()
            )

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

    # Rung-3 action: for every head whose INT8/FP16 rankings disagree on the
    # top-r positions, replace its output with a full FP16 dense attention.
    # Non-disagreeing heads keep their Phase-2 output exactly. torch.nonzero
    # already does the device→host transfer needed to shape the index
    # tensor, so a separate .any().item() guard would be a redundant sync.
    ranking_fallback_heads = 0
    if (
        ranking_fallback
        and ranking_fallback_mode == "full"
        and ranking_disagree_mask is not None
    ):
        disagree_heads = torch.nonzero(ranking_disagree_mask, as_tuple=True)[0]
        if disagree_heads.numel() > 0:
            output = recompute_heads_dense_fp16(
                cache=cache,
                q_all=q_all,
                output=output,
                head_indices=disagree_heads,
                gqa_group=gqa_group,
                q_scale=q_scale,
            )
            ranking_fallback_heads = int(disagree_heads.numel())

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
        # Adaptive K* telemetry (paper §3.3). Present only when enabled.
        if k_star is not None:
            stats["k_star_mean"] = float(k_star.float().mean().item())
            stats["k_star_min"] = int(k_star.min().item())
            stats["k_star_max"] = int(k_star.max().item())
            stats["tau_cov"] = float(tau_cov) if tau_cov is not None else 0.0
            stats["tau_cov_actual_mean"] = float(tau_cov_actual.mean().item())
            stats["tail_mass_int8_est_mean"] = float(tail_mass_est.mean().item())
            stats["tail_mass_int8_est_max"] = float(tail_mass_est.max().item())
            # Rung-1 fallback (expand K*) counters. Only relevant when adaptive
            # K* is active; zero on steps where no head hit the tail-mass gate.
            stats["rung1_triggered_heads"] = int(rung1_triggered_heads)
            stats["rung1_threshold"] = float(rung1_threshold)
            stats["rung1_multiplier"] = float(rung1_multiplier)
            # Exploration-budget telemetry (paper §6): blocks randomly added
            # to the attended set beyond adaptive K*. Does not affect the
            # certified bound; purely for monitoring.
            stats["exploration_rate"] = float(exploration_rate)
            stats["exploration_blocks"] = int(explored_blocks_count)
        # Score-consistency violation counters (paper §6). Always emitted
        # when the feature is enabled so runs can confirm the 0-count baseline.
        if score_consistency_check:
            stats["score_consistency_violation_heads"] = int(score_consistency_violation_heads)
            stats["delta_bound_mean"] = float(delta_bound_mean)
            stats["eps_guard"] = float(eps_guard)
        # Ranking-consistency fallback telemetry (Rung 3).
        # Populated by the detection block above when enabled; the trigger
        # count is still zero here because commit 2 is detection-only — the
        # per-head fallback action arrives in the next commit.
        if ranking_fallback:
            stats["ranking_heads_total"] = num_q_heads
            stats["ranking_disagree_r1"] = int(ranking_disagree_r1_heads)
            stats["ranking_disagree_r3"] = int(ranking_disagree_r3_heads)
            stats["ranking_fallback_triggered"] = int(ranking_fallback_heads)
            stats["ranking_r"] = int(ranking_r)
            stats["ranking_k"] = int(ranking_k)
            stats["ranking_fallback_mode"] = ranking_fallback_mode
            # Score-gap diagnostics (spec §5) — only emitted when we actually
            # computed FP16 scores for at least one block per head.
            if fp16_block_scores is not None and fp16_block_scores.shape[1] > 0:
                # Top-1/top-2 gap on the FP16 re-rank: measures ranking fragility.
                # Larger gap → more stable ranking → disagreement less likely.
                if fp16_block_scores.shape[1] >= 2:
                    sorted_fp16 = fp16_block_scores.sort(dim=1, descending=True).values
                    gap_top12 = (sorted_fp16[:, 0] - sorted_fp16[:, 1]).float()
                    stats["score_gap_top12_mean"] = float(gap_top12.mean().item())
                    stats["score_gap_top12_min"] = float(gap_top12.min().item())
                int8_top1 = m_b[:, :n_qblocks].gather(
                    1, top_block_indices[:, :1]
                ).squeeze(1).float()
                fp16_top1 = fp16_block_scores.gather(
                    1, fp16_block_scores.argsort(dim=1, descending=True)[:, :1]
                ).squeeze(1).float()
                stats["s_int8_top1_mean"] = float(int8_top1.mean().item())
                stats["s_fp16_top1_mean"] = float(fp16_top1.mean().item())
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
