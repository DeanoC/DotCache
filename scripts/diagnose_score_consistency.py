#!/usr/bin/env python3
"""Diagnose the score-consistency violation rate on one decode step.

Runs a certified prefill + one decode step and, via monkey-patches on the
inner helpers, snapshots (per layer, per head) the top-K INT8 scores, the
matching FP16 rescores, and the current Δ bound. Reports the distribution
of |FP16 − INT8| normalised by:

  (a) Δ_current : what compute_delta_bound returns today
  (b) Δ_alt     : drop the extra q_scale multiplication (hypothesised fix)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from dotcache.integrations.llama import (
    LlamaDotCacheModelAdapter,
    CertifiedAttentionState,
    _ensure_certified_imports,
)
from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
from dotcache.kernels import certified_attention as CA
from dotcache.config import DotCacheConfig


def build_prompt(tokenizer, context_tokens: int) -> str:
    FILLER = (
        "The history of mathematics spans thousands of years and encompasses many "
        "different cultures and civilizations. "
    )
    needle = "The secret code is 7429-DELTA."
    question = "\n\nBased on the above, what is the secret code?\nAnswer:"
    ft = len(tokenizer.encode(FILLER, add_special_tokens=False))
    nt = len(tokenizer.encode(needle, add_special_tokens=False))
    qt = len(tokenizer.encode(question, add_special_tokens=False))
    avail = context_tokens - nt - qt - 50
    nb = max(avail // ft, 2)
    parts = []
    for i in range(nb):
        if i == nb // 2:
            parts.append(f"\n[IMPORTANT] {needle}\n\n")
        parts.append(FILLER)
    return "".join(parts) + question


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--eps-guard", type=float, default=0.01)
    ap.add_argument("--output", default="benchmarks/results/perf_tests_20260422/diag_score_consistency.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", token=token,
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, cfg)
    device = next(model.parameters()).device

    prompt = build_prompt(tokenizer, args.context)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context).to(device)
    seq_len = ids["input_ids"].shape[1]
    print(f"Prefill seq_len = {seq_len}  head_dim = {head_dim}")

    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered = create_tiered_cache_from_model(past_kv, layer_ids)
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    # Buffers keyed by call sequence. We rely on compute_delta_bound being
    # called before score_consistency_violations within each layer's
    # certified_attention_layer invocation, so same index = same layer.
    deltas_cur: list[np.ndarray] = []
    deltas_alt: list[np.ndarray] = []
    int8_tops: list[np.ndarray] = []
    fp16_scores: list[np.ndarray] = []

    real_delta = CA.compute_delta_bound
    real_viol = CA.score_consistency_violations
    real_fp16 = CA.compute_fp16_block_scores

    def wrapped_delta(q_all, key_scales, gqa_group, q_scale):
        d_cur = real_delta(q_all, key_scales, gqa_group, q_scale)
        num_q_heads, hd = q_all.shape
        if key_scales.numel() == 0:
            d_alt = torch.zeros(num_q_heads, dtype=torch.float32, device=q_all.device)
        else:
            per_channel = key_scales.amax(dim=1)
            kv_per_h = torch.arange(num_q_heads, device=q_all.device) // gqa_group
            s_per_h = per_channel.index_select(0, kv_per_h)
            d_alt = (q_all.abs().float() * s_per_h.float()).sum(dim=1) / (2.0 * math.sqrt(hd))
        deltas_cur.append(d_cur.detach().cpu().numpy())
        deltas_alt.append(d_alt.detach().cpu().numpy())
        return d_cur

    def wrapped_fp16(cache, q_all, block_indices, n_scoring_blocks, gqa_group, q_scale):
        s = real_fp16(cache, q_all, block_indices, n_scoring_blocks, gqa_group, q_scale)
        return s

    def wrapped_viol(int8_scores, fp16_scores_t, delta_per_head, eps_guard=0.01):
        int8_tops.append(int8_scores.detach().cpu().numpy())
        fp16_scores.append(fp16_scores_t.detach().cpu().numpy())
        return real_viol(int8_scores, fp16_scores_t, delta_per_head, eps_guard)

    CA.compute_delta_bound = wrapped_delta
    CA.compute_fp16_block_scores = wrapped_fp16
    CA.score_consistency_violations = wrapped_viol

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        layer_epsilons={},
        default_epsilon=1e-4,
        collect_stats=True,
        append_kv=True,
        top_k_fp16_keys=4,
        ranking_fallback=True,
        ranking_r=1,
        tau_cov=0.995,
        k_min=2,
        k_max=128,
        score_consistency_check=True,
        eps_guard=args.eps_guard,
        exploration_rate=0.02,
    )
    adapter.set_mode("certified")
    cache_pos = torch.tensor([seq_len], dtype=torch.long, device=device)
    with torch.inference_mode():
        _ = model(
            input_ids=first_token, use_cache=False,
            cache_position=cache_pos,
            position_ids=cache_pos.unsqueeze(0),
        )

    CA.compute_delta_bound = real_delta
    CA.compute_fp16_block_scores = real_fp16
    CA.score_consistency_violations = real_viol

    assert len(deltas_cur) == len(int8_tops), f"delta/viol length mismatch: {len(deltas_cur)} vs {len(int8_tops)}"
    n_layers_seen = len(deltas_cur)
    print(f"Layers captured: {n_layers_seen}")

    # Aggregate |FP16 − INT8| and ratios.
    diffs_all: list[float] = []
    ratios_cur: list[float] = []
    ratios_alt: list[float] = []
    d_cur_flat: list[float] = []
    d_alt_flat: list[float] = []
    for i in range(n_layers_seen):
        i8 = int8_tops[i]       # [H, K]
        fp = fp16_scores[i]     # [H, K]
        dc = deltas_cur[i]      # [H]
        da = deltas_alt[i]      # [H]
        if i8.size == 0 or fp.size == 0:
            continue
        diff = np.abs(fp.astype(np.float64) - i8.astype(np.float64))
        H, K = diff.shape
        for h in range(H):
            d_cur_flat.append(float(dc[h]))
            d_alt_flat.append(float(da[h]))
            for k in range(K):
                d = float(diff[h, k])
                diffs_all.append(d)
                if dc[h] > 0:
                    ratios_cur.append(d / float(dc[h]))
                if da[h] > 0:
                    ratios_alt.append(d / float(da[h]))

    def pct(xs, p):
        if not xs:
            return float("nan")
        s = sorted(xs)
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]

    def frac_over(xs, thr):
        if not xs:
            return float("nan")
        return sum(1 for x in xs if x > thr) / len(xs)

    print(f"\n=== Diagnostic ===")
    print(f"head_dim = {head_dim}   √d = {math.sqrt(head_dim):.3f}")
    print(f"q_scale   = 1/√d = {1.0/math.sqrt(head_dim):.6f}")
    print(f"samples   = {len(diffs_all)} (heads × K-blocks across {n_layers_seen} layers)")
    print()
    print(f"|FP16 − INT8|:   median={pct(diffs_all,0.5):.5f}  p95={pct(diffs_all,0.95):.5f}  max={max(diffs_all):.5f}")
    print(f"Δ_current:       median={pct(d_cur_flat,0.5):.5f}  p95={pct(d_cur_flat,0.95):.5f}")
    print(f"Δ_alt (no ×q_s): median={pct(d_alt_flat,0.5):.5f}  p95={pct(d_alt_flat,0.95):.5f}")
    print(f"Ratio |diff|/Δ_current: median={pct(ratios_cur,0.5):.3f}  p95={pct(ratios_cur,0.95):.3f}")
    print(f"  fraction > 1 (would-fire vs Δ_current) = {frac_over(ratios_cur,1.0):.1%}")
    print(f"  fraction > (1 + ε/Δ) ≈ actual fire-rate at eps_guard={args.eps_guard}:")
    # Effective threshold per-sample is (Δ + ε); ratio = diff / Δ; fire when diff > Δ + ε
    # i.e. when ratio > 1 + ε/Δ. Use per-sample ε/Δ approximation.
    fires_cur = 0
    fires_alt = 0
    for i in range(n_layers_seen):
        i8 = int8_tops[i]; fp = fp16_scores[i]
        dc = deltas_cur[i]; da = deltas_alt[i]
        if i8.size == 0: continue
        diff = np.abs(fp.astype(np.float64) - i8.astype(np.float64))
        thr_cur = dc[:, None] + args.eps_guard
        thr_alt = da[:, None] + args.eps_guard
        fires_cur += int(((diff > thr_cur).any(axis=1)).sum())
        fires_alt += int(((diff > thr_alt).any(axis=1)).sum())
    total_heads = sum(int8_tops[i].shape[0] for i in range(n_layers_seen) if int8_tops[i].size > 0)
    print(f"  Actual (per-head) fire rate with Δ_current + ε={args.eps_guard}: {fires_cur/total_heads:.1%}")
    print(f"  Actual (per-head) fire rate with Δ_alt     + ε={args.eps_guard}: {fires_alt/total_heads:.1%}")
    print(f"Ratio |diff|/Δ_alt:     median={pct(ratios_alt,0.5):.3f}  p95={pct(ratios_alt,0.95):.3f}")

    summary = {
        "head_dim": head_dim,
        "n_layer_snapshots": n_layers_seen,
        "n_samples": len(diffs_all),
        "diff_median": pct(diffs_all, 0.5),
        "diff_p95": pct(diffs_all, 0.95),
        "diff_max": max(diffs_all) if diffs_all else 0,
        "delta_current_median": pct(d_cur_flat, 0.5),
        "delta_current_p95": pct(d_cur_flat, 0.95),
        "delta_alt_median": pct(d_alt_flat, 0.5),
        "delta_alt_p95": pct(d_alt_flat, 0.95),
        "ratio_current_median": pct(ratios_cur, 0.5),
        "ratio_current_p95": pct(ratios_cur, 0.95),
        "ratio_alt_median": pct(ratios_alt, 0.5),
        "ratio_alt_p95": pct(ratios_alt, 0.95),
        "fire_rate_current": fires_cur / total_heads if total_heads else 0,
        "fire_rate_alt": fires_alt / total_heads if total_heads else 0,
        "eps_guard": args.eps_guard,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
