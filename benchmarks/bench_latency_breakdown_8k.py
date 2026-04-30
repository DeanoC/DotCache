"""Test 2: per-step latency breakdown for the certified path at 8K context.

Runs a prefill + `decode-steps` certified decode steps, with phase timers
active. Each layer records μs into cert_state.phase_timings; the harness
snapshots that dict per step, computes the delta vs the previous step, and
accumulates a per-step vector of 7 phases:

  phase1_int8_scoring       fused_score_certify_multihead
  adaptive_selection        compute_adaptive_topk_mask + Rung-1 re-pick
  ranking_check             compute_fp16_block_scores + argsort (rung 3)
  h2d_pagein                FP16 key/value CPU→GPU transfers
  value_check               tier-2 residual mass + INT4 v_format decide
  phase2_fused_attend       the selective_attend kernel (Triton/SDPA)
  overhead_other            total step wall − sum(named phases)

Enabling phase_timings adds one torch.cuda.Event pair + sync per wrapped
region per layer (≈5 extra syncs per layer per step), so the absolute
throughput numbers here should NOT be compared to Test 1's tok/s; the
ratios between phases are what matter.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch


PHASES = (
    "phase1_int8_scoring",
    "adaptive_selection",
    "ranking_check",
    "h2d_pagein",
    "value_check",
    "phase2_fused_attend",
)


def build_prefill(tokenizer, context_tokens: int) -> str:
    FILLER = (
        "The history of mathematics spans thousands of years and encompasses many "
        "different cultures and civilizations. "
    )
    question = "\nContinue:"
    ft = len(tokenizer.encode(FILLER, add_special_tokens=False))
    qt = len(tokenizer.encode(question, add_special_tokens=False))
    avail = context_tokens - qt - 50
    nb = max(avail // ft, 2)
    return FILLER * nb + question


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--decode-steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=16)
    ap.add_argument("--output", default="benchmarks/results/perf_tests_20260422/test2_phase_breakdown.json")
    args = ap.parse_args()

    # DOTCACHE_V_TOL was attempted as a runtime override but the env var was
    # never read by any kernel — every cell silently used v_tolerance=0.5.
    # See docs/paper_code_audit_20260424.md. Use --v-tolerance on the paper
    # benches; this perf bench hardcodes 0.5 in the CertifiedAttentionState
    # construction below to match its prior measured behaviour.

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    from dotcache.config import DotCacheConfig

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
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

    prompt = build_prefill(tokenizer, args.context_length)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=args.context_length).to(device)
    seq_len = ids["input_ids"].shape[1]
    print(f"Prefill seq_len = {seq_len}")

    # Prefill dense → tiered cache
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    _env_cap = os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS")
    _cap = None if _env_cap is None or _env_cap == "" else int(_env_cap)
    tiered = create_tiered_cache_from_model(
        past_kv, layer_ids, fp16_key_cache_capacity=_cap,
    )
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    phase_timings: dict = {}
    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        layer_epsilons={},
        v_tolerance=0.5,
        collect_stats=True,
        append_kv=True,
        top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=True, eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
        phase_timings=phase_timings,
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token

    per_step_phases: list[dict[str, float]] = []
    per_step_total_ms: list[float] = []

    # Previous cumulative totals for diffing.
    prev_totals = {f"{p}_us": 0.0 for p in PHASES}

    total_evt_start = torch.cuda.Event(enable_timing=True)
    total_evt_end = torch.cuda.Event(enable_timing=True)

    for step in range(args.decode_steps):
        total_evt_start.record()
        with torch.inference_mode():
            out = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        total_evt_end.record()
        torch.cuda.synchronize()
        total_ms = total_evt_start.elapsed_time(total_evt_end)
        per_step_total_ms.append(total_ms)

        # Diff current cumulative phase totals vs last snapshot.
        step_phases: dict[str, float] = {}
        for p in PHASES:
            key = f"{p}_us"
            cur = float(phase_timings.get(key, 0.0))
            step_phases[p] = cur - prev_totals[key]
            prev_totals[key] = cur
        # overhead_other = total_step − sum(named phases), in μs.
        step_phases["overhead_other"] = max(
            0.0, total_ms * 1000.0 - sum(step_phases.values())
        )
        per_step_phases.append(step_phases)

        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    # Drop warmup.
    usable = per_step_phases[args.warmup_steps:]
    usable_total_ms = per_step_total_ms[args.warmup_steps:]

    summary = {"n_steps_measured": len(usable), "total_ms_mean": statistics.mean(usable_total_ms),
               "total_ms_p50": statistics.median(usable_total_ms),
               "total_ms_p95": sorted(usable_total_ms)[int(0.95 * (len(usable_total_ms) - 1))],
               "total_ms_p99": sorted(usable_total_ms)[int(0.99 * (len(usable_total_ms) - 1))]}

    # Per-phase mean, median, p95.
    for p in (*PHASES, "overhead_other"):
        vals = [s[p] for s in usable]
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        summary[f"{p}_us_mean"] = statistics.mean(vals)
        summary[f"{p}_us_p50"] = vals_sorted[n // 2] if n else 0.0
        summary[f"{p}_us_p95"] = vals_sorted[min(n - 1, int(0.95 * (n - 1)))]
    # Share of each phase (% of total), mean over steps.
    total_per_step = [sum(s.values()) for s in usable]
    for p in (*PHASES, "overhead_other"):
        frac = [s[p] / t if t > 0 else 0.0 for s, t in zip(usable, total_per_step)]
        summary[f"{p}_share_mean"] = statistics.mean(frac) if frac else 0.0

    payload = {
        "hardware": torch.cuda.get_device_name(0),
        "model": args.model,
        "context_length": args.context_length,
        "seq_len": seq_len,
        "decode_steps": args.decode_steps,
        "warmup_steps": args.warmup_steps,
        "summary": summary,
        "per_step_phases_us": per_step_phases,
        "per_step_total_ms": per_step_total_ms,
    }
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {p}")

    print("\n=== Per-phase breakdown (warmup dropped) ===")
    print(f"{'phase':<24} {'mean μs':>10} {'p50 μs':>10} {'p95 μs':>10} {'share':>8}")
    for phase in (*PHASES, "overhead_other"):
        print(f"{phase:<24} "
              f"{summary[f'{phase}_us_mean']:>10.1f} "
              f"{summary[f'{phase}_us_p50']:>10.1f} "
              f"{summary[f'{phase}_us_p95']:>10.1f} "
              f"{summary[f'{phase}_share_mean']*100:>7.1f}%")
    print(f"\nTotal step: mean {summary['total_ms_mean']:.2f}ms, "
          f"p50 {summary['total_ms_p50']:.2f}ms, p95 {summary['total_ms_p95']:.2f}ms, "
          f"p99 {summary['total_ms_p99']:.2f}ms")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
