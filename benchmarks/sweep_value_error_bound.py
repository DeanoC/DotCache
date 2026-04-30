"""Sweep: compare loose vs tight value-error bounds on a real cert decode.

Runs a cert decode with INT4-values cache (the only path where v_format
actually decides between INT4 and FP16 — FP16-values path skips the
check). Collects per-layer per-step:

  loose_bound = rho_max * eta_max  (legacy decide_v_format input)
  tight_bound = max_h Σ_b ρ_b η_b  (the tight per-block sum)

Reports the distribution, the ratio tight/loose, and the disagreement
rate at a sweep of v_tolerance thresholds — "disagreement" = legacy
loose check would pick FP16 but tight would allow INT4. Used to decide
whether flipping CertifiedAttentionState.value_error_mode default from
"loose" to "tight" is safe.

Usage:
    .venv/bin/python benchmarks/sweep_value_error_bound.py \
        --context 8192 --decode-steps 64 \
        --output benchmarks/results/value_error_sweep.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


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
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--output", default="benchmarks/results/value_error_sweep.json")
    ap.add_argument("--tolerances", default="0.05,0.1,0.25,0.5,1.0")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_int4v_from_model
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

    prompt = build_prefill(tokenizer, args.context)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context).to(device)
    seq_len = ids["input_ids"].shape[1]
    print(f"Prefill seq_len = {seq_len}")

    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered = create_tiered_cache_int4v_from_model(past_kv, layer_ids)
    del past_kv
    gc.collect(); torch.cuda.empty_cache()

    # collect_stats=True is required to exercise the value_check phase and
    # emit e_val_* telemetry. value_error_mode can stay "loose" — both
    # bounds are always computed when e_val_head != None.
    #
    # append_kv=False keeps the cache frozen at block-aligned prefill state
    # across all decode steps. The INT4-values attend kernel has a
    # pre-existing shape bug on trailing partial blocks during decode —
    # out of scope for this sweep. We only care about the v_format decision
    # bounds (computed before the attend kernel), not correct decode output.
    # Each step is effectively "what would v_format decide if we were
    # attending the prefill state with this step's query"? — which is
    # exactly the bound-comparison signal we want.
    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        layer_epsilons={},
        v_tolerance=0.5,
        collect_stats=True,
        append_kv=False,
        top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=False, eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
        value_error_mode="loose",   # v_format uses loose; sweep reports both.
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_tok

    # Per-(step, layer) records.
    records: list[dict] = []

    # Warmup — swallow the same attend-kernel shape errors the timed loop
    # does, since warmup is just about JIT compile and we're not reading its
    # output. With append_kv=False the cache doesn't change, so a crashed
    # warmup step leaves state identical to the pre-step snapshot.
    for _ in range(args.warmup_steps):
        adapter.certified_state.clear_step_stats()
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=current_input, use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            tid = out.logits[:, -1, :].argmax(dim=-1)
            current_input = tid.view(1, 1)
            cache_position = cache_position + 1
        except RuntimeError:
            pass

    print(f"Collecting {args.decode_steps} steps of v_format telemetry...")
    t0 = time.perf_counter()
    partial_steps = 0
    for step in range(args.decode_steps):
        adapter.certified_state.clear_step_stats()
        # The INT4-values attend kernel + its FP16 fallback both have
        # pre-existing shape issues around trailing-partial blocks. We
        # only care about the v_format decision telemetry (populated
        # BEFORE the attend kernel in each layer), so a forward() that
        # crashes mid-model is fine — step_stats has all layers that
        # reached decide_v_format before the crash.
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=current_input, use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            out_ok = True
        except RuntimeError as e:
            out = None
            out_ok = False
            if step == 0 and len(adapter.certified_state.step_stats) == 0:
                # Nothing captured at all — something is really wrong.
                print(f"ERROR step 0 captured 0 layers: {str(e)[:120]}")
                return 1
            partial_steps += 1
        # step_stats is list of per-layer dicts (layers that reached
        # value_check populate e_val_* fields; we filter to those).
        for layer_stats in adapter.certified_state.step_stats:
            if "e_val_max" not in layer_stats:
                continue  # non-INT4-values path — skip
            records.append({
                "step": step,
                "layer": layer_stats.get("layer"),
                "rho_max": layer_stats["rho_max"],
                "eta_int4": layer_stats["eta_int4"],
                "loose": layer_stats["int4_error_bound_loose"],
                "tight_max": layer_stats["e_val_max"],
                "tight_mean": layer_stats["e_val_mean"],
                "v_format": layer_stats.get("v_format"),
            })
        if out_ok:
            tid = out.logits[:, -1, :].argmax(dim=-1)
            current_input = tid.view(1, 1)
            cache_position = cache_position + 1
        else:
            # Can't advance; reuse the same input. Cache is frozen
            # (append_kv=False) so this just re-runs the same decision.
            pass
    elapsed = time.perf_counter() - t0
    print(f"{len(records)} (step, layer) records collected in {elapsed:.1f}s  "
          f"({partial_steps} steps had partial telemetry due to attend-kernel crashes)")

    if not records:
        print("ERROR: no e_val telemetry captured. Is the cache using INT4 values?")
        return 1

    # Analysis
    loose_vals = [r["loose"] for r in records]
    tight_vals = [r["tight_max"] for r in records]
    ratios = [t / l if l > 1e-12 else 0.0 for t, l in zip(tight_vals, loose_vals)]

    print(f"\n=== Summary over {len(records)} (step × layer) samples ===")
    def _stat(name, xs):
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        print(f"  {name:<22} mean={statistics.mean(xs):.4f}  "
              f"p50={xs_sorted[n//2]:.4f}  "
              f"p95={xs_sorted[min(n-1, int(0.95*(n-1)))]:.4f}  "
              f"p99={xs_sorted[min(n-1, int(0.99*(n-1)))]:.4f}  "
              f"max={max(xs):.4f}")
    _stat("loose bound", loose_vals)
    _stat("tight bound", tight_vals)
    _stat("tight / loose ratio", ratios)

    # Invariant check: tight ≤ loose + small numerical slack.
    violations = [(r, t, l) for r, t, l in zip(records, tight_vals, loose_vals) if t > l + 1e-5]
    print(f"\nInvariant tight ≤ loose: {len(records) - len(violations)}/{len(records)} samples pass")
    if violations:
        print(f"  WARN: {len(violations)} samples violate invariant (expected only under top-K ∩ skip ≠ ∅)")

    # Disagreement rate at various v_tolerance values.
    tolerances = [float(x) for x in args.tolerances.split(",")]
    print(f"\n=== Disagreement rate (loose picks FP16 but tight would allow INT4) ===")
    print(f"  {'tolerance':<12} {'loose→FP16':<12} {'tight→FP16':<12} {'flipped→INT4':<14} {'flip%':<8}")
    tolerance_summary = []
    for tol in tolerances:
        loose_fp16 = sum(1 for v in loose_vals if v >= tol)
        tight_fp16 = sum(1 for v in tight_vals if v >= tol)
        flipped = sum(1 for l, t in zip(loose_vals, tight_vals) if l >= tol and t < tol)
        flip_pct = 100.0 * flipped / len(records) if records else 0.0
        print(f"  {tol:<12.3f} {loose_fp16:<12} {tight_fp16:<12} {flipped:<14} {flip_pct:<7.2f}%")
        tolerance_summary.append({
            "tolerance": tol,
            "loose_fp16_decisions": loose_fp16,
            "tight_fp16_decisions": tight_fp16,
            "flipped_to_int4": flipped,
            "flip_pct": flip_pct,
        })

    payload = {
        "model": args.model,
        "context": args.context,
        "seq_len": seq_len,
        "decode_steps": args.decode_steps,
        "n_records": len(records),
        "summary": {
            "loose_mean": statistics.mean(loose_vals),
            "tight_mean": statistics.mean(tight_vals),
            "ratio_mean": statistics.mean(ratios),
            "ratio_p95": sorted(ratios)[min(len(ratios)-1, int(0.95*(len(ratios)-1)))],
            "invariant_violations": len(violations),
        },
        "tolerances": tolerance_summary,
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
