"""Context-length scaling benchmark for certified upper-bound modes.

Sweeps over context lengths [1K, 2K, 4K, 8K, 16K, 32K], running the three
bound lanes (spherical_only / interval / interval_ellip) at each length and
recording ms/step, cert_stop_rate, and score_ms/step.

16K and 32K are included intentionally (they were excluded from the MPS sweep
due to dense SDPA OOM at prefill).  On CUDA with HBM neither should OOM.
If GPU VRAM is tight, pass --max-dense-length 16384 to skip the dense
reference pass for 32K cases; the persistent model uses sparse attention so
it fits even when dense does not.

Key hypotheses being tested
---------------------------
1. cert_stop_rate ≈ 6.2% at every context length (property of the data,
   not the hardware).
2. ms/step saving from `interval` vs `spherical_only` *grows* with context
   length on CUDA (each skipped block costs more to process at longer
   contexts, and there are more blocks to skip).
3. If `interval` is positive-delta at all lengths, MPS overhead (not
   insufficient bound tightening) was the limiting factor on MPS.

cert_stop_rate definition
-------------------------
For each decode step we check whether any FA layer issued a certified early
exit (last_first_certified_stop_block_count > 0).  Rate = certified_steps /
total_steps.  Because the harness only exposes last-step cert-stop info, we
run decode_steps=1 in a tight loop over a single prefill, re-using the
internal KV cache state via the adapter.  See _decode_steps_with_cert_tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _persistent_base_config,
    _matching_prefix_length,
)
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.backends.metal.persistent_types import PersistentServingConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35TextModelAdapter,
    load_qwen35_text_only_from_pretrained,
    run_qwen35_attention_subset_persistent_serving_harness,
    run_qwen35_text_generation_harness,
    transformers_available,
)


_DEFAULT_CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

_LANES = [
    {
        "name": "spherical_only",
        "label": "Spherical only (baseline)",
        "enable_interval_bound": False,
        "enable_ellipsoidal_bound": False,
    },
    {
        "name": "interval",
        "label": "Interval bound (paper primary)",
        "enable_interval_bound": True,
        "enable_ellipsoidal_bound": False,
    },
    {
        "name": "interval_ellip",
        "label": "Interval + Ellipsoidal (full paper)",
        "enable_interval_bound": True,
        "enable_ellipsoidal_bound": True,
    },
]

# Minimal filler text used to pad prompts to the requested token count.
# Repeated until at least the target number of tokens is available.
_FILLER_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used to pad the context to the required token length for benchmarking purposes. "
    "Attention patterns at longer contexts reveal properties of the underlying key distribution. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep context lengths [1K–32K] for certified bound mode timing on CUDA. "
            "16K and 32K are included; use --max-dense-length 16384 to skip dense "
            "reference for 32K if VRAM is tight."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument(
        "--contexts",
        nargs="*",
        type=int,
        default=None,
        help="Context lengths to sweep (default: 1024 2048 4096 8192 16384 32768)",
    )
    parser.add_argument("--decode-steps", type=int, default=16,
                        help="Decode steps per case (default: 16)")
    parser.add_argument(
        "--max-dense-length",
        type=int,
        default=0,
        help=(
            "Skip the dense reference pass for contexts longer than this value. "
            "Set to 16384 to skip dense at 32K if GPU VRAM is tight. "
            "0 = no limit (run dense at all lengths)."
        ),
    )
    parser.add_argument("--lanes", nargs="*", default=None,
                        help="Subset of lane names to run (default: all three)")
    parser.add_argument(
        "--cert-tracking-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of single-step harness calls used to sample cert_stop_rate. "
            "Each call re-runs a full prefill, so this is expensive. "
            "0 (default) skips cert tracking entirely and reports last-step cert from "
            "the timing run instead.  Set to 16 to restore the original behaviour."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Decode steps to run as a throwaway warmup before the timed measurement. "
            "Warms Triton JIT compilation and CUDA kernel caches so the timing run "
            "does not include first-step compile overhead.  Default: 1."
        ),
    )
    parser.add_argument("--prompt-file", default=None,
                        help="Optional base text file for prompt (padded/truncated to each context length)")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _build_padded_prompt_text(base_text: str, *, target_tokens: int, tokenizer: Any) -> str:
    """Repeat *base_text* until it tokenises to at least *target_tokens*, then truncate."""
    chunk = base_text.strip()
    repeated = chunk
    # Rough estimate: 1 token ≈ 4 chars; overshoot by 2× to be safe
    while len(repeated) < target_tokens * 8:
        repeated = repeated + " " + chunk
    return repeated


def _build_context_inputs(
    tokenizer: Any,
    *,
    device: Any,
    base_text: str,
    target_length: int,
) -> tuple[Any, Any]:
    """Tokenise *base_text* padded to *target_length* tokens."""
    repeated_text = _build_padded_prompt_text(base_text, target_tokens=target_length, tokenizer=tokenizer)
    return _build_prompt_text_inputs(
        tokenizer,
        device=device,
        prompt_text=repeated_text,
        prompt_length=target_length,
    )


def _build_serving_config(lane: dict[str, Any]) -> PersistentServingConfig:
    config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
    config.enable_interval_bound = bool(lane["enable_interval_bound"])
    config.enable_ellipsoidal_bound = bool(lane["enable_ellipsoidal_bound"])
    return config


def _sum_by_layer(result: dict[str, Any], key: str) -> float:
    return float(sum(float(v) for v in result.get(key, {}).values() if v is not None))


def _has_cert_stop(result: dict[str, Any]) -> bool:
    """Return True if any FA layer had a certified exit in the last decode step."""
    return any(
        v is not None and int(v) > 0
        for v in result.get(
            "persistent_full_attention_last_first_certified_stop_block_count_by_layer", {}
        ).values()
    )


def _run_lane_at_context(
    *,
    lane: dict[str, Any],
    model,
    tokenizer,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    input_ids: Any,
    attention_mask: Any,
    decode_steps: int,
    dense_reference_ids: list[int] | None,
    warmup_steps: int = 1,
    cert_tracking_steps: int = 0,
) -> dict[str, Any]:
    """Run one lane at one context length; return per-step timing + cert_stop_rate.

    Execution order
    ---------------
    1. Warmup (warmup_steps > 0): throw-away harness call to prime Triton JIT
       and CUDA kernel caches so the timed run is not polluted by cold-start.
    2. Optional cert tracking (cert_tracking_steps > 0): N single-step harness
       calls to sample whether each decode step issues a certified early exit.
       Each call re-runs a full prefill, so keep N small (or 0).
       When cert_tracking_steps=0 the cert_stop flag is taken from the last
       step of the timing run instead (cheap, no extra prefill).
    3. Timed N-step run: one harness call with decode_steps=N for accurate
       avg ms/step (amortises CUDA synchronisation overhead across N steps).
    """
    serving_config = _build_serving_config(lane)

    # --- Warmup ---
    if warmup_steps > 0:
        run_qwen35_attention_subset_persistent_serving_harness(
            model,
            adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=warmup_steps,
            persistent_serving_config=serving_config,
        )

    # --- Optional cert tracking (each call re-runs prefill: expensive) ---
    cert_stop_count: int | None = None
    if cert_tracking_steps > 0:
        cert_stop_count = 0
        for _ in range(cert_tracking_steps):
            single = run_qwen35_attention_subset_persistent_serving_harness(
                model,
                adapter,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=1,
                persistent_serving_config=serving_config,
            )
            if _has_cert_stop(single):
                cert_stop_count += 1

    # --- Timed N-step run ---
    result = run_qwen35_attention_subset_persistent_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
        persistent_serving_config=serving_config,
    )

    decode_ms_per_step = float(result.get("persistent_decode_ms_per_step", 0.0))
    score_ms_per_step = (
        _sum_by_layer(result, "persistent_full_attention_score_ms_total_by_layer") / max(decode_steps, 1)
    )

    # cert_stop_rate: if cert tracking was enabled use the sampled rate;
    # otherwise report whether the timing run's last step issued a cert exit.
    if cert_tracking_steps > 0 and cert_stop_count is not None:
        cert_stop_rate: float | None = float(cert_stop_count) / max(cert_tracking_steps, 1)
    else:
        # Last-step cert from the timing run (not a true rate; 0.0 or 1.0)
        cert_stop_rate = 1.0 if _has_cert_stop(result) else 0.0
        cert_stop_count = int(cert_stop_rate)

    generated_ids = [int(t) for t in result.get("persistent_generated_ids", [])]
    exact_match: bool | None = None
    if dense_reference_ids is not None:
        dense_cmp = dense_reference_ids[:decode_steps]
        exact_match = bool(generated_ids == dense_cmp and len(generated_ids) >= decode_steps)

    return {
        "lane": str(lane["name"]),
        "decode_ms_per_step": float(decode_ms_per_step),
        "score_ms_per_step": float(score_ms_per_step),
        "cert_stop_rate": float(cert_stop_rate) if cert_stop_rate is not None else None,
        "cert_stop_count": int(cert_stop_count) if cert_stop_count is not None else None,
        "decode_steps": int(decode_steps),
        "exact_match_vs_dense": exact_match,
    }


def _summarize_context(lane_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute relative deltas vs spherical_only at this context length."""
    spherical = lane_results.get("spherical_only", {})
    sph_ms = float(spherical.get("decode_ms_per_step", 0.0))
    out: dict[str, Any] = {}
    for lane_name, r in lane_results.items():
        ms = float(r.get("decode_ms_per_step", 0.0))
        delta_pct = (ms - sph_ms) / sph_ms * 100.0 if sph_ms > 0 else 0.0
        out[lane_name] = {
            "decode_ms_per_step": ms,
            "score_ms_per_step": float(r.get("score_ms_per_step", 0.0)),
            "cert_stop_rate": float(r.get("cert_stop_rate", 0.0)),
            "cert_stop_count": int(r.get("cert_stop_count", 0)),
            "delta_vs_spherical_pct": float(delta_pct),
            "exact_match_vs_dense": r.get("exact_match_vs_dense"),
        }
    return out


def _render_markdown(
    *,
    payload: dict[str, Any],
    active_lanes: list[dict[str, Any]],
) -> str:
    context_lengths = [int(c) for c in payload.get("context_lengths", [])]
    summaries_by_ctx = payload.get("summaries_by_context", {})

    lines = [
        "# Qwen3.5 Bound Mode Context-Length Scaling",
        "",
        "Sweeps context lengths [1K–32K] for certified upper-bound modes on CUDA.",
        "16K and 32K are included (excluded from MPS due to dense-SDPA OOM at prefill).",
        "",
        "## Lane definitions",
        "",
        "| Lane | enable_interval_bound | enable_ellipsoidal_bound |",
        "|---|---|---|",
    ]
    for lane in active_lanes:
        lines.append(
            f"| `{lane['name']}` | {lane['enable_interval_bound']} | {lane['enable_ellipsoidal_bound']} |"
        )

    # ms/step table
    lines += [
        "",
        "## ms/step by context length",
        "",
        "| context | " + " | ".join(f"`{l['name']}`" for l in active_lanes) + " |",
        "|---" * (1 + len(active_lanes)) + "|",
    ]
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_ctx.get(ctx_key, {})
        row = f"| {ctx:,} |"
        for lane in active_lanes:
            ms = float(ctx_data.get(lane["name"], {}).get("decode_ms_per_step", 0.0))
            row += f" {ms:.2f} |"
        lines.append(row)

    # Speedup/delta table
    lines += [
        "",
        "## Δ ms/step vs spherical_only (negative = faster)",
        "",
        "| context | " + " | ".join(f"`{l['name']}`" for l in active_lanes if l["name"] != "spherical_only") + " |",
        "|---" * (1 + sum(1 for l in active_lanes if l["name"] != "spherical_only")) + "|",
    ]
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_ctx.get(ctx_key, {})
        row = f"| {ctx:,} |"
        for lane in active_lanes:
            if lane["name"] == "spherical_only":
                continue
            delta = float(ctx_data.get(lane["name"], {}).get("delta_vs_spherical_pct", 0.0))
            row += f" {delta:+.1f}% |"
        lines.append(row)

    # cert_stop_rate table
    lines += [
        "",
        "## cert_stop_rate by context length",
        "",
        "| context | " + " | ".join(f"`{l['name']}`" for l in active_lanes) + " |",
        "|---" * (1 + len(active_lanes)) + "|",
    ]
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_ctx.get(ctx_key, {})
        row = f"| {ctx:,} |"
        for lane in active_lanes:
            rate = float(ctx_data.get(lane["name"], {}).get("cert_stop_rate", 0.0))
            row += f" {rate:.3f} |"
        lines.append(row)

    lines += [
        "",
        "## score_ms/step by context length",
        "",
        "| context | " + " | ".join(f"`{l['name']}`" for l in active_lanes) + " |",
        "|---" * (1 + len(active_lanes)) + "|",
    ]
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_ctx.get(ctx_key, {})
        row = f"| {ctx:,} |"
        for lane in active_lanes:
            score_ms = float(ctx_data.get(lane["name"], {}).get("score_ms_per_step", 0.0))
            row += f" {score_ms:.2f} |"
        lines.append(row)

    cert_tracking_steps = int(payload.get("cert_tracking_steps", 0))
    warmup_steps = int(payload.get("warmup_steps", 0))
    lines += [
        "",
        "## Notes",
        "",
    ]
    if cert_tracking_steps > 0:
        lines.append(
            f"- cert_stop_rate: sampled from {cert_tracking_steps} single-step harness calls "
            "(each re-runs a full prefill)."
        )
    else:
        lines.append(
            "- cert_stop_rate: taken from the last decode step of the timed N-step run "
            "(0.0 or 1.0 per lane per context; not a true across-step rate). "
            "Use --cert-tracking-steps N to sample across N independent prefills."
        )
    lines += [
        f"- warmup_steps={warmup_steps}: throwaway harness call(s) before each timed run.",
        "- ms/step is from a single N-step run to amortise CUDA sync overhead.",
        "- Dense reference is skipped for contexts above --max-dense-length (default: none).",
    ]
    if int(payload.get("max_dense_length", 0)) > 0:
        lines.append(
            f"- Dense reference was skipped for contexts > {int(payload['max_dense_length']):,} tokens."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit(
            "bench_qwen35_bound_context_scaling.py requires the optional transformers dependencies"
        )

    context_lengths = [int(c) for c in (args.contexts or _DEFAULT_CONTEXT_LENGTHS)]
    max_dense_length = int(args.max_dense_length)

    active_lanes = (
        [lane for lane in _LANES if lane["name"] in args.lanes]
        if args.lanes
        else list(_LANES)
    )
    if not active_lanes:
        raise SystemExit(f"no valid lanes selected; choices: {[l['name'] for l in _LANES]}")

    # Base text for padding
    if args.prompt_file:
        base_text = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        base_text = _FILLER_PARAGRAPH * 500  # ~50K chars — enough for 32K tokens

    print(f"Loading model {args.model_id} ...")
    # Dense model for reference pass
    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    # Persistent model for bound lanes
    persistent_model, persistent_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    dotcache_config = real_mixed_probe_dotcache_config()
    dense_adapter = Qwen35TextModelAdapter(model=dense_model)
    persistent_adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=persistent_model,
        dotcache_config=dotcache_config,
        persistent_serving_config=_build_serving_config(active_lanes[0]),
        backend=str(args.backend),
    )

    summaries_by_context: dict[str, Any] = {}
    raw_by_context: dict[str, Any] = {}

    for ctx in context_lengths:
        print(f"\n{'='*60}")
        print(f"Context length: {ctx:,} tokens")
        print(f"{'='*60}")

        device = next(persistent_model.parameters()).device
        input_ids, attention_mask = _build_context_inputs(
            persistent_tokenizer,
            device=device,
            base_text=base_text,
            target_length=ctx,
        )
        actual_length = int(input_ids.shape[1])
        print(f"  Actual prompt length: {actual_length:,} tokens")

        # Dense reference pass (skip if over max_dense_length)
        dense_reference_ids: list[int] | None = None
        if max_dense_length <= 0 or actual_length <= max_dense_length:
            print(f"  [dense] running reference pass ...")
            dense_input_ids, dense_attention_mask = _build_context_inputs(
                dense_tokenizer,
                device=next(dense_model.parameters()).device,
                base_text=base_text,
                target_length=ctx,
            )
            dense_result = run_qwen35_text_generation_harness(
                dense_model,
                dense_adapter,
                input_ids=dense_input_ids,
                attention_mask=dense_attention_mask,
                max_new_tokens=int(args.decode_steps) + 1,
                tokenizer=dense_tokenizer,
            )
            all_dense_ids = [int(t) for t in dense_result.get("dense_generated_ids", [])]
            dense_reference_ids = all_dense_ids[: int(args.decode_steps)]
            print(f"  [dense] generated {len(all_dense_ids)} reference tokens")
        else:
            print(
                f"  [dense] SKIPPED (context {actual_length:,} > max_dense_length {max_dense_length:,})"
            )

        # Run all lanes
        lane_results: dict[str, dict[str, Any]] = {}
        for lane in active_lanes:
            print(f"  [{lane['name']}] running {args.decode_steps} steps ...")
            r = _run_lane_at_context(
                lane=lane,
                model=persistent_model,
                tokenizer=persistent_tokenizer,
                adapter=persistent_adapter,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=int(args.decode_steps),
                dense_reference_ids=dense_reference_ids,
                warmup_steps=int(args.warmup_steps),
                cert_tracking_steps=int(args.cert_tracking_steps),
            )
            lane_results[lane["name"]] = r
            print(
                f"  [{lane['name']}] {r['decode_ms_per_step']:.2f} ms/step, "
                f"cert_stop_rate={r['cert_stop_rate']:.3f} "
                f"({r['cert_stop_count']}/{r['decode_steps']}), "
                f"score={r['score_ms_per_step']:.2f} ms/step"
                + (
                    f", exact={r['exact_match_vs_dense']}"
                    if r["exact_match_vs_dense"] is not None
                    else ""
                )
            )

        ctx_summary = _summarize_context(lane_results)
        summaries_by_context[str(ctx)] = ctx_summary
        raw_by_context[str(ctx)] = lane_results

        # Print relative deltas
        spherical_ms = float(ctx_summary.get("spherical_only", {}).get("decode_ms_per_step", 0.0))
        if spherical_ms > 0:
            for lane in active_lanes:
                if lane["name"] == "spherical_only":
                    continue
                delta = float(ctx_summary[lane["name"]]["delta_vs_spherical_pct"])
                lane_ms = float(ctx_summary[lane["name"]]["decode_ms_per_step"])
                print(
                    f"  delta {lane['name']:20s} vs spherical: {delta:+.1f}% "
                    f"({lane_ms:.2f} vs {spherical_ms:.2f} ms/step)"
                )

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — ms/step by context")
    print(f"{'='*60}")
    header = f"{'ctx':>8}" + "".join(f"  {l['name']:>16}" for l in active_lanes)
    print(header)
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_context.get(ctx_key, {})
        row = f"{ctx:>8}"
        for lane in active_lanes:
            ms = float(ctx_data.get(lane["name"], {}).get("decode_ms_per_step", 0.0))
            row += f"  {ms:>16.2f}"
        print(row)

    print(f"\nFINAL SUMMARY — cert_stop_rate by context")
    print(header)
    for ctx in context_lengths:
        ctx_key = str(ctx)
        ctx_data = summaries_by_context.get(ctx_key, {})
        row = f"{ctx:>8}"
        for lane in active_lanes:
            rate = float(ctx_data.get(lane["name"], {}).get("cert_stop_rate", 0.0))
            row += f"  {rate:>16.3f}"
        print(row)

    payload: dict[str, Any] = {
        "context_lengths": context_lengths,
        "active_lanes": [lane["name"] for lane in active_lanes],
        "decode_steps": int(args.decode_steps),
        "max_dense_length": int(max_dense_length),
        "warmup_steps": int(args.warmup_steps),
        "cert_tracking_steps": int(args.cert_tracking_steps),
        "model_id": str(args.model_id),
        "device": str(args.device),
        "backend": str(args.backend),
        "summaries_by_context": summaries_by_context,
        "raw_by_context": raw_by_context,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {out_path}")

    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _render_markdown(payload=payload, active_lanes=active_lanes),
            encoding="utf-8",
        )
        print(f"MD   -> {out_path}")


if __name__ == "__main__":
    main()
