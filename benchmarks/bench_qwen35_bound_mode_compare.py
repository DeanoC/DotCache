"""Compare three certified upper-bound modes on the real mixed Stage 9 serving config.

Lanes
-----
spherical_only   -- existing baseline: Cauchy-Schwarz centroid+radius bound only
                    (enable_interval_bound=False, enable_ellipsoidal_bound=False)
interval         -- paper primary bound: per-dimension sign-conditional multiply
                    (enable_interval_bound=True,  enable_ellipsoidal_bound=False)
interval_ellip   -- full paper bounds: interval + first principal component
                    (enable_interval_bound=True,  enable_ellipsoidal_bound=True)

Each lane uses the canonical real-mixed Stage 9 serving config (M0/M3 mixed, certified
streaming, early exit at check_interval=16, priority_value_hybrid order mode).

Tighter upper bounds -> earlier certified exit -> fewer blocks processed -> faster decode.
All three lanes must produce exact-match-vs-dense = 1.0 for safety validity.

Key result fields extracted from harness result dicts
------------------------------------------------------
- persistent_decode_ms_per_step
- persistent_generated_ids
- persistent_full_attention_score_ms_total_by_layer
- persistent_full_attention_executed_m0_block_count_total_by_layer
- persistent_full_attention_executed_m3_block_count_total_by_layer
- persistent_full_attention_last_first_certified_stop_block_count_by_layer
- persistent_full_attention_last_checkpoint_count_by_layer
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _persistent_base_config,
    _resolve_prompt_records,
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
    _run_dense_prefill,
    _clone_qwen35_past_key_values,
    _run_dense_decode_step,
)
from dotcache.integrations.llama import _timed_call, _synchronize_device


_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/manifests/qwen35_stage9_repo_public_validation_broader_20260414.json"
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spherical / interval / interval+ellipsoidal certified bounds on MPS."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=str(_DEFAULT_MANIFEST))
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--lanes", nargs="*", default=None,
                        help="Subset of lane names to run (default: all three)")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Decode steps to run as a throwaway warmup once before any timed lane run. "
            "Primes Triton JIT and CUDA kernel caches.  Default: 1."
        ),
    )
    parser.add_argument(
        "--no-shared-prefill",
        action="store_true",
        default=False,
        help=(
            "Disable shared-prefill optimisation.  By default all three lanes share a "
            "single prefill per case (avoids 2 redundant prefill passes per case). "
            "Pass this flag to revert to the original per-lane prefill behaviour."
        ),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--execution-strategy",
        type=str,
        default=None,
        metavar="STRATEGY",
        help=(
            "Override the mixed-mode execution strategy for all lanes. "
            "Options: 'direct_m0' (fused compressed-domain scoring, default from "
            "serving config), 'cached_reconstruct' (widen to FP16 then standard "
            "attention).  When set, overrides the strategy in the base serving config."
        ),
    )
    parser.add_argument(
        "--max-k-comp-error",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Override max_k_comp_error for all lanes.  Controls which blocks use "
            "M0 (compressed) vs M3 (FP16 exact) in the mixed execution path. "
            "Set to 0.0 to force all blocks through the M3/FP16 path (Fused Fp "
            "condition for kernel comparison).  Default: use serving config value."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes for case dispatch.  Each worker "
            "loads the model independently and processes a subset of cases "
            "concurrently on the GPU.  With N workers, GPU utilisation improves "
            "because each worker's Python overhead can overlap with another "
            "worker's GPU kernels.  Requires enough VRAM for N model instances "
            "(Qwen3.5-0.8B at float16 ~1.6 GB each).  Default: 1."
        ),
    )
    return parser.parse_args()


_EXECUTION_STRATEGY_OVERRIDE: str | None = None
_MAX_K_COMP_ERROR_OVERRIDE: float | None = None


def _build_serving_config_for_lane(lane: dict[str, Any]) -> PersistentServingConfig:
    config = real_mixed_probe_serving_config(
        policy_path=str(_DEFAULT_POLICY_PATH),
    )
    config.enable_interval_bound = bool(lane["enable_interval_bound"])
    config.enable_ellipsoidal_bound = bool(lane["enable_ellipsoidal_bound"])
    if _EXECUTION_STRATEGY_OVERRIDE is not None:
        config.full_attention_mixed_mode_execution_strategy = _EXECUTION_STRATEGY_OVERRIDE
    if _MAX_K_COMP_ERROR_OVERRIDE is not None:
        config.full_attention_mixed_mode_execution_max_k_comp_error = _MAX_K_COMP_ERROR_OVERRIDE
    return config


def _sum_by_layer(result: dict[str, Any], key: str) -> float:
    return float(sum(float(v) for v in result.get(key, {}).values() if v is not None))


def _sum_by_layer_int(result: dict[str, Any], key: str) -> int:
    return int(sum(int(v) for v in result.get(key, {}).values() if v is not None))


def _extract_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    bound_spherical = _sum_by_layer_int(
        result, "persistent_full_attention_bound_spherical_active_count_by_layer"
    )
    bound_interval = _sum_by_layer_int(
        result, "persistent_full_attention_bound_interval_active_count_by_layer"
    )
    bound_ellipsoidal = _sum_by_layer_int(
        result, "persistent_full_attention_bound_ellipsoidal_active_count_by_layer"
    )
    bound_total = bound_spherical + bound_interval + bound_ellipsoidal
    return {
        "score_ms_total": _sum_by_layer(result, "persistent_full_attention_score_ms_total_by_layer"),
        "selection_ms_total": _sum_by_layer(result, "persistent_full_attention_selection_ms_total_by_layer"),
        "executed_m0_blocks_total": _sum_by_layer_int(
            result, "persistent_full_attention_executed_m0_block_count_total_by_layer"
        ),
        "executed_m3_blocks_total": _sum_by_layer_int(
            result, "persistent_full_attention_executed_m3_block_count_total_by_layer"
        ),
        # Certified exit: how many blocks were processed at the first certified stop
        "first_certified_stop_blocks": _sum_by_layer_int(
            result, "persistent_full_attention_last_first_certified_stop_block_count_by_layer"
        ),
        "last_checkpoint_count": _sum_by_layer_int(
            result, "persistent_full_attention_last_checkpoint_count_by_layer"
        ),
        # Bound winner counts: how many (block, q_head) evals each bound won
        "bound_spherical_count": bound_spherical,
        "bound_interval_count": bound_interval,
        "bound_ellipsoidal_count": bound_ellipsoidal,
        "bound_total_count": bound_total,
        # Fraction each bound provides the tightest upper bound
        "bound_spherical_frac": float(bound_spherical) / float(bound_total) if bound_total > 0 else 0.0,
        "bound_interval_frac": float(bound_interval) / float(bound_total) if bound_total > 0 else 0.0,
        "bound_ellipsoidal_frac": float(bound_ellipsoidal) / float(bound_total) if bound_total > 0 else 0.0,
        # Fallback ladder counts (per-layer sums)
        "fallback_process_more_count": _sum_by_layer_int(
            result, "persistent_full_attention_fallback_process_more_count_by_layer"
        ),
        "fallback_widen_count": _sum_by_layer_int(
            result, "persistent_full_attention_fallback_widen_count_by_layer"
        ),
        "fallback_disable_compression_count": _sum_by_layer_int(
            result, "persistent_full_attention_fallback_disable_compression_count_by_layer"
        ),
        "fallback_disable_pruning_count": _sum_by_layer_int(
            result, "persistent_full_attention_fallback_disable_pruning_count_by_layer"
        ),
        "dense_fallback_count": _sum_by_layer_int(
            result, "persistent_full_attention_dense_fallback_count_by_layer"
        ),
        # Format distribution histograms (per-layer dicts of mode -> count)
        "block_k_mode_histogram": result.get(
            "persistent_full_attention_block_k_mode_histogram_by_layer", {}
        ),
        "block_v_mode_histogram": result.get(
            "persistent_full_attention_block_v_mode_histogram_by_layer", {}
        ),
    }


def _run_all_lanes_for_case(
    *,
    lanes: list[dict[str, Any]],
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    input_ids: Any,
    attention_mask: Any,
    decode_steps: int,
    warmup_steps: int,
    dense_reference_ids: list[int],
    case_tag: str,
) -> dict[str, dict[str, Any]]:
    """Run all *lanes* on one prompt using a single shared prefill pass.

    The prefill (dense forward on the full prompt) is identical for every lane
    since only the decode-phase bound mode differs.  Running it once and cloning
    the resulting KV-cache state for each lane avoids (len(lanes)-1) redundant
    O(N²) prefill passes per case.

    A single warmup pass using the first lane is also done here before any
    timed measurement.
    """
    device = next(model.parameters()).device

    # Run one shared prefill
    prefill_outputs, _ = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    first_decode_input = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _decode_from_prefill_clone(serving_config: Any, n_steps: int) -> dict[str, Any]:
        """Load a fresh clone of the prefill cache into *adapter* and run *n_steps* decode steps."""
        pkv_clone = _clone_qwen35_past_key_values(prefill_outputs.past_key_values)
        adapter.persistent_serving_config = serving_config
        adapter.clear()
        adapter.load_attention_subset_persistent_prefill_cache(pkv_clone)
        adapter.set_mode("dotcache_attention_subset_persistent_experimental")
        adapter.configure_persistent_shortlist_policy_context(
            prompt_family=None,
            prompt_length=int(input_ids.shape[1]),
        )
        runtime_state = adapter.require_persistent_hybrid_runtime_state()

        current_input_ids = first_decode_input.clone()
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        decode_ms_total = 0.0
        generated_ids: list[int] = []

        for _ in range(n_steps):
            generated_ids.append(int(current_input_ids.item()))
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                ),
                device=device,
            )
            decode_ms_total += step_ms
            runtime_state.advance(outputs.past_key_values)
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = cache_position + 1

        return {
            "decode_ms_per_step": float(decode_ms_total / max(n_steps, 1)),
            "generated_ids": generated_ids,
            "runtime_state": runtime_state,
        }

    # Warmup: throw-away decode run using the first lane's config
    if warmup_steps > 0:
        _decode_from_prefill_clone(_build_serving_config_for_lane(lanes[0]), warmup_steps)

    # Timed runs — one per lane, each gets a fresh prefill clone
    results: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        serving_config = _build_serving_config_for_lane(lane)
        run = _decode_from_prefill_clone(serving_config, decode_steps)

        runtime_state = run["runtime_state"]
        decode_ms_per_step = run["decode_ms_per_step"]
        generated_ids = run["generated_ids"]
        dense_cmp = dense_reference_ids[:decode_steps]
        exact_match = bool(generated_ids == dense_cmp and len(generated_ids) >= decode_steps)

        # Collect telemetry from the adapter/runtime state directly
        tel = _extract_telemetry(runtime_state.summary())
        results[lane["name"]] = {
            "case_tag": case_tag,
            "decode_ms_per_step": float(decode_ms_per_step),
            "exact_match_vs_dense": bool(exact_match),
            "score_ms_total": float(tel["score_ms_total"]),
            "selection_ms_total": float(tel["selection_ms_total"]),
            "executed_m0_blocks_total": int(tel["executed_m0_blocks_total"]),
            "executed_m3_blocks_total": int(tel["executed_m3_blocks_total"]),
            "first_certified_stop_blocks": int(tel["first_certified_stop_blocks"]),
            "last_checkpoint_count": int(tel["last_checkpoint_count"]),
            "bound_spherical_count": int(tel["bound_spherical_count"]),
            "bound_interval_count": int(tel["bound_interval_count"]),
            "bound_ellipsoidal_count": int(tel["bound_ellipsoidal_count"]),
            "bound_total_count": int(tel["bound_total_count"]),
            "bound_spherical_frac": float(tel["bound_spherical_frac"]),
            "bound_interval_frac": float(tel["bound_interval_frac"]),
            "bound_ellipsoidal_frac": float(tel["bound_ellipsoidal_frac"]),
            "fallback_process_more_count": int(tel["fallback_process_more_count"]),
            "fallback_widen_count": int(tel["fallback_widen_count"]),
            "fallback_disable_compression_count": int(tel["fallback_disable_compression_count"]),
            "fallback_disable_pruning_count": int(tel["fallback_disable_pruning_count"]),
            "dense_fallback_count": int(tel["dense_fallback_count"]),
            "block_k_mode_histogram": tel["block_k_mode_histogram"],
            "block_v_mode_histogram": tel["block_v_mode_histogram"],
        }
        print(
            f"  [{lane['name']}] {case_tag}: "
            f"{decode_ms_per_step:.2f} ms/step, "
            f"exact={exact_match}, "
            f"M0={tel['executed_m0_blocks_total']} M3={tel['executed_m3_blocks_total']}, "
            f"cert_stop_blocks={tel['first_certified_stop_blocks']}, "
            f"checkpoints={tel['last_checkpoint_count']}"
            + (
                f"  bound: sph={tel['bound_spherical_frac']:.0%} int={tel['bound_interval_frac']:.0%}"
                + (f" ellip={tel['bound_ellipsoidal_frac']:.0%}" if tel["bound_ellipsoidal_count"] > 0 else "")
                if tel["bound_total_count"] > 0
                else ""
            )
        )

    # Reset adapter to dense mode so the next case's shared prefill (full
    # sequence forward pass) does not hit the persistent-mode batch=1/query=1
    # guard.
    adapter.set_mode("dense")
    adapter.clear()
    return results


def _run_one_lane(
    *,
    lane: dict[str, Any],
    prompt_records: list[dict[str, Any]],
    model,
    tokenizer,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    dense_results_by_case: dict[str, list[int]],
    decode_steps: int,
    warmup_steps: int = 0,
    _warmup_done: list[bool] | None = None,
) -> list[dict[str, Any]]:
    """Original per-lane runner (used when --no-shared-prefill is passed).

    Runs one warmup call on the first case of the first lane if warmup_steps > 0
    and *_warmup_done* is a single-element list whose first element is False.
    """
    serving_config = _build_serving_config_for_lane(lane)
    records: list[dict[str, Any]] = []
    for prompt_record in prompt_records:
        case_tag = str(prompt_record["case_tag"])
        prompt_text = Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
        device = next(model.parameters()).device
        input_ids, attention_mask = _build_prompt_text_inputs(
            tokenizer,
            device=device,
            prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        # Warmup: once before the very first timed run across all lanes
        if warmup_steps > 0 and _warmup_done is not None and not _warmup_done[0]:
            run_qwen35_attention_subset_persistent_serving_harness(
                model,
                adapter,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=warmup_steps,
                persistent_serving_config=serving_config,
            )
            _warmup_done[0] = True
        result = run_qwen35_attention_subset_persistent_serving_harness(
            model,
            adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=decode_steps,
            persistent_serving_config=serving_config,
        )
        decode_ms_per_step = float(result.get("persistent_decode_ms_per_step", 0.0))
        generated_ids = [int(t) for t in result.get("persistent_generated_ids", [])]
        dense_ids = dense_results_by_case.get(case_tag, [])
        # Compare first decode_steps tokens of dense vs persistent
        dense_cmp = dense_ids[:decode_steps]
        exact_match = bool(generated_ids == dense_cmp and len(generated_ids) >= decode_steps)
        tel = _extract_telemetry(result)
        bound_winner_str = (
            f"bound_wins: sph={tel['bound_spherical_frac']:.0%} "
            f"int={tel['bound_interval_frac']:.0%} "
            f"ellip={tel['bound_ellipsoidal_frac']:.0%}"
            if tel["bound_total_count"] > 0
            else "bound_wins: (no data)"
        )
        records.append({
            "case_tag": case_tag,
            "decode_ms_per_step": float(decode_ms_per_step),
            "exact_match_vs_dense": bool(exact_match),
            "score_ms_total": float(tel["score_ms_total"]),
            "selection_ms_total": float(tel["selection_ms_total"]),
            "executed_m0_blocks_total": int(tel["executed_m0_blocks_total"]),
            "executed_m3_blocks_total": int(tel["executed_m3_blocks_total"]),
            "first_certified_stop_blocks": int(tel["first_certified_stop_blocks"]),
            "last_checkpoint_count": int(tel["last_checkpoint_count"]),
            "bound_spherical_count": int(tel["bound_spherical_count"]),
            "bound_interval_count": int(tel["bound_interval_count"]),
            "bound_ellipsoidal_count": int(tel["bound_ellipsoidal_count"]),
            "bound_total_count": int(tel["bound_total_count"]),
            "bound_spherical_frac": float(tel["bound_spherical_frac"]),
            "bound_interval_frac": float(tel["bound_interval_frac"]),
            "bound_ellipsoidal_frac": float(tel["bound_ellipsoidal_frac"]),
        })
        print(
            f"  [{lane['name']}] {case_tag}: "
            f"{decode_ms_per_step:.2f} ms/step, "
            f"exact={exact_match}, "
            f"M0={tel['executed_m0_blocks_total']} M3={tel['executed_m3_blocks_total']}, "
            f"cert_stop_blocks={tel['first_certified_stop_blocks']}, "
            f"checkpoints={tel['last_checkpoint_count']}, "
            f"{bound_winner_str}"
        )
    return records


def _summarize_lane(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    total_bound = sum(r.get("bound_total_count", 0) for r in records)
    return {
        "case_count": int(n),
        "avg_ms_per_step": float(sum(r["decode_ms_per_step"] for r in records) / n),
        "exact_match_vs_dense": float(sum(int(r["exact_match_vs_dense"]) for r in records) / n),
        "avg_score_ms_per_case": float(sum(r["score_ms_total"] for r in records) / n),
        "avg_selection_ms_per_case": float(sum(r["selection_ms_total"] for r in records) / n),
        "avg_executed_m0_blocks_per_case": float(sum(r["executed_m0_blocks_total"] for r in records) / n),
        "avg_executed_m3_blocks_per_case": float(sum(r["executed_m3_blocks_total"] for r in records) / n),
        "avg_first_certified_stop_blocks": float(sum(r["first_certified_stop_blocks"] for r in records) / n),
        "avg_last_checkpoint_count": float(sum(r["last_checkpoint_count"] for r in records) / n),
        # Bound winner fractions (aggregated across all cases in the lane)
        "total_bound_spherical_count": int(sum(r.get("bound_spherical_count", 0) for r in records)),
        "total_bound_interval_count": int(sum(r.get("bound_interval_count", 0) for r in records)),
        "total_bound_ellipsoidal_count": int(sum(r.get("bound_ellipsoidal_count", 0) for r in records)),
        "total_bound_count": int(total_bound),
        "bound_spherical_frac": float(sum(r.get("bound_spherical_count", 0) for r in records)) / float(total_bound) if total_bound > 0 else 0.0,
        "bound_interval_frac": float(sum(r.get("bound_interval_count", 0) for r in records)) / float(total_bound) if total_bound > 0 else 0.0,
        "bound_ellipsoidal_frac": float(sum(r.get("bound_ellipsoidal_count", 0) for r in records)) / float(total_bound) if total_bound > 0 else 0.0,
    }


def _render_markdown(
    *,
    payload: dict[str, Any],
    active_lanes: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen3.5 Bound Mode Compare",
        "",
        "Compares three certified upper-bound modes under the real mixed Stage 9 serving config.",
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
    lines += ["", "## Summary", ""]

    summaries = payload.get("lane_summaries", {})
    lines += [
        "| Lane | avg ms/step | exact_match_vs_dense | score_ms/case | cert_stop_blocks/case | checkpoints/case |",
        "|---|---|---|---|---|---|",
    ]
    for lane in active_lanes:
        s = summaries.get(lane["name"], {})
        lines.append(
            f"| `{lane['name']}` "
            f"| {s.get('avg_ms_per_step', 0.0):.2f} "
            f"| {s.get('exact_match_vs_dense', 0.0):.3f} "
            f"| {s.get('avg_score_ms_per_case', 0.0):.2f} "
            f"| {s.get('avg_first_certified_stop_blocks', 0.0):.1f} "
            f"| {s.get('avg_last_checkpoint_count', 0.0):.1f} |"
        )

    # Speed-up vs spherical
    spherical_ms = summaries.get("spherical_only", {}).get("avg_ms_per_step", 0.0)
    spherical_cert = summaries.get("spherical_only", {}).get("avg_first_certified_stop_blocks", 0.0)
    if spherical_ms > 0:
        lines += ["", "### Speed-up vs spherical_only", ""]
        for lane in active_lanes:
            if lane["name"] == "spherical_only":
                continue
            lane_ms = summaries.get(lane["name"], {}).get("avg_ms_per_step", 0.0)
            lane_cert = summaries.get(lane["name"], {}).get("avg_first_certified_stop_blocks", 0.0)
            if lane_ms > 0:
                speedup_pct = (spherical_ms - lane_ms) / spherical_ms * 100.0
                cert_reduction = (
                    f", cert_stop_blocks {lane_cert:.1f} vs {spherical_cert:.1f} ({(spherical_cert - lane_cert):.1f} fewer)"
                    if spherical_cert > 0
                    else ""
                )
                lines.append(
                    f"- `{lane['name']}`: {speedup_pct:+.1f}% ({lane_ms:.2f} vs {spherical_ms:.2f} ms/step){cert_reduction}"
                )

    # Bound winner table (only when at least one lane has winner data)
    summaries = payload.get("lane_summaries", {})
    has_winner_data = any(
        summaries.get(lane["name"], {}).get("total_bound_count", 0) > 0
        for lane in active_lanes
    )
    if has_winner_data:
        lines += ["", "## Bound winner fractions", ""]
        lines.append("Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.")
        lines += [
            "",
            "| Lane | spherical | interval | ellipsoidal | total evals |",
            "|---|---|---|---|---|",
        ]
        for lane in active_lanes:
            s = summaries.get(lane["name"], {})
            total = s.get("total_bound_count", 0)
            if total > 0:
                lines.append(
                    f"| `{lane['name']}` "
                    f"| {s.get('bound_spherical_frac', 0):.1%} "
                    f"| {s.get('bound_interval_frac', 0):.1%} "
                    f"| {s.get('bound_ellipsoidal_frac', 0):.1%} "
                    f"| {total:,} |"
                )
            else:
                lines.append(f"| `{lane['name']}` | — | — | — | 0 |")

    # Per-case detail
    lines += ["", "## Per-case results", ""]
    first_lane_records = payload.get("lane_records", {}).get(active_lanes[0]["name"], [])
    case_tags = [r["case_tag"] for r in first_lane_records]
    for case_tag in case_tags:
        lines.append(f"### {case_tag}")
        lines.append("")
        for lane in active_lanes:
            records = payload.get("lane_records", {}).get(lane["name"], [])
            case_record = next((r for r in records if r["case_tag"] == case_tag), None)
            if case_record is None:
                continue
            bound_str = ""
            if case_record.get("bound_total_count", 0) > 0:
                bound_str = (
                    f", bound_wins: sph={case_record['bound_spherical_frac']:.0%}"
                    f"/int={case_record['bound_interval_frac']:.0%}"
                    f"/ellip={case_record['bound_ellipsoidal_frac']:.0%}"
                )
            lines.append(
                f"- `{lane['name']}`: {case_record['decode_ms_per_step']:.2f} ms/step, "
                f"exact={case_record['exact_match_vs_dense']}, "
                f"cert_stop={case_record['first_certified_stop_blocks']} blocks, "
                f"chkpts={case_record['last_checkpoint_count']}"
                f"{bound_str}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Multi-process worker infrastructure
# ---------------------------------------------------------------------------
# Worker state is stored in a process-local dict so the model is loaded once
# per worker process rather than once per task.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(
    model_id: str,
    device: str,
    backend: str,
    torch_dtype: str,
    lane_dicts: list[dict[str, Any]],
) -> None:
    """Initialiser called once when each worker process starts.

    Loads the persistent model + adapter and caches them in ``_WORKER_STATE``
    so subsequent ``_worker_run_case`` calls reuse the same in-memory model.
    """
    import os as _os
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ensure the repo root is on the path inside the spawned process.
    _repo = str(Path(__file__).resolve().parents[1])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from dotcache.integrations.qwen35 import (
        Qwen35AttentionSubsetDotCacheModelAdapter,
        load_qwen35_text_only_from_pretrained,
    )
    from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
        real_mixed_probe_dotcache_config,
    )

    model, tokenizer = load_qwen35_text_only_from_pretrained(
        model_id, device=device, torch_dtype=torch_dtype
    )
    dotcache_config = real_mixed_probe_dotcache_config()
    first_lane = lane_dicts[0]
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=dotcache_config,
        persistent_serving_config=_build_serving_config_for_lane(first_lane),
        backend=backend,
    )
    _WORKER_STATE["model"] = model
    _WORKER_STATE["tokenizer"] = tokenizer
    _WORKER_STATE["adapter"] = adapter
    print(
        f"[worker pid={_os.getpid()}] model loaded on {device}",
        flush=True,
    )


def _worker_run_case(
    args_tuple: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    """Run all lanes for a single case.  Called once per prompt_record by the pool.

    Returns a dict ``{lane_name: result_record}`` exactly like
    ``_run_all_lanes_for_case`` but is safe to call in a spawned subprocess
    because it uses only process-local (non-shared) model state.
    """
    (
        prompt_record,
        dense_reference_ids,
        decode_steps,
        warmup_steps,
        lane_dicts,
    ) = args_tuple

    from pathlib import Path as _Path

    model = _WORKER_STATE["model"]
    tokenizer = _WORKER_STATE["tokenizer"]
    adapter = _WORKER_STATE["adapter"]
    device = next(model.parameters()).device

    prompt_text = _Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
    input_ids, attention_mask = _build_prompt_text_inputs(
        tokenizer,
        device=device,
        prompt_text=prompt_text,
        prompt_length=int(prompt_record.get("prompt_length", 0)),
    )
    case_tag = str(prompt_record["case_tag"])
    return _run_all_lanes_for_case(
        lanes=lane_dicts,
        model=model,
        adapter=adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
        warmup_steps=warmup_steps,
        dense_reference_ids=dense_reference_ids,
        case_tag=case_tag,
    )


def main() -> None:
    global _EXECUTION_STRATEGY_OVERRIDE
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_bound_mode_compare.py requires the optional transformers dependencies")

    if args.execution_strategy is not None:
        _EXECUTION_STRATEGY_OVERRIDE = str(args.execution_strategy)
    if args.max_k_comp_error is not None:
        _MAX_K_COMP_ERROR_OVERRIDE = float(args.max_k_comp_error)

    active_lanes = (
        [lane for lane in _LANES if lane["name"] in args.lanes]
        if args.lanes
        else list(_LANES)
    )
    if not active_lanes:
        raise SystemExit(f"no valid lanes selected; choices: {[l['name'] for l in _LANES]}")

    prompt_records = _resolve_prompt_records(
        manifest_path=args.manifest_path,
        prompt_files=[str(p) for p in args.prompt_files],
        prompt_file_target_length=int(args.prompt_file_target_length),
    )
    if not prompt_records:
        raise SystemExit("no prompt records resolved; provide --manifest-path or --prompt-files")

    print(f"Loading model {args.model_id} ...")
    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
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
        persistent_serving_config=_build_serving_config_for_lane(active_lanes[0]),
        backend=str(args.backend),
    )

    # Step 1: Dense reference pass — generates ground-truth token sequences
    print("\n--- Dense reference pass ---")
    dense_results_by_case: dict[str, list[int]] = {}
    for prompt_record in prompt_records:
        case_tag = str(prompt_record["case_tag"])
        prompt_text = Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
        dense_device = next(dense_model.parameters()).device
        dense_input_ids, dense_attention_mask = _build_prompt_text_inputs(
            dense_tokenizer,
            device=dense_device,
            prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        dense_result = run_qwen35_text_generation_harness(
            dense_model,
            dense_adapter,
            input_ids=dense_input_ids,
            attention_mask=dense_attention_mask,
            max_new_tokens=int(args.decode_steps) + 1,  # +1 because first token comes from prefill
            tokenizer=dense_tokenizer,
        )
        # dense_generated_ids has max_new_tokens entries (1 from prefill + decode_steps from decode)
        all_dense_ids = [int(t) for t in dense_result.get("dense_generated_ids", [])]
        # Persistent harness generates decode_steps tokens starting from prefill argmax
        # Keep first decode_steps tokens for comparison
        dense_results_by_case[case_tag] = all_dense_ids[:int(args.decode_steps)]
        print(f"  [dense] {case_tag}: {len(all_dense_ids)} tokens generated, keeping first {args.decode_steps}")

    # Step 2: Per-lane serving runs
    use_shared_prefill = not bool(args.no_shared_prefill)
    n_workers = max(1, int(args.workers))
    print(
        f"\n--- Lane runs (shared_prefill={use_shared_prefill}, "
        f"warmup_steps={args.warmup_steps}, workers={n_workers}) ---"
    )
    lane_records: dict[str, list[dict[str, Any]]] = {lane["name"]: [] for lane in active_lanes}

    if use_shared_prefill:
        # All lanes share a single prefill per case — avoids (N_lanes-1) redundant
        # O(N²) prefill passes per case.
        if n_workers > 1:
            # Multi-process path: each worker loads the model independently and
            # processes its subset of cases.  Workers run concurrently so their
            # Python-side overhead can overlap with other workers' GPU kernels,
            # improving overall GPU utilisation.
            print(
                f"  Dispatching {len(prompt_records)} cases to {n_workers} workers …"
            )
            worker_tasks = [
                (
                    pr,
                    dense_results_by_case.get(str(pr["case_tag"]), []),
                    int(args.decode_steps),
                    int(args.warmup_steps),
                    active_lanes,
                )
                for pr in prompt_records
            ]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_worker_init,
                initargs=(
                    str(args.model_id),
                    str(args.device),
                    str(args.backend),
                    str(args.torch_dtype),
                    active_lanes,
                ),
            ) as pool:
                for case_lane_results in pool.map(_worker_run_case, worker_tasks):
                    for lane in active_lanes:
                        lane_records[lane["name"]].append(case_lane_results[lane["name"]])
        else:
            # Single-process path (default).
            for prompt_record in prompt_records:
                case_tag = str(prompt_record["case_tag"])
                prompt_text = Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
                device = next(persistent_model.parameters()).device
                input_ids, attention_mask = _build_prompt_text_inputs(
                    persistent_tokenizer,
                    device=device,
                    prompt_text=prompt_text,
                    prompt_length=int(prompt_record.get("prompt_length", 0)),
                )
                print(f"\n[case: {case_tag}]")
                case_lane_results = _run_all_lanes_for_case(
                    lanes=active_lanes,
                    model=persistent_model,
                    adapter=persistent_adapter,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decode_steps=int(args.decode_steps),
                    warmup_steps=int(args.warmup_steps),
                    dense_reference_ids=dense_results_by_case.get(case_tag, []),
                    case_tag=case_tag,
                )
                for lane in active_lanes:
                    lane_records[lane["name"]].append(case_lane_results[lane["name"]])
    else:
        # Original per-lane runner (--no-shared-prefill)
        _warmup_done: list[bool] = [False]
        for lane in active_lanes:
            print(f"\n--- Lane: {lane['label']} ---")
            lane_records[lane["name"]] = _run_one_lane(
                lane=lane,
                prompt_records=prompt_records,
                model=persistent_model,
                tokenizer=persistent_tokenizer,
                adapter=persistent_adapter,
                dense_results_by_case=dense_results_by_case,
                decode_steps=int(args.decode_steps),
                warmup_steps=int(args.warmup_steps),
                _warmup_done=_warmup_done,
            )

    # Step 3: Summarize
    lane_summaries = {
        lane["name"]: _summarize_lane(lane_records[lane["name"]])
        for lane in active_lanes
    }

    print("\n=== Summary ===")
    for lane in active_lanes:
        s = lane_summaries[lane["name"]]
        print(
            f"  {lane['name']:20s}: {s.get('avg_ms_per_step', 0):.2f} ms/step, "
            f"exact={s.get('exact_match_vs_dense', 0):.3f}, "
            f"score={s.get('avg_score_ms_per_case', 0):.2f} ms/case, "
            f"cert_stop={s.get('avg_first_certified_stop_blocks', 0):.1f} blocks, "
            f"chkpts={s.get('avg_last_checkpoint_count', 0):.1f}"
        )

    spherical_ms = lane_summaries.get("spherical_only", {}).get("avg_ms_per_step", 0.0)
    if spherical_ms > 0:
        for lane in active_lanes:
            if lane["name"] == "spherical_only":
                continue
            lane_ms = lane_summaries[lane["name"]].get("avg_ms_per_step", 0.0)
            if lane_ms > 0:
                speedup_pct = (spherical_ms - lane_ms) / spherical_ms * 100.0
                print(f"  {lane['name']} speedup vs spherical: {speedup_pct:+.1f}%")

    # Print bound winner fractions
    has_winner_data = any(
        lane_summaries.get(lane["name"], {}).get("total_bound_count", 0) > 0
        for lane in active_lanes
    )
    if has_winner_data:
        print("\n=== Bound winner fractions (which bound provides the tightest upper bound) ===")
        for lane in active_lanes:
            s = lane_summaries[lane["name"]]
            total = s.get("total_bound_count", 0)
            if total > 0:
                print(
                    f"  {lane['name']:20s}: "
                    f"spherical={s.get('bound_spherical_frac', 0):.1%}  "
                    f"interval={s.get('bound_interval_frac', 0):.1%}  "
                    f"ellipsoidal={s.get('bound_ellipsoidal_frac', 0):.1%}  "
                    f"(total evals: {total:,})"
                )
            else:
                print(f"  {lane['name']:20s}: (no bound winner data — interval/ellipsoidal both disabled)")

    payload: dict[str, Any] = {
        "lane_summaries": lane_summaries,
        "lane_records": lane_records,
        "active_lanes": [lane["name"] for lane in active_lanes],
        "manifest_path": str(args.manifest_path),
        "decode_steps": int(args.decode_steps),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {out_path}")

    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_markdown(payload=payload, active_lanes=active_lanes), encoding="utf-8")
        print(f"MD   -> {out_path}")


if __name__ == "__main__":
    main()
