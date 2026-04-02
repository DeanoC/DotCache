from __future__ import annotations

import argparse
import json
import os
import math


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serving-style DotCache benchmark for the Qwen3.5 full-attention subset.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--default-mode-k", choices=["M0", "M1", "M2", "M3", "M4", "T3"], default="M0")
    parser.add_argument("--default-mode-v", choices=["M0", "M1", "M3", "T3"], default="M0")
    parser.add_argument("--key-policy-tier", choices=["exact", "strict", "balanced", "aggressive"], default="exact")
    parser.add_argument("--value-policy-tier", choices=["exact", "strict", "balanced", "aggressive"], default="exact")
    parser.add_argument("--key-mode-override", action="append", default=[])
    parser.add_argument("--value-mode-override", action="append", default=[])
    parser.add_argument("--key-layer-sensitivity", action="append", default=[])
    parser.add_argument("--value-layer-sensitivity", action="append", default=[])
    parser.add_argument("--key-policy-override", action="append", default=[])
    parser.add_argument("--value-policy-override", action="append", default=[])
    parser.add_argument("--layer-profile", default=None)
    parser.add_argument("--quant-scheme-k", choices=["affine", "lut", "sketch", "project", "turbo3"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "lut", "turbo3"], default="affine")
    parser.add_argument("--escape-dtype", choices=["float16", "float32", "int8"], default="float16")
    parser.add_argument("--recent-page-escape-dtype", choices=["float16", "float32", "int8"], default="float16")
    parser.add_argument("--recent-window", type=int, default=128)
    parser.add_argument("--execution-recent-window", type=int, default=0)
    parser.add_argument("--execution-sink-window", type=int, default=0)
    parser.add_argument("--execution-recent-window-layer", action="append", default=[])
    parser.add_argument("--execution-recent-window-context-layer", action="append", default=[])
    parser.add_argument("--execution-relevance-top-k", type=int, default=0)
    parser.add_argument("--execution-relevance-top-k-layer", action="append", default=[])
    parser.add_argument("--execution-relevance-top-k-context-layer", action="append", default=[])
    parser.add_argument("--execution-full-context-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-disable-grouped-batching-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-recent-old-bonus-window", type=int, default=0)
    parser.add_argument("--execution-recent-old-bonus-strength", type=float, default=0.0)
    parser.add_argument("--execution-recent-old-bonus-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-relevance-mode", choices=["sketch", "envelope"], default="envelope")
    parser.add_argument("--execution-secondary-relevance-mode", choices=["", "sketch", "envelope"], default="")
    parser.add_argument("--execution-secondary-relevance-top-k", type=int, default=0)
    parser.add_argument("--execution-secondary-relevance-min-overlap", type=float, default=0.0)
    parser.add_argument("--execution-secondary-relevance-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-recent-neighbor-rescue-top-k", type=int, default=0)
    parser.add_argument("--execution-recent-neighbor-rescue-anchor-window", type=int, default=0)
    parser.add_argument("--execution-recent-neighbor-rescue-min-anchor-pages", type=int, default=0)
    parser.add_argument("--execution-recent-neighbor-rescue-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-exact-promote-top-k", type=int, default=0)
    parser.add_argument("--execution-exact-promote-min-margin-threshold", type=float, default=0.0)
    parser.add_argument("--execution-exact-promote-max-context", type=int, default=0)
    parser.add_argument("--execution-exact-promote-margin-threshold", type=float, default=0.0)
    parser.add_argument("--execution-exact-promote-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-exact-promote-union-rescue-top-k", type=int, default=0)
    parser.add_argument("--execution-grouped-decode-compact", action="store_true")
    parser.add_argument("--execution-grouped-mix-compact", action="store_true")
    parser.add_argument("--execution-grouped-mix-disable-packed-cuda", action="store_true")
    parser.add_argument("--execution-freeze-chunk-budget-during-decode", action="store_true")
    parser.add_argument("--execution-builtin-selector-cache", action="store_true")
    parser.add_argument("--execution-builtin-selector-score-all-pages", action="store_true")
    parser.add_argument("--execution-builtin-selector-candidate-only", action="store_true")
    parser.add_argument("--execution-builtin-selector-score-all-pages-min-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--execution-value-escape-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-value-escape-mode", choices=["M0", "M1", "M3", "T3"], default="M3")
    parser.add_argument("--execution-value-escape-old-only", action="store_true")
    parser.add_argument("--execution-value-escape-top-k", type=int, default=0)
    parser.add_argument("--execution-value-escape-prewarm", action="store_true")
    parser.add_argument("--execution-value-escape-prewarm-min-context", type=int, default=0)
    parser.add_argument("--execution-exact-refine-top-k", type=int, default=0)
    parser.add_argument("--execution-exact-refine-layer", type=int, action="append", default=[])
    parser.add_argument("--m2-sketch-dim-k", type=int, default=8)
    parser.add_argument("--m2-center-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-segment-count-k", type=int, default=1)
    parser.add_argument("--m2-adaptive-segments-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-adaptive-min-improvement-k", type=float, default=0.1)
    parser.add_argument("--m2-prefilter-top-k", type=int, default=0)
    parser.add_argument("--m2-prefilter-min-pages", type=int, default=8)
    parser.add_argument("--prefer-m4-project-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lut-refine-steps", type=int, default=0)
    parser.add_argument("--preconditioner", choices=["none", "tanh"], default="none")
    parser.add_argument("--precondition-strength", type=float, default=1.0)
    parser.add_argument("--m1-segment-count-k", type=int, default=1)
    parser.add_argument("--m1-segment-count-v", type=int, default=1)
    parser.add_argument("--m1-fallback-to-m0", action="store_true")
    parser.add_argument("--m1-error-threshold", type=float, default=0.2)
    parser.add_argument("--m1-token-p95-error-threshold", type=float, default=0.55)
    parser.add_argument("--learned-page-selector-path", default=None)
    parser.add_argument("--learned-page-selector-prompt-family", default=None)
    parser.add_argument("--learned-page-selector-prompt-variant", default=None)
    parser.add_argument("--learned-page-selector-scope", choices=["KV", "K", "V"], default="KV")
    parser.add_argument("--learned-page-selector-target-candidate", default="M3/affine/4/float16")
    parser.add_argument("--learned-page-selector-logit-offset", type=float, default=0.0)
    parser.add_argument("--prepared-chunk-cache-budget-ratio", type=float, default=None)
    parser.add_argument("--prepared-chunk-cache-min-bytes", type=int, default=None)
    parser.add_argument("--prepared-chunk-cache-max-bytes", type=int, default=None)
    parser.add_argument("--prepared-chunk-cache-min-page-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[1, 32])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--blas-num-threads", type=int, default=0)
    parser.add_argument("--scorer-diagnostic", action="store_true")
    parser.add_argument("--recall-analysis", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    return parser.parse_args()


def _build_exact_length_inputs(
    harness,
    *,
    prompt_unit: str,
    prompt_length: int,
):
    import torch

    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _run_case(
    harness,
    *,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    base_record: dict[str, object],
    continue_on_error: bool,
) -> None:
    record = _execute_case(
        harness,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        base_record=base_record,
        continue_on_error=continue_on_error,
    )
    print(json.dumps(record, sort_keys=True), flush=True)


def _execute_case(
    harness,
    *,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    base_record: dict[str, object],
    continue_on_error: bool,
) -> dict[str, object]:
    try:
        if bool(base_record.get("scorer_diagnostic", False)):
            record = harness.run_attention_subset_dotcache_serving_scorer_diagnostic(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=max_new_tokens,
                profile_backend=bool(base_record.get("profile_backend", False)),
                trace_python_allocations=bool(base_record.get("trace_python_allocations", False)),
            )
        elif bool(base_record.get("recall_analysis", False)):
            record = harness.run_attention_subset_dotcache_serving_recall_analysis(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=max_new_tokens,
                profile_backend=bool(base_record.get("profile_backend", False)),
            )
        elif bool(base_record.get("quality_check", False)):
            record = harness.run_attention_subset_dotcache_serving_quality(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=max_new_tokens,
                profile_backend=bool(base_record.get("profile_backend", False)),
                trace_python_allocations=bool(base_record.get("trace_python_allocations", False)),
            )
        else:
            record = harness.run_attention_subset_dotcache_serving(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decode_steps=max_new_tokens,
                profile_backend=bool(base_record.get("profile_backend", False)),
            )
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not continue_on_error:
            raise
        error_record = dict(base_record)
        effective_config = getattr(getattr(harness, "adapter", None), "dotcache_config", None)
        if effective_config is not None:
            error_record.update(
                {
                    "execution_recent_window": int(effective_config.execution_recent_window),
                    "execution_sink_window": int(effective_config.execution_sink_window),
                    "execution_recent_window_overrides": list(effective_config.execution_recent_window_overrides),
                    "execution_recent_window_context_overrides": list(
                        effective_config.execution_recent_window_context_overrides
                    ),
                    "execution_relevance_top_k": int(effective_config.execution_relevance_top_k),
                    "execution_relevance_top_k_overrides": list(effective_config.execution_relevance_top_k_overrides),
                    "execution_relevance_top_k_context_overrides": list(
                        effective_config.execution_relevance_top_k_context_overrides
                    ),
                    "execution_full_context_layers": list(effective_config.execution_full_context_layers),
                    "execution_disable_grouped_batching_layers": list(
                        effective_config.execution_disable_grouped_batching_layers
                    ),
                    "execution_recent_old_bonus_window": int(effective_config.execution_recent_old_bonus_window),
                    "execution_recent_old_bonus_strength": float(effective_config.execution_recent_old_bonus_strength),
                    "execution_recent_old_bonus_layers": list(effective_config.execution_recent_old_bonus_layers),
                    "execution_secondary_relevance_mode": str(effective_config.execution_secondary_relevance_mode),
                    "execution_secondary_relevance_top_k": int(effective_config.execution_secondary_relevance_top_k),
                    "execution_secondary_relevance_min_overlap": float(
                        effective_config.execution_secondary_relevance_min_overlap
                    ),
                    "execution_secondary_relevance_layers": list(effective_config.execution_secondary_relevance_layers),
                    "execution_recent_neighbor_rescue_top_k": int(effective_config.execution_recent_neighbor_rescue_top_k),
                    "execution_recent_neighbor_rescue_anchor_window": int(
                        effective_config.execution_recent_neighbor_rescue_anchor_window
                    ),
                    "execution_recent_neighbor_rescue_min_anchor_pages": int(
                        effective_config.execution_recent_neighbor_rescue_min_anchor_pages
                    ),
                    "execution_recent_neighbor_rescue_layers": list(
                        effective_config.execution_recent_neighbor_rescue_layers
                    ),
                    "execution_exact_promote_top_k": int(effective_config.execution_exact_promote_top_k),
                    "execution_exact_promote_min_margin_threshold": float(
                        effective_config.execution_exact_promote_min_margin_threshold
                    ),
                    "execution_exact_promote_max_context": int(effective_config.execution_exact_promote_max_context),
                    "execution_exact_promote_margin_threshold": float(
                        effective_config.execution_exact_promote_margin_threshold
                    ),
                    "execution_exact_promote_layers": list(effective_config.execution_exact_promote_layers),
                    "execution_exact_promote_union_rescue_top_k": int(
                        effective_config.execution_exact_promote_union_rescue_top_k
                    ),
                    "execution_grouped_decode_compact": bool(effective_config.execution_grouped_decode_compact),
                    "execution_grouped_mix_compact": bool(effective_config.execution_grouped_mix_compact),
                    "execution_grouped_mix_disable_packed_cuda": bool(
                        effective_config.execution_grouped_mix_disable_packed_cuda
                    ),
                    "execution_freeze_chunk_budget_during_decode": bool(
                        effective_config.execution_freeze_chunk_budget_during_decode
                    ),
                    "execution_builtin_selector_cache": bool(
                        effective_config.execution_builtin_selector_cache
                    ),
                    "execution_builtin_selector_score_all_pages": bool(
                        effective_config.execution_builtin_selector_score_all_pages
                    ),
                    "execution_builtin_selector_candidate_only": bool(
                        effective_config.execution_builtin_selector_candidate_only
                    ),
                    "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
                        effective_config.execution_builtin_selector_score_all_pages_min_candidate_fraction
                    ),
                    "execution_value_escape_layers": list(effective_config.execution_value_escape_layers),
                    "execution_value_escape_mode": str(effective_config.execution_value_escape_mode),
                    "execution_value_escape_old_only": bool(effective_config.execution_value_escape_old_only),
                    "execution_value_escape_top_k": int(effective_config.execution_value_escape_top_k),
                    "execution_value_escape_prewarm": bool(effective_config.execution_value_escape_prewarm),
                    "execution_value_escape_prewarm_min_context": int(
                        effective_config.execution_value_escape_prewarm_min_context
                    ),
                    "execution_relevance_mode": str(effective_config.execution_relevance_mode),
                    "serving_shortlist_heuristic_applied": bool(
                        getattr(getattr(harness, "adapter", None), "serving_shortlist_heuristic_applied", False)
                    ),
                }
            )
        error_record.update(
            {
                "benchmark": "qwen35_attention_subset_dotcache_serving",
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "prompt_length": int(input_ids.shape[1]),
            }
        )
        return error_record

    merged_record = dict(base_record)
    merged_record.update(record)
    return merged_record


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(fraction))) * float(len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - float(lower)
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _aggregate_record_values(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("records must be non-empty")
    template = records[-1]
    aggregated: dict[str, object] = {}
    for key, template_value in template.items():
        values = [record.get(key) for record in records if key in record]
        if isinstance(template_value, dict) and all(isinstance(value, dict) for value in values):
            aggregated[key] = _aggregate_record_values(values)  # type: ignore[arg-type]
            continue
        if _is_numeric(template_value) and all(_is_numeric(value) for value in values):
            numeric_values = [float(value) for value in values]
            mean_value = sum(numeric_values) / float(len(numeric_values))
            median_value = _percentile(numeric_values, 0.5)
            p95_value = _percentile(numeric_values, 0.95)
            if isinstance(template_value, int) and all(value.is_integer() for value in numeric_values):
                aggregated[key] = int(round(median_value))
            else:
                aggregated[key] = median_value
            aggregated[f"{key}_mean"] = mean_value
            aggregated[f"{key}_median"] = median_value
            aggregated[f"{key}_p95"] = p95_value
            continue
        aggregated[key] = template_value
    return aggregated


def _run_case_series(
    harness,
    *,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    base_record: dict[str, object],
    continue_on_error: bool,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    if warmup_runs <= 0 and measured_runs <= 1:
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            base_record=base_record,
            continue_on_error=continue_on_error,
        )
        return

    for warmup_index in range(max(0, int(warmup_runs))):
        warmup_record = _execute_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            base_record=base_record,
            continue_on_error=continue_on_error,
        )
        warmup_record.update(
            {
                "measurement_kind": "warmup",
                "measurement_index": int(warmup_index),
                "warmup_runs": int(warmup_runs),
                "measured_runs": int(measured_runs),
            }
        )
        print(json.dumps(warmup_record, sort_keys=True), flush=True)
        if warmup_record.get("status") == "error":
            return

    measured_records: list[dict[str, object]] = []
    for measurement_index in range(max(1, int(measured_runs))):
        trial_record = _execute_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            base_record=base_record,
            continue_on_error=continue_on_error,
        )
        trial_record.update(
            {
                "measurement_kind": "trial",
                "measurement_index": int(measurement_index),
                "warmup_runs": int(warmup_runs),
                "measured_runs": int(measured_runs),
            }
        )
        print(json.dumps(trial_record, sort_keys=True), flush=True)
        if trial_record.get("status") == "error":
            return
        measured_records.append(trial_record)

    aggregate_record = _aggregate_record_values(measured_records)
    aggregate_record.update(
        {
            "measurement_kind": "aggregate",
            "measurement_index": None,
            "warmup_runs": int(warmup_runs),
            "measured_runs": int(measured_runs),
        }
    )
    print(json.dumps(aggregate_record, sort_keys=True), flush=True)


def _resolve_args_from_layer_profile(args: argparse.Namespace) -> None:
    from dotcache.config_io import load_layer_profile

    if args.layer_profile is None:
        return
    profile = load_layer_profile(args.layer_profile)
    profile_model_id = profile.get("model_id")
    if profile_model_id not in (None, args.model_id):
        raise SystemExit(f"layer profile model_id {profile_model_id!r} does not match --model-id {args.model_id!r}")
    if args.default_mode_k == "M0":
        args.default_mode_k = str(profile.get("default_mode_k", args.default_mode_k))
    if args.default_mode_v == "M0":
        args.default_mode_v = str(profile.get("default_mode_v", args.default_mode_v))
    if args.key_policy_tier == "exact":
        args.key_policy_tier = str(profile.get("key_policy_tier", args.key_policy_tier))
    if args.value_policy_tier == "exact":
        args.value_policy_tier = str(profile.get("value_policy_tier", args.value_policy_tier))
    if not args.key_mode_override:
        args.key_mode_override = list(profile.get("key_mode_overrides", []))
    if not args.value_mode_override:
        args.value_mode_override = list(profile.get("value_mode_overrides", []))
    if not args.key_layer_sensitivity:
        args.key_layer_sensitivity = list(profile.get("key_layer_sensitivity", []))
    if not args.value_layer_sensitivity:
        args.value_layer_sensitivity = list(profile.get("value_layer_sensitivity", []))
    if not args.key_policy_override:
        args.key_policy_override = list(profile.get("key_policy_overrides", []))
    if not args.value_policy_override:
        args.value_policy_override = list(profile.get("value_policy_overrides", []))
    if args.recent_window == 128:
        args.recent_window = int(profile.get("recent_window", args.recent_window))
    if args.execution_recent_window == 0:
        args.execution_recent_window = int(profile.get("execution_recent_window", args.execution_recent_window))
    if args.execution_sink_window == 0:
        args.execution_sink_window = int(profile.get("execution_sink_window", args.execution_sink_window))
    if args.execution_relevance_top_k == 0:
        args.execution_relevance_top_k = int(profile.get("execution_relevance_top_k", args.execution_relevance_top_k))
    if not args.execution_relevance_top_k_layer:
        args.execution_relevance_top_k_layer = list(
            profile.get("execution_relevance_top_k_overrides", args.execution_relevance_top_k_layer)
        )
    if not args.execution_relevance_top_k_context_layer:
        args.execution_relevance_top_k_context_layer = list(
            profile.get(
                "execution_relevance_top_k_context_overrides",
                args.execution_relevance_top_k_context_layer,
            )
        )
    if args.execution_relevance_mode == "envelope":
        args.execution_relevance_mode = str(profile.get("execution_relevance_mode", args.execution_relevance_mode))
    if not args.execution_full_context_layer:
        args.execution_full_context_layer = list(
            profile.get("execution_full_context_layers", args.execution_full_context_layer)
        )
    if not args.execution_disable_grouped_batching_layer:
        args.execution_disable_grouped_batching_layer = list(
            profile.get("execution_disable_grouped_batching_layers", args.execution_disable_grouped_batching_layer)
        )
    if args.execution_builtin_selector_cache is False:
        args.execution_builtin_selector_cache = bool(
            profile.get("execution_builtin_selector_cache", args.execution_builtin_selector_cache)
        )
    if args.execution_builtin_selector_score_all_pages is False:
        args.execution_builtin_selector_score_all_pages = bool(
            profile.get(
                "execution_builtin_selector_score_all_pages",
                args.execution_builtin_selector_score_all_pages,
            )
        )
    if args.execution_builtin_selector_candidate_only is False:
        args.execution_builtin_selector_candidate_only = bool(
            profile.get(
                "execution_builtin_selector_candidate_only",
                args.execution_builtin_selector_candidate_only,
            )
        )
    if args.execution_builtin_selector_score_all_pages_min_candidate_fraction == 0.0:
        args.execution_builtin_selector_score_all_pages_min_candidate_fraction = float(
            profile.get(
                "execution_builtin_selector_score_all_pages_min_candidate_fraction",
                args.execution_builtin_selector_score_all_pages_min_candidate_fraction,
            )
        )
    if not args.execution_value_escape_layer:
        args.execution_value_escape_layer = [
            int(layer_id)
            for layer_id in profile.get("execution_value_escape_layers", args.execution_value_escape_layer)
        ]
    if args.execution_value_escape_mode == "M3":
        args.execution_value_escape_mode = str(
            profile.get("execution_value_escape_mode", args.execution_value_escape_mode)
        )
    if args.execution_value_escape_old_only is False:
        args.execution_value_escape_old_only = bool(
            profile.get("execution_value_escape_old_only", args.execution_value_escape_old_only)
        )
    if args.execution_value_escape_top_k == 0:
        args.execution_value_escape_top_k = int(
            profile.get("execution_value_escape_top_k", args.execution_value_escape_top_k)
        )
    if args.execution_value_escape_prewarm is False:
        args.execution_value_escape_prewarm = bool(
            profile.get("execution_value_escape_prewarm", args.execution_value_escape_prewarm)
        )
    if args.execution_value_escape_prewarm_min_context == 0:
        args.execution_value_escape_prewarm_min_context = int(
            profile.get(
                "execution_value_escape_prewarm_min_context",
                args.execution_value_escape_prewarm_min_context,
            )
        )
    if args.execution_exact_promote_top_k == 0:
        args.execution_exact_promote_top_k = int(
            profile.get("execution_exact_promote_top_k", args.execution_exact_promote_top_k)
        )
    if args.execution_exact_promote_margin_threshold == 0.0:
        args.execution_exact_promote_margin_threshold = float(
            profile.get(
                "execution_exact_promote_margin_threshold",
                args.execution_exact_promote_margin_threshold,
            )
        )
    if not args.execution_exact_promote_layer:
        args.execution_exact_promote_layer = list(
            profile.get("execution_exact_promote_layers", args.execution_exact_promote_layer)
        )
    if args.execution_exact_refine_top_k == 0:
        args.execution_exact_refine_top_k = int(
            profile.get("execution_exact_refine_top_k", args.execution_exact_refine_top_k)
        )
    if not args.execution_exact_refine_layer:
        args.execution_exact_refine_layer = list(
            profile.get("execution_exact_refine_layers", args.execution_exact_refine_layer)
        )


def _build_dotcache_config(args: argparse.Namespace, *, head_dim: int) -> DotCacheConfig:
    from dotcache.config import DotCacheConfig

    config_kwargs: dict[str, object] = {
        "head_dim": head_dim,
        "group_size": args.group_size,
        "bits_k": args.bits_k,
        "bits_v": args.bits_v,
        "default_mode_k": args.default_mode_k,
        "default_mode_v": args.default_mode_v,
        "key_policy_tier": args.key_policy_tier,
        "value_policy_tier": args.value_policy_tier,
        "key_mode_overrides": tuple(args.key_mode_override),
        "value_mode_overrides": tuple(args.value_mode_override),
        "key_layer_sensitivity": tuple(args.key_layer_sensitivity),
        "value_layer_sensitivity": tuple(args.value_layer_sensitivity),
        "key_policy_overrides": tuple(args.key_policy_override),
        "value_policy_overrides": tuple(args.value_policy_override),
        "quant_scheme_k": args.quant_scheme_k,
        "quant_scheme_v": args.quant_scheme_v,
        "escape_dtype": args.escape_dtype,
        "recent_page_escape_dtype": args.recent_page_escape_dtype,
        "recent_window": args.recent_window,
        "execution_recent_window": args.execution_recent_window,
        "execution_sink_window": args.execution_sink_window,
        "execution_recent_window_overrides": tuple(args.execution_recent_window_layer),
        "execution_recent_window_context_overrides": tuple(args.execution_recent_window_context_layer),
        "execution_relevance_top_k": args.execution_relevance_top_k,
        "execution_relevance_top_k_overrides": tuple(args.execution_relevance_top_k_layer),
        "execution_relevance_top_k_context_overrides": tuple(args.execution_relevance_top_k_context_layer),
        "execution_full_context_layers": tuple(args.execution_full_context_layer),
        "execution_disable_grouped_batching_layers": tuple(args.execution_disable_grouped_batching_layer),
        "execution_recent_old_bonus_window": args.execution_recent_old_bonus_window,
        "execution_recent_old_bonus_strength": args.execution_recent_old_bonus_strength,
        "execution_recent_old_bonus_layers": tuple(args.execution_recent_old_bonus_layer),
        "execution_relevance_mode": args.execution_relevance_mode,
        "execution_secondary_relevance_mode": args.execution_secondary_relevance_mode,
        "execution_secondary_relevance_top_k": args.execution_secondary_relevance_top_k,
        "execution_secondary_relevance_min_overlap": args.execution_secondary_relevance_min_overlap,
        "execution_secondary_relevance_layers": tuple(args.execution_secondary_relevance_layer),
        "execution_recent_neighbor_rescue_top_k": args.execution_recent_neighbor_rescue_top_k,
        "execution_recent_neighbor_rescue_anchor_window": args.execution_recent_neighbor_rescue_anchor_window,
        "execution_recent_neighbor_rescue_min_anchor_pages": args.execution_recent_neighbor_rescue_min_anchor_pages,
        "execution_recent_neighbor_rescue_layers": tuple(args.execution_recent_neighbor_rescue_layer),
        "execution_exact_promote_top_k": args.execution_exact_promote_top_k,
        "execution_exact_promote_min_margin_threshold": args.execution_exact_promote_min_margin_threshold,
        "execution_exact_promote_max_context": args.execution_exact_promote_max_context,
        "execution_exact_promote_margin_threshold": args.execution_exact_promote_margin_threshold,
        "execution_exact_promote_layers": tuple(args.execution_exact_promote_layer),
        "execution_exact_promote_union_rescue_top_k": args.execution_exact_promote_union_rescue_top_k,
        "execution_grouped_decode_compact": args.execution_grouped_decode_compact,
        "execution_grouped_mix_compact": args.execution_grouped_mix_compact,
        "execution_grouped_mix_disable_packed_cuda": args.execution_grouped_mix_disable_packed_cuda,
        "execution_freeze_chunk_budget_during_decode": args.execution_freeze_chunk_budget_during_decode,
        "execution_builtin_selector_cache": args.execution_builtin_selector_cache,
        "execution_builtin_selector_score_all_pages": args.execution_builtin_selector_score_all_pages,
        "execution_builtin_selector_candidate_only": args.execution_builtin_selector_candidate_only,
        "execution_builtin_selector_score_all_pages_min_candidate_fraction": (
            args.execution_builtin_selector_score_all_pages_min_candidate_fraction
        ),
        "execution_value_escape_layers": tuple(args.execution_value_escape_layer),
        "execution_value_escape_mode": args.execution_value_escape_mode,
        "execution_value_escape_old_only": args.execution_value_escape_old_only,
        "execution_value_escape_top_k": args.execution_value_escape_top_k,
        "execution_value_escape_prewarm": args.execution_value_escape_prewarm,
        "execution_value_escape_prewarm_min_context": args.execution_value_escape_prewarm_min_context,
        "execution_exact_refine_top_k": args.execution_exact_refine_top_k,
        "execution_exact_refine_layers": tuple(args.execution_exact_refine_layer),
        "m2_sketch_dim_k": args.m2_sketch_dim_k,
        "m2_center_k": args.m2_center_k,
        "m2_segment_count_k": args.m2_segment_count_k,
        "m2_adaptive_segments_k": args.m2_adaptive_segments_k,
        "m2_adaptive_min_improvement_k": args.m2_adaptive_min_improvement_k,
        "m2_prefilter_top_k": args.m2_prefilter_top_k,
        "m2_prefilter_min_pages": args.m2_prefilter_min_pages,
        "prefer_m4_project_k": args.prefer_m4_project_k,
        "lut_refine_steps": args.lut_refine_steps,
        "preconditioner": args.preconditioner,
        "precondition_strength": args.precondition_strength,
        "m1_segment_count_k": args.m1_segment_count_k,
        "m1_segment_count_v": args.m1_segment_count_v,
        "m1_fallback_to_m0": args.m1_fallback_to_m0,
        "m1_error_threshold": args.m1_error_threshold,
        "m1_token_p95_error_threshold": args.m1_token_p95_error_threshold,
        "tokens_per_page": args.tokens_per_page,
        "learned_page_selector_path": args.learned_page_selector_path,
        "learned_page_selector_prompt_family": args.learned_page_selector_prompt_family,
        "learned_page_selector_prompt_variant": args.learned_page_selector_prompt_variant,
        "learned_page_selector_scope": args.learned_page_selector_scope,
        "learned_page_selector_target_candidate": args.learned_page_selector_target_candidate,
        "learned_page_selector_logit_offset": args.learned_page_selector_logit_offset,
    }
    if args.prepared_chunk_cache_budget_ratio is not None:
        config_kwargs["prepared_chunk_cache_budget_ratio"] = args.prepared_chunk_cache_budget_ratio
    if args.prepared_chunk_cache_min_bytes is not None:
        config_kwargs["prepared_chunk_cache_min_bytes"] = args.prepared_chunk_cache_min_bytes
    if args.prepared_chunk_cache_max_bytes is not None:
        config_kwargs["prepared_chunk_cache_max_bytes"] = args.prepared_chunk_cache_max_bytes
    return DotCacheConfig(
        **config_kwargs,
    )


def _common_record(args: argparse.Namespace, *, max_position_embeddings: int) -> dict[str, object]:
    return {
        "benchmark": "qwen35_attention_subset_dotcache_serving",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "weight_quantization": args.weight_quantization,
        "tokens_per_page": args.tokens_per_page,
        "scorer_diagnostic": bool(args.scorer_diagnostic),
        "recall_analysis": bool(args.recall_analysis),
        "quality_check": bool(args.quality_check),
        "default_mode_k": args.default_mode_k,
        "default_mode_v": args.default_mode_v,
        "key_policy_tier": args.key_policy_tier,
        "value_policy_tier": args.value_policy_tier,
        "key_mode_overrides": list(args.key_mode_override),
        "value_mode_overrides": list(args.value_mode_override),
        "key_layer_sensitivity": list(args.key_layer_sensitivity),
        "value_layer_sensitivity": list(args.value_layer_sensitivity),
        "key_policy_overrides": list(args.key_policy_override),
        "value_policy_overrides": list(args.value_policy_override),
        "layer_profile": args.layer_profile,
        "quant_scheme_k": args.quant_scheme_k,
        "quant_scheme_v": args.quant_scheme_v,
        "bits_k": args.bits_k,
        "bits_v": args.bits_v,
        "escape_dtype": args.escape_dtype,
        "recent_page_escape_dtype": args.recent_page_escape_dtype,
        "recent_window": args.recent_window,
        "execution_recent_window": args.execution_recent_window,
        "execution_sink_window": args.execution_sink_window,
        "execution_recent_window_overrides": list(args.execution_recent_window_layer),
        "execution_recent_window_context_overrides": list(args.execution_recent_window_context_layer),
        "execution_relevance_top_k": args.execution_relevance_top_k,
        "execution_relevance_top_k_overrides": list(args.execution_relevance_top_k_layer),
        "execution_relevance_top_k_context_overrides": list(args.execution_relevance_top_k_context_layer),
        "execution_full_context_layers": list(args.execution_full_context_layer),
        "execution_disable_grouped_batching_layers": list(args.execution_disable_grouped_batching_layer),
        "execution_recent_old_bonus_window": args.execution_recent_old_bonus_window,
        "execution_recent_old_bonus_strength": args.execution_recent_old_bonus_strength,
        "execution_recent_old_bonus_layers": list(args.execution_recent_old_bonus_layer),
        "execution_relevance_mode": args.execution_relevance_mode,
        "execution_secondary_relevance_mode": args.execution_secondary_relevance_mode,
        "execution_secondary_relevance_top_k": args.execution_secondary_relevance_top_k,
        "execution_secondary_relevance_min_overlap": args.execution_secondary_relevance_min_overlap,
        "execution_secondary_relevance_layers": list(args.execution_secondary_relevance_layer),
        "execution_recent_neighbor_rescue_top_k": args.execution_recent_neighbor_rescue_top_k,
        "execution_recent_neighbor_rescue_anchor_window": args.execution_recent_neighbor_rescue_anchor_window,
        "execution_recent_neighbor_rescue_min_anchor_pages": args.execution_recent_neighbor_rescue_min_anchor_pages,
        "execution_recent_neighbor_rescue_layers": list(args.execution_recent_neighbor_rescue_layer),
        "execution_exact_promote_top_k": args.execution_exact_promote_top_k,
        "execution_exact_promote_min_margin_threshold": args.execution_exact_promote_min_margin_threshold,
        "execution_exact_promote_max_context": args.execution_exact_promote_max_context,
        "execution_exact_promote_margin_threshold": args.execution_exact_promote_margin_threshold,
        "execution_exact_promote_layers": list(args.execution_exact_promote_layer),
        "execution_exact_promote_union_rescue_top_k": args.execution_exact_promote_union_rescue_top_k,
        "execution_grouped_decode_compact": args.execution_grouped_decode_compact,
        "execution_grouped_mix_compact": args.execution_grouped_mix_compact,
        "execution_grouped_mix_disable_packed_cuda": args.execution_grouped_mix_disable_packed_cuda,
        "execution_freeze_chunk_budget_during_decode": args.execution_freeze_chunk_budget_during_decode,
        "execution_builtin_selector_cache": args.execution_builtin_selector_cache,
        "execution_builtin_selector_score_all_pages": args.execution_builtin_selector_score_all_pages,
        "execution_builtin_selector_candidate_only": args.execution_builtin_selector_candidate_only,
        "execution_builtin_selector_score_all_pages_min_candidate_fraction": (
            args.execution_builtin_selector_score_all_pages_min_candidate_fraction
        ),
        "execution_value_escape_layers": list(args.execution_value_escape_layer),
        "execution_value_escape_mode": args.execution_value_escape_mode,
        "execution_value_escape_old_only": bool(args.execution_value_escape_old_only),
        "execution_value_escape_top_k": int(args.execution_value_escape_top_k),
        "execution_value_escape_prewarm": bool(args.execution_value_escape_prewarm),
        "execution_value_escape_prewarm_min_context": int(args.execution_value_escape_prewarm_min_context),
        "execution_exact_refine_top_k": args.execution_exact_refine_top_k,
        "execution_exact_refine_layers": list(args.execution_exact_refine_layer),
        "m2_sketch_dim_k": args.m2_sketch_dim_k,
        "m2_center_k": args.m2_center_k,
        "m2_segment_count_k": args.m2_segment_count_k,
        "m2_adaptive_segments_k": args.m2_adaptive_segments_k,
        "m2_adaptive_min_improvement_k": args.m2_adaptive_min_improvement_k,
        "m2_prefilter_top_k": args.m2_prefilter_top_k,
        "m2_prefilter_min_pages": args.m2_prefilter_min_pages,
        "prefer_m4_project_k": args.prefer_m4_project_k,
        "lut_refine_steps": args.lut_refine_steps,
        "preconditioner": args.preconditioner,
        "precondition_strength": args.precondition_strength,
        "m1_segment_count_k": args.m1_segment_count_k,
        "m1_segment_count_v": args.m1_segment_count_v,
        "m1_fallback_to_m0": bool(args.m1_fallback_to_m0),
        "m1_error_threshold": args.m1_error_threshold,
        "m1_token_p95_error_threshold": args.m1_token_p95_error_threshold,
        "learned_page_selector_path": args.learned_page_selector_path,
        "learned_page_selector_prompt_family": args.learned_page_selector_prompt_family,
        "learned_page_selector_prompt_variant": args.learned_page_selector_prompt_variant,
        "learned_page_selector_scope": args.learned_page_selector_scope,
        "learned_page_selector_target_candidate": args.learned_page_selector_target_candidate,
        "learned_page_selector_logit_offset": args.learned_page_selector_logit_offset,
        "prepared_chunk_cache_budget_ratio": args.prepared_chunk_cache_budget_ratio,
        "prepared_chunk_cache_min_bytes": args.prepared_chunk_cache_min_bytes,
        "prepared_chunk_cache_max_bytes": args.prepared_chunk_cache_max_bytes,
        "prepared_chunk_cache_min_page_count": args.prepared_chunk_cache_min_page_count,
        "model_max_position_embeddings": max_position_embeddings,
        "text_only": True,
        "dotcache_ready": False,
        "hybrid_family": "qwen3_5",
        "profile_backend": bool(args.profile_backend),
        "trace_python_allocations": bool(args.trace_python_allocations),
        "blas_num_threads": int(args.blas_num_threads),
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
    }


def main() -> None:
    args = parse_args()
    if int(args.warmup_runs) < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    if int(args.measured_runs) <= 0:
        raise SystemExit("--measured-runs must be positive")
    if int(args.blas_num_threads) > 0:
        thread_count = str(int(args.blas_num_threads))
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count

    import torch
    from transformers import AutoConfig

    from dotcache.backends import configure_prepared_chunk_cache
    from dotcache.config_io import load_layer_profile
    from dotcache.integrations.llama import resolve_hf_auth_kwargs
    from dotcache.integrations.qwen35 import Qwen35AttentionSubsetDotCacheHarness, transformers_available

    if not transformers_available():
        raise SystemExit("bench_qwen35_attention_subset_dotcache_serving.py requires the optional transformers dependencies")

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    head_dim = int(text_config.hidden_size) // int(text_config.num_attention_heads)
    _resolve_args_from_layer_profile(args)
    if args.prepared_chunk_cache_min_page_count is not None:
        configure_prepared_chunk_cache(min_page_count=args.prepared_chunk_cache_min_page_count, clear=False)

    harness = Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config=_build_dotcache_config(args, head_dim=head_dim),
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    common_record = _common_record(args, max_position_embeddings=max_position_embeddings)

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        _run_case_series(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={**common_record, "prompt_mode": "repeat_count", "repeat_count": repeat_count},
            continue_on_error=args.continue_on_error,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        _run_case_series(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={**common_record, "prompt_mode": "exact_length", "prompt_length": prompt_length},
            continue_on_error=args.continue_on_error,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )


if __name__ == "__main__":
    main()
