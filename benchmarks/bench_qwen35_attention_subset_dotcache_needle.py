from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.config_io import load_layer_profile
from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35AttentionSubsetDotCacheHarness, transformers_available


DEFAULT_HAYSTACK_UNIT = (
    "Background note about schedules, archives, and logistics. This sentence is filler and not the answer.\n"
)


@dataclass(frozen=True)
class NeedlePromptBuild:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    needle_text: str
    question_text: str
    answer_text: str
    needle_token_start: int
    needle_token_end: int
    question_token_start: int
    filler_before_tokens: int
    filler_after_tokens: int
    needle_position_fraction_actual: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-Haystack serving benchmark for the Qwen3.5 full-attention DotCache subset."
    )
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
    parser.add_argument("--execution-relevance-top-k", type=int, default=0)
    parser.add_argument("--execution-relevance-top-k-layer", action="append", default=[])
    parser.add_argument("--execution-relevance-top-k-context-layer", action="append", default=[])
    parser.add_argument("--execution-full-context-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-disable-grouped-batching-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-relevance-mode", choices=["sketch", "envelope"], default="envelope")
    parser.add_argument("--execution-builtin-selector-cache", action="store_true")
    parser.add_argument("--execution-builtin-selector-score-all-pages", action="store_true")
    parser.add_argument("--execution-builtin-selector-candidate-only", action="store_true")
    parser.add_argument(
        "--execution-builtin-selector-score-all-pages-min-candidate-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument("--execution-value-escape-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-value-escape-mode", choices=["M0", "M1", "M3", "T3"], default="M3")
    parser.add_argument("--execution-value-escape-old-only", action="store_true")
    parser.add_argument("--execution-value-escape-top-k", type=int, default=0)
    parser.add_argument("--execution-value-escape-prewarm", action="store_true")
    parser.add_argument("--execution-value-escape-prewarm-min-context", type=int, default=0)
    parser.add_argument("--execution-exact-promote-top-k", type=int, default=0)
    parser.add_argument("--execution-exact-promote-min-margin-threshold", type=float, default=0.0)
    parser.add_argument("--execution-exact-promote-max-context", type=int, default=0)
    parser.add_argument("--execution-exact-promote-margin-threshold", type=float, default=0.0)
    parser.add_argument("--execution-exact-promote-layer", type=int, action="append", default=[])
    parser.add_argument("--execution-exact-promote-union-rescue-top-k", type=int, default=0)
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
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--needle-key", default="hidden passphrase")
    parser.add_argument("--needle-value", default="crimson-velvet-472")
    parser.add_argument("--needle-position-fraction", type=float, default=0.5)
    parser.add_argument("--haystack-unit", default=DEFAULT_HAYSTACK_UNIT)
    parser.add_argument(
        "--needle-template",
        default="Important detail: the {needle_key} is {needle_value}. Remember it exactly.\n",
    )
    parser.add_argument(
        "--question-template",
        default="Question: What is the {needle_key}? Answer with the exact value only.\nAnswer:",
    )
    return parser.parse_args()


def _apply_missing_serving_defaults(args: argparse.Namespace) -> None:
    defaults: dict[str, object] = {
        "execution_recent_window_layer": [],
        "execution_recent_window_context_layer": [],
        "execution_recent_old_bonus_window": 0,
        "execution_recent_old_bonus_strength": 0.0,
        "execution_recent_old_bonus_layer": [],
        "execution_secondary_relevance_mode": "",
        "execution_secondary_relevance_top_k": 0,
        "execution_secondary_relevance_min_overlap": 0.0,
        "execution_secondary_relevance_layer": [],
        "execution_recent_neighbor_rescue_top_k": 0,
        "execution_recent_neighbor_rescue_anchor_window": 0,
        "execution_recent_neighbor_rescue_min_anchor_pages": 0,
        "execution_recent_neighbor_rescue_layer": [],
        "execution_grouped_decode_compact": False,
        "execution_grouped_mix_compact": False,
        "execution_grouped_mix_disable_packed_cuda": False,
        "execution_freeze_chunk_budget_during_decode": False,
        "scorer_diagnostic": False,
        "recall_analysis": False,
        "blas_num_threads": 0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _resolve_args_from_layer_profile(args: argparse.Namespace) -> None:
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
    if args.execution_exact_promote_min_margin_threshold == 0.0:
        args.execution_exact_promote_min_margin_threshold = float(
            profile.get(
                "execution_exact_promote_min_margin_threshold",
                args.execution_exact_promote_min_margin_threshold,
            )
        )
    if args.execution_exact_promote_max_context == 0:
        args.execution_exact_promote_max_context = int(
            profile.get("execution_exact_promote_max_context", args.execution_exact_promote_max_context)
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
    if args.execution_exact_promote_union_rescue_top_k == 0:
        args.execution_exact_promote_union_rescue_top_k = int(
            profile.get(
                "execution_exact_promote_union_rescue_top_k",
                args.execution_exact_promote_union_rescue_top_k,
            )
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
    return DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k=args.default_mode_k,
        default_mode_v=args.default_mode_v,
        key_policy_tier=args.key_policy_tier,
        value_policy_tier=args.value_policy_tier,
        key_mode_overrides=tuple(args.key_mode_override),
        value_mode_overrides=tuple(args.value_mode_override),
        key_layer_sensitivity=tuple(args.key_layer_sensitivity),
        value_layer_sensitivity=tuple(args.value_layer_sensitivity),
        key_policy_overrides=tuple(args.key_policy_override),
        value_policy_overrides=tuple(args.value_policy_override),
        quant_scheme_k=args.quant_scheme_k,
        quant_scheme_v=args.quant_scheme_v,
        escape_dtype=args.escape_dtype,
        recent_page_escape_dtype=args.recent_page_escape_dtype,
        recent_window=args.recent_window,
        execution_recent_window=args.execution_recent_window,
        execution_sink_window=args.execution_sink_window,
        execution_recent_window_overrides=tuple(args.execution_recent_window_layer),
        execution_recent_window_context_overrides=tuple(args.execution_recent_window_context_layer),
        execution_relevance_top_k=args.execution_relevance_top_k,
        execution_relevance_top_k_overrides=tuple(args.execution_relevance_top_k_layer),
        execution_relevance_top_k_context_overrides=tuple(args.execution_relevance_top_k_context_layer),
        execution_full_context_layers=tuple(args.execution_full_context_layer),
        execution_disable_grouped_batching_layers=tuple(args.execution_disable_grouped_batching_layer),
        execution_recent_old_bonus_window=args.execution_recent_old_bonus_window,
        execution_recent_old_bonus_strength=args.execution_recent_old_bonus_strength,
        execution_recent_old_bonus_layers=tuple(args.execution_recent_old_bonus_layer),
        execution_relevance_mode=args.execution_relevance_mode,
        execution_secondary_relevance_mode=args.execution_secondary_relevance_mode,
        execution_secondary_relevance_top_k=args.execution_secondary_relevance_top_k,
        execution_secondary_relevance_min_overlap=args.execution_secondary_relevance_min_overlap,
        execution_secondary_relevance_layers=tuple(args.execution_secondary_relevance_layer),
        execution_recent_neighbor_rescue_top_k=args.execution_recent_neighbor_rescue_top_k,
        execution_recent_neighbor_rescue_anchor_window=args.execution_recent_neighbor_rescue_anchor_window,
        execution_recent_neighbor_rescue_min_anchor_pages=args.execution_recent_neighbor_rescue_min_anchor_pages,
        execution_recent_neighbor_rescue_layers=tuple(args.execution_recent_neighbor_rescue_layer),
        execution_exact_promote_top_k=args.execution_exact_promote_top_k,
        execution_exact_promote_min_margin_threshold=args.execution_exact_promote_min_margin_threshold,
        execution_exact_promote_max_context=args.execution_exact_promote_max_context,
        execution_exact_promote_margin_threshold=args.execution_exact_promote_margin_threshold,
        execution_exact_promote_layers=tuple(args.execution_exact_promote_layer),
        execution_exact_promote_union_rescue_top_k=args.execution_exact_promote_union_rescue_top_k,
        execution_grouped_decode_compact=args.execution_grouped_decode_compact,
        execution_grouped_mix_compact=args.execution_grouped_mix_compact,
        execution_grouped_mix_disable_packed_cuda=args.execution_grouped_mix_disable_packed_cuda,
        execution_freeze_chunk_budget_during_decode=args.execution_freeze_chunk_budget_during_decode,
        execution_builtin_selector_cache=args.execution_builtin_selector_cache,
        execution_builtin_selector_score_all_pages=args.execution_builtin_selector_score_all_pages,
        execution_builtin_selector_candidate_only=args.execution_builtin_selector_candidate_only,
        execution_builtin_selector_score_all_pages_min_candidate_fraction=(
            args.execution_builtin_selector_score_all_pages_min_candidate_fraction
        ),
        execution_value_escape_layers=tuple(args.execution_value_escape_layer),
        execution_value_escape_mode=args.execution_value_escape_mode,
        execution_value_escape_old_only=args.execution_value_escape_old_only,
        execution_value_escape_top_k=args.execution_value_escape_top_k,
        execution_value_escape_prewarm=args.execution_value_escape_prewarm,
        execution_value_escape_prewarm_min_context=args.execution_value_escape_prewarm_min_context,
        execution_exact_refine_top_k=args.execution_exact_refine_top_k,
        execution_exact_refine_layers=tuple(args.execution_exact_refine_layer),
        m2_sketch_dim_k=args.m2_sketch_dim_k,
        m2_center_k=args.m2_center_k,
        m2_segment_count_k=args.m2_segment_count_k,
        m2_adaptive_segments_k=args.m2_adaptive_segments_k,
        m2_adaptive_min_improvement_k=args.m2_adaptive_min_improvement_k,
        m2_prefilter_top_k=args.m2_prefilter_top_k,
        m2_prefilter_min_pages=args.m2_prefilter_min_pages,
        prefer_m4_project_k=args.prefer_m4_project_k,
        lut_refine_steps=args.lut_refine_steps,
        preconditioner=args.preconditioner,
        precondition_strength=args.precondition_strength,
        m1_segment_count_k=args.m1_segment_count_k,
        m1_segment_count_v=args.m1_segment_count_v,
        m1_fallback_to_m0=args.m1_fallback_to_m0,
        m1_error_threshold=args.m1_error_threshold,
        m1_token_p95_error_threshold=args.m1_token_p95_error_threshold,
        tokens_per_page=args.tokens_per_page,
    )


def _encode_text(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"text encoded to an empty token sequence: {text!r}")
    return [int(token_id) for token_id in token_ids]


def _repeat_trim_ids(unit_ids: list[int], target_length: int) -> list[int]:
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if target_length == 0:
        return []
    if not unit_ids:
        raise ValueError("unit_ids must be non-empty when target_length is positive")
    token_ids: list[int] = []
    while len(token_ids) < target_length:
        token_ids.extend(unit_ids)
    return token_ids[:target_length]


def build_needle_prompt_inputs(
    tokenizer,
    *,
    device,
    prompt_length: int,
    needle_position_fraction: float,
    haystack_unit: str,
    needle_key: str,
    needle_value: str,
    needle_template: str,
    question_template: str,
) -> NeedlePromptBuild:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if not 0.0 <= float(needle_position_fraction) <= 1.0:
        raise ValueError("needle_position_fraction must be in [0, 1]")

    bos_ids = [int(tokenizer.bos_token_id)] if getattr(tokenizer, "bos_token_id", None) is not None else []
    haystack_ids = _encode_text(tokenizer, haystack_unit)
    needle_text = needle_template.format(needle_key=needle_key, needle_value=needle_value)
    question_text = question_template.format(needle_key=needle_key)
    needle_ids = _encode_text(tokenizer, needle_text)
    question_ids = _encode_text(tokenizer, question_text)

    reserved = len(bos_ids) + len(needle_ids) + len(question_ids)
    if reserved >= prompt_length:
        raise ValueError(
            f"prompt_length={prompt_length} is too small for bos+needle+question payload of {reserved} tokens"
        )

    filler_budget = prompt_length - reserved
    filler_before_tokens = int(round(filler_budget * float(needle_position_fraction)))
    filler_before_tokens = max(0, min(filler_budget, filler_before_tokens))
    filler_after_tokens = filler_budget - filler_before_tokens

    token_ids = (
        bos_ids
        + _repeat_trim_ids(haystack_ids, filler_before_tokens)
        + needle_ids
        + _repeat_trim_ids(haystack_ids, filler_after_tokens)
        + question_ids
    )
    if len(token_ids) != prompt_length:
        raise AssertionError(f"constructed prompt length {len(token_ids)} did not match target {prompt_length}")

    needle_token_start = len(bos_ids) + filler_before_tokens
    needle_token_end = needle_token_start + len(needle_ids)
    question_token_start = prompt_length - len(question_ids)
    actual_position = float(needle_token_start / max(prompt_length - len(question_ids), 1))

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return NeedlePromptBuild(
        input_ids=input_ids,
        attention_mask=attention_mask,
        needle_text=needle_text,
        question_text=question_text,
        answer_text=needle_value,
        needle_token_start=needle_token_start,
        needle_token_end=needle_token_end,
        question_token_start=question_token_start,
        filler_before_tokens=filler_before_tokens,
        filler_after_tokens=filler_after_tokens,
        needle_position_fraction_actual=actual_position,
    )


def normalize_needle_text(text: str) -> str:
    compact = " ".join(text.strip().lower().split())
    return compact.strip(" \t\r\n'\"`.,;:!?()[]{}")


def score_needle_answer(generated_text: str, expected_answer: str) -> dict[str, object]:
    stripped = generated_text.strip()
    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    normalized_expected = normalize_needle_text(expected_answer)
    normalized_first_line = normalize_needle_text(first_line)
    normalized_full_text = normalize_needle_text(stripped)
    exact_match = bool(normalized_expected) and normalized_first_line == normalized_expected
    prefix_match = bool(normalized_expected) and normalized_first_line.startswith(normalized_expected)
    contains_match = bool(normalized_expected) and normalized_expected in normalized_full_text
    return {
        "needle_expected_answer": expected_answer,
        "needle_generated_text": stripped,
        "needle_generated_first_line": first_line,
        "needle_answer_exact_match": exact_match,
        "needle_answer_prefix_match": prefix_match,
        "needle_answer_contains_match": contains_match,
        "needle_answer_correct": prefix_match or contains_match,
    }


def _decode_generated_text(tokenizer, generated_ids: list[int]) -> str:
    if tokenizer is None or not generated_ids:
        return ""
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded if isinstance(decoded, str) else ""


def _build_base_record(args: argparse.Namespace, *, max_position_embeddings: int) -> dict[str, object]:
    return {
        "benchmark": "qwen35_attention_subset_dotcache_needle",
        "benchmark_task": "needle_in_a_haystack",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "weight_quantization": args.weight_quantization,
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
        "execution_relevance_top_k": args.execution_relevance_top_k,
        "execution_relevance_top_k_overrides": list(args.execution_relevance_top_k_layer),
        "execution_relevance_top_k_context_overrides": list(args.execution_relevance_top_k_context_layer),
        "execution_full_context_layers": list(args.execution_full_context_layer),
        "execution_disable_grouped_batching_layers": list(args.execution_disable_grouped_batching_layer),
        "execution_relevance_mode": args.execution_relevance_mode,
        "execution_builtin_selector_cache": bool(args.execution_builtin_selector_cache),
        "execution_builtin_selector_score_all_pages": bool(args.execution_builtin_selector_score_all_pages),
        "execution_builtin_selector_candidate_only": bool(args.execution_builtin_selector_candidate_only),
        "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
            args.execution_builtin_selector_score_all_pages_min_candidate_fraction
        ),
        "execution_value_escape_layers": list(args.execution_value_escape_layer),
        "execution_value_escape_mode": args.execution_value_escape_mode,
        "execution_value_escape_old_only": bool(args.execution_value_escape_old_only),
        "execution_value_escape_top_k": int(args.execution_value_escape_top_k),
        "execution_value_escape_prewarm": bool(args.execution_value_escape_prewarm),
        "execution_value_escape_prewarm_min_context": int(args.execution_value_escape_prewarm_min_context),
        "execution_exact_promote_top_k": int(args.execution_exact_promote_top_k),
        "execution_exact_promote_min_margin_threshold": float(args.execution_exact_promote_min_margin_threshold),
        "execution_exact_promote_max_context": int(args.execution_exact_promote_max_context),
        "execution_exact_promote_margin_threshold": float(args.execution_exact_promote_margin_threshold),
        "execution_exact_promote_layers": list(args.execution_exact_promote_layer),
        "execution_exact_promote_union_rescue_top_k": int(args.execution_exact_promote_union_rescue_top_k),
        "execution_exact_refine_top_k": int(args.execution_exact_refine_top_k),
        "execution_exact_refine_layers": list(args.execution_exact_refine_layer),
        "m2_sketch_dim_k": int(args.m2_sketch_dim_k),
        "m2_center_k": bool(args.m2_center_k),
        "m2_segment_count_k": int(args.m2_segment_count_k),
        "m2_adaptive_segments_k": bool(args.m2_adaptive_segments_k),
        "m2_adaptive_min_improvement_k": float(args.m2_adaptive_min_improvement_k),
        "m2_prefilter_top_k": int(args.m2_prefilter_top_k),
        "m2_prefilter_min_pages": int(args.m2_prefilter_min_pages),
        "prefer_m4_project_k": bool(args.prefer_m4_project_k),
        "lut_refine_steps": int(args.lut_refine_steps),
        "preconditioner": args.preconditioner,
        "precondition_strength": float(args.precondition_strength),
        "m1_segment_count_k": int(args.m1_segment_count_k),
        "m1_segment_count_v": int(args.m1_segment_count_v),
        "m1_fallback_to_m0": bool(args.m1_fallback_to_m0),
        "m1_error_threshold": float(args.m1_error_threshold),
        "m1_token_p95_error_threshold": float(args.m1_token_p95_error_threshold),
        "prompt_mode": "needle_in_a_haystack",
        "prompt_length": int(args.prompt_length),
        "max_new_tokens": int(args.max_new_tokens),
        "quality_check": bool(args.quality_check),
        "profile_backend": bool(args.profile_backend),
        "trace_python_allocations": bool(args.trace_python_allocations),
        "tokens_per_page": int(args.tokens_per_page),
        "text_only": True,
        "dotcache_ready": False,
        "hybrid_family": "qwen3_5",
        "model_max_position_embeddings": int(max_position_embeddings),
        "needle_key": args.needle_key,
        "needle_value": args.needle_value,
        "needle_position_fraction_requested": float(args.needle_position_fraction),
        "needle_haystack_unit": args.haystack_unit,
        "needle_template": args.needle_template,
        "needle_question_template": args.question_template,
    }


def _run_case(
    harness: Qwen35AttentionSubsetDotCacheHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.quality_check:
        return harness.run_attention_subset_dotcache_serving_quality(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=args.max_new_tokens,
            profile_backend=bool(args.profile_backend),
            trace_python_allocations=bool(args.trace_python_allocations),
        )
    return harness.run_attention_subset_dotcache_serving(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=args.max_new_tokens,
        profile_backend=bool(args.profile_backend),
    )


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_attention_subset_dotcache_needle.py requires the optional transformers dependencies")
    _apply_missing_serving_defaults(args)

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    head_dim = int(text_config.hidden_size) // int(text_config.num_attention_heads)
    _resolve_args_from_layer_profile(args)

    harness = Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config=_build_dotcache_config(args, head_dim=head_dim),
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )

    prompt_build = build_needle_prompt_inputs(
        harness.tokenizer,
        device=harness.adapter.device,
        prompt_length=args.prompt_length,
        needle_position_fraction=args.needle_position_fraction,
        haystack_unit=args.haystack_unit,
        needle_key=args.needle_key,
        needle_value=args.needle_value,
        needle_template=args.needle_template,
        question_template=args.question_template,
    )

    result = _run_case(
        harness,
        input_ids=prompt_build.input_ids,
        attention_mask=prompt_build.attention_mask,
        args=args,
    )

    generated_text = _decode_generated_text(harness.tokenizer, list(result.get("dotcache_generated_ids", [])))
    answer_score = score_needle_answer(generated_text, prompt_build.answer_text)
    record = _build_base_record(args, max_position_embeddings=max_position_embeddings)
    record.update(result)
    record.update(answer_score)
    record.update(
        {
            "needle_prompt_text": prompt_build.needle_text,
            "needle_question_text": prompt_build.question_text,
            "needle_token_start": int(prompt_build.needle_token_start),
            "needle_token_end": int(prompt_build.needle_token_end),
            "needle_question_token_start": int(prompt_build.question_token_start),
            "needle_filler_before_tokens": int(prompt_build.filler_before_tokens),
            "needle_filler_after_tokens": int(prompt_build.filler_after_tokens),
            "needle_position_fraction_actual": float(prompt_build.needle_position_fraction_actual),
        }
    )
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
