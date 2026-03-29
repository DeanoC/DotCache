from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.config_io import load_layer_profile
from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35AttentionSubsetDotCacheHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced DotCache loss harness for the Qwen3.5 full-attention subset.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
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
    parser.add_argument("--execution-relevance-mode", choices=["sketch", "envelope"], default="envelope")
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
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--profile-backend", action="store_true")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35AttentionSubsetDotCacheHarness,
    *,
    prompt_unit: str,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < sequence_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:sequence_length]

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


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
        execution_relevance_top_k=args.execution_relevance_top_k,
        execution_relevance_top_k_overrides=tuple(args.execution_relevance_top_k_layer),
        execution_relevance_top_k_context_overrides=tuple(args.execution_relevance_top_k_context_layer),
        execution_relevance_mode=args.execution_relevance_mode,
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


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_attention_subset_dotcache_loss.py requires the optional transformers dependencies")
    if args.prefix_length <= 0 or args.prefix_length >= args.sequence_length:
        raise SystemExit("prefix_length must be in [1, sequence_length)")
    if args.eval_steps <= 0 or args.prefix_length + args.eval_steps > args.sequence_length:
        raise SystemExit("prefix_length + eval_steps must be <= sequence_length")

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    head_dim = int(text_config.hidden_size) // int(text_config.num_attention_heads)
    _resolve_args_from_layer_profile(args)

    harness = Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config=_build_dotcache_config(args, head_dim=head_dim),
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        sequence_length=args.sequence_length,
    )
    result = harness.evaluate_attention_subset_dotcache_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
        profile_backend=args.profile_backend,
    )
    result.update(
        {
            "benchmark": "qwen35_attention_subset_dotcache_loss",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_unit": args.prompt_unit,
            "tokens_per_page": args.tokens_per_page,
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
            "execution_relevance_mode": args.execution_relevance_mode,
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
            "prompt_mode": "exact_length",
            "prompt_length": int(args.prefix_length),
            "text_only": True,
            "dotcache_ready": False,
            "hybrid_family": "qwen3_5",
            "status": "ok",
        }
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
