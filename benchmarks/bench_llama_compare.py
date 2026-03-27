from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense KV vs DotCache on one loaded Llama-family model.")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--default-mode-k", choices=["M0", "M1", "M2", "M3"], default="M0")
    parser.add_argument("--default-mode-v", choices=["M0", "M1", "M3"], default="M0")
    parser.add_argument("--quant-scheme-k", choices=["affine", "lut", "sketch"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "lut"], default="affine")
    parser.add_argument("--m2-sketch-dim-k", type=int, default=8)
    parser.add_argument("--m2-center-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-segment-count-k", type=int, default=1)
    parser.add_argument("--m2-adaptive-segments-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-adaptive-min-improvement-k", type=float, default=0.1)
    parser.add_argument("--m2-prefilter-top-k", type=int, default=0)
    parser.add_argument("--m2-prefilter-min-pages", type=int, default=8)
    parser.add_argument("--lut-refine-steps", type=int, default=0)
    parser.add_argument("--preconditioner", choices=["none", "tanh"], default="none")
    parser.add_argument("--precondition-strength", type=float, default=1.0)
    parser.add_argument("--m1-segment-count-k", type=int, default=1)
    parser.add_argument("--m1-segment-count-v", type=int, default=1)
    parser.add_argument("--m1-fallback-to-m0", action="store_true")
    parser.add_argument("--m1-error-threshold", type=float, default=0.2)
    parser.add_argument("--m1-token-p95-error-threshold", type=float, default=0.55)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[1, 32, 64])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[])
    parser.add_argument("--include-max-practical", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: LlamaDotCacheHarness,
    *,
    prompt_unit: str,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    harness: LlamaDotCacheHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    base_record: dict[str, object],
    continue_on_error: bool,
    profile: bool,
) -> None:
    try:
        record = harness.generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            profile=profile,
        )
    except Exception as exc:  # pragma: no cover - benchmark-only failure path
        if not continue_on_error:
            raise
        error_record = dict(base_record)
        error_record.update(
            {
                "benchmark": "llama_compare",
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "prompt_length": int(input_ids.shape[1]),
            }
        )
        print(json.dumps(error_record, sort_keys=True), flush=True)
        return

    record.update(base_record)
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_llama_compare.py requires the optional transformers dependencies")

    model_config = AutoConfig.from_pretrained(args.model_id)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    dotcache_config = DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k=args.default_mode_k,
        default_mode_v=args.default_mode_v,
        quant_scheme_k=args.quant_scheme_k,
        quant_scheme_v=args.quant_scheme_v,
        m2_sketch_dim_k=args.m2_sketch_dim_k,
        m2_center_k=args.m2_center_k,
        m2_segment_count_k=args.m2_segment_count_k,
        m2_adaptive_segments_k=args.m2_adaptive_segments_k,
        m2_adaptive_min_improvement_k=args.m2_adaptive_min_improvement_k,
        m2_prefilter_top_k=args.m2_prefilter_top_k,
        m2_prefilter_min_pages=args.m2_prefilter_min_pages,
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
    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    max_position_embeddings = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    target_prompt_lengths = list(args.target_prompt_lengths)
    if args.include_max_practical:
        if max_position_embeddings <= 0:
            raise ValueError("model config does not expose max_position_embeddings")
        target_prompt_lengths.append(max_position_embeddings - args.max_new_tokens)

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "llama_compare",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "tokens_per_page": args.tokens_per_page,
                "default_mode_k": args.default_mode_k,
                "default_mode_v": args.default_mode_v,
                "quant_scheme_k": args.quant_scheme_k,
                "quant_scheme_v": args.quant_scheme_v,
                "m2_sketch_dim_k": args.m2_sketch_dim_k,
                "m2_center_k": args.m2_center_k,
                "m2_segment_count_k": args.m2_segment_count_k,
                "m2_adaptive_segments_k": args.m2_adaptive_segments_k,
                "m2_adaptive_min_improvement_k": args.m2_adaptive_min_improvement_k,
                "m2_prefilter_top_k": args.m2_prefilter_top_k,
                "m2_prefilter_min_pages": args.m2_prefilter_min_pages,
                "lut_refine_steps": args.lut_refine_steps,
                "preconditioner": args.preconditioner,
                "precondition_strength": args.precondition_strength,
                "m1_segment_count_k": args.m1_segment_count_k,
                "m1_segment_count_v": args.m1_segment_count_v,
                "m1_fallback_to_m0": bool(args.m1_fallback_to_m0),
                "m1_error_threshold": args.m1_error_threshold,
                "m1_token_p95_error_threshold": args.m1_token_p95_error_threshold,
                "prompt_mode": "repeat_count",
                "repeat_count": repeat_count,
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
            },
            continue_on_error=args.continue_on_error,
            profile=args.profile,
        )

    for prompt_length in sorted(set(length for length in target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "llama_compare",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "tokens_per_page": args.tokens_per_page,
                "default_mode_k": args.default_mode_k,
                "default_mode_v": args.default_mode_v,
                "quant_scheme_k": args.quant_scheme_k,
                "quant_scheme_v": args.quant_scheme_v,
                "m2_sketch_dim_k": args.m2_sketch_dim_k,
                "m2_center_k": args.m2_center_k,
                "m2_segment_count_k": args.m2_segment_count_k,
                "m2_adaptive_segments_k": args.m2_adaptive_segments_k,
                "m2_adaptive_min_improvement_k": args.m2_adaptive_min_improvement_k,
                "m2_prefilter_top_k": args.m2_prefilter_top_k,
                "m2_prefilter_min_pages": args.m2_prefilter_min_pages,
                "lut_refine_steps": args.lut_refine_steps,
                "preconditioner": args.preconditioner,
                "precondition_strength": args.precondition_strength,
                "m1_segment_count_k": args.m1_segment_count_k,
                "m1_segment_count_v": args.m1_segment_count_v,
                "m1_fallback_to_m0": bool(args.m1_fallback_to_m0),
                "m1_error_threshold": args.m1_error_threshold,
                "m1_token_p95_error_threshold": args.m1_token_p95_error_threshold,
                "prompt_mode": "exact_length",
                "requested_prompt_length": prompt_length,
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
            },
            continue_on_error=args.continue_on_error,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
