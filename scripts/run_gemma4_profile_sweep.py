#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotcache.integrations import (
    Gemma4TextHarness,
    gemma4_sliding_attention_source_layers,
    gemma4_text_recommended_dotcache_config,
)
from dotcache.integrations.llama import resolve_hf_auth_kwargs

DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Gemma 4 DotCache profile and sliding-layer sensitivity sweep.")
    parser.add_argument("--model-id", default="google/gemma-4-E2B")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--max-new-tokens-list", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--profiles", nargs="+", default=["aggressive", "value_exact", "balanced", "exact"])
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tokens-per-page", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--extra-exact-key-layers", type=int, nargs="*", default=[])
    parser.add_argument("--extra-exact-value-layers", type=int, nargs="*", default=[])
    parser.add_argument("--scan-sliding-key-layers", action="store_true")
    parser.add_argument("--scan-profile", default="balanced")
    parser.add_argument("--scan-prompt-length", type=int, default=0)
    parser.add_argument("--scan-max-new-tokens", type=int, default=0)
    return parser.parse_args()


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _build_exact_length_inputs(
    harness: Gemma4TextHarness,
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
    harness: Gemma4TextHarness,
    *,
    model_id: str,
    profile: str,
    prompt_length: int,
    max_new_tokens: int,
    prompt_unit: str,
    bits_k: int,
    bits_v: int,
    group_size: int,
    tokens_per_page: int,
    extra_exact_key_layers: tuple[int, ...] = (),
    extra_exact_value_layers: tuple[int, ...] = (),
) -> dict[str, Any]:
    config = gemma4_text_recommended_dotcache_config(
        harness.model,
        bits_k=bits_k,
        bits_v=bits_v,
        group_size=group_size,
        tokens_per_page=tokens_per_page,
        profile=profile,
        extra_exact_key_layers=extra_exact_key_layers,
        extra_exact_value_layers=extra_exact_value_layers,
    )
    harness.adapter.reconfigure(config)
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=prompt_unit,
        prompt_length=prompt_length,
    )
    result = harness.generate_greedy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        profile=False,
    )
    return {
        "benchmark": "gemma4_text_dotcache_sweep",
        "model_id": model_id,
        "profile": profile,
        "prompt_mode": "exact_length",
        "prompt_length": int(prompt_length),
        "max_new_tokens": int(max_new_tokens),
        "prompt_unit": prompt_unit,
        "bits_k": int(bits_k),
        "bits_v": int(bits_v),
        "group_size": int(group_size),
        "tokens_per_page": int(tokens_per_page),
        "extra_exact_key_layers": list(extra_exact_key_layers),
        "extra_exact_value_layers": list(extra_exact_value_layers),
        "greedy_token_agreement_rate": float(result["greedy_token_agreement_rate"]),
        "teacher_forced_logit_max_abs_error": float(result["teacher_forced_logit_max_abs_error"]),
        "teacher_forced_logit_max_rel_error": float(result["teacher_forced_logit_max_rel_error"]),
        "resident_bytes": int(result["resident_bytes"]),
        "kv_resident_bytes": int(result["kv_resident_bytes"]),
        "decode_ms_per_step": float(result["decode_ms_per_step"]),
        "m0_pages": int(result.get("m0_pages", 0)),
        "m3_pages": int(result.get("m3_pages", 0)),
        "dense_generated_ids": list(result["dense_generated_ids"]),
        "dotcache_generated_ids": list(result["dotcache_generated_ids"]),
        "dense_text": result.get("dense_text"),
        "dotcache_text": result.get("dotcache_text"),
    }


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "gemma4_profile_sweep.jsonl"
    if results_path.exists() and not args.append:
        results_path.unlink()

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    seed_config = gemma4_text_recommended_dotcache_config(
        model_config,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        group_size=args.group_size,
        tokens_per_page=args.tokens_per_page,
        profile="balanced",
    )
    harness = Gemma4TextHarness.from_pretrained(
        args.model_id,
        seed_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    extra_exact_key_layers = tuple(sorted({int(layer_idx) for layer_idx in args.extra_exact_key_layers}))
    extra_exact_value_layers = tuple(sorted({int(layer_idx) for layer_idx in args.extra_exact_value_layers}))

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        for max_new_tokens in sorted(set(length for length in args.max_new_tokens_list if length > 0)):
            for profile in args.profiles:
                record = _run_case(
                    harness,
                    model_id=args.model_id,
                    profile=profile,
                    prompt_length=prompt_length,
                    max_new_tokens=max_new_tokens,
                    prompt_unit=args.prompt_unit,
                    bits_k=args.bits_k,
                    bits_v=args.bits_v,
                    group_size=args.group_size,
                    tokens_per_page=args.tokens_per_page,
                    extra_exact_key_layers=extra_exact_key_layers,
                    extra_exact_value_layers=extra_exact_value_layers,
                )
                _append_record(results_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
                _release_device_cache()

    if args.scan_sliding_key_layers:
        scan_prompt_length = int(args.scan_prompt_length or max(args.target_prompt_lengths))
        scan_max_new_tokens = int(args.scan_max_new_tokens or max(args.max_new_tokens_list))
        for layer_idx in gemma4_sliding_attention_source_layers(harness.model):
            record = _run_case(
                harness,
                model_id=args.model_id,
                profile=args.scan_profile,
                prompt_length=scan_prompt_length,
                max_new_tokens=scan_max_new_tokens,
                prompt_unit=args.prompt_unit,
                bits_k=args.bits_k,
                bits_v=args.bits_v,
                group_size=args.group_size,
                tokens_per_page=args.tokens_per_page,
                extra_exact_key_layers=(int(layer_idx),),
            )
            record["scan_mode"] = "single_sliding_key_layer"
            _append_record(results_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            _release_device_cache()


if __name__ == "__main__":
    main()
