#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_qwen35_attention_subset_dotcache_needle import (
    _apply_missing_serving_defaults,
    _decode_generated_text,
    _repeat_trim_ids,
    build_needle_prompt_inputs,
    score_needle_answer,
)
from benchmarks.bench_qwen35_attention_subset_dotcache_serving import (
    _aggregate_record_values,
    _build_dotcache_config,
    _resolve_args_from_layer_profile,
)
from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35AttentionSubsetDotCacheHarness, transformers_available
DEFAULT_SELECTOR_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "qwen35_selector_qwen35_9b_suite_20260401"
    / "serving_selector_artifact"
    / "linear_selector_model.json"
)
DEFAULT_REASONING_FILLER = (
    "Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters. "
)
DEFAULT_INSTRUCTION_FILLER = (
    "Operations guidance references staffing, inventory checks, maintenance windows, and shipping manifests in long planning documents. "
)


def _strip_chat_artifacts(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").strip()
    cleaned = re.sub(r"(?is)<\|im_end\|>", " ", cleaned)
    cleaned = re.sub(r"(?is)<\|endoftext\|>", " ", cleaned)
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        line = re.sub(r"(?i)(user|assistant)\s*$", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _strip_think_blocks(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*thinking process:\s*$", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact Qwen real-task selector comparison suite.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--layer-profile", default=None)
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
    parser.add_argument("--execution-relevance-mode", choices=["sketch", "envelope"], default="envelope")
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
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--selector-artifact", default=str(DEFAULT_SELECTOR_ARTIFACT))
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--profiles", nargs="+", choices=["exact", "quality", "systems"], default=["exact", "quality", "systems"])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--max-new-tokens-retrieval", type=int, default=12)
    parser.add_argument("--max-new-tokens-reasoning", type=int, default=12)
    parser.add_argument("--max-new-tokens-instruction", type=int, default=12)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _build_suffix_task_inputs(
    tokenizer,
    *,
    device: torch.device,
    prompt_length: int,
    filler_unit: str,
    suffix_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    bos_ids = [int(tokenizer.bos_token_id)] if getattr(tokenizer, "bos_token_id", None) is not None else []
    filler_ids = tokenizer(filler_unit, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    if not filler_ids:
        raise ValueError("filler_unit tokenized to an empty sequence")
    if not suffix_ids:
        raise ValueError("suffix_text tokenized to an empty sequence")
    reserved = len(bos_ids) + len(suffix_ids)
    if reserved >= prompt_length:
        raise ValueError(f"prompt_length={prompt_length} is too small for reserved suffix of {reserved} tokens")
    token_ids = bos_ids + _repeat_trim_ids([int(token_id) for token_id in filler_ids], prompt_length - reserved) + [
        int(token_id) for token_id in suffix_ids
    ]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _build_reasoning_inputs(tokenizer, *, device: torch.device, prompt_length: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    answer = "48"
    suffix = (
        "A clerk solves a budget worksheet step by step.\n"
        "Start with 17. Add 26. Subtract 9. Add 14.\n"
        "What is the final total?\n"
        "Answer with the exact integer only.\n"
        "Answer:"
    )
    input_ids, attention_mask = _build_suffix_task_inputs(
        tokenizer,
        device=device,
        prompt_length=prompt_length,
        filler_unit=DEFAULT_REASONING_FILLER,
        suffix_text=suffix,
    )
    return input_ids, attention_mask, answer


def _build_instruction_inputs(
    tokenizer,
    *,
    device: torch.device,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    answer = "STATUS: READY\nCOLOR: BLUE"
    suffix = (
        "Follow these instructions exactly.\n"
        "1. Reply with exactly two lines.\n"
        "2. First line must be: STATUS: READY\n"
        "3. Second line must be: COLOR: BLUE\n"
        "4. Do not add any other words, punctuation, or explanation.\n"
        "Response:"
    )
    input_ids, attention_mask = _build_suffix_task_inputs(
        tokenizer,
        device=device,
        prompt_length=prompt_length,
        filler_unit=DEFAULT_INSTRUCTION_FILLER,
        suffix_text=suffix,
    )
    return input_ids, attention_mask, answer


def _score_reasoning(generated_text: str, expected_answer: str) -> dict[str, object]:
    stripped = generated_text.strip()
    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    no_think = _strip_think_blocks(stripped)
    all_numbers = re.findall(r"-?\d+", no_think)
    predicted = all_numbers[-1] if all_numbers else ""
    match = re.search(r"-?\d+", first_line)
    predicted = match.group(0) if match else ""
    if all_numbers:
        predicted = all_numbers[-1]
    correct = predicted == expected_answer
    return {
        "task_expected_answer": expected_answer,
        "task_generated_text": stripped,
        "task_generated_first_line": first_line,
        "task_generated_text_cleaned": no_think,
        "task_generated_value": predicted,
        "task_success": bool(correct),
        "task_metric_name": "exact_integer_match",
        "task_metric_value": 1.0 if correct else 0.0,
    }


def _normalize_instruction_lines(text: str) -> list[str]:
    return [line.strip().upper() for line in text.splitlines() if line.strip()]


def _score_instruction(generated_text: str, expected_answer: str) -> dict[str, object]:
    stripped = generated_text.strip()
    cleaned = _strip_chat_artifacts(stripped)
    expected_lines = _normalize_instruction_lines(expected_answer)
    observed_lines = _normalize_instruction_lines(cleaned)
    correct = observed_lines == expected_lines
    return {
        "task_expected_answer": expected_answer,
        "task_generated_text": stripped,
        "task_generated_text_cleaned": cleaned,
        "task_generated_lines": observed_lines,
        "task_success": bool(correct),
        "task_metric_name": "exact_constraint_following",
        "task_metric_value": 1.0 if correct else 0.0,
    }


def _score_retrieval(generated_text: str, expected_answer: str) -> dict[str, object]:
    needle_score = score_needle_answer(generated_text, expected_answer)
    return {
        **needle_score,
        "task_success": bool(needle_score["needle_answer_correct"]),
        "task_metric_name": "answer_correct",
        "task_metric_value": 1.0 if needle_score["needle_answer_correct"] else 0.0,
        "task_expected_answer": expected_answer,
        "task_generated_text": generated_text.strip(),
    }


def _run_quality_case(
    harness: Qwen35AttentionSubsetDotCacheHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
) -> dict[str, Any]:
    return harness.run_attention_subset_dotcache_serving_quality(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
        profile_backend=False,
        trace_python_allocations=False,
    )


def _build_harness(args: argparse.Namespace, *, profile: str) -> Qwen35AttentionSubsetDotCacheHarness:
    config_args = argparse.Namespace(**vars(args))
    _apply_missing_serving_defaults(config_args)
    _resolve_args_from_layer_profile(config_args)
    if profile == "exact":
        config_args.learned_page_selector_path = None
        config_args.learned_page_selector_prompt_family = None
        config_args.learned_page_selector_prompt_variant = None
        config_args.learned_page_selector_profile = "quality"
        config_args.learned_page_selector_scope = "KV"
        config_args.learned_page_selector_target_candidate = "M3/affine/4/float16"
        config_args.learned_page_selector_logit_offset = 0.0
    else:
        config_args.learned_page_selector_path = args.selector_artifact
        config_args.learned_page_selector_prompt_family = {
            "quality": "cache",
            "systems": "cache",
        }[profile]
        config_args.learned_page_selector_prompt_variant = "locality"
        config_args.learned_page_selector_profile = profile
        config_args.learned_page_selector_scope = "KV"
        config_args.learned_page_selector_target_candidate = "M3/affine/4/float16"
        config_args.learned_page_selector_logit_offset = 0.0
    config_args.prepared_chunk_cache_budget_ratio = None
    config_args.prepared_chunk_cache_min_bytes = None
    config_args.prepared_chunk_cache_max_bytes = None
    auth_kwargs = resolve_hf_auth_kwargs()
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **auth_kwargs)
    text_config = getattr(model_config, "text_config", model_config)
    head_dim = int(getattr(text_config, "head_dim", int(text_config.hidden_size) // int(text_config.num_attention_heads)))
    dotcache_config = _build_dotcache_config(config_args, head_dim=head_dim)
    return Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )


def _task_specs(
    harness: Qwen35AttentionSubsetDotCacheHarness,
    *,
    prompt_length: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tokenizer = harness.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is unavailable")
    device = harness.adapter.device
    retrieval = build_needle_prompt_inputs(
        tokenizer,
        device=device,
        prompt_length=prompt_length,
        needle_position_fraction=0.5,
        haystack_unit=(
            "Background memo about permit backlogs, bridge closures, zoning appeals, and archive indexing. "
        ),
        needle_key="archive code",
        needle_value="RIVER-58142",
        needle_template="Important detail: the {needle_key} is {needle_value}. Remember it exactly.\n",
        question_template="Question: What is the {needle_key}? Answer with the exact value only.\nAnswer:",
    )
    reasoning_ids, reasoning_mask, reasoning_answer = _build_reasoning_inputs(
        tokenizer,
        device=device,
        prompt_length=prompt_length,
    )
    instruction_ids, instruction_mask, instruction_answer = _build_instruction_inputs(
        tokenizer,
        device=device,
        prompt_length=prompt_length,
    )
    return [
        {
            "task_name": "retrieval_passkey",
            "task_family": "retrieval",
            "task_variant": "passkey",
            "task_prompt_preview": "Question: What is the archive code? Answer with the exact value only.",
            "input_ids": retrieval.input_ids,
            "attention_mask": retrieval.attention_mask,
            "decode_steps": int(args.max_new_tokens_retrieval),
            "expected_answer": retrieval.answer_text,
            "score_fn": lambda text, expected=retrieval.answer_text: _score_retrieval(text, expected),
        },
        {
            "task_name": "reasoning_arithmetic",
            "task_family": "reasoning",
            "task_variant": "arithmetic",
            "task_prompt_preview": "Start with 17. Add 26. Subtract 9. Add 14. What is the final total?",
            "input_ids": reasoning_ids,
            "attention_mask": reasoning_mask,
            "decode_steps": int(args.max_new_tokens_reasoning),
            "expected_answer": reasoning_answer,
            "score_fn": lambda text, expected=reasoning_answer: _score_reasoning(text, expected),
        },
        {
            "task_name": "instruction_constraints",
            "task_family": "instruction",
            "task_variant": "constraints",
            "task_prompt_preview": "Reply with exactly two lines: STATUS: READY and COLOR: BLUE.",
            "input_ids": instruction_ids,
            "attention_mask": instruction_mask,
            "decode_steps": int(args.max_new_tokens_instruction),
            "expected_answer": instruction_answer,
            "score_fn": lambda text, expected=instruction_answer: _score_instruction(text, expected),
        },
    ]


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("run_qwen35_task_selector_compare.py requires the optional transformers dependencies")
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        output_path.unlink()

    for profile in args.profiles:
        harness = _build_harness(args, profile=profile)
        for prompt_length in args.prompt_lengths:
            task_specs = _task_specs(harness, prompt_length=prompt_length, args=args)
            for task_spec in task_specs:
                for warmup_index in range(max(0, int(args.warmup_runs))):
                    warmup_record = _run_quality_case(
                        harness,
                        input_ids=task_spec["input_ids"],
                        attention_mask=task_spec["attention_mask"],
                        decode_steps=int(task_spec["decode_steps"]),
                    )
                    warmup_record.update(
                        {
                            "benchmark": "qwen35_task_selector_compare",
                            "task_name": task_spec["task_name"],
                            "task_family": task_spec["task_family"],
                            "task_variant": task_spec["task_variant"],
                            "task_prompt_preview": task_spec["task_prompt_preview"],
                            "selector_profile": profile,
                            "measurement_kind": "warmup",
                            "measurement_index": int(warmup_index),
                            "warmup_runs": int(args.warmup_runs),
                            "measured_runs": int(args.measured_runs),
                        }
                    )
                    decoded = _decode_generated_text(harness.tokenizer, list(warmup_record.get("dotcache_generated_ids", [])))
                    warmup_record.update(task_spec["score_fn"](decoded))
                    if output_path is not None:
                        _append_record(output_path, warmup_record)
                    print(json.dumps(warmup_record, sort_keys=True), flush=True)

                measured_records: list[dict[str, Any]] = []
                for measurement_index in range(max(1, int(args.measured_runs))):
                    record = _run_quality_case(
                        harness,
                        input_ids=task_spec["input_ids"],
                        attention_mask=task_spec["attention_mask"],
                        decode_steps=int(task_spec["decode_steps"]),
                    )
                    decoded = _decode_generated_text(harness.tokenizer, list(record.get("dotcache_generated_ids", [])))
                    record.update(
                        {
                            "benchmark": "qwen35_task_selector_compare",
                            "task_name": task_spec["task_name"],
                            "task_family": task_spec["task_family"],
                            "task_variant": task_spec["task_variant"],
                            "task_prompt_preview": task_spec["task_prompt_preview"],
                            "selector_profile": profile,
                            "measurement_kind": "trial",
                            "measurement_index": int(measurement_index),
                            "warmup_runs": int(args.warmup_runs),
                            "measured_runs": int(args.measured_runs),
                            "prompt_length_requested": int(prompt_length),
                        }
                    )
                    record.update(task_spec["score_fn"](decoded))
                    if output_path is not None:
                        _append_record(output_path, record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    measured_records.append(record)

                aggregate_record = _aggregate_record_values(measured_records)
                aggregate_record.update(
                    {
                        "benchmark": "qwen35_task_selector_compare",
                        "task_name": task_spec["task_name"],
                        "task_family": task_spec["task_family"],
                        "task_variant": task_spec["task_variant"],
                        "task_prompt_preview": task_spec["task_prompt_preview"],
                        "selector_profile": profile,
                        "measurement_kind": "aggregate",
                        "measurement_index": None,
                        "warmup_runs": int(args.warmup_runs),
                        "measured_runs": int(args.measured_runs),
                        "prompt_length_requested": int(prompt_length),
                    }
                )
                if output_path is not None:
                    _append_record(output_path, aggregate_record)
                print(json.dumps(aggregate_record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
