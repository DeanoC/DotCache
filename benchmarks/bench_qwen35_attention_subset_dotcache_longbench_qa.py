from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import string
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35AttentionSubsetDotCacheHarness, transformers_available

from benchmarks.bench_qwen35_attention_subset_dotcache_needle import (
    _apply_missing_serving_defaults,
    _build_dotcache_config,
    _decode_generated_text,
    _resolve_args_from_layer_profile,
    _run_case,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONGBENCH_ZIP_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip?download=true"

SUPPORTED_DATASETS = {
    "hotpotqa": {
        "prompt": (
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "The following are given passages.\n{context}\n\n"
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 32,
    },
    "2wikimqa": {
        "prompt": (
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "The following are given passages.\n{context}\n\n"
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 32,
    },
    "multifieldqa_en": {
        "prompt": (
            "Read the following text and answer briefly.\n\n"
            "{context}\n\n"
            "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 64,
    },
    "qasper": {
        "prompt": (
            "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. "
            'If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
            "Do not provide any explanation.\n\n"
            "Article: {context}\n\n"
            ' Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
            "Question: {input}\n\n"
            "Answer:"
        ),
        "max_new_tokens": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongBench QA serving benchmark for the Qwen3.5 full-attention DotCache subset."
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
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--longbench-dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--longbench-row-index", type=int, required=True)
    parser.add_argument("--longbench-cache-dir", default=str(REPO_ROOT / "benchmarks" / "cache" / "longbench"))
    parser.add_argument("--longbench-zip-url", default=DEFAULT_LONGBENCH_ZIP_URL)
    parser.add_argument("--longbench-max-prompt-tokens", type=int, default=0)
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return f1_score(normalized_prediction.split(), normalized_ground_truth.split())


def score_longbench_answers(prediction: str, answers: list[str]) -> dict[str, object]:
    stripped = prediction.strip()
    exact = False
    best_f1 = 0.0
    best_answer = ""
    for answer in answers:
        score = qa_f1_score(stripped, answer)
        if score > best_f1:
            best_f1 = score
            best_answer = answer
        if normalize_answer(stripped) == normalize_answer(answer):
            exact = True
    return {
        "longbench_generated_text": stripped,
        "longbench_answer_exact_match": exact,
        "longbench_qa_f1_max": float(best_f1),
        "longbench_best_matching_answer": best_answer,
    }


def clean_longbench_generated_text(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*assistant\s*$", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*answer:\s*$", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _ensure_longbench_zip(cache_dir: Path, zip_url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "data.zip"
    if zip_path.exists():
        return zip_path
    tmp_path = cache_dir / "data.zip.tmp"
    with urllib.request.urlopen(zip_url, timeout=120) as response, tmp_path.open("wb") as out_handle:
        shutil.copyfileobj(response, out_handle)
    os.replace(tmp_path, zip_path)
    return zip_path


def _load_longbench_row(zip_path: Path, dataset: str, row_index: int) -> dict[str, Any]:
    member_name = f"data/{dataset}.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            for index, raw_line in enumerate(handle):
                if index == row_index:
                    return json.loads(raw_line)
    raise IndexError(f"row index {row_index} out of range for {dataset}")


def _truncate_prompt_middle(tokenizer, prompt_text: str, max_prompt_tokens: int) -> tuple[str, int, bool]:
    input_ids = tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
    original_len = len(input_ids)
    if max_prompt_tokens <= 0 or original_len <= max_prompt_tokens:
        return prompt_text, original_len, False
    half = max_prompt_tokens // 2
    trimmed_ids = input_ids[:half] + input_ids[-(max_prompt_tokens - half) :]
    truncated_prompt = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return truncated_prompt, original_len, True


def _encode_prompt(tokenizer, prompt_text: str, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    encoded = tokenizer(prompt_text, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask, int(input_ids.shape[-1])


def _build_base_record(
    args: argparse.Namespace,
    *,
    max_position_embeddings: int,
    row: dict[str, Any],
    prompt_length_tokens: int,
    prompt_length_tokens_original: int,
    prompt_was_truncated: bool,
) -> dict[str, object]:
    return {
        "benchmark": "qwen35_attention_subset_dotcache_longbench_qa",
        "benchmark_task": "longbench_qa",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "weight_quantization": args.weight_quantization,
        "layer_profile": args.layer_profile,
        "prompt_mode": "longbench_qa",
        "prompt_length": prompt_length_tokens,
        "profile_backend": bool(args.profile_backend),
        "quality_check": bool(args.quality_check),
        "text_only": True,
        "dotcache_ready": False,
        "hybrid_family": "qwen3_5",
        "model_max_position_embeddings": int(max_position_embeddings),
        "longbench_dataset": args.longbench_dataset,
        "longbench_row_index": int(args.longbench_row_index),
        "longbench_row_length": int(row["length"]),
        "longbench_row_language": row["language"],
        "longbench_answers": list(row["answers"]),
        "longbench_all_classes": list(row.get("all_classes") or []),
        "longbench_input": row["input"],
        "longbench_id": row["_id"],
        "longbench_prompt_token_length": int(prompt_length_tokens),
        "longbench_prompt_token_length_original": int(prompt_length_tokens_original),
        "longbench_prompt_was_truncated": bool(prompt_was_truncated),
        "max_new_tokens": int(args.max_new_tokens),
    }


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit(
            "bench_qwen35_attention_subset_dotcache_longbench_qa.py requires the optional transformers dependencies"
        )
    _apply_missing_serving_defaults(args)

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    head_dim = int(text_config.hidden_size) // int(text_config.num_attention_heads)
    _resolve_args_from_layer_profile(args)

    dataset_config = SUPPORTED_DATASETS[args.longbench_dataset]
    if args.max_new_tokens == 0:
        args.max_new_tokens = int(dataset_config["max_new_tokens"])

    harness = Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config=_build_dotcache_config(args, head_dim=head_dim),
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )

    zip_path = _ensure_longbench_zip(Path(args.longbench_cache_dir), args.longbench_zip_url)
    row = _load_longbench_row(zip_path, args.longbench_dataset, args.longbench_row_index)
    prompt_text = str(dataset_config["prompt"]).format(context=row["context"], input=row["input"])

    effective_max_prompt_tokens = int(args.longbench_max_prompt_tokens)
    if effective_max_prompt_tokens <= 0 and max_position_embeddings > 0:
        effective_max_prompt_tokens = max(max_position_embeddings - int(args.max_new_tokens), 1)
    prompt_text, original_prompt_tokens, prompt_was_truncated = _truncate_prompt_middle(
        harness.tokenizer,
        prompt_text,
        effective_max_prompt_tokens,
    )
    input_ids, attention_mask, prompt_length_tokens = _encode_prompt(
        harness.tokenizer,
        prompt_text,
        device=harness.adapter.device,
    )

    result = _run_case(
        harness,
        input_ids=input_ids,
        attention_mask=attention_mask,
        args=args,
    )

    generated_text = _decode_generated_text(harness.tokenizer, list(result.get("dotcache_generated_ids", [])))
    answer_score = score_longbench_answers(generated_text, list(row["answers"]))
    cleaned_generated_text = clean_longbench_generated_text(generated_text)
    cleaned_answer_score = score_longbench_answers(cleaned_generated_text, list(row["answers"]))
    record = _build_base_record(
        args,
        max_position_embeddings=max_position_embeddings,
        row=row,
        prompt_length_tokens=prompt_length_tokens,
        prompt_length_tokens_original=original_prompt_tokens,
        prompt_was_truncated=prompt_was_truncated,
    )
    record.update(result)
    record.update(answer_score)
    record.update(
        {
            "longbench_generated_text_cleaned": cleaned_generated_text,
            "longbench_chat_artifact_cleaned": bool(cleaned_generated_text != str(generated_text).strip()),
            "longbench_answer_exact_match_cleaned": cleaned_answer_score["longbench_answer_exact_match"],
            "longbench_qa_f1_max_cleaned": cleaned_answer_score["longbench_qa_f1_max"],
            "longbench_best_matching_answer_cleaned": cleaned_answer_score["longbench_best_matching_answer"],
        }
    )
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
