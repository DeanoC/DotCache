#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.bench_qwen35_attention_subset_dotcache_serving import _aggregate_record_values
from benchmarks.bench_qwen35_attention_subset_dotcache_longbench_qa import (
    DEFAULT_LONGBENCH_ZIP_URL,
    _ensure_longbench_zip,
    build_longbench_record,
    clean_longbench_generated_text,
    load_longbench_harness_from_args,
    parse_args as parse_benchmark_args,
    score_longbench_answers,
)
from dotcache.longbench_v1 import build_prompt_specs_from_zip

try:
    import torch
except ImportError:  # pragma: no cover - optional in import-only test environments
    torch = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTOR_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "qwen35_selector_qwen35_9b_suite_20260401"
    / "serving_selector_artifact"
    / "linear_selector_model.json"
)
SHARED_SELECTOR_ARTIFACT = Path(
    "/workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact Qwen LongBench selector-profile comparison suite.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--layer-profile", default=None)
    parser.add_argument("--selector-artifact", default=str(DEFAULT_SELECTOR_ARTIFACT))
    parser.add_argument(
        "--comparison-cases",
        nargs="+",
        choices=["dense", "exact", "quality", "systems", "streaming_sink_recent", "quest_like"],
        default=["dense", "exact", "quality", "systems", "streaming_sink_recent", "quest_like"],
    )
    parser.add_argument(
        "--prompt-pack",
        default=str(REPO_ROOT / "configs" / "prompt_packs" / "qwen35_cuda_longbench_qa_pack_v1.json"),
    )
    parser.add_argument(
        "--prompt-pack-preset",
        choices=["original_full_suite", "original_stratified_16_per_dataset", "original_stratified_32_per_dataset"],
        default=None,
    )
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--longbench-cache-dir", default=str(REPO_ROOT / "benchmarks" / "cache" / "longbench"))
    parser.add_argument("--longbench-zip-url", default=DEFAULT_LONGBENCH_ZIP_URL)
    parser.add_argument("--max-prompt-tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--quality-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _load_prompt_specs(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("prompt_specs")
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"prompt pack {path} must be a non-empty JSON list or manifest object")
    prompt_specs: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"prompt pack item #{index} is not an object")
        normalized = dict(item)
        prompt_id = str(normalized.get("prompt_id") or f"prompt_{index}")
        dataset = str(normalized.get("dataset") or "").strip()
        if not dataset:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define dataset")
        row_index = int(normalized.get("row_index", -1))
        if row_index < 0:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define a non-negative row_index")
        normalized["prompt_id"] = prompt_id
        normalized["dataset"] = dataset
        normalized["row_index"] = row_index
        prompt_specs.append(normalized)
    return prompt_specs


def _case_requires_selector_artifact(case: str) -> bool:
    return case in {"quality", "systems"}


def _resolve_selector_artifact(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if candidate == DEFAULT_SELECTOR_ARTIFACT and SHARED_SELECTOR_ARTIFACT.exists():
        return SHARED_SELECTOR_ARTIFACT.resolve()
    return candidate


def _case_extra_args(case: str, *, selector_artifact: str) -> list[str]:
    if case == "dense":
        return [
            "--learned-page-selector-profile",
            "quality",
        ]
    if case == "exact":
        return [
            "--learned-page-selector-profile",
            "quality",
        ]
    if case == "quality":
        return [
            "--learned-page-selector-path",
            selector_artifact,
            "--learned-page-selector-prompt-family",
            "cache",
            "--learned-page-selector-prompt-variant",
            "locality",
            "--learned-page-selector-profile",
            "quality",
        ]
    if case == "systems":
        return [
            "--learned-page-selector-path",
            selector_artifact,
            "--learned-page-selector-prompt-family",
            "cache",
            "--learned-page-selector-prompt-variant",
            "locality",
            "--learned-page-selector-profile",
            "systems",
        ]
    if case == "streaming_sink_recent":
        return [
            "--execution-recent-window",
            "1024",
            "--execution-sink-window",
            "256",
            "--learned-page-selector-profile",
            "quality",
        ]
    if case == "quest_like":
        return [
            "--execution-recent-window",
            "1024",
            "--execution-sink-window",
            "256",
            "--execution-relevance-top-k",
            "4",
            "--execution-relevance-mode",
            "envelope",
            "--learned-page-selector-profile",
            "quality",
        ]
    raise ValueError(f"unsupported case: {case}")


def _resolve_prompt_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt_pack_preset in {
        "original_full_suite",
        "original_stratified_16_per_dataset",
        "original_stratified_32_per_dataset",
    }:
        zip_path = _ensure_longbench_zip(Path(args.longbench_cache_dir), str(args.longbench_zip_url))
        if args.prompt_pack_preset == "original_full_suite":
            return build_prompt_specs_from_zip(zip_path)
        if args.prompt_pack_preset == "original_stratified_16_per_dataset":
            return build_prompt_specs_from_zip(zip_path, stratified_limit_per_dataset=16)
        return build_prompt_specs_from_zip(zip_path, stratified_limit_per_dataset=32)
    return _load_prompt_specs(args.prompt_pack)


def _apply_prompt_shard(
    prompt_specs: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise SystemExit(f"prompt shard count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit(
            f"prompt shard index must be in [0, {shard_count}), got {shard_index}"
        )
    if shard_count == 1:
        return list(prompt_specs)
    return [spec for index, spec in enumerate(prompt_specs) if index % shard_count == shard_index]


def _apply_comparison_view(payload: dict[str, Any], *, case: str) -> dict[str, Any]:
    record = dict(payload)
    answers = list(record.get("longbench_answers") or [])
    if case == "dense":
        generated_text = str(record.get("dense_text", "")).strip()
        answer_score = score_longbench_answers(generated_text, answers)
        cleaned_generated_text = clean_longbench_generated_text(generated_text)
        cleaned_answer_score = score_longbench_answers(cleaned_generated_text, answers)
        record.update(
            {
                "comparison_generated_text": generated_text,
                "comparison_generated_text_cleaned": cleaned_generated_text,
                "comparison_answer_exact_match": answer_score["longbench_answer_exact_match"],
                "comparison_answer_exact_match_cleaned": cleaned_answer_score["longbench_answer_exact_match"],
                "comparison_qa_f1_max": answer_score["longbench_qa_f1_max"],
                "comparison_qa_f1_max_cleaned": cleaned_answer_score["longbench_qa_f1_max"],
                "comparison_best_matching_answer": answer_score["longbench_best_matching_answer"],
                "comparison_best_matching_answer_cleaned": cleaned_answer_score["longbench_best_matching_answer"],
                "comparison_generated_text_scored": answer_score["longbench_generated_text_scored"],
                "comparison_generated_text_scored_cleaned": cleaned_answer_score["longbench_generated_text_scored"],
                "comparison_chat_artifact_cleaned": bool(cleaned_generated_text != generated_text),
                "comparison_decode_ms_per_step": float(record.get("dense_decode_ms_per_step", 0.0) or 0.0),
                "comparison_decode_ms_per_step_p95": float(
                    record.get("dense_decode_ms_per_step_p95", record.get("dense_decode_ms_per_step", 0.0)) or 0.0
                ),
                "comparison_teacher_forced_perplexity_ratio": 1.0,
                "comparison_teacher_forced_logit_rmse": 0.0,
                "comparison_official_score": None,
            }
        )
        return record

    record.update(
        {
            "comparison_generated_text": str(record.get("longbench_generated_text", "")).strip(),
            "comparison_generated_text_cleaned": str(record.get("longbench_generated_text_cleaned", "")).strip(),
            "comparison_answer_exact_match": bool(record.get("longbench_answer_exact_match")),
            "comparison_answer_exact_match_cleaned": bool(record.get("longbench_answer_exact_match_cleaned")),
            "comparison_qa_f1_max": float(record.get("longbench_qa_f1_max", 0.0) or 0.0),
            "comparison_qa_f1_max_cleaned": float(record.get("longbench_qa_f1_max_cleaned", 0.0) or 0.0),
            "comparison_best_matching_answer": str(record.get("longbench_best_matching_answer", "")),
            "comparison_best_matching_answer_cleaned": str(record.get("longbench_best_matching_answer_cleaned", "")),
            "comparison_generated_text_scored": str(record.get("longbench_generated_text_scored", "")),
            "comparison_generated_text_scored_cleaned": str(record.get("longbench_generated_text_scored_cleaned", "")),
            "comparison_chat_artifact_cleaned": bool(record.get("longbench_chat_artifact_cleaned")),
            "comparison_decode_ms_per_step": float(record.get("dotcache_decode_ms_per_step", 0.0) or 0.0),
            "comparison_decode_ms_per_step_p95": float(
                record.get("dotcache_decode_ms_per_step_p95", record.get("dotcache_decode_ms_per_step", 0.0)) or 0.0
            ),
            "comparison_teacher_forced_perplexity_ratio": record.get("teacher_forced_perplexity_ratio"),
            "comparison_teacher_forced_logit_rmse": record.get("teacher_forced_logit_rmse"),
            "comparison_official_score": record.get("longbench_official_score"),
        }
    )
    return record


def _benchmark_command(
    args: argparse.Namespace,
    *,
    case: str,
    prompt_spec: dict[str, Any],
    max_prompt_tokens: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_longbench_qa.py"),
        "--model-id",
        args.model_id,
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--weight-quantization",
        args.weight_quantization,
        "--longbench-dataset",
        prompt_spec["dataset"],
        "--longbench-row-index",
        str(prompt_spec["row_index"]),
        "--longbench-max-prompt-tokens",
        str(max_prompt_tokens),
    ]
    if args.layer_profile:
        command.extend(["--layer-profile", args.layer_profile])
    if args.profile_backend:
        command.append("--profile-backend")
    if args.trace_python_allocations:
        command.append("--trace-python-allocations")
    if args.quality_check:
        command.append("--quality-check")
    command.extend(_case_extra_args(case, selector_artifact=str(args.selector_artifact)))
    return command


def _benchmark_namespace(
    args: argparse.Namespace,
    *,
    case: str,
    prompt_spec: dict[str, Any],
    max_prompt_tokens: int,
) -> argparse.Namespace:
    command = _benchmark_command(args, case=case, prompt_spec=prompt_spec, max_prompt_tokens=max_prompt_tokens)
    return parse_benchmark_args(command[2:])


def _empty_torch_cache() -> None:
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def _run_single_in_process(
    base_args: argparse.Namespace,
    *,
    benchmark_args: argparse.Namespace,
    harness: Any,
    max_position_embeddings: int,
    zip_path: Path,
    case: str,
    prompt_spec: dict[str, Any],
    max_prompt_tokens: int,
) -> dict[str, Any]:
    command = _benchmark_command(base_args, case=case, prompt_spec=prompt_spec, max_prompt_tokens=max_prompt_tokens)
    benchmark_args.longbench_dataset = str(prompt_spec["dataset"])
    benchmark_args.longbench_row_index = int(prompt_spec["row_index"])
    benchmark_args.longbench_max_prompt_tokens = int(max_prompt_tokens)

    started_at = time.monotonic()
    try:
        payload = build_longbench_record(
            benchmark_args,
            harness,
            max_position_embeddings=max_position_embeddings,
            zip_path=zip_path,
            dataset=str(prompt_spec["dataset"]),
            row_index=int(prompt_spec["row_index"]),
            max_prompt_tokens=int(max_prompt_tokens),
        )
    except Exception as exc:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_longbench_qa",
            "benchmark_task": "longbench_qa",
            "model_id": base_args.model_id,
            "backend": base_args.backend,
            "device": base_args.device,
            "torch_dtype": base_args.torch_dtype,
            "prompt_mode": "longbench_qa",
            "longbench_dataset": prompt_spec["dataset"],
            "longbench_row_index": int(prompt_spec["row_index"]),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runner_exception_traceback": traceback.format_exc()[-4000:],
        }
    finally:
        gc.collect()
        _empty_torch_cache()

    elapsed = time.monotonic() - started_at
    payload = _apply_comparison_view(payload, case=case)
    payload.update(
        {
            "comparison_case": case,
            "evaluation_prompt_id": prompt_spec["prompt_id"],
            "comparison_max_prompt_tokens": int(max_prompt_tokens),
            "prompt_shard_count": int(base_args.prompt_shard_count),
            "prompt_shard_index": int(base_args.prompt_shard_index),
            "runner_timeout_seconds": int(base_args.timeout_seconds),
            "runner_wall_time_s": elapsed,
            "runner_command": command,
            "runner_execution_mode": "in_process_session",
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    args.selector_artifact = str(_resolve_selector_artifact(str(args.selector_artifact)))
    if any(_case_requires_selector_artifact(case) for case in args.comparison_cases):
        selector_artifact = Path(args.selector_artifact)
        if not selector_artifact.is_file():
            raise SystemExit(
                "selector artifact required for quality/systems but not found: "
                f"{selector_artifact}"
            )
    prompt_specs = _apply_prompt_shard(
        _resolve_prompt_specs(args),
        shard_count=int(args.prompt_shard_count),
        shard_index=int(args.prompt_shard_index),
    )
    if not prompt_specs:
        raise SystemExit(
            "resolved prompt shard is empty for "
            f"shard {args.prompt_shard_index} of {args.prompt_shard_count}"
        )
    zip_path = _ensure_longbench_zip(Path(args.longbench_cache_dir), str(args.longbench_zip_url))
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    session_prompt_spec = prompt_specs[0]
    for case in args.comparison_cases:
        benchmark_args = _benchmark_namespace(
            args,
            case=case,
            prompt_spec=session_prompt_spec,
            max_prompt_tokens=int(args.max_prompt_tokens[0]),
        )
        harness, max_position_embeddings = load_longbench_harness_from_args(benchmark_args)
        for max_prompt_tokens in args.max_prompt_tokens:
            for prompt_spec in prompt_specs:
                for warmup_index in range(max(0, int(args.warmup_runs))):
                    warmup = _run_single_in_process(
                        args,
                        benchmark_args=benchmark_args,
                        harness=harness,
                        max_position_embeddings=max_position_embeddings,
                        zip_path=zip_path,
                        case=case,
                        prompt_spec=prompt_spec,
                        max_prompt_tokens=int(max_prompt_tokens),
                    )
                    warmup.update(
                        {
                            "measurement_kind": "warmup",
                            "measurement_index": int(warmup_index),
                            "warmup_runs": int(args.warmup_runs),
                            "measured_runs": int(args.measured_runs),
                        }
                    )
                    _append_record(output_path, warmup)
                    print(json.dumps(warmup, sort_keys=True), flush=True)
                    if warmup.get("status") == "error":
                        raise SystemExit(
                            "benchmark warmup failed for "
                            f"{case} / {prompt_spec['prompt_id']} / {max_prompt_tokens}: "
                            f"{warmup.get('error_type', 'UnknownError')}: "
                            f"{warmup.get('error_message', 'no error message')}"
                        )

                measured_records: list[dict[str, Any]] = []
                for measurement_index in range(max(1, int(args.measured_runs))):
                    record = _run_single_in_process(
                        args,
                        benchmark_args=benchmark_args,
                        harness=harness,
                        max_position_embeddings=max_position_embeddings,
                        zip_path=zip_path,
                        case=case,
                        prompt_spec=prompt_spec,
                        max_prompt_tokens=int(max_prompt_tokens),
                    )
                    record.update(
                        {
                            "measurement_kind": "trial",
                            "measurement_index": int(measurement_index),
                            "warmup_runs": int(args.warmup_runs),
                            "measured_runs": int(args.measured_runs),
                        }
                    )
                    _append_record(output_path, record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if record.get("status") == "error":
                        raise SystemExit(
                            "benchmark trial failed for "
                            f"{case} / {prompt_spec['prompt_id']} / {max_prompt_tokens}: "
                            f"{record.get('error_type', 'UnknownError')}: "
                            f"{record.get('error_message', 'no error message')}"
                        )
                    measured_records.append(record)

                aggregate = _aggregate_record_values(measured_records)
                aggregate.update(
                    {
                        "measurement_kind": "aggregate",
                        "warmup_runs": int(args.warmup_runs),
                        "measured_runs": int(args.measured_runs),
                        "comparison_case": case,
                        "evaluation_prompt_id": prompt_spec["prompt_id"],
                        "comparison_max_prompt_tokens": int(max_prompt_tokens),
                        "prompt_shard_count": int(args.prompt_shard_count),
                        "prompt_shard_index": int(args.prompt_shard_index),
                    }
                )
                _append_record(output_path, aggregate)
                print(json.dumps(aggregate, sort_keys=True), flush=True)
        del harness
        gc.collect()
        _empty_torch_cache()


if __name__ == "__main__":
    main()
