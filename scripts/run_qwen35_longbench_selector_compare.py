#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.bench_qwen35_attention_subset_dotcache_serving import _aggregate_record_values


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
        choices=["exact", "quality", "systems", "streaming_sink_recent"],
        default=["exact", "quality", "systems", "streaming_sink_recent"],
    )
    parser.add_argument(
        "--prompt-pack",
        default=str(REPO_ROOT / "configs" / "prompt_packs" / "qwen35_cuda_longbench_qa_pack_v1.json"),
    )
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
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"prompt pack {path} must be a non-empty JSON list")
    prompt_specs: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"prompt pack item #{index} is not an object")
        prompt_id = str(item.get("prompt_id") or f"prompt_{index}")
        dataset = str(item.get("dataset") or "").strip()
        if not dataset:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define dataset")
        row_index = int(item.get("row_index", -1))
        if row_index < 0:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define a non-negative row_index")
        prompt_specs.append({"prompt_id": prompt_id, "dataset": dataset, "row_index": row_index})
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
    raise ValueError(f"unsupported case: {case}")


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


def _run_single(
    args: argparse.Namespace,
    *,
    case: str,
    prompt_spec: dict[str, Any],
    max_prompt_tokens: int,
) -> dict[str, Any]:
    command = _benchmark_command(args, case=case, prompt_spec=prompt_spec, max_prompt_tokens=max_prompt_tokens)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout_text, stderr_text = process.communicate()
    elapsed = time.monotonic() - started_at

    payload: dict[str, Any] | None = None
    for raw_line in stdout_text.splitlines():
        stripped = raw_line.strip().replace("\x00", "")
        if not stripped.startswith("{"):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if (
            candidate.get("prompt_mode") == "longbench_qa"
            and candidate.get("longbench_dataset") == prompt_spec["dataset"]
            and int(candidate.get("longbench_row_index", -1)) == int(prompt_spec["row_index"])
        ):
            payload = candidate

    if payload is None:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_longbench_qa",
            "benchmark_task": "longbench_qa",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_mode": "longbench_qa",
            "longbench_dataset": prompt_spec["dataset"],
            "longbench_row_index": int(prompt_spec["row_index"]),
            "status": "error",
            "error_type": "TimeoutExpired" if timed_out else "NoLongBenchRow",
            "error_message": (
                f"timed out after {args.timeout_seconds}s"
                if timed_out
                else f"command exited {process.returncode} without LongBench row"
            ),
        }

    payload = dict(payload)
    payload.update(
        {
            "comparison_case": case,
            "evaluation_prompt_id": prompt_spec["prompt_id"],
            "comparison_max_prompt_tokens": int(max_prompt_tokens),
            "runner_timeout_seconds": int(args.timeout_seconds),
            "runner_wall_time_s": elapsed,
            "runner_command": command,
        }
    )
    if stderr_text.strip():
        payload["runner_stderr_tail"] = stderr_text.strip()[-4000:]
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
    prompt_specs = _load_prompt_specs(args.prompt_pack)
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    for max_prompt_tokens in args.max_prompt_tokens:
        for prompt_spec in prompt_specs:
            for case in args.comparison_cases:
                for warmup_index in range(max(0, int(args.warmup_runs))):
                    warmup = _run_single(
                        args,
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
                    record = _run_single(
                        args,
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
                    }
                )
                _append_record(output_path, aggregate)
                print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
