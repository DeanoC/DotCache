#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_REPO_ROOT = Path("/workspace/DotCache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manifest-driven Qwen results matrix.")
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "configs" / "benchmark_matrices" / "qwen_results_matrix_v1.json"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        choices=["task_compare", "longbench", "backend_truth"],
        default=[],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _python_bin() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("matrix manifest must be a JSON object")
    return payload


def _resolve_selector_artifact(path_str: str) -> str:
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    repo_candidate = (REPO_ROOT / path_str).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    shared_candidate = (SHARED_REPO_ROOT / path_str).resolve()
    if shared_candidate.exists():
        return str(shared_candidate)
    return str(candidate)


def _selected_models(manifest: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    models = list(manifest.get("models", []))
    if not requested:
        return models
    requested_set = {value.strip() for value in requested if value.strip()}
    return [item for item in models if str(item.get("model_key", "")).strip() in requested_set]


def _selected_benchmarks(requested: list[str]) -> tuple[str, ...]:
    if not requested:
        return ("task_compare", "longbench", "backend_truth")
    return tuple(requested)


def _task_compare_command(
    defaults: dict[str, Any],
    model: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    run_jsonl = output_dir / f"{model['model_key']}_task_selector_compare.jsonl"
    report_md = output_dir / "task_selector_compare.md"
    report_json = output_dir / "task_selector_compare.json"
    prompt_lengths = model.get("task_compare", {}).get("prompt_lengths", defaults["task_prompt_lengths"])
    command = [
        _python_bin(),
        str(REPO_ROOT / "scripts" / "run_qwen35_task_selector_compare.py"),
        "--model-id",
        str(model["model_id"]),
        "--backend",
        str(defaults["backend"]),
        "--device",
        str(defaults["device"]),
        "--torch-dtype",
        str(defaults["torch_dtype"]),
        "--selector-artifact",
        _resolve_selector_artifact(str(model["selector_artifact"])),
        "--prompt-lengths",
        *[str(value) for value in prompt_lengths],
        "--warmup-runs",
        str(int(defaults["warmup_runs"])),
        "--measured-runs",
        str(int(defaults["measured_runs"])),
        "--output",
        str(run_jsonl),
    ]
    if model.get("layer_profile"):
        command.extend(["--layer-profile", str(model["layer_profile"])])
    report_command = [
        _python_bin(),
        str(REPO_ROOT / "scripts" / "report_qwen35_task_selector_compare.py"),
        "--input",
        str(run_jsonl),
        "--markdown-output",
        str(report_md),
        "--json-output",
        str(report_json),
    ]
    return command, report_command


def _longbench_command(
    defaults: dict[str, Any],
    model: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    run_jsonl = output_dir / f"{model['model_key']}_longbench_selector_compare.jsonl"
    report_md = output_dir / "longbench_selector_compare.md"
    report_json = output_dir / "longbench_selector_compare.json"
    max_prompt_tokens = model.get("longbench", {}).get("max_prompt_tokens", defaults["longbench_max_prompt_tokens"])
    command = [
        _python_bin(),
        str(REPO_ROOT / "scripts" / "run_qwen35_longbench_selector_compare.py"),
        "--model-id",
        str(model["model_id"]),
        "--backend",
        str(defaults["backend"]),
        "--device",
        str(defaults["device"]),
        "--torch-dtype",
        str(defaults["torch_dtype"]),
        "--selector-artifact",
        _resolve_selector_artifact(str(model["selector_artifact"])),
        "--max-prompt-tokens",
        *[str(value) for value in max_prompt_tokens],
        "--warmup-runs",
        str(int(defaults["warmup_runs"])),
        "--measured-runs",
        str(int(defaults["measured_runs"])),
        "--output",
        str(run_jsonl),
    ]
    if model.get("layer_profile"):
        command.extend(["--layer-profile", str(model["layer_profile"])])
    report_command = [
        _python_bin(),
        str(REPO_ROOT / "scripts" / "report_qwen35_longbench_selector_compare.py"),
        "--input",
        str(run_jsonl),
        "--markdown-output",
        str(report_md),
        "--json-output",
        str(report_json),
        "--title",
        f"{model['model_key']} LongBench Selector Compare",
    ]
    return command, report_command


def _backend_truth_command(
    defaults: dict[str, Any],
    model: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    prompt_lengths = model.get("backend_truth", {}).get("prompt_lengths", defaults["backend_prompt_lengths"])
    max_new_tokens = int(model.get("backend_truth", {}).get("max_new_tokens", defaults["backend_max_new_tokens"]))
    tokens_per_page = int(model.get("backend_truth", {}).get("tokens_per_page", defaults["backend_tokens_per_page"]))
    command = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_qwen35_9b_backend_truth.sh"),
        str(output_dir),
        "--warmup-runs",
        str(int(defaults["warmup_runs"])),
        "--measured-runs",
        str(int(defaults["measured_runs"])),
    ]
    env = {
        "LEARNED_SELECTOR_ARTIFACT": _resolve_selector_artifact(str(model["selector_artifact"])),
        "QWEN35_BACKEND_TRUTH_MODEL_ID": str(model["model_id"]),
        "QWEN35_BACKEND_TRUTH_MAX_NEW_TOKENS": str(max_new_tokens),
        "QWEN35_BACKEND_TRUTH_TOKENS_PER_PAGE": str(tokens_per_page),
        "QWEN35_BACKEND_TRUTH_PROMPT_LENGTHS": " ".join(str(value) for value in prompt_lengths),
    }
    return command, env


def _run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    command_env.setdefault("PYTHONPATH", str(REPO_ROOT))
    subprocess.run(command, cwd=str(REPO_ROOT), env=command_env, check=True)


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(manifest.get("defaults", {}))
    selected_models = _selected_models(manifest, args.models)
    selected_benchmarks = _selected_benchmarks(args.benchmarks)

    run_records: list[dict[str, Any]] = []
    for model in selected_models:
        model_key = str(model["model_key"])
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        for benchmark_name in selected_benchmarks:
            if benchmark_name not in model:
                continue
            benchmark_dir = model_dir / benchmark_name
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            if benchmark_name == "task_compare":
                run_command, report_command = _task_compare_command(defaults, model, output_dir=benchmark_dir)
                run_records.append(
                    {
                        "model_key": model_key,
                        "model_id": model["model_id"],
                        "benchmark": benchmark_name,
                        "run_command": run_command,
                        "report_command": report_command,
                    }
                )
                if not args.dry_run:
                    _run_command(run_command)
                    _run_command(report_command)
            elif benchmark_name == "longbench":
                run_command, report_command = _longbench_command(defaults, model, output_dir=benchmark_dir)
                run_records.append(
                    {
                        "model_key": model_key,
                        "model_id": model["model_id"],
                        "benchmark": benchmark_name,
                        "run_command": run_command,
                        "report_command": report_command,
                    }
                )
                if not args.dry_run:
                    _run_command(run_command)
                    _run_command(report_command)
            elif benchmark_name == "backend_truth":
                run_command, env = _backend_truth_command(defaults, model, output_dir=benchmark_dir)
                run_records.append(
                    {
                        "model_key": model_key,
                        "model_id": model["model_id"],
                        "benchmark": benchmark_name,
                        "run_command": run_command,
                        "environment": env,
                    }
                )
                if not args.dry_run:
                    _run_command(run_command, env=env)

    manifest_record = {
        "title": manifest.get("title", "Qwen Results Matrix"),
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir),
        "models": [str(item.get("model_key")) for item in selected_models],
        "benchmarks": list(selected_benchmarks),
        "dry_run": bool(args.dry_run),
        "runs": run_records,
    }
    manifest_path = output_dir / "matrix_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest_record, indent=2, sort_keys=True), encoding="utf-8")

    if args.dry_run:
        for run in run_records:
            print(f"[{run['model_key']}:{run['benchmark']}]")
            if "environment" in run:
                for key, value in sorted(run["environment"].items()):
                    print(f"  env {key}={shlex.quote(str(value))}")
            print("  " + " ".join(shlex.quote(str(part)) for part in run["run_command"]))
            report_command = run.get("report_command")
            if report_command:
                print("  " + " ".join(shlex.quote(str(part)) for part in report_command))
    else:
        print(manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
