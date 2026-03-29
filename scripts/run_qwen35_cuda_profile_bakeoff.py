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


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUDA bakeoff sweeps across one or more Qwen3.5 attention-subset layer profiles."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="One or more layer-profile YAML paths.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["serving", "quality", "scorer"],
        default=["serving", "quality", "scorer"],
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmarks" / "results" / "qwen35_cuda_profile_bakeoff"),
    )
    return parser.parse_args()


def _profile_label(profile_path: Path) -> str:
    return profile_path.stem


def _mode_args(mode: str) -> list[str]:
    if mode == "serving":
        return []
    if mode == "quality":
        return ["--quality-check"]
    if mode == "scorer":
        return ["--scorer-diagnostic"]
    raise ValueError(f"unsupported mode: {mode}")


def _run_command(command: list[str], *, timeout_seconds: int) -> tuple[str, str, bool, int, float]:
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
        stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout_text, stderr_text = process.communicate()
    elapsed = time.monotonic() - started_at
    return stdout_text, stderr_text, timed_out, int(process.returncode), elapsed


def _parse_rows(stdout_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in stdout_text.splitlines():
        stripped = raw_line.strip().replace("\x00", "")
        if not stripped.startswith("{"):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            rows.append(candidate)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    mode: str,
    output_path: Path,
    command: list[str],
    returncode: int,
    timed_out: bool,
    elapsed_s: float,
    stderr_text: str,
) -> dict[str, Any]:
    prompt_lengths = [int(row.get("prompt_length", -1)) for row in rows if row.get("prompt_length") is not None]
    statuses = [str(row.get("status", "ok")) for row in rows]
    return {
        "profile": profile,
        "mode": mode,
        "output_path": str(output_path),
        "row_count": len(rows),
        "prompt_lengths": prompt_lengths,
        "statuses": statuses,
        "timed_out": timed_out,
        "returncode": returncode,
        "wall_time_s": elapsed_s,
        "command": command,
        "stderr_tail": stderr_text.strip()[-4000:] if stderr_text.strip() else "",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    for raw_profile in args.profiles:
        profile_path = Path(raw_profile)
        if not profile_path.is_absolute():
            profile_path = (REPO_ROOT / profile_path).resolve()
        profile_label = _profile_label(profile_path)
        for mode in args.modes:
            output_path = output_dir / f"{profile_label}_{mode}.jsonl"
            command = [
                sys.executable,
                str(BENCHMARK),
                "--model-id",
                args.model_id,
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--torch-dtype",
                args.torch_dtype,
                "--profile-backend",
                "--default-mode-k",
                "M0",
                "--default-mode-v",
                "M0",
                "--layer-profile",
                str(profile_path),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--target-prompt-lengths",
                *[str(context) for context in args.contexts],
                "--continue-on-error",
                *_mode_args(mode),
            ]
            stdout_text, stderr_text, timed_out, returncode, elapsed_s = _run_command(
                command,
                timeout_seconds=args.timeout_seconds,
            )
            rows = _parse_rows(stdout_text)
            _write_jsonl(output_path, rows)
            summary = _summarize_rows(
                rows,
                profile=profile_label,
                mode=mode,
                output_path=output_path,
                command=command,
                returncode=returncode,
                timed_out=timed_out,
                elapsed_s=elapsed_s,
                stderr_text=stderr_text,
            )
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True))
                handle.write("\n")
            print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
