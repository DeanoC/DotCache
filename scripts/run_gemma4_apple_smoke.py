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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Gemma 4 Apple MPS smoke with a structured summary record.")
    parser.add_argument("--model-id", default="google/gemma-4-E2B")
    parser.add_argument("--prompt", default="Cache locality on Apple Silicon is")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmarks" / "results" / f"gemma4_apple_smoke_{time.strftime('%Y%m%d')}"),
    )
    return parser.parse_args()


def _probe_command(args: argparse.Namespace, *, probe_output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "probe_gemma4_text.py"),
        "--model-id",
        args.model_id,
        "--device-map",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--run-dotcache",
        "--dotcache-backend",
        "torch_mps",
        "--dotcache-profile",
        "balanced",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt",
        args.prompt,
        "--output-path",
        str(probe_output_path),
    ]


def _load_probe_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_summary(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_output_path = output_dir / "dotcache_mps_balanced.json"
    summary_path = output_dir / "smoke_runner.json"

    command = _probe_command(args, probe_output_path=probe_output_path)
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
    elapsed_s = time.monotonic() - started_at

    probe_record = _load_probe_record(probe_output_path)
    status = "timeout" if timed_out else ("completed" if process.returncode == 0 else "error")
    summary = {
        "status": status,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "timeout_seconds": int(args.timeout_seconds),
        "elapsed_s": float(elapsed_s),
        "returncode": int(process.returncode) if process.returncode is not None else None,
        "probe_output_path": _display_path(probe_output_path),
        "runner_command": command,
        "probe_record": probe_record,
    }
    if timed_out:
        summary["error_type"] = "TimeoutExpired"
        summary["error_message"] = f"timed out after {args.timeout_seconds}s"
    elif probe_record is None:
        summary["error_type"] = "MissingProbeRecord"
        summary["error_message"] = "probe did not write a JSON record"
    elif probe_record.get("status") != "ok":
        summary["error_type"] = "ProbeError"
        summary["error_message"] = str(probe_record.get("error_message", "probe returned a non-ok status"))
    if stdout_text.strip():
        summary["stdout_tail"] = stdout_text.strip()[-4000:]
    if stderr_text.strip():
        summary["stderr_tail"] = stderr_text.strip()[-4000:]

    _write_summary(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))

    if timed_out or probe_record is None:
        return 1
    return 0 if probe_record.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
