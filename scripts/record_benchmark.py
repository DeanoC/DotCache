#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark command and append its output to a JSONL history file.")
    parser.add_argument("--label", required=True, help="Short experiment label.")
    parser.add_argument("--notes", default="", help="Optional free-form notes about the run.")
    parser.add_argument("--output", default="benchmarks/results/history.jsonl", help="JSONL file to append to.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run. Prefix it with --.")
    return parser.parse_args()


def _parse_stdout(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            raw_lines.append(stripped)
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
        else:
            raw_lines.append(stripped)
    return records, raw_lines


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("record_benchmark.py requires a command after --")

    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)

    records, raw_stdout = _parse_stdout(completed.stdout)
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "notes": args.notes,
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "command": " ".join(command),
        "returncode": completed.returncode,
        "records": records,
        "raw_stdout": raw_stdout,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
