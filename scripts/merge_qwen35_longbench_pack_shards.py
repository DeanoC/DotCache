#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Qwen LongBench pack outputs and build reports.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pack", default="original_suite")
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def model_slug(model_id: str) -> str:
    value = str(model_id).split("/")[-1].lower()
    return value.replace(".", "p")


def shard_jsonl_path(*, output_dir: Path, model_id: str, pack: str, shard_count: int, shard_index: int) -> Path:
    slug = model_slug(model_id)
    suffix = f".shard{int(shard_index):02d}-of-{int(shard_count):02d}"
    return output_dir / f"{slug}_longbench_{pack}{suffix}.jsonl"


def merged_output_paths(*, output_dir: Path, model_id: str, pack: str) -> dict[str, Path]:
    slug = model_slug(model_id)
    return {
        "jsonl": output_dir / f"{slug}_longbench_{pack}.jsonl",
        "markdown": output_dir / "longbench_selector_compare.md",
        "json": output_dir / "longbench_selector_compare.json",
        "workbook_markdown": output_dir / "longbench_failure_workbook.md",
        "workbook_json": output_dir / "longbench_failure_workbook.json",
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.input_dir).resolve()
    shard_count = int(args.shard_count)
    if shard_count <= 0:
        raise SystemExit(f"shard count must be positive, got {shard_count}")
    shard_paths = [
        shard_jsonl_path(
            output_dir=output_dir,
            model_id=args.model_id,
            pack=args.pack,
            shard_count=shard_count,
            shard_index=shard_index,
        )
        for shard_index in range(shard_count)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing shard outputs:\n" + "\n".join(missing))

    output_paths = merged_output_paths(output_dir=output_dir, model_id=args.model_id, pack=args.pack)
    combined_jsonl_path = output_paths["jsonl"]
    with combined_jsonl_path.open("w", encoding="utf-8") as destination:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)

    child_env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT)]
    if child_env.get("PYTHONPATH"):
        pythonpath_entries.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    report_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "report_qwen35_longbench_selector_compare.py"),
        "--input",
        str(combined_jsonl_path),
        "--markdown-output",
        str(output_paths["markdown"]),
        "--json-output",
        str(output_paths["json"]),
        "--title",
        f"{args.model_id} LongBench {args.pack} Pack Compare",
    ]
    workbook_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "report_qwen35_longbench_failure_workbook.py"),
        "--input",
        str(combined_jsonl_path),
        "--markdown-output",
        str(output_paths["workbook_markdown"]),
        "--json-output",
        str(output_paths["workbook_json"]),
        "--title",
        f"{args.model_id} LongBench Failure Workbook",
    ]
    subprocess.run(report_command, cwd=str(REPO_ROOT), env=child_env, check=True)
    subprocess.run(workbook_command, cwd=str(REPO_ROOT), env=child_env, check=True)
    print(combined_jsonl_path)
    print(output_paths["markdown"])
    print(output_paths["json"])
    print(output_paths["workbook_markdown"])
    print(output_paths["workbook_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
