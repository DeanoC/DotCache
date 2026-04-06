#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.bench_qwen35_attention_subset_dotcache_longbench_qa import (
    DEFAULT_LONGBENCH_ZIP_URL,
    _ensure_longbench_zip,
)
from dotcache.longbench_v1 import build_prompt_specs_from_zip


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_CASES = ["exact", "quality", "systems", "streaming_sink_recent", "quest_like"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report live per-shard throughput and ETA for sharded Qwen LongBench runs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pack", default="original_suite")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--comparison-cases", nargs="+", default=list(DEFAULT_COMPARISON_CASES))
    parser.add_argument("--max-prompt-tokens", type=int, nargs="+", default=[4096])
    parser.add_argument(
        "--prompt-pack-preset",
        choices=["original_full_suite", "original_stratified_16_per_dataset", "original_stratified_32_per_dataset"],
        default="original_full_suite",
    )
    parser.add_argument("--longbench-cache-dir", default=str(REPO_ROOT / "benchmarks" / "cache" / "longbench"))
    parser.add_argument("--longbench-zip-url", default=DEFAULT_LONGBENCH_ZIP_URL)
    parser.add_argument("--recent-aggregates", type=int, default=5)
    return parser.parse_args()


def model_slug(model_id: str) -> str:
    value = str(model_id).split("/")[-1].lower()
    return value.replace(".", "p")


def shard_jsonl_path(*, output_dir: Path, model_id: str, pack: str, shard_count: int, shard_index: int) -> Path:
    slug = model_slug(model_id)
    suffix = f".shard{int(shard_index):02d}-of-{int(shard_count):02d}"
    return output_dir / f"{slug}_longbench_{pack}{suffix}.jsonl"


def apply_prompt_shard(
    prompt_specs: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    return [spec for index, spec in enumerate(prompt_specs) if index % shard_count == shard_index]


def load_prompt_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    zip_path = _ensure_longbench_zip(Path(args.longbench_cache_dir), str(args.longbench_zip_url))
    if args.prompt_pack_preset == "original_full_suite":
        return build_prompt_specs_from_zip(zip_path)
    if args.prompt_pack_preset == "original_stratified_16_per_dataset":
        return build_prompt_specs_from_zip(zip_path, stratified_limit_per_dataset=16)
    if args.prompt_pack_preset == "original_stratified_32_per_dataset":
        return build_prompt_specs_from_zip(zip_path, stratified_limit_per_dataset=32)
    raise SystemExit(f"unsupported prompt pack preset: {args.prompt_pack_preset}")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "-"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def load_aggregate_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("measurement_kind") == "aggregate":
                rows.append(payload)
    return rows


def compute_shard_summary(
    *,
    aggregate_rows: list[dict[str, Any]],
    total_prompt_count_for_shard: int,
    case_count: int,
    context_count: int,
    recent_aggregates: int,
) -> dict[str, Any]:
    total_expected_aggregates = int(total_prompt_count_for_shard * case_count * context_count)
    completed_aggregates = len(aggregate_rows)
    remaining_aggregates = max(total_expected_aggregates - completed_aggregates, 0)
    recent_rows = aggregate_rows[-max(1, int(recent_aggregates)) :] if aggregate_rows else []
    recent_mean_wall_s = None
    if recent_rows:
        recent_mean_wall_s = sum(float(row.get("runner_wall_time_s", 0.0)) for row in recent_rows) / float(len(recent_rows))
    eta_seconds = None
    if recent_mean_wall_s is not None:
        eta_seconds = float(recent_mean_wall_s) * float(remaining_aggregates)
    last_row = aggregate_rows[-1] if aggregate_rows else None
    return {
        "total_expected_aggregates": total_expected_aggregates,
        "completed_aggregates": completed_aggregates,
        "remaining_aggregates": remaining_aggregates,
        "recent_mean_wall_s": recent_mean_wall_s,
        "eta_seconds": eta_seconds,
        "last_dataset": last_row.get("longbench_dataset") if last_row else None,
        "last_row_index": last_row.get("longbench_row_index") if last_row else None,
        "last_case": last_row.get("comparison_case") if last_row else None,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.input_dir).resolve()
    prompt_specs = load_prompt_specs(args)
    total_prompt_count = len(prompt_specs)
    case_count = len(args.comparison_cases)
    context_count = len(args.max_prompt_tokens)
    total_expected_aggregates = int(total_prompt_count * case_count * context_count)

    shard_summaries: list[dict[str, Any]] = []
    for shard_index in range(int(args.shard_count)):
        shard_prompt_specs = apply_prompt_shard(
            prompt_specs,
            shard_count=int(args.shard_count),
            shard_index=shard_index,
        )
        shard_path = shard_jsonl_path(
            output_dir=output_dir,
            model_id=args.model_id,
            pack=args.pack,
            shard_count=int(args.shard_count),
            shard_index=shard_index,
        )
        aggregate_rows = load_aggregate_rows(shard_path)
        summary = compute_shard_summary(
            aggregate_rows=aggregate_rows,
            total_prompt_count_for_shard=len(shard_prompt_specs),
            case_count=case_count,
            context_count=context_count,
            recent_aggregates=int(args.recent_aggregates),
        )
        summary.update(
            {
                "shard_index": shard_index,
                "prompt_count": len(shard_prompt_specs),
                "path": shard_path,
            }
        )
        shard_summaries.append(summary)

    total_completed_aggregates = sum(int(summary["completed_aggregates"]) for summary in shard_summaries)
    total_remaining_aggregates = max(total_expected_aggregates - total_completed_aggregates, 0)
    shard_etas = [float(summary["eta_seconds"]) for summary in shard_summaries if summary["eta_seconds"] is not None]
    overall_eta_seconds = max(shard_etas) if shard_etas else None

    print(f"Input dir: {output_dir}")
    print(
        "Overall: "
        f"{total_completed_aggregates}/{total_expected_aggregates} aggregate units complete "
        f"({(100.0 * total_completed_aggregates / max(total_expected_aggregates, 1)):.2f}%), "
        f"remaining {total_remaining_aggregates}, "
        f"ETA {format_duration(overall_eta_seconds)}"
    )
    print("Shard status:")
    for summary in shard_summaries:
        last_marker = "-"
        if summary["last_dataset"] is not None:
            last_marker = (
                f"{summary['last_dataset']}#{summary['last_row_index']}:{summary['last_case']}"
            )
        print(
            f"  shard {int(summary['shard_index']):02d}: "
            f"prompts={int(summary['prompt_count'])} "
            f"aggregates={int(summary['completed_aggregates'])}/{int(summary['total_expected_aggregates'])} "
            f"recent_mean_wall_s="
            f"{('-' if summary['recent_mean_wall_s'] is None else f'{float(summary['recent_mean_wall_s']):.2f}') } "
            f"eta={format_duration(summary['eta_seconds'])} "
            f"last={last_marker}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
