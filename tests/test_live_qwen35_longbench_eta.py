from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "live_qwen35_longbench_eta.py"
SPEC = importlib.util.spec_from_file_location("live_qwen35_longbench_eta", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_apply_prompt_shard_matches_round_robin_distribution() -> None:
    specs = [{"prompt_id": f"p{index}"} for index in range(10)]

    shard = MODULE.apply_prompt_shard(specs, shard_count=4, shard_index=2)

    assert [row["prompt_id"] for row in shard] == ["p2", "p6"]


def test_compute_shard_summary_uses_recent_aggregate_mean_for_eta() -> None:
    rows = [
        {"runner_wall_time_s": 10.0, "longbench_dataset": "a", "longbench_row_index": 0, "comparison_case": "exact"},
        {"runner_wall_time_s": 20.0, "longbench_dataset": "a", "longbench_row_index": 1, "comparison_case": "exact"},
        {"runner_wall_time_s": 30.0, "longbench_dataset": "a", "longbench_row_index": 2, "comparison_case": "quality"},
    ]

    summary = MODULE.compute_shard_summary(
        aggregate_rows=rows,
        total_prompt_count_for_shard=4,
        case_count=2,
        context_count=1,
        recent_aggregates=2,
    )

    assert summary["total_expected_aggregates"] == 8
    assert summary["completed_aggregates"] == 3
    assert summary["remaining_aggregates"] == 5
    assert summary["recent_mean_wall_s"] == 25.0
    assert summary["eta_seconds"] == 125.0
    assert summary["last_dataset"] == "a"
    assert summary["last_row_index"] == 2
    assert summary["last_case"] == "quality"


def test_format_duration_handles_short_and_long_intervals() -> None:
    assert MODULE.format_duration(None) == "-"
    assert MODULE.format_duration(59) == "59s"
    assert MODULE.format_duration(61) == "1m 01s"
    assert MODULE.format_duration(3661) == "1h 01m 01s"
    assert MODULE.format_duration(90061) == "1d 01h 01m"
