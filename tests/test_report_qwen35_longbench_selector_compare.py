from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_qwen35_longbench_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("report_qwen35_longbench_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_summarizes_longbench_rows_with_official_score_and_parity() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 120.0,
            "dotcache_decode_ms_per_step_p95": 125.0,
            "effective_bytes_per_token": 8.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_rmse": 0.20,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 80.0,
            "dotcache_decode_ms_per_step_p95": 82.0,
            "effective_bytes_per_token": 7.5,
            "teacher_forced_perplexity_ratio": 1.10,
            "teacher_forced_logit_rmse": 0.10,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.95,
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 40.0,
            "dotcache_decode_ms_per_step_p95": 41.0,
            "effective_bytes_per_token": 4.0,
            "teacher_forced_perplexity_ratio": 1.05,
            "teacher_forced_logit_rmse": 0.08,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.60,
            "longbench_answer_exact_match_cleaned": False,
            "longbench_qa_f1_max_cleaned": 0.5,
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 62.0,
            "effective_bytes_per_token": 3.0,
            "teacher_forced_perplexity_ratio": 1.20,
            "teacher_forced_logit_rmse": 0.30,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quest_like",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.93,
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 0.9,
            "dotcache_decode_ms_per_step": 55.0,
            "dotcache_decode_ms_per_step_p95": 56.0,
            "effective_bytes_per_token": 4.2,
            "teacher_forced_perplexity_ratio": 1.04,
            "teacher_forced_logit_rmse": 0.09,
        },
    ]
    trial_rows = [
        {
            "measurement_kind": "trial",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.95,
            "longbench_generated_text_cleaned": "bridge",
            "measurement_index": 0,
        }
    ]

    payload, markdown = MODULE.build_report(rows, title="Qwen LongBench Selector Compare", trial_rows=trial_rows)

    by_case = {row["comparison_case"]: row for row in payload["rows"]}
    assert by_case["systems"]["mean_decode_ms_per_step"] == 40.0
    assert by_case["systems"]["mean_official_score"] == 0.95
    assert any(row["comparison_case"] == "quest_like" for row in payload["rows"])
    assert any(row["comparison_case"] == "quest_like" for row in payload["parity_rows"])
    assert any(row["comparison_case"] == "quest_like" for row in payload["confidence_rows"])
    assert "| 4096 | systems | 1 | 0.950 | 1.000 | 1.000 | 40.000 | 41.000 | 4.000 | 1.050 | 0.080 | 0.950 |" in markdown
    assert "Task Family Breakdown" in markdown
    assert "## Parity" in markdown
    assert "## Confidence" in markdown
    assert "bridge" in markdown


def test_build_report_rejects_missing_expected_case() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "dotcache_decode_ms_per_step": 120.0,
            "dotcache_decode_ms_per_step_p95": 125.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "dotcache_decode_ms_per_step": 80.0,
            "dotcache_decode_ms_per_step_p95": 82.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.5,
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 62.0,
        },
    ]

    with pytest.raises(SystemExit, match="missing cases by context: 4096: systems"):
        MODULE.build_report(rows, title="Qwen LongBench Selector Compare")


def test_build_report_accepts_missing_optional_quality_case() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "dotcache_decode_ms_per_step": 120.0,
            "dotcache_decode_ms_per_step_p95": 125.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.95,
            "dotcache_decode_ms_per_step": 40.0,
            "dotcache_decode_ms_per_step_p95": 41.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.60,
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 62.0,
        },
    ]

    payload, markdown = MODULE.build_report(rows, title="Reduced Courtroom")

    assert [row["comparison_case"] for row in payload["rows"]] == ["exact", "streaming_sink_recent", "systems"]
    assert "## Confidence" in markdown


def test_build_report_renders_missing_optional_metrics_as_dash() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": case,
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "trec",
            "longbench_task_family": "classification",
            "longbench_official_score": 0.25,
            "longbench_answer_exact_match_cleaned": None,
            "longbench_qa_f1_max_cleaned": None,
            "dotcache_decode_ms_per_step": 10.0,
            "dotcache_decode_ms_per_step_p95": 11.0,
            "effective_bytes_per_token": 3.0,
            "teacher_forced_perplexity_ratio": None,
            "teacher_forced_logit_rmse": 0.50,
        }
        for case in ("exact", "quality", "systems", "streaming_sink_recent")
    ]

    payload, markdown = MODULE.build_report(rows, title="Missing Optional Metrics")

    assert payload["rows"][0]["mean_teacher_forced_perplexity_ratio"] is None
    assert "| 4096 | exact | 1 | 0.250 | - | - | 10.000 | 11.000 | 3.000 | - | 0.500 | 0.250 |" in markdown
