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
            "comparison_case": "dense",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 160.0,
            "comparison_decode_ms_per_step_p95": 165.0,
            "comparison_teacher_forced_perplexity_ratio": 1.0,
            "comparison_teacher_forced_logit_rmse": 0.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 120.0,
            "comparison_decode_ms_per_step_p95": 125.0,
            "effective_bytes_per_token": 8.0,
            "comparison_teacher_forced_perplexity_ratio": 1.0,
            "comparison_teacher_forced_logit_rmse": 0.20,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 80.0,
            "comparison_decode_ms_per_step_p95": 82.0,
            "effective_bytes_per_token": 7.5,
            "comparison_teacher_forced_perplexity_ratio": 1.10,
            "comparison_teacher_forced_logit_rmse": 0.10,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.95,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 40.0,
            "comparison_decode_ms_per_step_p95": 41.0,
            "effective_bytes_per_token": 4.0,
            "comparison_teacher_forced_perplexity_ratio": 1.05,
            "comparison_teacher_forced_logit_rmse": 0.08,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.60,
            "comparison_generated_text_cleaned": "tunnel",
            "comparison_answer_exact_match_cleaned": False,
            "comparison_qa_f1_max_cleaned": 0.5,
            "comparison_decode_ms_per_step": 60.0,
            "comparison_decode_ms_per_step_p95": 62.0,
            "effective_bytes_per_token": 3.0,
            "comparison_teacher_forced_perplexity_ratio": 1.20,
            "comparison_teacher_forced_logit_rmse": 0.30,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quest_like",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.93,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 0.9,
            "comparison_decode_ms_per_step": 55.0,
            "comparison_decode_ms_per_step_p95": 56.0,
            "effective_bytes_per_token": 4.2,
            "comparison_teacher_forced_perplexity_ratio": 1.04,
            "comparison_teacher_forced_logit_rmse": 0.09,
        },
    ]
    trial_rows = [
        {
            "measurement_kind": "trial",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "dense",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "measurement_index": 0,
        }
    ]

    payload, markdown = MODULE.build_report(rows, title="Qwen LongBench Selector Compare", trial_rows=trial_rows)

    by_case = {row["comparison_case"]: row for row in payload["rows"]}
    assert by_case["dense"]["mean_matches_dense_output"] == 1.0
    assert by_case["systems"]["mean_matches_dense_output"] == 1.0
    assert by_case["systems"]["mean_decode_ms_per_step"] == 40.0
    assert by_case["systems"]["mean_official_score"] == 0.95
    assert any(row["comparison_case"] == "quest_like" for row in payload["rows"])
    assert any(row["comparison_case"] == "quest_like" for row in payload["parity_rows"])
    assert any(row["comparison_case"] == "quest_like" for row in payload["confidence_rows"])
    assert "| 4096 | systems | 1 | 0.950 | 1.000 | 1.000 | 1.000 | 40.000 | 41.000 | 4.000 | 1.050 | 0.080 | 0.950 |" in markdown
    assert "Task Family Breakdown" in markdown
    assert "## Parity" in markdown
    assert "## Confidence" in markdown
    assert "## Sample Outputs" in markdown
    assert "bridge" in markdown


def test_build_report_rejects_missing_expected_case() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "dense",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 160.0,
            "comparison_decode_ms_per_step_p95": 165.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 1.0,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 120.0,
            "comparison_decode_ms_per_step_p95": 125.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 1.0,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 80.0,
            "comparison_decode_ms_per_step_p95": 82.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 0.5,
            "comparison_generated_text_cleaned": "tunnel",
            "comparison_answer_exact_match_cleaned": False,
            "comparison_qa_f1_max_cleaned": 0.5,
            "comparison_decode_ms_per_step": 60.0,
            "comparison_decode_ms_per_step_p95": 62.0,
        },
    ]

    with pytest.raises(SystemExit, match="missing cases by context: 4096: systems"):
        MODULE.build_report(rows, title="Qwen LongBench Selector Compare")


def test_build_report_accepts_missing_optional_cases() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "dense",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 160.0,
            "comparison_decode_ms_per_step_p95": 165.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 1.0,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 120.0,
            "comparison_decode_ms_per_step_p95": 125.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 0.95,
            "comparison_generated_text_cleaned": "bridge",
            "comparison_answer_exact_match_cleaned": True,
            "comparison_qa_f1_max_cleaned": 1.0,
            "comparison_decode_ms_per_step": 40.0,
            "comparison_decode_ms_per_step_p95": 41.0,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_official_score": 0.6,
            "comparison_generated_text_cleaned": "tunnel",
            "comparison_answer_exact_match_cleaned": False,
            "comparison_qa_f1_max_cleaned": 0.5,
            "comparison_decode_ms_per_step": 60.0,
            "comparison_decode_ms_per_step_p95": 62.0,
        },
    ]

    payload, markdown = MODULE.build_report(rows, title="Reduced Courtroom")

    assert [row["comparison_case"] for row in payload["rows"]] == ["dense", "exact", "streaming_sink_recent", "systems"]
    assert "## Confidence" in markdown


def test_build_report_renders_missing_optional_metrics_as_dash() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": case,
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "comparison_generated_text_cleaned": "bridge" if case == "dense" else "tunnel",
            "comparison_answer_exact_match_cleaned": case == "dense",
            "comparison_qa_f1_max_cleaned": 1.0 if case == "dense" else 0.25,
            "comparison_decode_ms_per_step": 10.0,
            "comparison_decode_ms_per_step_p95": 11.0,
            "comparison_teacher_forced_perplexity_ratio": 1.0 if case == "dense" else None,
            "comparison_teacher_forced_logit_rmse": 0.0 if case == "dense" else 0.50,
        }
        for case in ("dense", "exact", "quality", "systems", "streaming_sink_recent")
    ]

    payload, markdown = MODULE.build_report(rows, title="Missing Optional Metrics")

    by_case = {row["comparison_case"]: row for row in payload["rows"]}
    assert by_case["exact"]["mean_teacher_forced_perplexity_ratio"] is None
    assert "| 4096 | exact | 1 | - | 0.000 | 0.000 | 0.250 | 10.000 | 11.000 | - | - | 0.500 | - |" in markdown
