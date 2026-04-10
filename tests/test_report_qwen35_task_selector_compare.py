from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_qwen35_task_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("report_qwen35_task_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_summarizes_task_rows() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "dense",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-58142",
            "task_decode_ms_per_step": 120.0,
        },
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "exact",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-58142",
            "dotcache_decode_ms_per_step": 100.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_rmse": 0.6,
        },
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "quality",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-5814X",
            "dotcache_decode_ms_per_step": 80.0,
            "teacher_forced_perplexity_ratio": 1.1,
            "teacher_forced_logit_rmse": 0.4,
        },
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "systems",
            "task_metric_value": 1.0,
            "dotcache_decode_ms_per_step": 40.0,
            "teacher_forced_perplexity_ratio": 1.05,
            "teacher_forced_logit_rmse": 0.3,
        },
    ]
    trial_rows = [
        {
            "measurement_kind": "trial",
            "measurement_index": 0,
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "dense",
            "task_metric_value": 1.0,
            "task_expected_answer": "RIVER-58142",
            "task_generated_text": "RIVER-58142",
            "task_generated_text_cleaned": "RIVER-58142",
            "task_generated_first_line": "RIVER-58142",
            "task_generated_token_ids": [1, 2, 3],
            "decode_steps": 3,
        }
    ]
    payload, markdown = MODULE.build_report(rows, title="Qwen Task Selector Compare", trial_rows=trial_rows)
    assert payload["rows"][0]["dense_success"] == 1.0
    assert payload["rows"][0]["dense_matches_dense_output"] == 1.0
    assert payload["rows"][0]["exact_matches_dense_output"] == 1.0
    assert payload["rows"][0]["quality_matches_dense_output"] == 0.0
    assert payload["rows"][0]["quality_vs_dense_speedup"] == 1.5
    assert payload["rows"][0]["systems_vs_quality_speedup"] == 2.0
    assert payload["rows"][0]["systems_teacher_forced_perplexity_ratio"] == 1.05
    assert payload["rows"][0]["systems_teacher_forced_logit_rmse"] == 0.3
    assert "retrieval_passkey" in markdown
    assert "dense_success" in markdown
    assert "quality_matches_dense_output" in markdown
    assert "2.000" in markdown
    assert "1.050" in markdown
    assert "Sample Outputs" in markdown
    assert "RIVER-58142" in markdown
    assert "matches_dense_output" in markdown
    assert "cap_hit" in markdown
    assert "yes" in markdown


def test_build_report_can_merge_reference_dense_and_exact_rows() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "quality",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-58142",
            "dotcache_decode_ms_per_step": 80.0,
            "teacher_forced_perplexity_ratio": 1.02,
            "teacher_forced_logit_rmse": 0.4,
        }
    ]
    reference_rows = [
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "dense",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-58142",
            "task_decode_ms_per_step": 120.0,
        },
        {
            "measurement_kind": "aggregate",
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "exact",
            "task_metric_value": 1.0,
            "task_generated_text_cleaned": "RIVER-58142",
            "dotcache_decode_ms_per_step": 100.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_rmse": 0.6,
        },
    ]

    payload, markdown = MODULE.build_report(
        rows,
        title="Qwen Task Selector Compare",
        reference_rows=reference_rows,
    )

    summary = payload["rows"][0]
    assert summary["dense_success"] == 1.0
    assert summary["exact_success"] == 1.0
    assert summary["quality_matches_dense_output"] == 1.0
    assert summary["quality_vs_dense_speedup"] == 1.5
    assert "dense_success" in markdown
    assert "exact_success" in markdown
