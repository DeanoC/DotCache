from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_qwen35_longbench_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("report_qwen35_longbench_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_summarizes_longbench_rows() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 120.0,
            "dotcache_decode_ms_per_step_p95": 125.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_rmse": 0.20,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 80.0,
            "dotcache_decode_ms_per_step_p95": 82.0,
            "teacher_forced_perplexity_ratio": 1.10,
            "teacher_forced_logit_rmse": 0.10,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 40.0,
            "dotcache_decode_ms_per_step_p95": 41.0,
            "teacher_forced_perplexity_ratio": 1.05,
            "teacher_forced_logit_rmse": 0.08,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": False,
            "longbench_qa_f1_max_cleaned": 0.5,
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 62.0,
            "teacher_forced_perplexity_ratio": 1.20,
            "teacher_forced_logit_rmse": 0.30,
        },
    ]
    trial_rows = [
        {
            "measurement_kind": "trial",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "longbench_generated_text_cleaned": "bridge",
            "measurement_index": 0,
        }
    ]

    payload, markdown = MODULE.build_report(rows, title="Qwen LongBench Selector Compare", trial_rows=trial_rows)

    by_case = {row["comparison_case"]: row for row in payload["rows"]}
    assert by_case["systems"]["mean_decode_ms_per_step"] == 40.0
    assert "systems_vs_quality_speedup" not in payload
    assert "| 4096 | systems | 1 | 1.000 | 1.000 | 40.000 | 41.000 | 1.050 | 0.080 |" in markdown
    assert "| 4096 | 1.500 | 3.000 | 2.000 | 2.000 | 0.000 | 0.000 |" in markdown
    assert "Sample Outputs" in markdown
    assert "bridge" in markdown


def test_build_report_rejects_missing_expected_case() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 120.0,
            "dotcache_decode_ms_per_step_p95": 125.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_rmse": 0.20,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "quality",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": True,
            "longbench_qa_f1_max_cleaned": 1.0,
            "dotcache_decode_ms_per_step": 80.0,
            "dotcache_decode_ms_per_step_p95": 82.0,
            "teacher_forced_perplexity_ratio": 1.10,
            "teacher_forced_logit_rmse": 0.10,
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "streaming_sink_recent",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_answer_exact_match_cleaned": False,
            "longbench_qa_f1_max_cleaned": 0.5,
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 62.0,
            "teacher_forced_perplexity_ratio": 1.20,
            "teacher_forced_logit_rmse": 0.30,
        },
    ]

    with pytest.raises(SystemExit, match="missing cases by context: 4096: systems"):
        MODULE.build_report(rows, title="Qwen LongBench Selector Compare")
