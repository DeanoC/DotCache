from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_qwen35_longbench_failure_workbook.py"
SPEC = importlib.util.spec_from_file_location("report_qwen35_longbench_failure_workbook", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_classifies_systems_misses() -> None:
    rows = [
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 1.0,
            "effective_bytes_per_token": 8.0,
            "longbench_generated_text_cleaned": "bridge answer",
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "hotpot_0",
            "longbench_dataset": "hotpotqa",
            "longbench_task_family": "qa",
            "longbench_official_score": 0.0,
            "longbench_official_score_raw": 0.0,
            "longbench_official_score_cleaning_delta": 0.0,
            "effective_bytes_per_token": 4.0,
            "longbench_generated_text_cleaned": "wrong",
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "trec_0",
            "longbench_dataset": "trec",
            "longbench_task_family": "classification",
            "longbench_official_score": 1.0,
            "effective_bytes_per_token": 8.0,
            "longbench_generated_text_cleaned": "Sports",
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "trec_0",
            "longbench_dataset": "trec",
            "longbench_task_family": "classification",
            "longbench_official_score": 0.5,
            "longbench_official_score_raw": 0.1,
            "longbench_official_score_cleaning_delta": 0.4,
            "longbench_chat_artifact_cleaned": True,
            "effective_bytes_per_token": 4.0,
            "longbench_generated_text_cleaned": "Sports",
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "exact",
            "evaluation_prompt_id": "samsum_0",
            "longbench_dataset": "samsum",
            "longbench_task_family": "summarization",
            "longbench_official_score": 0.6,
            "effective_bytes_per_token": 8.0,
            "longbench_generated_text_cleaned": "summary",
        },
        {
            "measurement_kind": "aggregate",
            "comparison_max_prompt_tokens": 4096,
            "comparison_case": "systems",
            "evaluation_prompt_id": "samsum_0",
            "longbench_dataset": "samsum",
            "longbench_task_family": "summarization",
            "longbench_official_score": 0.4,
            "longbench_official_score_raw": 0.4,
            "longbench_official_score_cleaning_delta": 0.0,
            "effective_bytes_per_token": 7.9,
            "longbench_generated_text_cleaned": "short summary",
        },
    ]

    payload, markdown = MODULE.build_report(rows, title="Failure Workbook")

    by_prompt = {row["evaluation_prompt_id"]: row for row in payload["rows"]}
    assert by_prompt["hotpot_0"]["classification"] == "selection_miss"
    assert by_prompt["trec_0"]["classification"] == "write_format_damage"
    assert by_prompt["samsum_0"]["classification"] == "downstream_under_attention"
    assert "## Workbook" in markdown
    assert "selection_miss" in markdown
    assert "write_format_damage" in markdown
    assert "downstream_under_attention" in markdown
