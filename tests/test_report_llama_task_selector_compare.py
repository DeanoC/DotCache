from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_llama_task_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("report_llama_task_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_summarizes_speedup_and_errors() -> None:
    rows = [
        {
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "exact",
            "task_metric_value": 1.0,
            "decode_ms_per_step": 120.0,
            "teacher_forced_perplexity_ratio": 1.0,
            "teacher_forced_logit_max_abs_error": 0.9,
        },
        {
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "quality",
            "task_metric_value": 1.0,
            "decode_ms_per_step": 80.0,
            "teacher_forced_perplexity_ratio": 1.1,
            "teacher_forced_logit_max_abs_error": 0.4,
        },
        {
            "task_name": "retrieval_passkey",
            "prompt_length_requested": 1024,
            "selector_profile": "systems",
            "task_metric_value": 1.0,
            "decode_ms_per_step": 40.0,
            "teacher_forced_perplexity_ratio": 1.05,
            "teacher_forced_logit_max_abs_error": 0.3,
        },
    ]
    payload, markdown = MODULE.build_report(rows, trial_rows=[])
    assert payload["rows"][0]["systems_vs_quality_speedup"] == 2.0
    assert payload["rows"][0]["systems_teacher_forced_perplexity_ratio"] == 1.05
    assert payload["rows"][0]["systems_teacher_forced_logit_max_abs_error"] == 0.3
    assert "Llama 3.2 3B Task Selector Compare" in markdown
    assert "1.050" in markdown
