from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_qwen_results_matrix.py"
SPEC = importlib.util.spec_from_file_location("report_qwen_results_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_renders_task_longbench_and_backend_tables(tmp_path: Path) -> None:
    manifest = {
        "title": "Qwen Matrix Test",
        "models": [
            {"model_key": "qwen35_9b", "model_id": "Qwen/Qwen3.5-9B"},
            {"model_key": "qwen35_27b", "model_id": "Qwen/Qwen3.5-27B"},
        ],
    }

    task_dir = tmp_path / "qwen35_9b" / "task_compare"
    task_dir.mkdir(parents=True)
    (task_dir / "task_selector_compare.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_name": "retrieval_passkey",
                        "prompt_length": 1024,
                        "exact_success": 1.0,
                        "quality_success": 1.0,
                        "systems_success": 1.0,
                        "quality_decode_ms_per_step": 100.0,
                        "systems_decode_ms_per_step": 25.0,
                        "systems_vs_quality_speedup": 4.0,
                        "quality_teacher_forced_logit_rmse": 0.4,
                        "systems_teacher_forced_logit_rmse": 0.3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    longbench_dir = tmp_path / "qwen35_9b" / "longbench"
    longbench_dir.mkdir(parents=True)
    (longbench_dir / "longbench_selector_compare.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "comparison_case": "systems",
                        "max_prompt_tokens": 4096,
                        "mean_exact_match": 0.25,
                        "mean_qa_f1": 0.44,
                        "mean_decode_ms_per_step": 90.0,
                        "p95_decode_ms_per_step": 91.0,
                        "mean_teacher_forced_perplexity_ratio": 1.02,
                        "mean_teacher_forced_logit_rmse": 0.02,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    backend_dir = tmp_path / "qwen35_27b" / "backend_truth"
    backend_dir.mkdir(parents=True)
    (backend_dir / "backend_truth_report.json").write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "prompt_length": 2048,
                        "speedups": {"learned_selector": {"vs_exact": 3.6, "vs_shortlist": 2.8}},
                        "variants": {
                            "exact": {"decode_ms_per_step": 820.0},
                            "shortlist_base": {"decode_ms_per_step": 640.0},
                            "learned_selector": {
                                "decode_ms_per_step": 225.0,
                                "m3_fraction": 0.995,
                                "selector_us_per_invocation": 24.8,
                                "score_ms_step": 77.8,
                                "mix_ms_step": 66.1,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload, markdown = MODULE.build_report(manifest, output_dir=tmp_path)
    assert payload["title"] == "Qwen Matrix Test"
    assert len(payload["task_rows"]) == 1
    assert len(payload["longbench_rows"]) == 1
    assert len(payload["backend_rows"]) == 1
    assert "Task Compare Matrix" in markdown
    assert "LongBench Matrix" in markdown
    assert "Backend Truth Matrix" in markdown
    assert "Qwen/Qwen3.5-27B" in markdown
