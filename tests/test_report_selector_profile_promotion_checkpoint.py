from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_selector_profile_promotion_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("report_selector_profile_promotion_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_captures_qwen_and_llama_calls() -> None:
    payload, markdown = MODULE.build_report(
        qwen_matrix_payload={
            "task_rows": [
                {
                    "model_key": "qwen35_9b",
                    "task_name": "retrieval_passkey",
                    "prompt_length": 1024,
                    "exact_success": 1.0,
                    "quality_decode_ms_per_step": 120.0,
                    "systems_decode_ms_per_step": 40.0,
                    "systems_vs_quality_speedup": 3.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                },
                {
                    "model_key": "qwen35_4b",
                    "task_name": "instruction_constraints",
                    "prompt_length": 1024,
                    "exact_success": 1.0,
                    "quality_decode_ms_per_step": 90.0,
                    "systems_decode_ms_per_step": 30.0,
                    "systems_vs_quality_speedup": 3.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                },
                {
                    "model_key": "qwen35_27b",
                    "task_name": "reasoning_arithmetic",
                    "prompt_length": 2048,
                    "exact_success": 1.0,
                    "quality_decode_ms_per_step": 300.0,
                    "systems_decode_ms_per_step": 100.0,
                    "systems_vs_quality_speedup": 3.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                },
            ],
            "longbench_rows": [
                {
                    "model_key": "qwen35_9b",
                    "comparison_case": "exact",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 600.0,
                    "mean_qa_f1": 0.29,
                },
                {
                    "model_key": "qwen35_9b",
                    "comparison_case": "quality",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 610.0,
                    "mean_qa_f1": 0.29,
                },
                {
                    "model_key": "qwen35_9b",
                    "comparison_case": "systems",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 95.0,
                    "mean_qa_f1": 0.29,
                },
                {
                    "model_key": "qwen35_9b",
                    "comparison_case": "streaming_sink_recent",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 250.0,
                    "mean_qa_f1": 0.23,
                },
            ],
            "backend_rows": [
                {
                    "model_key": "qwen35_27b",
                    "prompt_length": 1024,
                    "speedups": {"learned_selector": {"vs_exact": 3.0, "vs_shortlist": 3.1}},
                    "variants": {
                        "exact": {"decode_ms_per_step": 500.0},
                        "shortlist_base": {"decode_ms_per_step": 520.0},
                        "learned_selector": {
                            "decode_ms_per_step": 170.0,
                            "m3_fraction": 0.995,
                            "score_ms_step": 44.0,
                            "mix_ms_step": 38.0,
                            "selector_us_per_invocation": 25.0,
                        },
                    },
                }
            ],
        },
        qwen_quality_payload={
            "comparisons": [
                {
                    "prompt_length": 1024,
                    "systems_vs_quality_decode_speedup": 1.5,
                    "variants": {
                        "quality": {
                            "decode_ms_per_step": 80.0,
                            "teacher_forced_token_agreement_rate": 1.0,
                            "teacher_forced_logit_rmse": 0.4,
                            "m3_fraction": 0.95,
                        },
                        "systems": {
                            "decode_ms_per_step": 40.0,
                            "teacher_forced_token_agreement_rate": 1.0,
                            "teacher_forced_logit_rmse": 0.3,
                            "m3_fraction": 0.99,
                        },
                    },
                }
            ]
        },
        llama_task_payload={
            "rows": [
                {
                    "task_name": "retrieval_passkey",
                    "prompt_length": 1024,
                    "quality_decode_ms_per_step": 60.0,
                    "systems_decode_ms_per_step": 60.0,
                    "systems_vs_quality_speedup": 1.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                }
            ]
        },
        qwen_9b_longbench_medium_payload={
            "rows": [
                {
                    "comparison_case": "exact",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 614.0,
                    "mean_exact_match": 0.167,
                    "mean_qa_f1": 0.270,
                    "mean_teacher_forced_perplexity_ratio": 1.012,
                },
                {
                    "comparison_case": "quality",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 574.0,
                    "mean_exact_match": 0.167,
                    "mean_qa_f1": 0.270,
                    "mean_teacher_forced_perplexity_ratio": 1.013,
                },
                {
                    "comparison_case": "systems",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 91.6,
                    "mean_exact_match": 0.167,
                    "mean_qa_f1": 0.270,
                    "mean_teacher_forced_perplexity_ratio": 1.012,
                },
                {
                    "comparison_case": "streaming_sink_recent",
                    "max_prompt_tokens": 4096,
                    "mean_decode_ms_per_step": 258.0,
                    "mean_exact_match": 0.167,
                    "mean_qa_f1": 0.270,
                    "mean_teacher_forced_perplexity_ratio": 1.307,
                },
            ]
        },
    )
    assert payload["promotion_calls"]["qwen"]["default_profile"] == "systems"
    assert payload["promotion_calls"]["llama"]["default_profile"] == "quality_or_systems_equivalent"
    assert "Qwen3.5 4B" in markdown
    assert "Qwen3.5 27B" in markdown
    assert "Promote systems as default" in markdown
    assert "Llama 3.2 3B" in markdown
    assert "Qwen LongBench External Check" in markdown
    assert "Qwen 9B LongBench Medium Check" in markdown
    assert "Qwen Backend Check" in markdown
