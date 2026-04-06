from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_qwen35_cuda_longbench_hotpot_diagnostic import _render


def test_render_includes_case_and_layer_sections() -> None:
    rows = [
        {
            "evaluation_prompt_id": "hotpot_case_order",
            "longbench_dataset": "hotpotqa",
            "longbench_row_index": 0,
            "runner_case": "exact",
            "longbench_qa_f1_max": 0.375,
            "longbench_answer_exact_match": False,
            "dotcache_decode_ms_per_step": 743.05,
            "longbench_generated_text": "Miller v. California",
        },
        {
            "evaluation_prompt_id": "hotpot_case_order",
            "longbench_dataset": "hotpotqa",
            "longbench_row_index": 0,
            "runner_case": "shortlist_base",
            "longbench_qa_f1_max": 0.125,
            "longbench_answer_exact_match": False,
            "dotcache_decode_ms_per_step": 174.91,
            "longbench_generated_text": "Gates v. Collier",
            "scorer_worst_layer_id": "11",
            "scorer_exact_top_recall_mean_by_layer": {"23": 0.7},
            "scorer_rank_correlation_mean_by_layer": {"23": 0.95},
            "scorer_layer_records": [
                {
                    "layer_id": 23,
                    "groups": [
                        {
                            "exact_top_recall": 0.7,
                            "approx_exact_top_recall": 0.7,
                            "score_rank_correlation": 0.95,
                            "score_value_correlation": 0.96,
                            "missed_exact_page_ranges": [
                                {"token_start": 31056, "token_end": 31072},
                                {"token_start": 31056, "token_end": 31072},
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    markdown = _render(rows, prompt_id="hotpot_case_order", dataset="hotpotqa", top_missed_pages=4)

    assert "## Case Summary" in markdown
    assert "shortlist_base" in markdown
    assert "## shortlist_base Layers" in markdown
    assert "`31056:31072` missed `2` times" in markdown
