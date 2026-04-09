from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_mps_paged_attention_family_compare import _build_report, _render_markdown


def test_build_report_summarizes_family_compare() -> None:
    compare_rows = [
        {
            "label": "experimental_approx_8_128_c8",
            "engine": "mps_experimental",
            "avg_total_step_time_ms": 43.6,
            "avg_tokens_processed": 242.3,
            "avg_selected_page_count": 6.9,
            "avg_processed_page_count": 6.9,
            "pass_rate": 1.0,
            "max_abs_error": 4.2e-6,
            "max_rel_error": 7.1e-4,
            "count": 108,
        },
        {
            "label": "baseline_approx_8_128_c2",
            "engine": "torch_mps_baseline",
            "avg_total_step_time_ms": 45.5,
            "avg_tokens_processed": 242.3,
            "avg_selected_page_count": 6.9,
            "avg_processed_page_count": 6.9,
            "pass_rate": 1.0,
            "max_abs_error": 4.2e-6,
            "max_rel_error": 7.1e-4,
            "count": 108,
        },
        {
            "label": "baseline_approx_8_64_c2",
            "engine": "torch_mps_baseline",
            "avg_total_step_time_ms": 39.9,
            "avg_tokens_processed": 178.3,
            "avg_selected_page_count": 5.9,
            "avg_processed_page_count": 5.9,
            "pass_rate": 0.9907,
            "max_abs_error": 4.0e-6,
            "max_rel_error": 7.7e-3,
            "count": 108,
        },
    ]
    corpus_summary = {
        "case_count": 3,
        "counts_by_prompt_mode": {"prompt_file": 3},
        "snapshot_count": 108,
        "layer_ids": [3, 7, 11, 15, 19, 23],
        "kv_head_ids": [0, 1],
        "resolved_step_indices": [0, 1, 3],
    }
    corpus_manifest = {
        "output_dir": "/tmp/corpus",
        "records": [
            {
                "prompt_mode": "prompt_file",
                "case_tag": "readme",
                "prompt_length": 4096,
                "prompt_file_path": "/tmp/readme.md",
                "prefill_ms": 4200.0,
                "dense_decode_ms_per_step": 122.0,
                "decode_steps": 4,
                "paged_attention_snapshot_corpus_count": 36,
                "paged_attention_snapshot_corpus_resolved_step_indices": [0, 1, 3],
            }
        ],
    }

    report = _build_report(
        compare_rows,
        corpus_summary=corpus_summary,
        corpus_manifest=corpus_manifest,
        title="Test Family Report",
        compare_input="/tmp/family.json",
    )

    assert report["coverage"]["snapshot_count"] == 108
    assert report["normalized_comparison"]["avg_dense_decode_ms_per_step"] == 122.0
    assert report["recommendations"]["best_fully_passing_overall"]["label"] == "experimental_approx_8_128_c8"
    assert report["recommendations"]["best_near_perfect_overall"]["label"] == "experimental_approx_8_128_c8"
    assert report["recommendations"]["best_fast_tradeoff_overall"]["label"] == "baseline_approx_8_64_c2"
    assert report["matched_families"][0]["family_key"] == "approx|topk=8|recent=128"

    markdown = _render_markdown(report)
    assert "## Coverage" in markdown
    assert "## Normalized Comparison" in markdown
    assert "experimental_approx_8_128_c8" in markdown
    assert "baseline_approx_8_64_c2" in markdown
    assert "Approx Budget" in markdown
