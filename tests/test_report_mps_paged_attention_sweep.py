from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_mps_paged_attention_sweep import _build_report, _render_markdown


def test_build_report_summarizes_coverage_and_recommendations() -> None:
    payload = {
        "candidate_records": [
            {
                "engine": "mps_experimental",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "snapshot_name": "qwen35_attention_subset_prompt512_layer03_kv00_step00.npz",
                "snapshot_path": "/tmp/a.npz",
                "num_pages": 9,
                "tokens_per_page": 64,
                "total_tokens": 513,
                "total_step_time_ms": 1.8,
                "tokens_processed": 449.0,
                "processed_page_count": 8.0,
                "max_abs_error": 1e-5,
                "max_rel_error": 1e-4,
            },
            {
                "engine": "mps_experimental",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "snapshot_name": "qwen35_attention_subset_prompt2048_layer11_kv01_step00.npz",
                "snapshot_path": "/tmp/b.npz",
                "num_pages": 33,
                "tokens_per_page": 64,
                "total_tokens": 2049,
                "total_step_time_ms": 2.0,
                "tokens_processed": 449.0,
                "processed_page_count": 8.0,
                "max_abs_error": 1.2e-5,
                "max_rel_error": 1.1e-4,
            },
            {
                "engine": "torch_mps_baseline",
                "config_key": "topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001",
                "snapshot_name": "qwen35_attention_subset_prompt512_layer03_kv00_step00.npz",
                "snapshot_path": "/tmp/a.npz",
                "num_pages": 9,
                "tokens_per_page": 64,
                "total_tokens": 513,
                "total_step_time_ms": 2.1,
                "tokens_processed": 663.0,
                "processed_page_count": 11.0,
                "max_abs_error": 1e-5,
                "max_rel_error": 1e-4,
            },
            {
                "engine": "torch_mps_baseline",
                "config_key": "topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001",
                "snapshot_name": "qwen35_attention_subset_prompt2048_layer11_kv01_step00.npz",
                "snapshot_path": "/tmp/b.npz",
                "num_pages": 33,
                "tokens_per_page": 64,
                "total_tokens": 2049,
                "total_step_time_ms": 2.3,
                "tokens_processed": 663.0,
                "processed_page_count": 11.0,
                "max_abs_error": 1.1e-5,
                "max_rel_error": 1.1e-4,
            },
        ],
        "aggregate_records": [
            {
                "engine": "mps_experimental",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 1.9,
                "avg_tokens_processed": 449.0,
                "avg_processed_page_count": 8.0,
                "max_abs_error": 1.2e-5,
                "max_rel_error": 1.1e-4,
            },
            {
                "engine": "torch_mps_baseline",
                "config_key": "topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 2.2,
                "avg_tokens_processed": 663.0,
                "avg_processed_page_count": 11.0,
                "max_abs_error": 1.1e-5,
                "max_rel_error": 1.1e-4,
            },
            {
                "engine": "torch_mps_baseline",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 2.1,
                "avg_tokens_processed": 449.0,
                "avg_processed_page_count": 8.0,
                "max_abs_error": 1.2e-5,
                "max_rel_error": 1.1e-4,
            },
            {
                "engine": "mps_experimental",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 1.9,
                "avg_tokens_processed": 449.0,
                "avg_processed_page_count": 8.0,
                "max_abs_error": 1.2e-5,
                "max_rel_error": 1.1e-4,
            },
        ],
        "recommendation_records": [
            {
                "engine": "mps_experimental",
                "config_key": "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 1.9,
                "avg_tokens_processed": 449.0,
                "avg_processed_page_count": 8.0,
                "avg_score_time_ms": 0.4,
                "avg_selection_time_ms": 0.2,
                "avg_attention_time_ms": 1.3,
                "pass_count": 2,
                "snapshot_count": 2,
                "pass_rate": 1.0,
                "max_abs_error": 1.2e-5,
                "max_rel_error": 1.1e-4,
                "record_type": "recommendation",
                "recommendation_reason": "fastest_full_pass",
            },
            {
                "engine": "torch_mps_baseline",
                "config_key": "topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001",
                "avg_total_step_time_ms": 2.2,
                "avg_tokens_processed": 663.0,
                "avg_processed_page_count": 11.0,
                "avg_score_time_ms": 0.4,
                "avg_selection_time_ms": 0.2,
                "avg_attention_time_ms": 1.5,
                "pass_count": 2,
                "snapshot_count": 2,
                "pass_rate": 1.0,
                "max_abs_error": 1.1e-5,
                "max_rel_error": 1.1e-4,
                "record_type": "recommendation",
                "recommendation_reason": "fastest_full_pass",
            },
        ],
    }

    report = _build_report(payload, title="Test Report", input_path="/tmp/input.json")

    assert report["coverage"]["snapshot_count"] == 2
    assert report["coverage"]["prompt_lengths"] == [512, 2048]
    assert report["coverage"]["layer_ids"] == [3, 11]
    assert report["coverage"]["kv_head_ids"] == [0, 1]
    assert report["recommendation_comparison"]["speedup_ratio"] > 1.0
    assert report["matched_speedups"][0]["config_key"] == "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001"

    markdown = _render_markdown(report)
    assert "## Coverage" in markdown
    assert "mps_experimental" in markdown
    assert "topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001" in markdown
