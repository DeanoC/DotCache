from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_certified_snapshot_compare import (
    _build_variant_config,
    _summarize_variant_records,
)


def test_build_variant_config_applies_conservative_bound_modes() -> None:
    default = _build_variant_config(
        variant="default",
        block_size=16,
        check_interval=4,
        mass_eps=1e-3,
        value_eps=2e-3,
        min_processed_blocks=2,
    )
    region_caps = _build_variant_config(
        variant="region_caps",
        block_size=16,
        check_interval=4,
        mass_eps=1e-3,
        value_eps=2e-3,
        min_processed_blocks=2,
    )
    cluster8 = _build_variant_config(
        variant="cluster8",
        block_size=16,
        check_interval=4,
        mass_eps=1e-3,
        value_eps=2e-3,
        min_processed_blocks=2,
    )

    assert default.enable_early_exit is True
    assert default.full_attention_check_interval == 4
    assert default.full_attention_mass_eps == 1e-3
    assert default.full_attention_value_eps == 2e-3
    assert default.full_attention_min_processed_blocks == 2
    assert default.full_attention_region_residual_caps is False
    assert default.full_attention_residual_cluster_count == 0
    assert region_caps.full_attention_region_residual_caps is True
    assert cluster8.full_attention_residual_cluster_count == 8


def test_summarize_variant_records_aggregates_first_stop_metrics() -> None:
    summary = _summarize_variant_records(
        [
            {
                "streaming_processed_block_count": 10,
                "streaming_processed_token_count": 160,
                "streaming_checkpoint_count": 5,
                "streaming_first_certified_stop_block_count": 8,
                "streaming_first_certified_stop_token_count": 128,
                "streaming_first_true_certified_stop_block_count": 6,
                "streaming_first_true_certified_stop_token_count": 96,
                "streaming_truth_max_beta_over_true_beta_ratio": 12.0,
                "streaming_truth_max_delta_over_true_delta_ratio": 8.0,
                "streaming_truth_min_stop_ratio": 0.8,
                "streaming_truth_bound_stop_ratio_at_true_frontier": 11.0,
                "streaming_truth_frontier_block_count": 6,
                "streaming_truth_min_nonterminal_stop_ratio": 0.8,
                "streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": 11.0,
                "streaming_truth_nonterminal_frontier_block_count": 6,
                "streaming_truth_nonterminal_frontier_true_beta_ratio": 0.7,
                "streaming_truth_nonterminal_frontier_true_delta_ratio": 0.8,
                "streaming_truth_nonterminal_frontier_bound_beta_ratio": 10.0,
                "streaming_truth_nonterminal_frontier_bound_delta_ratio": 11.0,
                "streaming_max_abs_error": 0.01,
                "full_block_count": 20,
                "full_token_count": 320,
            },
            {
                "streaming_processed_block_count": 12,
                "streaming_processed_token_count": 192,
                "streaming_checkpoint_count": 6,
                "streaming_first_certified_stop_block_count": None,
                "streaming_first_certified_stop_token_count": None,
                "streaming_first_true_certified_stop_block_count": None,
                "streaming_first_true_certified_stop_token_count": None,
                "streaming_truth_max_beta_over_true_beta_ratio": 20.0,
                "streaming_truth_max_delta_over_true_delta_ratio": 10.0,
                "streaming_truth_min_stop_ratio": 1.6,
                "streaming_truth_bound_stop_ratio_at_true_frontier": 14.0,
                "streaming_truth_frontier_block_count": 12,
                "streaming_truth_min_nonterminal_stop_ratio": 1.6,
                "streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": 14.0,
                "streaming_truth_nonterminal_frontier_block_count": 12,
                "streaming_truth_nonterminal_frontier_true_beta_ratio": 1.6,
                "streaming_truth_nonterminal_frontier_true_delta_ratio": 1.2,
                "streaming_truth_nonterminal_frontier_bound_beta_ratio": 14.0,
                "streaming_truth_nonterminal_frontier_bound_delta_ratio": 9.0,
                "streaming_max_abs_error": 0.02,
                "full_block_count": 20,
                "full_token_count": 320,
            },
        ]
    )

    assert summary["snapshot_count"] == 2
    assert summary["streaming_certified_stop_rate"] == 0.5
    assert summary["streaming_true_certified_stop_rate"] == 0.5
    assert summary["avg_streaming_processed_block_count"] == 11.0
    assert summary["avg_streaming_first_stop_block_count"] == 8.0
    assert summary["avg_streaming_first_true_stop_block_count"] == 6.0
    assert summary["avg_streaming_checkpoint_count"] == 5.5
    assert summary["avg_streaming_first_stop_fraction"] == 0.4
    assert summary["avg_streaming_first_true_stop_fraction"] == 0.3
    assert summary["avg_streaming_processed_fraction"] == 0.55
    assert summary["avg_streaming_truth_max_beta_over_true_beta_ratio"] == 16.0
    assert summary["avg_streaming_truth_max_delta_over_true_delta_ratio"] == 9.0
    assert summary["max_streaming_truth_max_beta_over_true_beta_ratio"] == 20.0
    assert summary["max_streaming_truth_max_delta_over_true_delta_ratio"] == 10.0
    assert summary["avg_streaming_truth_min_stop_ratio"] == pytest.approx(1.2)
    assert summary["avg_streaming_truth_bound_stop_ratio_at_true_frontier"] == 12.5
    assert summary["avg_streaming_truth_frontier_block_fraction"] == pytest.approx(0.45)
    assert summary["avg_streaming_truth_min_nonterminal_stop_ratio"] == pytest.approx(1.2)
    assert summary["avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier"] == 12.5
    assert summary["avg_streaming_truth_nonterminal_frontier_block_fraction"] == pytest.approx(0.45)
    assert summary["avg_streaming_truth_nonterminal_frontier_true_beta_ratio"] == pytest.approx(1.15)
    assert summary["avg_streaming_truth_nonterminal_frontier_true_delta_ratio"] == pytest.approx(1.0)
    assert summary["avg_streaming_truth_nonterminal_frontier_bound_beta_ratio"] == pytest.approx(12.0)
    assert summary["avg_streaming_truth_nonterminal_frontier_bound_delta_ratio"] == pytest.approx(10.0)
    assert summary["streaming_truth_nonterminal_frontier_true_beta_dominant_rate"] == pytest.approx(0.5)
    assert summary["streaming_truth_nonterminal_frontier_true_delta_dominant_rate"] == pytest.approx(0.5)
    assert summary["streaming_truth_nonterminal_frontier_bound_beta_dominant_rate"] == pytest.approx(0.5)
    assert summary["streaming_truth_nonterminal_frontier_bound_delta_dominant_rate"] == pytest.approx(0.5)
    assert summary["max_streaming_max_abs_error"] == 0.02
