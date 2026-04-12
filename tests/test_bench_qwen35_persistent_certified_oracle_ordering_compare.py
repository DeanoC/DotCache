from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_certified_oracle_ordering_compare import (
    _build_config,
    _summarize_variant_records,
    _variant_score,
)


def test_build_config_applies_conservative_oracle_order_defaults() -> None:
    config = _build_config(
        block_size=16,
        check_interval=8,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )
    assert config.enable_priority is True
    assert config.enable_early_exit is True
    assert config.block_size == 16
    assert config.full_attention_check_interval == 8
    assert config.full_attention_mass_eps == 1e-2
    assert config.full_attention_value_eps == 2e-2
    assert config.full_attention_min_processed_blocks == 8


def test_summarize_variant_records_aggregates_oracle_frontier_metrics() -> None:
    summary = _summarize_variant_records(
        [
            {
                "streaming_first_true_certified_stop_block_count": 6,
                "streaming_first_true_certified_stop_token_count": 96,
                "streaming_truth_min_nonterminal_stop_ratio": 0.8,
                "streaming_truth_nonterminal_frontier_block_count": 6,
                "full_block_count": 20,
                "full_token_count": 320,
            },
            {
                "streaming_first_true_certified_stop_block_count": 8,
                "streaming_first_true_certified_stop_token_count": 128,
                "streaming_truth_min_nonterminal_stop_ratio": 1.6,
                "streaming_truth_nonterminal_frontier_block_count": 10,
                "full_block_count": 20,
                "full_token_count": 320,
            },
        ]
    )
    assert summary["snapshot_count"] == 2
    assert summary["avg_streaming_first_true_stop_fraction"] == pytest.approx(0.35)
    assert summary["avg_streaming_truth_min_nonterminal_stop_ratio"] == pytest.approx(1.2)
    assert summary["avg_streaming_truth_nonterminal_frontier_block_fraction"] == pytest.approx(0.4)


def test_variant_score_prefers_lower_oracle_frontier_metrics() -> None:
    better = {
        "avg_streaming_truth_min_nonterminal_stop_ratio": 3.0,
        "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.8,
    }
    worse = {
        "avg_streaming_truth_min_nonterminal_stop_ratio": 5.0,
        "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.9,
    }
    assert _variant_score(better) < _variant_score(worse)
