from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_certified_ordering_compare import (
    _build_ordering_config,
    _variant_score,
)


def test_build_ordering_config_applies_interval_and_ordering_mode() -> None:
    upper_ci16 = _build_ordering_config(
        variant="upper_ci16",
        block_size=16,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )
    priority_ci8 = _build_ordering_config(
        variant="priority_ci8",
        block_size=16,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )
    hybrid_ci4 = _build_ordering_config(
        variant="hybrid_ci4",
        block_size=16,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )
    proxy_ci16 = _build_ordering_config(
        variant="proxy_ci16",
        block_size=16,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )
    proxy_refineall_ci16 = _build_ordering_config(
        variant="proxy_refineall_ci16",
        block_size=16,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
    )

    assert upper_ci16.enable_priority is True
    assert upper_ci16.full_attention_check_interval == 16
    assert upper_ci16.full_attention_optional_use_upper_bounds_first is True
    assert upper_ci16.full_attention_mass_eps == 1e-2
    assert upper_ci16.full_attention_value_eps == 2e-2
    assert upper_ci16.full_attention_min_processed_blocks == 8

    assert priority_ci8.enable_priority is True
    assert priority_ci8.full_attention_check_interval == 8
    assert priority_ci8.full_attention_optional_use_upper_bounds_first is False
    assert priority_ci8.full_attention_streaming_order_mode == "shortlist"
    assert hybrid_ci4.full_attention_check_interval == 4
    assert hybrid_ci4.full_attention_streaming_order_mode == "priority_value_hybrid"
    assert hybrid_ci4.full_attention_streaming_priority_value_upper_weight == 0.25
    assert proxy_ci16.full_attention_check_interval == 16
    assert proxy_ci16.full_attention_streaming_order_mode == "residual_proxy"
    assert proxy_ci16.full_attention_refine_top_k == 0
    assert proxy_refineall_ci16.full_attention_refine_top_k == 100000


def test_variant_score_prefers_better_oracle_frontier_summary() -> None:
    better = {
        "avg_streaming_truth_min_nonterminal_stop_ratio": 5.0,
        "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.80,
        "avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": 40.0,
    }
    worse = {
        "avg_streaming_truth_min_nonterminal_stop_ratio": 7.0,
        "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.90,
        "avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": 200.0,
    }

    assert _variant_score(better) < _variant_score(worse)
