from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_exact_value_signal_export import (
    _build_config,
    _summarize_rows,
)


def test_build_config_applies_signal_export_settings() -> None:
    config = _build_config(
        block_size=16,
        check_interval=4,
        mass_eps=1e-2,
        value_eps=2e-2,
        min_processed_blocks=8,
        order_mode="residual_proxy",
        key_centroid_count_by_layer={19: 8, 23: 16},
    )
    assert config.enable_priority is True
    assert config.enable_early_exit is True
    assert config.block_size == 16
    assert config.full_attention_check_interval == 4
    assert config.full_attention_mass_eps == 1e-2
    assert config.full_attention_value_eps == 2e-2
    assert config.full_attention_min_processed_blocks == 8
    assert config.full_attention_streaming_order_mode == "residual_proxy"
    assert config.full_attention_key_centroid_count_by_layer == {19: 8, 23: 16}


def test_summarize_rows_aggregates_topk_recall() -> None:
    rows = [
        {
            "snapshot_path": "a",
            "checkpoint_index": 0,
            "block_id": 0,
            "upper_bound": 4.0,
            "priority_score": 1.0,
            "proxy_score": 1.0,
            "value_upper_score": 3.0,
            "exact_value_score": 5.0,
        },
        {
            "snapshot_path": "a",
            "checkpoint_index": 0,
            "block_id": 1,
            "upper_bound": 3.0,
            "priority_score": 5.0,
            "proxy_score": 4.0,
            "value_upper_score": 2.0,
            "exact_value_score": 4.0,
        },
        {
            "snapshot_path": "a",
            "checkpoint_index": 0,
            "block_id": 2,
            "upper_bound": 1.0,
            "priority_score": 2.0,
            "proxy_score": 2.0,
            "value_upper_score": 5.0,
            "exact_value_score": 1.0,
        },
        {
            "snapshot_path": "b",
            "checkpoint_index": 0,
            "block_id": 0,
            "upper_bound": 5.0,
            "priority_score": 2.0,
            "proxy_score": 3.0,
            "value_upper_score": 1.0,
            "exact_value_score": 3.0,
        },
        {
            "snapshot_path": "b",
            "checkpoint_index": 0,
            "block_id": 1,
            "upper_bound": 2.0,
            "priority_score": 4.0,
            "proxy_score": 5.0,
            "value_upper_score": 4.0,
            "exact_value_score": 2.0,
        },
    ]
    summary = _summarize_rows(rows, top_k=1)
    assert summary["row_count"] == 5
    assert summary["checkpoint_count"] == 2
    assert summary["snapshot_count"] == 2
    assert summary["avg_rows_per_checkpoint"] == pytest.approx(2.5)
    assert summary["avg_topk_recall_upper_bound"] == pytest.approx(1.0)
    assert summary["avg_topk_recall_priority_score"] == pytest.approx(0.0)
    assert summary["avg_topk_recall_proxy_score"] == pytest.approx(0.0)
    assert summary["avg_topk_recall_value_upper_score"] == pytest.approx(0.0)
