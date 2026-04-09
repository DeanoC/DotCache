from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_mps_paged_attention_snapshot_sweep import (
    _aggregate_records,
    _config_records,
    _recommend_records,
    _resolve_snapshot_paths,
)
from dotcache.backends.mps_persistent_experimental import build_synthetic_snapshot, save_paged_attention_snapshot


def test_resolve_snapshot_paths_accepts_paths_and_globs(tmp_path: Path) -> None:
    first = tmp_path / "one.npz"
    second = tmp_path / "two.npz"
    save_paged_attention_snapshot(first, build_synthetic_snapshot(num_pages=2, tokens_per_page=4, head_dim=8, seed=1))
    save_paged_attention_snapshot(second, build_synthetic_snapshot(num_pages=2, tokens_per_page=4, head_dim=8, seed=2))

    resolved = _resolve_snapshot_paths([str(first)], [str(tmp_path / "*.npz")])

    assert resolved == [first.resolve(), second.resolve()]


def test_config_records_builds_unique_grid() -> None:
    args = SimpleNamespace(
        top_ks=[4],
        recent_windows=[256],
        sink_windows=[128],
        page_chunk_sizes=[2],
        early_exit_modes=["off", "on"],
        early_exit_epsilons=[1e-4, 1e-3],
    )

    configs = _config_records(args)

    assert len(configs) == 3
    assert any(config.early_exit is False for config in configs)
    assert sum(1 for config in configs if config.early_exit) == 2


def test_aggregate_and_recommend_records_choose_fast_full_pass() -> None:
    candidate_records = [
        {
            "record_type": "candidate",
            "engine": "mps_experimental",
            "config_key": "fast",
            "snapshot_path": "/tmp/a.npz",
            "total_step_time_ms": 2.0,
            "score_time_ms": 0.5,
            "selection_time_ms": 0.5,
            "attention_time_ms": 1.0,
            "selected_page_count": 4.0,
            "processed_page_count": 4.0,
            "tokens_processed": 64.0,
            "early_exit_rate": 0.0,
            "max_abs_error": 1e-5,
            "max_rel_error": 1e-5,
        },
        {
            "record_type": "candidate",
            "engine": "mps_experimental",
            "config_key": "fast",
            "snapshot_path": "/tmp/b.npz",
            "total_step_time_ms": 3.0,
            "score_time_ms": 0.75,
            "selection_time_ms": 0.75,
            "attention_time_ms": 1.5,
            "selected_page_count": 4.0,
            "processed_page_count": 4.0,
            "tokens_processed": 64.0,
            "early_exit_rate": 0.0,
            "max_abs_error": 2e-5,
            "max_rel_error": 2e-5,
        },
        {
            "record_type": "candidate",
            "engine": "mps_experimental",
            "config_key": "slower",
            "snapshot_path": "/tmp/a.npz",
            "total_step_time_ms": 5.0,
            "score_time_ms": 1.0,
            "selection_time_ms": 1.0,
            "attention_time_ms": 3.0,
            "selected_page_count": 8.0,
            "processed_page_count": 8.0,
            "tokens_processed": 128.0,
            "early_exit_rate": 0.0,
            "max_abs_error": 1e-6,
            "max_rel_error": 1e-6,
        },
        {
            "record_type": "candidate",
            "engine": "mps_experimental",
            "config_key": "slower",
            "snapshot_path": "/tmp/b.npz",
            "total_step_time_ms": 5.5,
            "score_time_ms": 1.1,
            "selection_time_ms": 1.1,
            "attention_time_ms": 3.3,
            "selected_page_count": 8.0,
            "processed_page_count": 8.0,
            "tokens_processed": 128.0,
            "early_exit_rate": 0.0,
            "max_abs_error": 1e-6,
            "max_rel_error": 1e-6,
        },
    ]

    aggregates = _aggregate_records(candidate_records, max_abs_error_threshold=1e-3, max_rel_error_threshold=1e-3)
    recommendations = _recommend_records(aggregates)

    assert len(aggregates) == 2
    assert recommendations[0]["config_key"] == "fast"
    assert recommendations[0]["recommendation_reason"] == "fastest_full_pass"
