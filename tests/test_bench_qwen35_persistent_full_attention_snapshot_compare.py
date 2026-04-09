from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import _build_summary, _resolve_snapshot_paths


def test_resolve_snapshot_paths_collects_manifest_and_explicit_paths(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.write_bytes(b"")
    second.write_bytes(b"")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_records": [
                    {"paged_attention_snapshot_path": str(first)},
                    {"paged_attention_snapshot_path": str(second)},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_snapshot_paths(
        snapshot_paths=[str(first)],
        snapshot_glob=None,
        manifest_path=str(manifest),
    )

    assert resolved == [first.resolve(), second.resolve()]


def test_build_summary_aggregates_snapshot_compare_records() -> None:
    summary = _build_summary(
        [
            {
                "selected_block_count": 4,
                "selected_token_count": 64,
                "full_block_count": 8,
                "full_token_count": 128,
                "max_abs_error": 0.01,
                "max_rel_error": 0.1,
            },
            {
                "selected_block_count": 2,
                "selected_token_count": 32,
                "full_block_count": 8,
                "full_token_count": 128,
                "max_abs_error": 0.02,
                "max_rel_error": 0.2,
            },
        ]
    )

    assert summary["snapshot_count"] == 2
    assert summary["max_abs_error"] == 0.02
    assert summary["max_rel_error"] == 0.2
    assert summary["avg_selected_block_count"] == 3.0
    assert summary["avg_selected_token_count"] == 48.0
    assert summary["avg_selected_fraction"] == 0.375
