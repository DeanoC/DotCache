from __future__ import annotations

import json
from pathlib import Path

from scripts.report_persistent_full_attention_snapshot_compare import _build_report, _build_markdown


def _write_payload(path: Path, *, enable_priority: bool, recent: int, topk: int, selected_tokens: float, max_abs: float) -> None:
    payload = {
        "config": {
            "block_size": 16,
            "enable_priority": enable_priority,
            "sink_block_count": 1,
            "recent_block_count": recent,
            "exploration_blocks_per_region": 1,
            "optional_top_k": topk,
        },
        "summary": {
            "snapshot_count": 2,
            "max_abs_error": max_abs,
            "max_rel_error": 123.0,
            "avg_selected_block_count": 10.0,
            "avg_selected_token_count": selected_tokens,
            "avg_full_block_count": 20.0,
            "avg_full_token_count": 320.0,
            "avg_selected_fraction": selected_tokens / 320.0,
        },
        "records": [
            {
                "snapshot_path": "/tmp/corpus/readme/layer19_kv00_step+00.npz",
                "selected_token_count": int(selected_tokens),
                "full_token_count": 320,
                "max_abs_error": max_abs,
                "max_rel_error": 123.0,
            },
            {
                "snapshot_path": "/tmp/corpus/world/layer15_kv01_step+01.npz",
                "selected_token_count": int(selected_tokens),
                "full_token_count": 320,
                "max_abs_error": max_abs / 2.0,
                "max_rel_error": 10.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_selects_best_priority_and_most_aggressive(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    p1 = tmp_path / "p1.json"
    p2 = tmp_path / "p2.json"
    _write_payload(full, enable_priority=False, recent=1, topk=0, selected_tokens=320.0, max_abs=0.0)
    _write_payload(p1, enable_priority=True, recent=64, topk=128, selected_tokens=240.0, max_abs=0.1)
    _write_payload(p2, enable_priority=True, recent=8, topk=8, selected_tokens=64.0, max_abs=5.0)

    report = _build_report([full, p1, p2])

    assert report["best_priority_candidate"]["label"] == "priority_recent64_topk128_sink1_explore1"
    assert report["most_aggressive_candidate"]["label"] == "priority_recent8_topk8_sink1_explore1"
    assert report["cases"][0]["label"] == "full_coverage"


def test_build_markdown_includes_config_table_and_worst_slices(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    p1 = tmp_path / "p1.json"
    _write_payload(full, enable_priority=False, recent=1, topk=0, selected_tokens=320.0, max_abs=0.0)
    _write_payload(p1, enable_priority=True, recent=64, topk=128, selected_tokens=240.0, max_abs=0.1)

    report = _build_report([full, p1])
    markdown = _build_markdown("Persistent Report", report)

    assert "## Config Sweep" in markdown
    assert "priority_recent64_topk128_sink1_explore1" in markdown
    assert "readme/layer19_kv00_step+00.npz" in markdown
