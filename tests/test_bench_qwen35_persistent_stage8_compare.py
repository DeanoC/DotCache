from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_stage8_compare import (
    _resolve_prompt_records_from_corpus_manifest,
    _resolve_snapshot_records_from_corpus_manifest,
    _summarize_replay_pair_records,
    _summarize_serving_pair_records,
)


def test_resolve_prompt_records_from_corpus_manifest_reads_prompt_paths(tmp_path) -> None:
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("alpha", encoding="utf-8")
    prompt_b.write_text("beta", encoding="utf-8")
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"case_tag": "alpha_case", "prompt_file_path": str(prompt_a), "prompt_length": 128},
                    {"case_tag": "beta_case", "prompt_file_path": str(prompt_b), "prompt_length": 64},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _resolve_prompt_records_from_corpus_manifest(str(manifest))

    assert [record["case_tag"] for record in records] == ["alpha_case", "beta_case"]
    assert [record["prompt_length"] for record in records] == [128, 64]


def test_resolve_snapshot_records_from_corpus_manifest_expands_child_manifests(tmp_path) -> None:
    snap_a = tmp_path / "a.npz"
    snap_b = tmp_path / "b.npz"
    snap_a.write_bytes(b"")
    snap_b.write_bytes(b"")
    child_manifest = tmp_path / "child_manifest.json"
    child_manifest.write_text(
        json.dumps(
            {
                "snapshot_records": [
                    {
                        "paged_attention_snapshot_path": str(snap_a),
                        "paged_attention_snapshot_layer_id": 3,
                        "paged_attention_snapshot_kv_head_id": 1,
                        "paged_attention_snapshot_step_index": 0,
                    },
                    {
                        "paged_attention_snapshot_path": str(snap_b),
                        "paged_attention_snapshot_layer_id": 7,
                        "paged_attention_snapshot_kv_head_id": 0,
                        "paged_attention_snapshot_step_index": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_tag": "paper",
                        "paged_attention_snapshot_corpus_manifest_path": str(child_manifest),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _resolve_snapshot_records_from_corpus_manifest(str(manifest))

    assert len(records) == 2
    assert records[0]["case_tag"] == "paper"
    assert records[0]["layer_id"] == 3
    assert records[1]["kv_head_id"] == 0


def test_summarize_replay_pair_records_aggregates_stage8_metrics() -> None:
    summary = _summarize_replay_pair_records(
        [
            {
                "selection_changed": True,
                "baseline_max_abs_error": 0.10,
                "stage8_max_abs_error": 0.08,
                "baseline_selected_token_count": 64,
                "stage8_selected_token_count": 64,
                "stage8_selected_m0_metadata_block_count": 2,
                "stage8_compression_invalid_block_count": 0,
                "stage8_metadata_m0_block_count": 3,
            },
            {
                "selection_changed": False,
                "baseline_max_abs_error": 0.20,
                "stage8_max_abs_error": 0.15,
                "baseline_selected_token_count": 32,
                "stage8_selected_token_count": 48,
                "stage8_selected_m0_metadata_block_count": 4,
                "stage8_compression_invalid_block_count": 1,
                "stage8_metadata_m0_block_count": 5,
            },
        ]
    )

    assert summary["snapshot_count"] == 2
    assert summary["selection_changed_rate"] == 0.5
    assert summary["baseline_avg_max_abs_error"] == pytest.approx(0.15)
    assert summary["stage8_avg_selected_m0_metadata_block_count"] == 3.0
    assert summary["stage8_avg_compression_invalid_block_count"] == 0.5


def test_summarize_serving_pair_records_aggregates_latency_and_fallbacks() -> None:
    summary = _summarize_serving_pair_records(
        [
            {
                "dense_decode_ms_per_step": 10.0,
                "baseline_decode_ms_per_step": 8.0,
                "stage8_decode_ms_per_step": 7.0,
                "baseline_matches_dense_exact": True,
                "stage8_matches_dense_exact": True,
                "stage8_matches_baseline_exact": True,
                "stage8_selected_m0_metadata_block_count_total": 12,
                "stage8_dense_fallback_count_total": 0,
                "stage8_compression_rerank_count_total": 1,
            },
            {
                "dense_decode_ms_per_step": 12.0,
                "baseline_decode_ms_per_step": 9.0,
                "stage8_decode_ms_per_step": 11.0,
                "baseline_matches_dense_exact": False,
                "stage8_matches_dense_exact": True,
                "stage8_matches_baseline_exact": False,
                "stage8_selected_m0_metadata_block_count_total": 8,
                "stage8_dense_fallback_count_total": 2,
                "stage8_compression_rerank_count_total": 3,
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["baseline_avg_ms_per_step"] == 8.5
    assert summary["stage8_avg_ms_per_step"] == 9.0
    assert summary["stage8_beats_baseline_latency_rate"] == 0.5
    assert summary["stage8_matches_dense_exact_rate"] == 1.0
    assert summary["stage8_avg_selected_m0_metadata_block_count"] == 10.0
    assert summary["stage8_avg_dense_fallback_count"] == 1.0
