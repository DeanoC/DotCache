from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_paged_attention_replay_corpus import _build_summary, _case_tag


def test_case_tag_uses_prompt_mode_fields() -> None:
    assert _case_tag({"repeat_count": 7}) == "repeat0007"
    assert _case_tag({"prompt_file_label": "aae_dotcache_spec"}) == "aae_dotcache_spec"
    assert _case_tag({"prompt_length": 1024}) == "prompt01024"


def test_build_summary_counts_cases_and_snapshot_axes() -> None:
    summary = _build_summary(
        [
            {
                "status": "ok",
                "prompt_mode": "exact_length",
                "paged_attention_snapshot_corpus_count": 12,
                "paged_attention_snapshot_corpus_layer_ids": [3, 7],
                "paged_attention_snapshot_corpus_kv_head_ids": [0, 1],
                "paged_attention_snapshot_corpus_resolved_step_indices": [0, 3],
            },
            {
                "status": "ok",
                "prompt_mode": "repeat_count",
                "paged_attention_snapshot_corpus_count": 6,
                "paged_attention_snapshot_corpus_layer_ids": [11],
                "paged_attention_snapshot_corpus_kv_head_ids": [1],
                "paged_attention_snapshot_corpus_resolved_step_indices": [0],
            },
            {
                "status": "error",
                "prompt_mode": "exact_length",
            },
            {
                "status": "ok",
                "prompt_mode": "prompt_file",
                "paged_attention_snapshot_corpus_count": 4,
                "paged_attention_snapshot_corpus_layer_ids": [15],
                "paged_attention_snapshot_corpus_kv_head_ids": [0],
                "paged_attention_snapshot_corpus_resolved_step_indices": [1],
            },
        ]
    )

    assert summary["case_count"] == 4
    assert summary["success_case_count"] == 3
    assert summary["error_case_count"] == 1
    assert summary["snapshot_count"] == 22
    assert summary["counts_by_prompt_mode"] == {"exact_length": 1, "prompt_file": 1, "repeat_count": 1}
    assert summary["layer_ids"] == [3, 7, 11, 15]
    assert summary["kv_head_ids"] == [0, 1]
    assert summary["resolved_step_indices"] == [0, 1, 3]
