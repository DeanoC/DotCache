from __future__ import annotations

import json

import numpy as np

from dotcache.integrations.llama import LlamaReplayRecord
from dotcache.integrations.qwen35 import (
    build_attention_subset_page_trace_records,
    build_attention_subset_prefill_page_trace_records,
    export_attention_subset_page_traces,
)
from dotcache.page_oracle import load_page_trace


def _make_record(*, step_index: int, token_index: int) -> LlamaReplayRecord:
    query_states = np.array(
        [
            [1.0 + step_index, 0.0],
            [3.0 + step_index, 0.0],
            [0.0, 2.0 + step_index],
            [0.0, 4.0 + step_index],
        ],
        dtype=np.float32,
    )
    key_states = np.array(
        [
            [10.0 + step_index, 11.0 + step_index],
            [20.0 + step_index, 21.0 + step_index],
        ],
        dtype=np.float32,
    )
    value_states = key_states + 100.0
    return LlamaReplayRecord(
        step_index=step_index,
        layer_id=3,
        token_index=token_index,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        context_states=np.zeros((8,), dtype=np.float32),
        output_states=np.zeros((8,), dtype=np.float32),
    )


def test_build_attention_subset_page_trace_records_groups_streams_into_pages() -> None:
    per_step_records = [[_make_record(step_index=0, token_index=100)], [_make_record(step_index=1, token_index=101)], [_make_record(step_index=2, token_index=102)]]
    page_traces = build_attention_subset_page_trace_records(
        per_step_records,
        q_head_to_kv_head=np.array([0, 0, 1, 1], dtype=np.int32),
        tokens_per_page=2,
        kinds=("K", "V"),
    )

    assert len(page_traces) == 8
    first_trace = next(trace for trace in page_traces if trace.kind == "K" and trace.kv_head_id == 0 and trace.token_start == 100)
    assert first_trace.values.shape == (2, 2)
    assert first_trace.token_age == 1
    assert np.allclose(first_trace.query, np.array([2.5, 0.0], dtype=np.float32))


def test_export_attention_subset_page_traces_writes_manifest_and_npz_files(tmp_path) -> None:
    per_step_records = [[_make_record(step_index=0, token_index=100)], [_make_record(step_index=1, token_index=101)]]
    prefill_tensors = {
        3: (
            np.array(
                [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
                dtype=np.float32,
            ),
            np.array(
                [[[[11.0, 12.0], [13.0, 14.0]], [[15.0, 16.0], [17.0, 18.0]]]],
                dtype=np.float32,
            ),
        )
    }
    manifest = export_attention_subset_page_traces(
        per_step_records,
        q_head_to_kv_head=np.array([0, 0, 1, 1], dtype=np.int32),
        output_dir=tmp_path,
        tokens_per_page=2,
        kinds=("K",),
        prefill_tensors=prefill_tensors,
        prefill_token_count=2,
    )

    assert manifest["page_trace_count"] == 4
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["page_trace_count"] == 4
    assert loaded_manifest["page_trace_counts_by_stage"] == {"decode": 2, "prefill": 2}

    first_trace_path = manifest["page_trace_paths"][0]
    loaded_trace = load_page_trace(first_trace_path)
    assert loaded_trace.kind == "K"
    assert loaded_trace.layer_id == 3
    assert loaded_trace.values.shape == (2, 2)


def test_build_attention_subset_prefill_page_trace_records_emits_queryless_pages() -> None:
    prefill_tensors = {
        3: (
            np.array(
                [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]],
                dtype=np.float32,
            ),
            np.array(
                [[[[21.0, 22.0], [23.0, 24.0], [25.0, 26.0]], [[27.0, 28.0], [29.0, 30.0], [31.0, 32.0]]]],
                dtype=np.float32,
            ),
        )
    }

    traces = build_attention_subset_prefill_page_trace_records(
        prefill_tensors,
        tokens_per_page=2,
        kinds=("K", "V"),
        max_token_index=5,
    )

    assert len(traces) == 8
    first_prefill = next(trace for trace in traces if trace.kind == "K" and trace.kv_head_id == 0 and trace.token_start == 0)
    assert first_prefill.query is None
    assert first_prefill.values.shape == (2, 2)
    assert first_prefill.token_age == 4
    assert "stage=prefill" in first_prefill.notes
