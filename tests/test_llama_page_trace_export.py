import json

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.integrations.llama import (
    LlamaReplayRecord,
    build_llama_page_trace_records,
    build_llama_prefill_page_trace_records,
    export_llama_page_traces,
)


def _sample_replay_record(*, token_index: int, layer_id: int = 0) -> LlamaReplayRecord:
    return LlamaReplayRecord(
        step_index=token_index,
        layer_id=layer_id,
        token_index=token_index,
        query_states=np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        key_states=np.asarray([[10.0 + token_index, 20.0], [30.0 + token_index, 40.0]], dtype=np.float32),
        value_states=np.asarray([[50.0 + token_index, 60.0], [70.0 + token_index, 80.0]], dtype=np.float32),
        context_states=np.asarray([0.0, 0.0], dtype=np.float32),
        output_states=np.asarray([0.0, 0.0], dtype=np.float32),
    )


def test_build_llama_page_trace_records_groups_decode_records() -> None:
    traces = build_llama_page_trace_records(
        [[_sample_replay_record(token_index=4)], [_sample_replay_record(token_index=5)]],
        q_head_to_kv_head=np.asarray([0, 1], dtype=np.int64),
        tokens_per_page=2,
        kinds=("K", "V"),
    )

    assert len(traces) == 4
    first = traces[0]
    assert first.kind == "K"
    assert first.layer_id == 0
    assert first.kv_head_id == 0
    assert first.token_start == 4
    assert first.token_count == 2
    assert np.allclose(first.query, np.asarray([1.0, 3.0], dtype=np.float32))
    assert any(note == "stage=decode" for note in first.notes)


def test_build_llama_prefill_page_trace_records_groups_prefill_layers() -> None:
    layer_keys = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        dtype=torch.float32,
    )
    layer_values = torch.tensor(
        [[[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]],
        dtype=torch.float32,
    )

    traces = build_llama_prefill_page_trace_records(
        [(layer_keys, layer_values)],
        tokens_per_page=2,
        kinds=("K",),
        max_token_index=3,
    )

    assert len(traces) == 2
    assert traces[0].kind == "K"
    assert traces[0].token_age == 2
    assert any(note == "stage=prefill" for note in traces[0].notes)


def test_export_llama_page_traces_writes_manifest(tmp_path) -> None:
    manifest = export_llama_page_traces(
        [[_sample_replay_record(token_index=4)]],
        q_head_to_kv_head=np.asarray([0, 1], dtype=np.int64),
        output_dir=tmp_path,
        tokens_per_page=1,
        kinds=("K",),
    )

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["page_trace_count"] == manifest["page_trace_count"] == 2
    assert payload["page_trace_counts_by_kind"] == {"K": 2}
    assert payload["page_trace_counts_by_stage"] == {"decode": 2}
