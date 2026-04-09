from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dotcache.backends.metal import (
    PersistentFullAttentionState,
    PersistentLinearAttentionState,
    PersistentServingConfig,
    PersistentStepTelemetry,
)


class _FakeLayerStructuredCache:
    def __init__(self, *, conv_states=None, recurrent_states=None):
        self.conv_states = list(conv_states or [])
        self.recurrent_states = list(recurrent_states or [])


def test_persistent_full_attention_state_appends_and_decodes_exactly() -> None:
    config = PersistentServingConfig(block_size=2)
    prefill_tensors = {
        3: (
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
            torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0, 0], dtype=np.int32),
        config=config,
    )
    state.append_step(
        3,
        torch.tensor([[[1.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[4.0, 5.0]]], dtype=torch.float32),
        token_index=2,
    )
    context = state.decode_layer(
        3,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        query_scale=1.0,
    )
    assert tuple(context.shape) == (2, 2)
    assert state.layers[3].append_count == 1
    assert int(state.layers[3].key_cache.shape[1]) == 3
    assert state.layers[3].block_token_starts.tolist() == [0, 2]
    assert state.layers[3].block_token_counts.tolist() == [2, 1]

    manual_logits_q0 = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    manual_weights_q0 = torch.softmax(manual_logits_q0, dim=0)
    manual_q0 = manual_weights_q0 @ torch.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(context[0], manual_q0, atol=1e-6, rtol=1e-6)


def test_persistent_linear_attention_state_syncs_cache_roundtrip() -> None:
    cache = _FakeLayerStructuredCache(
        conv_states=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)],
        recurrent_states=[torch.tensor([[3.0, 4.0]], dtype=torch.float32)],
    )
    linear_state = PersistentLinearAttentionState.from_native_cache(
        cache=cache,
        layer_ids=[0],
        device=torch.device("cpu"),
        telemetry=PersistentStepTelemetry(),
    )
    cache.conv_states[0] = torch.tensor([[9.0, 9.0]], dtype=torch.float32)
    cache.recurrent_states[0] = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
    linear_state.sync_layer_into_cache(cache, 0)
    assert torch.allclose(cache.conv_states[0], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.allclose(cache.recurrent_states[0], torch.tensor([[3.0, 4.0]], dtype=torch.float32))

    cache.conv_states[0] = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    cache.recurrent_states[0] = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
    linear_state.sync_layer_from_cache(cache, 0)
    assert torch.allclose(linear_state.layers[0].conv_state, torch.tensor([[5.0, 6.0]], dtype=torch.float32))
    assert torch.allclose(linear_state.layers[0].recurrent_state, torch.tensor([[7.0, 8.0]], dtype=torch.float32))
