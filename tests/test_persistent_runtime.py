from __future__ import annotations

import copy
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

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


def _tiny_qwen35_model() -> Qwen3_5ForConditionalGeneration:
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 1,
            "num_heads": 4,
        },
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


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
    assert state.layers[3].block_region_ids.tolist() == [0, 1]
    assert tuple(state.layers[3].block_k_center.shape) == (2, 1, 2)
    assert tuple(state.layers[3].block_k_radius.shape) == (2, 1)
    assert tuple(state.layers[3].block_v_norm_max.shape) == (2, 1)
    assert np.count_nonzero(state.layers[3].metadata_valid) == 2

    manual_logits_q0 = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    manual_weights_q0 = torch.softmax(manual_logits_q0, dim=0)
    manual_q0 = manual_weights_q0 @ torch.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(context[0], manual_q0, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state.layers[3].block_k_center[0, 0],
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        state.layers[3].block_k_radius[:, 0],
        torch.tensor([2**-0.5, 0.0], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        state.layers[3].block_v_norm_max[:, 0],
        torch.tensor([3.0, 41**0.5], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_persistent_full_attention_state_builds_and_refreshes_block_metadata() -> None:
    config = PersistentServingConfig(block_size=2)
    prefill_tensors = {
        5: (
            torch.tensor(
                [[[[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 1.0], [2.0, 2.0], [0.0, 3.0], [0.0, 5.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    layer = state.layers[5]

    assert layer.block_token_starts.tolist() == [0, 2]
    assert layer.block_token_counts.tolist() == [2, 2]
    assert layer.block_region_ids.tolist() == [0, 1]
    assert torch.allclose(layer.block_k_center[0, 0], torch.tensor([2.0, 0.0], dtype=torch.float32))
    assert torch.allclose(layer.block_k_center[1, 0], torch.tensor([0.0, 3.0], dtype=torch.float32))
    assert torch.allclose(layer.block_k_radius[:, 0], torch.tensor([1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(
        layer.block_v_norm_max[:, 0],
        torch.tensor([8**0.5, 5.0], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(layer.block_prev_attention_ema, torch.zeros((2,), dtype=torch.float32))
    assert torch.allclose(layer.block_k_comp_error, torch.zeros_like(layer.block_k_comp_error))

    state.append_step(
        5,
        torch.tensor([[[6.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[6.0, 8.0]]], dtype=torch.float32),
        token_index=4,
    )
    layer = state.layers[5]
    assert layer.block_token_starts.tolist() == [0, 2, 4]
    assert layer.block_token_counts.tolist() == [2, 2, 1]
    assert layer.block_region_ids.tolist() == [0, 2, 1]
    assert torch.allclose(layer.block_k_center[2, 0], torch.tensor([6.0, 0.0], dtype=torch.float32))
    assert torch.allclose(layer.block_k_radius[2, 0], torch.tensor(0.0, dtype=torch.float32))
    assert torch.allclose(layer.block_v_norm_max[2, 0], torch.tensor(10.0, dtype=torch.float32))
    summary = state.summary()
    assert summary["persistent_full_attention_block_count_by_layer"]["5"] == 3
    assert summary["persistent_full_attention_metadata_valid_blocks_by_layer"]["5"] == 3


def test_persistent_full_attention_state_scores_and_selects_blocks() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=1,
        full_attention_optional_top_k=1,
    )
    prefill_tensors = {
        7: (
            torch.tensor(
                [[[[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 4.0], [2.5, 0.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 4.0], [2.5, 0.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    state.layers[7].block_prev_attention_ema[4] = 0.25

    selection = state.select_blocks(
        7,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == [0, 5]
    assert len(selection["exploration_block_ids"]) == 2
    assert any(block_id in selection["exploration_block_ids"] for block_id in [1, 2])
    assert any(block_id in selection["exploration_block_ids"] for block_id in [3, 4])
    assert len(selection["optional_block_ids"]) == 1
    assert 4 in selection["selected_block_ids"]
    assert selection["selected_block_ids"] == sorted(selection["selected_block_ids"])
    assert selection["upper_bounds"].shape[0] == 6
    assert selection["priority_scores"].shape[0] == 6

    gathered_keys, gathered_values, token_counts = state.gather_selected_blocks(7, selection["selected_block_ids"])
    assert int(gathered_keys.shape[1]) == sum(token_counts)
    assert gathered_keys.shape == gathered_values.shape

    fake_weights = torch.ones((1, 1, 1, sum(token_counts)), dtype=torch.float32) / float(sum(token_counts))
    state.update_block_attention_ema(
        7,
        selected_block_ids=selection["selected_block_ids"],
        selected_block_token_counts=token_counts,
        attn_weights=fake_weights,
    )
    assert float(state.layers[7].block_prev_attention_ema.sum().item()) > 0.0


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


def test_persistent_linear_attention_decode_matches_native_qwen_decode_step() -> None:
    torch.manual_seed(0)
    model = _tiny_qwen35_model()
    text_model = model.model.language_model
    linear_layer = text_model.layers[0].linear_attn

    input_ids = torch.tensor([[1, 9, 12, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    native_cache = copy.deepcopy(prefill_outputs.past_key_values)
    persistent_cache = copy.deepcopy(prefill_outputs.past_key_values)
    telemetry = PersistentStepTelemetry()
    linear_state = PersistentLinearAttentionState.from_native_cache(
        cache=persistent_cache,
        layer_ids=[0],
        device=torch.device("cpu"),
        telemetry=telemetry,
    )

    hidden_states = torch.randn((1, 1, model.config.text_config.hidden_size), dtype=torch.float32)
    with torch.no_grad():
        native_output = linear_layer(
            hidden_states=hidden_states,
            cache_params=native_cache,
            attention_mask=None,
        )
        persistent_output = linear_state.decode_layer(
            0,
            layer_module=linear_layer,
            hidden_states=hidden_states,
            attention_mask=None,
        )

    assert torch.allclose(persistent_output, native_output, atol=1e-5, rtol=1e-5)
    assert linear_state.layers[0].has_previous_state is True
    assert linear_state.layers[0].direct_compute_count == 1
    assert torch.allclose(
        linear_state.layers[0].conv_state,
        native_cache.layers[0].conv_states.to(dtype=linear_state.layers[0].conv_state.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        linear_state.layers[0].recurrent_state,
        native_cache.layers[0].recurrent_states.to(dtype=linear_state.layers[0].recurrent_state.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
