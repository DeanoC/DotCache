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
from dotcache.config import DotCacheConfig
from dotcache.backends.metal.persistent_runtime import (
    _StreamingResidualUpperTracker,
    _build_block_token_index_arrays,
    _build_block_token_index_tensors,
    _residual_value_upper_for_blocks,
    _resolve_streaming_proxy_scores,
    _resolve_streaming_value_upper_scores,
)


class _FakeLayerStructuredCache:
    def __init__(self, *, conv_states=None, recurrent_states=None):
        self.conv_states = list(conv_states or [])
        self.recurrent_states = list(recurrent_states or [])


def _tiny_qwen35_model() -> Qwen3_5ForConditionalGeneration:
    with torch.random.fork_rng():
        torch.manual_seed(0)
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
    assert state.layers[3].block_region_ids.tolist() == [0, 2]
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


def test_block_token_index_tensors_match_numpy_builder() -> None:
    token_starts = np.asarray([0, 16, 32], dtype=np.int64)
    token_counts = np.asarray([16, 8, 3], dtype=np.int64)
    local_starts = np.asarray([0, 16, 24], dtype=np.int64)

    expected_global, expected_local = _build_block_token_index_arrays(
        token_starts=token_starts,
        token_counts=token_counts,
        local_starts=local_starts,
    )
    actual_global, actual_local = _build_block_token_index_tensors(
        token_starts=torch.as_tensor(token_starts, dtype=torch.int64),
        token_counts=torch.as_tensor(token_counts, dtype=torch.int64),
        local_starts=torch.as_tensor(local_starts, dtype=torch.int64),
    )

    assert np.array_equal(actual_global.cpu().numpy(), expected_global)
    assert np.array_equal(actual_local.cpu().numpy(), expected_local)


def test_residual_value_upper_uses_signed_component_box_bound() -> None:
    config = PersistentServingConfig(block_size=2)
    prefill_tensors = {
        4: (
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    residual_mass_upper, residual_value_upper = _residual_value_upper_for_blocks(
        state=state.layers[4],
        block_ids=[0],
        kv_head_idx=0,
        q_vec=torch.tensor([1.0, 0.0], dtype=torch.float32),
        q_norm=1.0,
        query_scale=1.0,
        m_value=0.0,
        upper_bounds=torch.tensor([0.0], dtype=torch.float32),
        use_region_caps=False,
        residual_cluster_count=0,
    )
    assert residual_mass_upper == pytest.approx(2.0, abs=1e-8)
    assert residual_value_upper == pytest.approx(1.0, abs=1e-6)


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
    assert layer.block_region_ids.tolist() == [0, 2]
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
    assert np.count_nonzero(layer.block_compression_metadata_valid) == 0

    state.append_step(
        5,
        torch.tensor([[[6.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[6.0, 8.0]]], dtype=torch.float32),
        token_index=4,
    )
    layer = state.layers[5]
    assert layer.block_token_starts.tolist() == [0, 2, 4]
    assert layer.block_token_counts.tolist() == [2, 2, 1]
    assert layer.block_region_ids.tolist() == [0, 1, 2]
    assert torch.allclose(layer.block_k_center[2, 0], torch.tensor([6.0, 0.0], dtype=torch.float32))
    assert torch.allclose(layer.block_k_radius[2, 0], torch.tensor(0.0, dtype=torch.float32))
    assert torch.allclose(layer.block_v_norm_max[2, 0], torch.tensor(10.0, dtype=torch.float32))
    summary = state.summary()
    assert summary["persistent_full_attention_block_count_by_layer"]["5"] == 3
    assert summary["persistent_full_attention_metadata_valid_blocks_by_layer"]["5"] == 3


def test_persistent_full_attention_state_assigns_dotcache_modes_and_comp_error() -> None:
    config = PersistentServingConfig(block_size=2, enable_compression=True)
    dotcache_config = DotCacheConfig(
        head_dim=4,
        group_size=4,
        bits_k=2,
        bits_v=4,
        tokens_per_page=2,
        default_mode_k="M0",
        default_mode_v="M3",
    )
    prefill_tensors = {
        3: (
            torch.tensor([[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4]]]], dtype=torch.float32),
            torch.tensor([[[[0.5, 1.5, 0.0, 1.0], [1.5, 0.5, 1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
        dotcache_config=dotcache_config,
    )
    layer = state.layers[3]
    assert layer.block_k_mode.tolist() == [["M0"]]
    assert layer.block_v_mode.tolist() == [["M3"]]
    assert float(layer.block_k_comp_error[0, 0].item()) > 0.0
    assert float(layer.block_compression_metadata_valid[0, 0]) == pytest.approx(1.0)


def test_persistent_full_attention_state_mode_aware_priority_penalizes_m0() -> None:
    serving_config = PersistentServingConfig(
        block_size=2,
        enable_compression=True,
        full_attention_mode_cost_weight=0.10,
    )
    prefill_tensors = {
        3: (
            torch.tensor([[[[1.125, -0.375], [0.875, 0.625], [0.5, 0.5], [0.5, -0.5]]]], dtype=torch.float32),
            torch.tensor([[[[0.5, 1.5], [1.5, 0.5], [1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
        )
    }
    m3_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=serving_config,
        dotcache_config=DotCacheConfig(
            head_dim=2,
            group_size=2,
            bits_k=4,
            bits_v=4,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
    )
    m0_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=serving_config,
        dotcache_config=DotCacheConfig(
            head_dim=2,
            group_size=2,
            bits_k=2,
            bits_v=4,
            tokens_per_page=2,
            default_mode_k="M0",
            default_mode_v="M3",
        ),
    )
    query = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    m3_scores = m3_state.score_blocks(3, query, query_scale=1.0)
    m0_scores = m0_state.score_blocks(3, query, query_scale=1.0)
    assert torch.all(m0_scores["priority_scores"] < m3_scores["priority_scores"])
    assert torch.all(m0_scores["upper_bounds"] >= m3_scores["upper_bounds"])


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
    assert selection["soft_recent_block_ids"] == []
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
    certificate = state.certify_selected_blocks(
        7,
        query=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
        selected_block_ids=selection["selected_block_ids"],
        upper_bounds=selection["upper_bounds"],
    )
    assert certificate["processed_block_count"] == len(selection["selected_block_ids"])
    assert certificate["remaining_block_count"] >= 0
    assert certificate["beta_upper"] >= 0.0
    assert certificate["delta_upper"] >= 0.0


def test_persistent_full_attention_state_allows_soft_recent_blocks_to_compete() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=4,
        full_attention_mandatory_recent_block_count=2,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=3,
    )
    prefill_tensors = {
        9: (
            torch.tensor(
                [[[[5.0, 0.0], [4.8, 0.0], [4.6, 0.0], [4.4, 0.0], [0.2, 0.0], [0.2, 0.0], [0.2, 0.0], [0.2, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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

    selection = state.select_blocks(
        9,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == [0, 6, 7]
    assert selection["soft_recent_block_ids"] == [4, 5]
    assert 1 in selection["selected_block_ids"]
    assert 2 in selection["selected_block_ids"]
    assert 3 in selection["selected_block_ids"]
    assert 4 not in selection["mandatory_block_ids"]
    assert 5 not in selection["mandatory_block_ids"]


def test_persistent_full_attention_state_optional_admission_can_prefer_upper_bounds() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_optional_use_upper_bounds_first=True,
    )
    prefill_tensors = {
        10: (
            torch.tensor(
                [[[[2.0, 0.0], [0.1, 0.0], [3.0, 0.0], [0.0, 0.2]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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
    layer = state.layers[10]
    layer.block_prev_attention_ema[1] = 0.5
    selection = state.select_blocks(
        10,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == [0, 3]
    assert selection["optional_block_ids"] == [2]
    assert 1 not in selection["optional_block_ids"]


def test_persistent_full_attention_state_optional_quota_preserves_upper_bound_candidate() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_upper_bound_quota=1,
    )
    prefill_tensors = {
        11: (
            torch.tensor(
                [[[[2.0, 0.0], [0.1, 0.0], [3.0, 0.0], [0.0, 0.2], [1.9, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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
    layer = state.layers[11]
    layer.block_prev_attention_ema[1] = 0.5
    layer.block_prev_attention_ema[4] = 0.25
    selection = state.select_blocks(
        11,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == [0, 4]
    assert 2 in selection["optional_block_ids"]
    assert len(selection["optional_block_ids"]) == 2


def test_persistent_full_attention_state_optional_region_quotas_keep_mid_blocks() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=3,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_far_quota=1,
        full_attention_optional_mid_quota=1,
        full_attention_optional_near_quota=1,
    )
    prefill_tensors = {
        12: (
            torch.tensor(
                [[[[5.0, 0.0], [4.9, 0.0], [4.8, 0.0], [4.7, 0.0], [4.6, 0.0], [0.2, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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
    selection = state.select_blocks(
        12,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    chosen_regions = {int(state.layers[12].block_region_ids[block_id]) for block_id in selection["optional_block_ids"]}
    assert chosen_regions == {0, 1, 2}


def test_persistent_full_attention_state_bootstrap_far_anchor_reserves_far_block() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_far_anchor_quota=1,
    )
    prefill_tensors = {
        14: (
            torch.tensor(
                [[[[0.0, 0.0], [0.8, 0.0], [0.7, 0.0], [1.1, 0.0], [1.3, 0.0], [1.2, 0.0], [1.15, 0.0], [1.05, 0.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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
    state.layers[14].block_prev_attention_ema[1] = 2.0

    selection = state.select_blocks(
        14,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    optional_regions = [int(state.layers[14].block_region_ids[block_id]) for block_id in selection["optional_block_ids"]]
    assert len(selection["optional_block_ids"]) == 2
    assert 0 in optional_regions


def test_persistent_full_attention_state_far_anchor_margin_prevents_weak_replacement() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_far_anchor_quota=1,
        full_attention_optional_far_anchor_priority_margin=5.0,
        full_attention_optional_far_anchor_upper_bound_margin=5.0,
    )
    prefill_tensors = {
        15: (
            torch.tensor(
                [[[[0.0, 0.0], [0.8, 0.0], [0.7, 0.0], [1.1, 0.0], [1.3, 0.0], [1.2, 0.0], [1.15, 0.0], [1.05, 0.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.ones((1, 1, 9, 2), dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )

    selection = state.select_blocks(
        15,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    optional_regions = [int(state.layers[15].block_region_ids[block_id]) for block_id in selection["optional_block_ids"]]
    assert optional_regions.count(0) == 0


def test_persistent_full_attention_state_priority_value_norm_weight_can_change_ranking() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_priority_recency_weight=0.0,
        full_attention_priority_value_norm_weight=0.5,
    )
    prefill_tensors = {
        13: (
            torch.tensor(
                [[[[2.0, 0.0], [1.9, 0.0], [1.8, 0.0], [0.1, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [10.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]],
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
    selection = state.select_blocks(
        13,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == [0, 3]
    assert selection["optional_block_ids"] == [1]


def test_persistent_full_attention_state_full_coverage_certificate_is_zero_residual() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_priority=False,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_optional_top_k=0,
    )
    prefill_tensors = {
        16: (
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    selection = state.select_blocks(16, query, query_scale=1.0)
    certificate = state.certify_selected_blocks(
        16,
        query=query,
        query_scale=1.0,
        selected_block_ids=selection["selected_block_ids"],
        upper_bounds=selection["upper_bounds"],
    )
    assert certificate["remaining_block_count"] == 0
    assert certificate["remaining_token_count"] == 0
    assert certificate["beta_upper"] == pytest.approx(0.0, abs=1e-8)
    assert certificate["delta_upper"] == pytest.approx(0.0, abs=1e-8)
    assert certificate["instability_flag"] is False
    assert certificate["fallback_recommended"] is False


def test_persistent_full_attention_stream_decode_matches_exact_full_output() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=1,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        17: (
            torch.tensor([[[[2.0, 0.0], [1.0, 0.0], [0.0, 1.5], [0.0, 0.5]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 1.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    exact_output = state.decode_layer(17, query, query_scale=1.0)
    streamed = state.stream_decode_layer(17, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert torch.allclose(streamed["output"], exact_output, atol=1e-6, rtol=1e-6)
    assert streamed["final_checkpoint"] is not None
    assert streamed["final_checkpoint"]["beta_upper"] == pytest.approx(0.0, abs=1e-8)
    assert streamed["final_checkpoint"]["delta_upper"] == pytest.approx(0.0, abs=1e-8)


def test_persistent_full_attention_stream_decode_block_attention_masses_match_token_weights() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_exploration_blocks_per_region=1,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        117: (
            torch.tensor([[[[2.0, 0.0], [1.0, 0.0], [0.0, 1.5], [0.0, 0.5]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 1.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed_with_weights = state.stream_decode_layer(
        117,
        query,
        query_scale=1.0,
        check_interval=1,
        stop_on_certificate=False,
    )
    streamed_no_weights = state.stream_decode_layer(
        117,
        query,
        query_scale=1.0,
        check_interval=1,
        stop_on_certificate=False,
        return_attn_weights=False,
    )

    assert streamed_no_weights["attn_weights"] is None
    expected_block_masses = torch.zeros_like(streamed_no_weights["block_attention_masses"])
    collapsed = streamed_with_weights["attn_weights"].to(dtype=torch.float32).mean(dim=(0, 2))
    offset = 0
    for block_id, token_count in zip(
        streamed_with_weights["processed_block_ids"],
        streamed_with_weights["processed_block_token_counts"],
        strict=True,
    ):
        expected_block_masses[:, int(block_id)] = collapsed[:, offset : offset + int(token_count)].sum(dim=-1)
        offset += int(token_count)
    assert torch.allclose(
        streamed_no_weights["block_attention_masses"],
        expected_block_masses,
        atol=1e-6,
        rtol=1e-6,
    )


def test_persistent_full_attention_stream_decode_can_find_certified_stop() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.05,
        full_attention_value_eps=0.05,
        full_attention_min_processed_blocks=1,
    )
    prefill_tensors = {
        18: (
            torch.tensor([[[[5.0, 0.0], [0.01, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(18, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert streamed["first_certified_stop"] is not None
    assert streamed["first_certified_stop"]["processed_block_count"] == 1
    assert streamed["first_certified_stop"]["beta_upper"] < 0.05
    summary = state.summary()
    assert summary["persistent_full_attention_last_first_certified_stop_block_count_by_layer"]["18"] == 1
    assert summary["persistent_full_attention_last_checkpoint_count_by_layer"]["18"] == len(
        streamed["checkpoint_records"]
    )


def test_persistent_full_attention_stream_decode_mixed_direct_m0_can_find_certified_stop() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        enable_early_exit=True,
        enable_compression=True,
        enable_full_attention_mixed_mode_execution=True,
        full_attention_mixed_mode_execution_strategy="direct_m0",
        full_attention_mixed_mode_execution_max_k_comp_error=10.0,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.05,
        full_attention_value_eps=0.05,
        full_attention_min_processed_blocks=1,
    )
    dotcache_config = DotCacheConfig(
        head_dim=4,
        group_size=4,
        bits_k=2,
        bits_v=4,
        tokens_per_page=1,
        default_mode_k="M0",
        default_mode_v="M3",
    )
    prefill_tensors = {
        27: (
            torch.tensor(
                [[[[5.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
        dotcache_config=dotcache_config,
    )
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(27, query, query_scale=1.0, check_interval=1, stop_on_certificate=True)
    assert streamed["first_certified_stop"] is not None
    assert streamed["first_certified_stop"]["processed_block_count"] == 1
    assert streamed["processed_block_ids"] == [0]
    selected_output, _attn_weights, _token_counts, executed_mode_counts = state.decode_selected_blocks(
        27,
        block_ids=[0],
        query=query,
        query_scale=1.0,
    )
    assert executed_mode_counts["M0"] == 1
    assert torch.allclose(streamed["output"], selected_output, atol=1e-5, rtol=1e-5)


def test_persistent_full_attention_stream_decode_summary_only_certificate_mode() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.05,
        full_attention_value_eps=0.05,
        full_attention_min_processed_blocks=1,
    )
    prefill_tensors = {
        28: (
            torch.tensor([[[[5.0, 0.0], [0.01, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(
        28,
        query,
        query_scale=1.0,
        check_interval=1,
        stop_on_certificate=True,
        return_attn_weights=False,
        return_checkpoint_records=False,
        return_checkpoint_per_head=False,
        return_certificate_summary_only=True,
    )
    assert streamed["checkpoint_records"] == []
    assert streamed["first_certified_stop"] is not None
    assert streamed["final_checkpoint"] is not None
    assert streamed["first_certified_stop"]["processed_block_count"] == 1
    assert streamed["final_checkpoint"]["processed_block_count"] == 1
    assert state.summary()["persistent_full_attention_last_checkpoint_count_by_layer"]["28"] == 0


def test_persistent_full_attention_stream_decode_requires_mandatory_coverage() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=1,
        full_attention_mandatory_recent_block_count=1,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=0,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.05,
        full_attention_value_eps=0.05,
        full_attention_min_processed_blocks=1,
    )
    prefill_tensors = {
        19: (
            torch.tensor([[[[5.0, 0.0], [0.01, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(19, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)

    assert streamed["checkpoint_records"][0]["mandatory_complete"] is False
    assert streamed["checkpoint_records"][0]["certified_can_stop"] is False
    assert streamed["first_certified_stop"] is not None
    assert streamed["first_certified_stop"]["processed_block_count"] == 2


def test_persistent_full_attention_stream_decode_disables_stop_for_invalid_metadata() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=1,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.50,
        full_attention_value_eps=0.50,
        full_attention_min_processed_blocks=1,
    )
    prefill_tensors = {
        20: (
            torch.tensor([[[[5.0, 0.0], [0.01, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    state.layers[20].metadata_valid[:] = 0.0
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    exact_output = state.decode_layer(20, query, query_scale=1.0)
    streamed = state.stream_decode_layer(20, query, query_scale=1.0, check_interval=1, stop_on_certificate=True)

    assert streamed["first_certified_stop"] is None
    assert streamed["processed_block_count"] == 2
    assert streamed["final_checkpoint"] is not None
    assert streamed["final_checkpoint"]["instability_flag"] is True
    assert streamed["final_checkpoint"]["mandatory_complete"] is True
    assert torch.allclose(streamed["output"], exact_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        streamed["attn_weights"].sum(dim=-1),
        torch.ones_like(streamed["attn_weights"].sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )
    summary = state.summary()
    assert summary["persistent_full_attention_last_first_certified_stop_block_count_by_layer"]["20"] is None
    assert summary["persistent_full_attention_last_checkpoint_count_by_layer"]["20"] == len(
        streamed["checkpoint_records"]
    )


def test_persistent_full_attention_stream_decode_checkpoint_bounds_match_explicit_recompute() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        21: (
            torch.tensor([[[[3.0, 0.0], [1.0, 0.0], [0.0, 2.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(21, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    processing_order = [int(block_id) for block_id in streamed["processing_order_block_ids"]]
    upper_bounds = streamed["selection"]["upper_bounds"]

    for checkpoint in streamed["checkpoint_records"]:
        processed_count = int(checkpoint["processed_block_count"])
        unresolved_ids = processing_order[processed_count:]
        expected_mass, expected_value = _residual_value_upper_for_blocks(
            state=state.layers[21],
            block_ids=unresolved_ids,
            kv_head_idx=0,
            q_vec=query[0],
            q_norm=float(torch.linalg.vector_norm(query[0]).item()),
            query_scale=1.0,
            m_value=float(checkpoint["per_head"][0]["m"]),
            upper_bounds=upper_bounds,
            use_region_caps=False,
            residual_cluster_count=0,
        )
        assert checkpoint["per_head"][0]["residual_mass_upper"] == pytest.approx(expected_mass, rel=1e-6, abs=1e-6)
        assert checkpoint["per_head"][0]["residual_value_upper"] == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-6,
        )


def test_streaming_residual_upper_tracker_batch_bounds_match_scalar() -> None:
    config = PersistentServingConfig(block_size=1)
    prefill_tensors = {
        29: (
            torch.tensor([[[[3.0, 0.0], [1.0, 0.0], [0.0, 2.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0, 0], dtype=np.int32),
        config=config,
    )
    upper_bounds = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)
    tracker = _StreamingResidualUpperTracker.from_state(
        state=state.layers[29],
        q_head_to_kv_head=np.asarray([0, 0], dtype=np.int32),
        upper_bounds=upper_bounds,
        num_heads=2,
    )
    tracker.mark_processed_blocks([0], m_values=[1.25, 0.75])
    batch_mass, batch_value = tracker.bounds_for_all_q_heads()
    for q_head_idx in range(2):
        scalar_mass, scalar_value = tracker.bounds_for_q_head(q_head_idx)
        assert batch_mass[q_head_idx] == pytest.approx(scalar_mass, rel=1e-6, abs=1e-6)
        assert batch_value[q_head_idx] == pytest.approx(scalar_value, rel=1e-6, abs=1e-6)


def test_persistent_full_attention_stream_decode_residual_proxy_reorders_non_mandatory_blocks() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_streaming_order_mode="residual_proxy",
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=0,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        22: (
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [0.1, 0.0], [5.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(22, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert streamed["processing_order_block_ids"] == [0, 2, 1]


def test_persistent_full_attention_residual_proxy_envelope_uses_tighter_value_proxy() -> None:
    prefill_tensors = {
        22: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[3.0, 4.0], [-3.0, 4.0], [5.0, 0.0], [5.0, 0.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=2),
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    score_result = state.score_blocks(22, query, query_scale=1.0)
    default_scores = _resolve_streaming_proxy_scores(
        state=state.layers[22],
        config=PersistentServingConfig(block_size=2, full_attention_streaming_order_mode="residual_proxy"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
        layer_id=22,
        mode="residual_proxy",
    )
    envelope_scores = _resolve_streaming_proxy_scores(
        state=state.layers[22],
        config=PersistentServingConfig(block_size=2, full_attention_streaming_order_mode="residual_proxy_envelope"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
        layer_id=22,
        mode="residual_proxy_envelope",
    )
    assert float(default_scores[0].item()) == pytest.approx(float(default_scores[1].item()), abs=1e-6)
    assert float(envelope_scores[1].item()) > float(envelope_scores[0].item())


def test_persistent_full_attention_residual_proxy_value_weight_by_layer_changes_ranking() -> None:
    prefill_tensors = {
        23: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [0.1, 0.0], [5.0, 0.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=1),
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    score_result = state.score_blocks(23, query, query_scale=1.0)
    base_scores = _resolve_streaming_proxy_scores(
        state=state.layers[23],
        config=PersistentServingConfig(block_size=1),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
        layer_id=23,
        mode="residual_proxy",
    )
    weighted_scores = _resolve_streaming_proxy_scores(
        state=state.layers[23],
        config=PersistentServingConfig(
            block_size=1,
            full_attention_streaming_proxy_value_weight_by_layer={23: 8.0},
        ),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
        layer_id=23,
        mode="residual_proxy",
    )
    assert float(base_scores[1].item()) > float(base_scores[2].item())
    assert float(weighted_scores[2].item()) > float(weighted_scores[1].item())


def test_persistent_full_attention_exact_value_rerank_reorders_later_tranches() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_streaming_order_mode="residual_proxy",
        full_attention_streaming_exact_value_rerank_layers=[24],
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=0,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        24: (
            torch.tensor(
                [[[[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.8, 0.0], [0.8, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [-5.0, 0.0], [3.0, 0.0], [3.0, 0.0]]]],
                dtype=torch.float32,
            ),
        ),
        25: (
            torch.tensor(
                [[[[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.8, 0.0], [0.8, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [-5.0, 0.0], [3.0, 0.0], [3.0, 0.0]]]],
                dtype=torch.float32,
            ),
        ),
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    reranked = state.stream_decode_layer(24, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    untouched = state.stream_decode_layer(25, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert reranked["processing_order_block_ids"] == [0, 2, 1]
    assert untouched["processing_order_block_ids"] == [0, 1, 2]


def test_persistent_full_attention_exact_value_rerank_can_be_gated_by_remaining_blocks() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_streaming_order_mode="residual_proxy",
        full_attention_streaming_exact_value_rerank_layers=[24],
        full_attention_streaming_exact_value_rerank_max_remaining_blocks=1,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=0,
        full_attention_check_interval=1,
    )
    prefill_tensors = {
        24: (
            torch.tensor(
                [[[[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.8, 0.0], [0.8, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [-5.0, 0.0], [3.0, 0.0], [3.0, 0.0]]]],
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
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(24, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert streamed["processing_order_block_ids"] == [0, 1, 2]


def test_persistent_full_attention_priority_value_hybrid_reorders_non_mandatory_blocks() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_streaming_order_mode="priority_value_hybrid",
        full_attention_streaming_priority_value_upper_weight=0.25,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=0,
        full_attention_check_interval=1,
        full_attention_priority_value_norm_weight=0.0,
    )
    prefill_tensors = {
        26: (
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [0.1, 0.0], [3.0, 4.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(26, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert streamed["processing_order_block_ids"] == [0, 2, 1]


def test_persistent_full_attention_residual_value_upper_scores_rank_by_boxed_value_upper() -> None:
    prefill_tensors = {
        22: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[4.0, 0.0], [-4.0, 0.0], [3.0, 4.0], [3.0, 4.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=2),
    )
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    score_result = state.score_blocks(22, query, query_scale=1.0)
    value_upper_scores = _resolve_streaming_value_upper_scores(
        state=state.layers[22],
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
    )
    assert float(value_upper_scores[1].item()) > float(value_upper_scores[0].item())


def test_persistent_full_attention_caches_streaming_value_upper_log_scores() -> None:
    prefill_tensors = {
        23: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.5, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=2),
    )
    score_result = state.score_blocks(23, torch.tensor([[1.0, 0.0]], dtype=torch.float32), query_scale=1.0)
    layer = state.layers[23]
    cached_log_scores = layer.block_streaming_value_upper_log_cache.clone()
    resolved_scores = _resolve_streaming_value_upper_scores(
        state=layer,
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
    )
    assert torch.allclose(
        resolved_scores - score_result["upper_bounds"].to(dtype=torch.float32),
        cached_log_scores,
        atol=1e-6,
        rtol=1e-6,
    )

    state.append_step(
        23,
        torch.tensor([[[0.25, 0.0]]], dtype=torch.float32),
        torch.tensor([[[6.0, 8.0]]], dtype=torch.float32),
        token_index=4,
    )
    layer = state.layers[23]
    score_result = state.score_blocks(23, torch.tensor([[1.0, 0.0]], dtype=torch.float32), query_scale=1.0)
    refreshed_scores = _resolve_streaming_value_upper_scores(
        state=layer,
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        upper_bounds=score_result["upper_bounds"],
    )
    assert int(layer.block_streaming_value_upper_log_cache.shape[0]) == 3
    assert torch.allclose(
        refreshed_scores - score_result["upper_bounds"].to(dtype=torch.float32),
        layer.block_streaming_value_upper_log_cache,
        atol=1e-6,
        rtol=1e-6,
    )


def test_persistent_full_attention_stream_decode_streaming_refine_top_k_tightens_stop() -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        enable_early_exit=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_check_interval=1,
        full_attention_mass_eps=0.05,
        full_attention_value_eps=0.05,
        full_attention_min_processed_blocks=1,
        full_attention_streaming_refine_top_k=1,
    )
    prefill_tensors = {
        23: (
            torch.tensor([[[[5.0, 0.0], [0.01, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    state.layers[23].block_k_radius[1, 0] = 10.0
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    streamed = state.stream_decode_layer(23, query, query_scale=1.0, check_interval=1, stop_on_certificate=False)
    assert streamed["first_certified_stop"] is not None
    assert streamed["first_certified_stop"]["processed_block_count"] == 1
    assert streamed["checkpoint_records"][0]["beta_upper"] < 0.05


def test_persistent_full_attention_refine_top_k_tightens_upper_bound() -> None:
    config = PersistentServingConfig(
        block_size=2,
        full_attention_refine_top_k=1,
    )
    prefill_tensors = {
        19: (
            torch.tensor([[[[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32),
        )
    }
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    coarse_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=2, full_attention_refine_top_k=0),
    )
    refined_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    coarse_scores = coarse_state.score_blocks(19, query, query_scale=1.0)
    refined_scores = refined_state.score_blocks(19, query, query_scale=1.0)
    assert float(refined_scores["upper_bounds"][0].item()) <= float(coarse_scores["upper_bounds"][0].item())
    assert float(refined_scores["upper_bounds"][0].item()) == pytest.approx(3.0, abs=1e-6)


def test_persistent_full_attention_refine_top_k_by_layer_is_layer_specific() -> None:
    prefill_tensors = {
        20: (
            torch.tensor(
                [[[[3.0, 0.0], [2.9, 0.0], [0.0, 2.0], [0.0, 1.9]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        ),
        21: (
            torch.tensor(
                [[[[3.0, 0.0], [2.9, 0.0], [0.0, 2.0], [0.0, 1.9]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        ),
    }
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(
            block_size=4,
            full_attention_refine_top_k=0,
            full_attention_refine_top_k_by_layer={20: 1},
        ),
    )
    layer20_scores = state.score_blocks(20, query, query_scale=1.0)
    layer21_scores = state.score_blocks(21, query, query_scale=1.0)
    assert float(layer20_scores["upper_bounds"][0].item()) == pytest.approx(3.0, abs=1e-6)
    assert float(layer21_scores["upper_bounds"][0].item()) > 3.0


def test_persistent_full_attention_multi_centroid_tightens_upper_bound() -> None:
    prefill_tensors = {
        20: (
            torch.tensor(
                [[[[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    # Disable interval bound to isolate multi-centroid feature: interval is exact on
    # axis-aligned data, which would equalise coarse and centroid scores.
    coarse_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=4, full_attention_key_centroid_count=1, enable_interval_bound=False),
    )
    centroid_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=4, full_attention_key_centroid_count=2, enable_interval_bound=False),
    )
    coarse_scores = coarse_state.score_blocks(20, query, query_scale=1.0)
    centroid_scores = centroid_state.score_blocks(20, query, query_scale=1.0)
    assert float(centroid_scores["upper_bounds"][0].item()) < float(coarse_scores["upper_bounds"][0].item())
    assert float(centroid_scores["upper_bounds"][0].item()) == pytest.approx(3.0, abs=1e-6)


def test_persistent_full_attention_key_centroid_count_by_layer_is_layer_specific() -> None:
    prefill_tensors = {
        20: (
            torch.tensor(
                [[[[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        ),
        21: (
            torch.tensor(
                [[[[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        ),
    }
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(
            block_size=4,
            full_attention_key_centroid_count=1,
            full_attention_key_centroid_count_by_layer={20: 2},
        ),
    )
    layer20_scores = state.score_blocks(20, query, query_scale=1.0)
    layer21_scores = state.score_blocks(21, query, query_scale=1.0)
    assert float(layer20_scores["upper_bounds"][0].item()) == pytest.approx(3.0, abs=1e-6)
    assert float(layer21_scores["upper_bounds"][0].item()) > 3.0


def test_persistent_full_attention_value_centroid_count_by_layer_is_layer_specific() -> None:
    prefill_tensors = {
        20: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
        ),
        21: (
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]],
                dtype=torch.float32,
            ),
        ),
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(
            block_size=4,
            full_attention_value_centroid_count=1,
            full_attention_value_centroid_count_by_layer={20: 2},
        ),
    )
    assert int(state.layers[20].block_v_subcenters.shape[2]) == 2
    assert int(state.layers[21].block_v_subcenters.shape[2]) == 1


def test_persistent_full_attention_residual_clusters_tighten_omitted_tail() -> None:
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors={
            22: (
                torch.tensor(
                    [[[[0.5, 0.0], [0.6, 0.0], [0.4, 0.0], [0.5, 0.0], [0.55, 0.0], [0.45, 0.0]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=2, full_attention_residual_cluster_count=1),
    )
    layer_state = state.layers[22]
    upper_bounds = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float32)
    q_vec = torch.tensor([1.0, 0.0], dtype=torch.float32)
    mass_without_caps, value_without_caps = _residual_value_upper_for_blocks(
        state=layer_state,
        block_ids=[0, 1, 2],
        kv_head_idx=0,
        q_vec=q_vec,
        q_norm=1.0,
        query_scale=1.0,
        m_value=5.0,
        upper_bounds=upper_bounds,
        use_region_caps=False,
        residual_cluster_count=0,
    )
    mass_with_caps, value_with_caps = _residual_value_upper_for_blocks(
        state=layer_state,
        block_ids=[0, 1, 2],
        kv_head_idx=0,
        q_vec=q_vec,
        q_norm=1.0,
        query_scale=1.0,
        m_value=5.0,
        upper_bounds=upper_bounds,
        use_region_caps=False,
        residual_cluster_count=1,
    )
    assert mass_with_caps < mass_without_caps
    assert value_with_caps < value_without_caps


def test_persistent_full_attention_probe_refine_tightens_upper_bound_safely() -> None:
    # Disable interval bound to isolate probe-refine feature: interval is exact on
    # axis-aligned data (3.0), which makes probe-refinement a no-op and the 3.1
    # assertion (probed spherical bound) meaningless.
    config = PersistentServingConfig(
        block_size=4,
        full_attention_probe_refine_top_k=1,
        full_attention_probe_sample_count=2,
        enable_interval_bound=False,
    )
    prefill_tensors = {
        21: (
            torch.tensor(
                [[[[3.0, 0.0], [2.9, 0.0], [0.0, 2.0], [0.0, 1.9]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        )
    }
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    coarse_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=PersistentServingConfig(block_size=4, full_attention_probe_refine_top_k=0, enable_interval_bound=False),
    )
    probe_state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
    )
    coarse_scores = coarse_state.score_blocks(21, query, query_scale=1.0)
    probe_scores = probe_state.score_blocks(21, query, query_scale=1.0)
    exact_max = 3.0
    refined_upper = float(probe_scores["upper_bounds"][0].item())
    assert refined_upper <= float(coarse_scores["upper_bounds"][0].item())
    assert refined_upper >= exact_max
    assert refined_upper == pytest.approx(3.1, abs=1e-6)


@pytest.mark.parametrize("diversity_strategy", ["greedy", "window_suppress"])
def test_persistent_full_attention_state_optional_diversity_spreads_clustered_picks(
    diversity_strategy: str,
) -> None:
    config = PersistentServingConfig(
        block_size=1,
        enable_priority=True,
        full_attention_sink_block_count=0,
        full_attention_recent_block_count=0,
        full_attention_exploration_blocks_per_region=0,
        full_attention_optional_top_k=2,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_diversity_weight=1.0,
        full_attention_optional_diversity_radius=2,
        full_attention_optional_diversity_strategy=diversity_strategy,
        full_attention_priority_prev_attention_weight=0.0,
        full_attention_priority_recency_weight=0.0,
        full_attention_priority_value_norm_weight=0.0,
    )
    prefill_tensors = {
        14: (
            torch.tensor(
                [[[[5.0, 0.0], [4.9, 0.0], [4.8, 0.0], [4.7, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]],
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

    selection = state.select_blocks(
        14,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        query_scale=1.0,
    )

    assert selection["mandatory_block_ids"] == []
    assert selection["optional_block_ids"] == [0, 2]


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
