from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dotcache.backends.metal import PersistentFullAttentionState, PersistentServingConfig
from dotcache.config import DotCacheConfig


def test_stage8_persistent_full_attention_builds_block_metadata_without_dotcache_modes() -> None:
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


def test_stage8_persistent_full_attention_assigns_dotcache_modes_and_comp_error() -> None:
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


def test_stage8_persistent_full_attention_mode_aware_priority_penalizes_m0() -> None:
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
