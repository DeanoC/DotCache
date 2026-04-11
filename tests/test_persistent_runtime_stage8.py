from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dotcache.backends.metal import PersistentFullAttentionState, PersistentServingConfig
from dotcache.backends.metal.persistent_runtime import _resolve_mixed_score_dtype
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


def test_stage8_persistent_full_attention_uses_prefill_block_metadata_overrides() -> None:
    config = PersistentServingConfig(block_size=2, enable_compression=True)
    prefill_tensors = {
        3: (
            torch.tensor([[[[1.125, -0.375], [0.875, 0.625]]]], dtype=torch.float32),
            torch.tensor([[[[0.5, 1.5], [1.5, 0.5]]]], dtype=torch.float32),
        )
    }
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
        dotcache_config=DotCacheConfig(
            head_dim=2,
            group_size=2,
            bits_k=4,
            bits_v=4,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.25]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0]], dtype=np.float32),
            }
        },
    )
    layer = state.layers[3]
    assert layer.block_k_mode.tolist() == [["M0"]]
    assert float(layer.block_k_comp_error[0, 0].item()) == pytest.approx(0.25)
    state.append_step(
        3,
        torch.tensor([[[0.5, -0.5]]], dtype=torch.float32),
        torch.tensor([[[0.5, 0.5]]], dtype=torch.float32),
        token_index=2,
    )
    layer = state.layers[3]
    assert layer.block_k_mode.tolist() == [["M0"], ["M3"]]
    assert float(layer.block_k_comp_error[0, 0].item()) == pytest.approx(0.25)


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


def test_stage8_persistent_full_attention_8bit_m0_lowers_key_comp_error() -> None:
    serving_config = PersistentServingConfig(block_size=2, enable_compression=True)
    prefill_tensors = {
        3: (
            torch.tensor(
                [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[0.5, 1.5, 0.0, 1.0], [1.5, 0.5, 1.0, 0.0], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                dtype=torch.float32,
            ),
        )
    }
    state_4bit = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=serving_config,
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=4,
            bits_v=4,
            tokens_per_page=2,
            default_mode_k="M0",
            default_mode_v="M3",
        ),
    )
    state_8bit = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors=prefill_tensors,
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=serving_config,
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=8,
            bits_v=4,
            tokens_per_page=2,
            default_mode_k="M0",
            default_mode_v="M3",
        ),
    )

    assert torch.all(state_8bit.layers[3].block_k_comp_error <= state_4bit.layers[3].block_k_comp_error)
    assert torch.any(state_8bit.layers[3].block_k_comp_error < state_4bit.layers[3].block_k_comp_error)


def test_stage8_persistent_full_attention_mixed_execution_only_reconstructs_m0_blocks() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_compression=True,
        enable_full_attention_mixed_mode_execution=True,
        full_attention_mixed_mode_execution_allow_value_m0=True,
        full_attention_mixed_mode_execution_max_k_comp_error=0.30,
    )
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors={
            3: (
                torch.tensor(
                    [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[0.4, 1.2, -0.3, 0.8], [1.4, 0.6, 0.9, -0.2], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=2,
            bits_v=2,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"], ["M3"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M0"], ["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.25], [0.0]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0], [1.0]], dtype=np.float32),
            }
        },
    )
    exact_keys, exact_values, token_counts = state.gather_selected_blocks(3, [0, 1])
    mixed_keys, mixed_values, mixed_token_counts, executed_mode_counts = state.prepare_selected_execution_tensors(3, [0, 1])
    blockwise_keys, blockwise_values, blockwise_token_counts, blockwise_mode_counts = state.prepare_selected_execution_tensors(
        3,
        [0, 1],
        config_override=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="blockwise_qdq",
            full_attention_mixed_mode_execution_allow_value_m0=True,
            full_attention_mixed_mode_execution_max_k_comp_error=0.30,
        ),
    )

    assert mixed_token_counts == token_counts
    assert blockwise_token_counts == token_counts
    assert executed_mode_counts == {"M0": 1, "M3": 1}
    assert blockwise_mode_counts == {"M0": 1, "M3": 1}
    assert not torch.allclose(mixed_keys[:, :2, :], exact_keys[:, :2, :])
    assert not torch.allclose(mixed_values[:, :2, :], exact_values[:, :2, :])
    assert torch.allclose(mixed_keys[:, 2:, :], exact_keys[:, 2:, :])
    assert torch.allclose(mixed_values[:, 2:, :], exact_values[:, 2:, :])
    assert torch.allclose(mixed_keys, blockwise_keys)
    assert torch.allclose(mixed_values, blockwise_values)
    summary = state.summary()
    assert summary["persistent_full_attention_executed_m0_block_count_total_by_layer"]["3"] == 2
    assert summary["persistent_full_attention_executed_m3_block_count_total_by_layer"]["3"] == 2
    assert summary["persistent_full_attention_mixed_execution_cache_refresh_ms_total_by_layer"]["3"] >= 0.0


def test_stage8_mixed_score_dtype_auto_prefers_accelerator_dtype() -> None:
    config = PersistentServingConfig(
        enable_compression=True,
        enable_full_attention_mixed_mode_execution=True,
        full_attention_mixed_mode_execution_strategy="direct_m0",
        full_attention_mixed_mode_score_dtype="auto",
    )

    assert _resolve_mixed_score_dtype(config=config, device=torch.device("mps")) == torch.float16
    assert _resolve_mixed_score_dtype(config=config, device=torch.device("cpu")) == torch.float32


def test_stage8_persistent_full_attention_mixed_execution_is_conservative_by_default() -> None:
    config = PersistentServingConfig(
        block_size=2,
        enable_compression=True,
        enable_full_attention_mixed_mode_execution=True,
        full_attention_mixed_mode_execution_max_k_comp_error=0.10,
    )
    state = PersistentFullAttentionState.from_prefill_tensors(
        prefill_tensors={
            3: (
                torch.tensor(
                    [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[0.4, 1.2, -0.3, 0.8], [1.4, 0.6, 0.9, -0.2], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        config=config,
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=2,
            bits_v=2,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"], ["M3"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M0"], ["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.25], [0.0]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0], [1.0]], dtype=np.float32),
            }
        },
    )
    exact_keys, exact_values, token_counts = state.gather_selected_blocks(3, [0, 1])
    mixed_keys, mixed_values, mixed_token_counts, executed_mode_counts = state.prepare_selected_execution_tensors(3, [0, 1])

    assert mixed_token_counts == token_counts
    assert executed_mode_counts == {"M0": 0, "M3": 2}
    assert torch.allclose(mixed_keys, exact_keys)
    assert torch.allclose(mixed_values, exact_values)


def test_stage8_direct_m0_decode_matches_cached_reconstruct_for_key_only_execution() -> None:
    base_kwargs = dict(
        prefill_tensors={
            3: (
                torch.tensor(
                    [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[0.4, 1.2, -0.3, 0.8], [1.4, 0.6, 0.9, -0.2], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=8,
            bits_v=8,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"], ["M3"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M3"], ["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.15], [0.0]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0], [1.0]], dtype=np.float32),
            }
        },
    )
    cached_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="cached_reconstruct",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    direct_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="direct_m0",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    query = torch.tensor([[0.25, -0.5, 0.75, 0.1]], dtype=torch.float32)

    cached_output, cached_weights, cached_token_counts, cached_mode_counts = cached_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )
    direct_output, direct_weights, direct_token_counts, direct_mode_counts = direct_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )

    assert cached_token_counts == direct_token_counts == [2, 2]
    assert cached_mode_counts == direct_mode_counts == {"M0": 1, "M3": 1}
    assert torch.allclose(direct_output, cached_output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(direct_weights, cached_weights, atol=1e-4, rtol=1e-4)


def test_stage8_direct_m0_decode_batches_multiple_m0_blocks_per_head() -> None:
    base_kwargs = dict(
        prefill_tensors={
            3: (
                torch.tensor(
                    [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[0.4, 1.2, -0.3, 0.8], [1.4, 0.6, 0.9, -0.2], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=8,
            bits_v=8,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"], ["M0"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M3"], ["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.13], [0.13]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0], [1.0]], dtype=np.float32),
            }
        },
    )
    cached_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="cached_reconstruct",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    direct_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="direct_m0",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    query = torch.tensor([[0.25, -0.5, 0.75, 0.1]], dtype=torch.float32)

    cached_output, cached_weights, cached_token_counts, cached_mode_counts = cached_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )
    direct_output, direct_weights, direct_token_counts, direct_mode_counts = direct_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )

    assert cached_token_counts == direct_token_counts == [2, 2]
    assert cached_mode_counts == direct_mode_counts == {"M0": 2, "M3": 0}
    assert torch.allclose(direct_output, cached_output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(direct_weights, cached_weights, atol=1e-4, rtol=1e-4)


def test_stage8_direct_m0_metal_packed_strategy_falls_back_to_torch_on_cpu() -> None:
    base_kwargs = dict(
        prefill_tensors={
            3: (
                torch.tensor(
                    [[[[1.125, -0.375, 0.2, 0.9], [0.875, 0.625, -0.1, 0.4], [0.5, 0.5, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5]]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[[0.4, 1.2, -0.3, 0.8], [1.4, 0.6, 0.9, -0.2], [1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]]]],
                    dtype=torch.float32,
                ),
            )
        },
        device=torch.device("cpu"),
        q_head_to_kv_head=np.asarray([0], dtype=np.int32),
        dotcache_config=DotCacheConfig(
            head_dim=4,
            group_size=4,
            bits_k=8,
            bits_v=8,
            tokens_per_page=2,
            default_mode_k="M3",
            default_mode_v="M3",
        ),
        prefill_block_metadata_by_layer={
            3: {
                "block_k_mode": np.asarray([["M0"], ["M0"]], dtype="<U2"),
                "block_v_mode": np.asarray([["M3"], ["M3"]], dtype="<U2"),
                "block_k_comp_error": np.asarray([[0.13], [0.13]], dtype=np.float32),
                "block_compression_metadata_valid": np.asarray([[1.0], [1.0]], dtype=np.float32),
            }
        },
    )
    torch_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="direct_m0",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    metal_state = PersistentFullAttentionState.from_prefill_tensors(
        config=PersistentServingConfig(
            block_size=2,
            enable_compression=True,
            enable_full_attention_mixed_mode_execution=True,
            full_attention_mixed_mode_execution_strategy="direct_m0_metal_packed",
            full_attention_mixed_mode_execution_allow_value_m0=False,
            full_attention_mixed_mode_execution_max_k_comp_error=0.20,
        ),
        **base_kwargs,
    )
    query = torch.tensor([[0.25, -0.5, 0.75, 0.1]], dtype=torch.float32)
    torch_output, torch_weights, torch_token_counts, torch_mode_counts = torch_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )
    metal_output, metal_weights, metal_token_counts, metal_mode_counts = metal_state.decode_selected_blocks(
        3,
        block_ids=[0, 1],
        query=query,
        query_scale=1.0,
    )

    assert metal_token_counts == torch_token_counts == [2, 2]
    assert metal_mode_counts == torch_mode_counts == {"M0": 2, "M3": 0}
    assert torch.allclose(metal_output, torch_output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(metal_weights, torch_weights, atol=1e-4, rtol=1e-4)
