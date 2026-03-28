from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dotcache.attention_runtime import decode_step
from dotcache.backends.torch_mps import (
    PreparedPageTorch,
    _get_grouped_prepared_chunk_mps,
    clear_prepared_chunk_cache,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.integrations.vllm_adapter import (
    VLLM_V1_MULTIPROCESSING_ENV,
    VllmAdapterConfig,
    VllmPagedKVCache,
    configure_vllm_inprocess_runtime,
    install_dotcache_on_vllm_model,
)
from dotcache.integrations.vllm_adapter.compat import require_supported_vllm_version
from dotcache.model_kv_cache import default_q_head_to_kv_head


def _encode_blocks_for_head(
    block_rows: np.ndarray,
    config: DotCacheConfig,
    *,
    kind: str,
    layer_id: int,
    kv_head_id: int,
    live_token_count: int | None = None,
) -> list:
    pages = []
    for block_index in range(block_rows.shape[0]):
        rows = block_rows[block_index]
        token_count = config.tokens_per_page if live_token_count is None or block_index + 1 != block_rows.shape[0] else live_token_count
        mode = None if token_count == config.tokens_per_page else "M3"
        pages.append(
            encode_page(
                rows[:token_count],
                config,
                kind=kind,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=block_index * config.tokens_per_page,
                mode=mode,
                build_runtime_metadata=False,
            )
        )
    return pages


def test_vllm_adapter_config_requires_block_size_alignment() -> None:
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    with pytest.raises(ValueError, match="tokens_per_page must equal the vLLM block_size"):
        VllmAdapterConfig(dotcache_config=config, block_size=8)


def test_require_supported_vllm_version_accepts_supported_minor(monkeypatch) -> None:
    monkeypatch.setattr("dotcache.integrations.vllm_adapter.compat.get_vllm_version", lambda: "0.18.0")
    assert require_supported_vllm_version(supported_minor="0.18") == "0.18.0"


def test_require_supported_vllm_version_rejects_unknown_minor(monkeypatch) -> None:
    monkeypatch.setattr("dotcache.integrations.vllm_adapter.compat.get_vllm_version", lambda: "0.17.2")
    with pytest.raises(RuntimeError, match="Unsupported vLLM version"):
        require_supported_vllm_version(supported_minor="0.18")


def test_configure_vllm_inprocess_runtime_sets_default(monkeypatch) -> None:
    monkeypatch.delenv(VLLM_V1_MULTIPROCESSING_ENV, raising=False)
    assert configure_vllm_inprocess_runtime() == "0"
    assert os.environ[VLLM_V1_MULTIPROCESSING_ENV] == "0"


def test_configure_vllm_inprocess_runtime_rejects_multiprocess_setting(monkeypatch) -> None:
    monkeypatch.setenv(VLLM_V1_MULTIPROCESSING_ENV, "1")
    with pytest.raises(RuntimeError, match="requires the in-process runtime"):
        configure_vllm_inprocess_runtime()


def test_vllm_block_table_mapping_tracks_stable_block_ownership() -> None:
    rng = np.random.default_rng(1201)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = VllmPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_size=4,
        backend="cpu_ref",
    )
    key_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)
    value_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)

    cache.sync_layer_blocks(0, key_blocks, value_blocks, block_ids=[11, 42])

    key_entry = cache.block_entry(0, 1, 42, "K")
    value_entry = cache.block_entry(0, 1, 42, "V")
    assert key_entry.finalized is True
    assert value_entry.finalized is True
    assert key_entry.token_start == 4
    assert key_entry.token_count == 4
    assert key_entry.page.header.layer_id == 0
    assert key_entry.page.header.kv_head_id == 1
    assert key_entry.page.header.kind == "K"
    assert value_entry.page.header.kind == "V"


def test_vllm_finalized_blocks_decode_matches_quantized_reference() -> None:
    rng = np.random.default_rng(1202)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = VllmPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_size=4,
        backend="cpu_ref",
    )
    key_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)
    value_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.sync_layer_blocks(0, key_blocks, value_blocks, block_ids=[3, 7])
    outputs = cache.decode_layer(0, queries, mapping)

    expected_outputs = []
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        key_pages = _encode_blocks_for_head(key_blocks[kv_head_id], config, kind="K", layer_id=0, kv_head_id=kv_head_id)
        value_pages = _encode_blocks_for_head(value_blocks[kv_head_id], config, kind="V", layer_id=0, kv_head_id=kv_head_id)
        expected_outputs.append(decode_step(queries[q_head_id], key_pages, value_pages, backend="cpu_ref")[2])

    np.testing.assert_allclose(outputs, np.stack(expected_outputs, axis=0), atol=1e-5, rtol=1e-5)


def test_vllm_live_partial_block_stays_visible_without_finalization() -> None:
    rng = np.random.default_rng(1203)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = VllmPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_size=4,
        backend="cpu_ref",
    )
    key_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)
    value_blocks = rng.normal(size=(2, 2, 4, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.sync_layer_blocks(0, key_blocks, value_blocks, block_ids=[5, 9], live_block_token_count=2)
    live_entry = cache.block_entry(0, 0, 9, "K")
    assert live_entry.finalized is False
    assert live_entry.token_count == 2
    assert live_entry.page.header.mode_default == "M3"

    outputs = cache.decode_layer(0, queries, mapping)
    expected_outputs = []
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        key_pages = _encode_blocks_for_head(
            key_blocks[kv_head_id],
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            live_token_count=2,
        )
        value_pages = _encode_blocks_for_head(
            value_blocks[kv_head_id],
            config,
            kind="V",
            layer_id=0,
            kv_head_id=kv_head_id,
            live_token_count=2,
        )
        expected_outputs.append(decode_step(queries[q_head_id], key_pages, value_pages, backend="cpu_ref")[2])

    np.testing.assert_allclose(outputs, np.stack(expected_outputs, axis=0), atol=1e-5, rtol=1e-5)


def test_vllm_append_tokens_torch_chunks_prefill_on_block_boundaries(monkeypatch) -> None:
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = VllmPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_size=4,
        backend="cpu_ref",
    )
    key_rows = torch.zeros((10, 2, config.head_dim), dtype=torch.float32)
    value_rows = torch.zeros((10, 2, config.head_dim), dtype=torch.float32)
    positions = torch.arange(10, dtype=torch.long)
    calls: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

    def _record_append(layer_id, key_step, value_step, token_index, *, trace=None):
        del layer_id, trace
        calls.append((int(token_index), tuple(key_step.shape), tuple(value_step.shape)))

    monkeypatch.setattr(cache.model_kv_cache, "append_step_torch", _record_append)
    cache.append_tokens_torch(0, key_rows, value_rows, positions)

    assert calls == [
        (0, (2, 4, config.head_dim), (2, 4, config.head_dim)),
        (4, (2, 4, config.head_dim), (2, 4, config.head_dim)),
        (8, (2, 2, config.head_dim), (2, 2, config.head_dim)),
    ]


def test_grouped_prepared_chunk_cache_reuses_stable_m0_page_batches() -> None:
    clear_prepared_chunk_cache()
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    rng = np.random.default_rng(1206)

    def _prepared_page(*, kv_head_id: int, token_start: int, cache_uid: int) -> PreparedPageTorch:
        encoded = encode_page(
            rng.normal(size=(4, config.head_dim)).astype(np.float32),
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            token_start=token_start,
            build_runtime_metadata=False,
        )
        return PreparedPageTorch(
            device_type="cuda",
            source_page=encoded,
            header=encoded.header,
                payload=torch.from_numpy(encoded.payload.astype(np.int32, copy=True)),
            scales=torch.from_numpy(encoded.scales.copy()),
            bias=torch.from_numpy(encoded.bias.copy()),
            unpack_shifts=torch.arange(8, dtype=torch.int32) * encoded.header.bits,
            unpack_mask=torch.tensor((1 << encoded.header.bits) - 1, dtype=torch.int32),
            cache_uid=cache_uid,
        )

    pages_by_group = [
        [
            _prepared_page(kv_head_id=0, token_start=0, cache_uid=11),
            _prepared_page(kv_head_id=0, token_start=4, cache_uid=12),
            _prepared_page(kv_head_id=0, token_start=8, cache_uid=13),
            _prepared_page(kv_head_id=0, token_start=12, cache_uid=14),
        ],
        [
            _prepared_page(kv_head_id=1, token_start=0, cache_uid=21),
            _prepared_page(kv_head_id=1, token_start=4, cache_uid=22),
            _prepared_page(kv_head_id=1, token_start=8, cache_uid=23),
            _prepared_page(kv_head_id=1, token_start=12, cache_uid=24),
        ],
    ]

    first = _get_grouped_prepared_chunk_mps(pages_by_group)
    second = _get_grouped_prepared_chunk_mps(pages_by_group)

    assert first is not None
    assert second is first
    assert first.payload_groups == ()
    assert first.codes_groups is not None
    assert first.codes_groups[0].shape[0] == len(pages_by_group)
    assert first.codes_groups[0].shape[1] == len(pages_by_group[0])
    assert first.codes_groups[0].shape[-1] == config.group_size


def test_grouped_prepared_chunk_cache_builds_fused_cuda_view_for_two_group64() -> None:
    clear_prepared_chunk_cache()
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    rng = np.random.default_rng(1207)

    def _prepared_page(*, kv_head_id: int, token_start: int, cache_uid: int) -> PreparedPageTorch:
        encoded = encode_page(
            rng.normal(size=(4, config.head_dim)).astype(np.float32),
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            token_start=token_start,
            build_runtime_metadata=False,
        )
        return PreparedPageTorch(
            device_type="cuda",
            source_page=encoded,
            header=encoded.header,
            payload=torch.from_numpy(encoded.payload.astype(np.int32, copy=True)),
            scales=torch.from_numpy(encoded.scales.copy()),
            bias=torch.from_numpy(encoded.bias.copy()),
            unpack_shifts=torch.arange(8, dtype=torch.int32) * encoded.header.bits,
            unpack_mask=torch.tensor((1 << encoded.header.bits) - 1, dtype=torch.int32),
            cache_uid=cache_uid,
        )

    pages_by_group = [
        [
            _prepared_page(kv_head_id=0, token_start=0, cache_uid=111),
            _prepared_page(kv_head_id=0, token_start=4, cache_uid=112),
            _prepared_page(kv_head_id=0, token_start=8, cache_uid=113),
            _prepared_page(kv_head_id=0, token_start=12, cache_uid=114),
        ],
        [
            _prepared_page(kv_head_id=1, token_start=0, cache_uid=121),
            _prepared_page(kv_head_id=1, token_start=4, cache_uid=122),
            _prepared_page(kv_head_id=1, token_start=8, cache_uid=123),
            _prepared_page(kv_head_id=1, token_start=12, cache_uid=124),
        ],
    ]

    try:
        first = _get_grouped_prepared_chunk_mps(pages_by_group)
        second = _get_grouped_prepared_chunk_mps(pages_by_group)
    finally:
        clear_prepared_chunk_cache()

    assert first is not None
    assert second is first
    assert first.fused_scaled_codes is not None
    assert first.codes_groups is None
    assert first.scales_groups is None
    assert first.bias_groups is not None
    assert tuple(first.fused_scaled_codes.shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert first.fused_scaled_codes.dtype == torch.float16
    assert first.bias_groups[0].dtype == torch.float16


def test_grouped_prepared_chunk_cache_builds_packed_cuda_view_for_four_group128() -> None:
    clear_prepared_chunk_cache()
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    rng = np.random.default_rng(1208)

    def _prepared_page(*, kv_head_id: int, token_start: int, cache_uid: int) -> PreparedPageTorch:
        encoded = encode_page(
            rng.normal(size=(4, config.head_dim)).astype(np.float32),
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            token_start=token_start,
            build_runtime_metadata=False,
        )
        return PreparedPageTorch(
            device_type="cuda",
            source_page=encoded,
            header=encoded.header,
            payload=torch.from_numpy(encoded.payload.astype(np.int32, copy=True)),
            scales=torch.from_numpy(encoded.scales.copy()),
            bias=torch.from_numpy(encoded.bias.copy()),
            unpack_shifts=torch.arange(8, dtype=torch.int32) * encoded.header.bits,
            unpack_mask=torch.tensor((1 << encoded.header.bits) - 1, dtype=torch.int32),
            cache_uid=cache_uid,
        )

    pages_by_group = [
        [
            _prepared_page(kv_head_id=group_id, token_start=0, cache_uid=200 + group_id * 10 + 1),
            _prepared_page(kv_head_id=group_id, token_start=4, cache_uid=200 + group_id * 10 + 2),
            _prepared_page(kv_head_id=group_id, token_start=8, cache_uid=200 + group_id * 10 + 3),
            _prepared_page(kv_head_id=group_id, token_start=12, cache_uid=200 + group_id * 10 + 4),
        ]
        for group_id in range(4)
    ]

    try:
        first = _get_grouped_prepared_chunk_mps(pages_by_group)
        second = _get_grouped_prepared_chunk_mps(pages_by_group)
    finally:
        clear_prepared_chunk_cache()

    assert first is not None
    assert second is first
    assert first.payload_groups != ()
    assert first.codes_groups is None
    assert first.scales_groups is not None
    assert first.bias_groups is not None
    assert first.fused_scaled_codes is None
    assert tuple(first.payload_groups[0].shape) == (len(pages_by_group), len(pages_by_group[0]), config.tokens_per_page, 4)
    assert first.scales_groups[0].dtype == torch.float32
    assert first.bias_groups[0].dtype == torch.float32


def test_grouped_prepared_chunk_cache_builds_m2_cuda_view() -> None:
    clear_prepared_chunk_cache()
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    rng = np.random.default_rng(1209)

    def _prepared_page(*, kv_head_id: int, token_start: int, cache_uid: int) -> PreparedPageTorch:
        encoded = encode_page(
            rng.normal(size=(4, config.head_dim)).astype(np.float32),
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            token_start=token_start,
            mode="M2",
            quant_scheme="sketch",
            build_runtime_metadata=False,
        )
        return PreparedPageTorch(
            device_type="cuda",
            source_page=encoded,
            header=encoded.header,
            m2_sketch=torch.from_numpy(encoded.m2_sketch.copy()),
            m2_basis=torch.from_numpy(encoded.m2_basis.copy()),
            m2_mean=torch.from_numpy(encoded.m2_mean.copy()),
            cache_uid=cache_uid,
        )

    pages_by_group = [
        [
            _prepared_page(kv_head_id=0, token_start=0, cache_uid=301),
            _prepared_page(kv_head_id=0, token_start=4, cache_uid=302),
            _prepared_page(kv_head_id=0, token_start=8, cache_uid=303),
            _prepared_page(kv_head_id=0, token_start=12, cache_uid=304),
        ],
        [
            _prepared_page(kv_head_id=1, token_start=0, cache_uid=311),
            _prepared_page(kv_head_id=1, token_start=4, cache_uid=312),
            _prepared_page(kv_head_id=1, token_start=8, cache_uid=313),
            _prepared_page(kv_head_id=1, token_start=12, cache_uid=314),
        ],
    ]

    try:
        first = _get_grouped_prepared_chunk_mps(pages_by_group)
        second = _get_grouped_prepared_chunk_mps(pages_by_group)
    finally:
        clear_prepared_chunk_cache()

    assert first is not None
    assert second is first
    assert first.m2_sketch_groups is not None
    assert first.m2_basis_groups is not None
    assert first.m2_mean_groups is not None
    assert first.m2_sketch_tensor is not None
    assert first.m2_basis_tensor is not None
    assert first.m2_mean_tensor is not None
    assert first.codes_groups is None
    assert tuple(first.m2_sketch_groups[0].shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert tuple(first.m2_basis_groups[0].shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert tuple(first.m2_mean_groups[0].shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert tuple(first.m2_sketch_tensor.shape[:4]) == (
        len(pages_by_group),
        len(pages_by_group[0]),
        config.tokens_per_page,
        config.head_dim // config.group_size,
    )
    assert tuple(first.m2_basis_tensor.shape[:3]) == (
        len(pages_by_group),
        len(pages_by_group[0]),
        config.head_dim // config.group_size,
    )
    assert tuple(first.m2_mean_tensor.shape[:3]) == (
        len(pages_by_group),
        len(pages_by_group[0]),
        config.head_dim // config.group_size,
    )


def test_grouped_prepared_chunk_cache_builds_m4_cuda_view() -> None:
    clear_prepared_chunk_cache()
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    rng = np.random.default_rng(1210)

    def _prepared_page(*, kv_head_id: int, token_start: int, cache_uid: int) -> PreparedPageTorch:
        encoded = encode_page(
            rng.normal(size=(4, config.head_dim)).astype(np.float32),
            config,
            kind="K",
            layer_id=0,
            kv_head_id=kv_head_id,
            token_start=token_start,
            mode="M4",
            quant_scheme="project",
            build_runtime_metadata=False,
        )
        return PreparedPageTorch(
            device_type="cuda",
            source_page=encoded,
            header=encoded.header,
            m2_sketch=torch.from_numpy(encoded.m2_sketch.copy()),
            m2_mean=torch.from_numpy(encoded.m2_mean.copy()),
            cache_uid=cache_uid,
        )

    pages_by_group = [
        [
            _prepared_page(kv_head_id=0, token_start=0, cache_uid=321),
            _prepared_page(kv_head_id=0, token_start=4, cache_uid=322),
            _prepared_page(kv_head_id=0, token_start=8, cache_uid=323),
            _prepared_page(kv_head_id=0, token_start=12, cache_uid=324),
        ],
        [
            _prepared_page(kv_head_id=1, token_start=0, cache_uid=331),
            _prepared_page(kv_head_id=1, token_start=4, cache_uid=332),
            _prepared_page(kv_head_id=1, token_start=8, cache_uid=333),
            _prepared_page(kv_head_id=1, token_start=12, cache_uid=334),
        ],
    ]

    try:
        first = _get_grouped_prepared_chunk_mps(pages_by_group)
        second = _get_grouped_prepared_chunk_mps(pages_by_group)
    finally:
        clear_prepared_chunk_cache()

    assert first is not None
    assert second is first
    assert first.m2_sketch_groups is not None
    assert first.m2_basis_groups is None
    assert first.m2_mean_groups is not None
    assert first.m2_sketch_tensor is not None
    assert first.m2_basis_tensor is None
    assert first.m2_mean_tensor is not None
    assert tuple(first.m2_sketch_groups[0].shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert tuple(first.m2_mean_groups[0].shape[:2]) == (len(pages_by_group), len(pages_by_group[0]))
    assert tuple(first.m2_sketch_tensor.shape[:4]) == (
        len(pages_by_group),
        len(pages_by_group[0]),
        config.tokens_per_page,
        config.head_dim // config.group_size,
    )
    assert tuple(first.m2_mean_tensor.shape[:3]) == (
        len(pages_by_group),
        len(pages_by_group[0]),
        config.head_dim // config.group_size,
    )


torch = pytest.importorskip("torch")


class _FakeQKVProj(torch.nn.Module):
    def __init__(self, hidden_size: int, q_size: int, kv_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, q_size + 2 * kv_size, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states), None


class _FakeOProj(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states), None


class _FakeRotary(torch.nn.Module):
    def forward(self, positions, query, key):
        del positions
        return query, key


class _FakeVllmLlamaAttention(torch.nn.Module):
    def __init__(self, *, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_proj = _FakeQKVProj(hidden_size, self.q_size, self.kv_size)
        self.o_proj = _FakeOProj(hidden_size)
        self.rotary_emb = _FakeRotary()
        self.forward_call_count = 0
        self.single_token_forward_call_count = 0
        self.register_buffer("_dense_keys", torch.zeros((self.num_kv_heads, 0, self.head_dim), dtype=torch.float32))
        self.register_buffer("_dense_values", torch.zeros((self.num_kv_heads, 0, self.head_dim), dtype=torch.float32))
        self._mapping = torch.from_numpy(default_q_head_to_kv_head(self.num_heads, self.num_kv_heads))

    def clear_cache(self) -> None:
        self._dense_keys = torch.zeros((self.num_kv_heads, 0, self.head_dim), dtype=torch.float32, device=self._dense_keys.device)
        self._dense_values = torch.zeros((self.num_kv_heads, 0, self.head_dim), dtype=torch.float32, device=self._dense_values.device)

    def forward(self, positions, hidden_states):
        del positions
        token_count = int(hidden_states.shape[0])
        self.forward_call_count += 1
        if token_count == 1:
            self.single_token_forward_call_count += 1
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query, key = self.rotary_emb(None, query, key)
        query_rows = query.view(token_count, self.num_heads, self.head_dim)
        key_rows = key.view(token_count, self.num_kv_heads, self.head_dim)
        value_rows = value.view(token_count, self.num_kv_heads, self.head_dim)
        outputs = []

        dense_keys = self._dense_keys
        dense_values = self._dense_values
        for token_index in range(token_count):
            dense_keys = torch.cat([dense_keys, key_rows[token_index].unsqueeze(1)], dim=1)
            dense_values = torch.cat([dense_values, value_rows[token_index].unsqueeze(1)], dim=1)
            head_outputs = []
            for q_head_id, kv_head_id in enumerate(self._mapping.tolist()):
                logits = torch.matmul(dense_keys[kv_head_id], query_rows[token_index, q_head_id]) * self.scaling
                weights = torch.softmax(logits, dim=0)
                context = torch.matmul(weights.unsqueeze(0), dense_values[kv_head_id]).squeeze(0)
                head_outputs.append(context)
            outputs.append(torch.cat(head_outputs, dim=0))
        self._dense_keys = dense_keys
        self._dense_values = dense_values
        projected, _ = self.o_proj(torch.stack(outputs, dim=0))
        return projected


class _FakeLayer(torch.nn.Module):
    def __init__(self, *, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.self_attn = _FakeVllmLlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )


class _FakeInnerModel(torch.nn.Module):
    def __init__(self, *, hidden_size: int, num_heads: int, num_kv_heads: int, num_hidden_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _FakeLayer(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)
                for _ in range(num_hidden_layers)
            ]
        )


class _FakeVllmExecutorModel(torch.nn.Module):
    def __init__(self, *, hidden_size: int = 32, num_heads: int = 4, num_kv_heads: int = 2, num_hidden_layers: int = 1) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )
        self.model = _FakeInnerModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_hidden_layers=num_hidden_layers,
        )


def test_vllm_active_matches_shadow_dotcache_output_for_finalized_prefill_case() -> None:
    torch.manual_seed(1204)
    config = DotCacheConfig(head_dim=8, group_size=8, bits_k=4, bits_v=4, tokens_per_page=4)
    shadow_model = _FakeVllmExecutorModel(hidden_size=32, num_heads=4, num_kv_heads=2, num_hidden_layers=1)
    active_model = deepcopy(shadow_model)

    shadow_adapter = install_dotcache_on_vllm_model(
        shadow_model,
        config,
        block_size=4,
        backend="cpu_ref",
        mode="dotcache_shadow",
    )
    active_adapter = install_dotcache_on_vllm_model(
        active_model,
        config,
        block_size=4,
        backend="cpu_ref",
        mode="dotcache_active",
    )

    prefill_positions = torch.arange(4, dtype=torch.long)
    prefill_hidden = torch.randn(4, 32, dtype=torch.float32)
    decode_positions = torch.tensor([4], dtype=torch.long)
    decode_hidden = torch.randn(1, 32, dtype=torch.float32)

    shadow_model.model.layers[0].self_attn(prefill_positions, prefill_hidden)
    active_model.model.layers[0].self_attn(prefill_positions, prefill_hidden)

    shadow_output = shadow_model.model.layers[0].self_attn(decode_positions, decode_hidden)
    active_output = active_model.model.layers[0].self_attn(decode_positions, decode_hidden)
    shadow_dotcache_output = shadow_adapter.last_dotcache_output(0)

    assert tuple(shadow_output.shape) == (1, 32)
    np.testing.assert_allclose(
        active_output.detach().cpu().numpy(),
        shadow_dotcache_output.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert shadow_adapter.shadow_output_max_abs_error >= 0.0
    assert active_adapter.resident_bytes >= 0


def test_vllm_active_decode_skips_dense_attention_forward() -> None:
    torch.manual_seed(1205)
    config = DotCacheConfig(head_dim=8, group_size=8, bits_k=4, bits_v=4, tokens_per_page=4)
    shadow_model = _FakeVllmExecutorModel(hidden_size=32, num_heads=4, num_kv_heads=2, num_hidden_layers=1)
    active_model = deepcopy(shadow_model)

    shadow_adapter = install_dotcache_on_vllm_model(
        shadow_model,
        config,
        block_size=4,
        backend="cpu_ref",
        mode="dotcache_shadow",
    )
    active_adapter = install_dotcache_on_vllm_model(
        active_model,
        config,
        block_size=4,
        backend="cpu_ref",
        mode="dotcache_active",
    )

    prefill_positions = torch.arange(4, dtype=torch.long)
    prefill_hidden = torch.randn(4, 32, dtype=torch.float32)
    decode_inputs = [
        (torch.tensor([4], dtype=torch.long), torch.randn(1, 32, dtype=torch.float32)),
        (torch.tensor([5], dtype=torch.long), torch.randn(1, 32, dtype=torch.float32)),
    ]

    shadow_layer = shadow_model.model.layers[0].self_attn
    active_layer = active_model.model.layers[0].self_attn
    shadow_base_attention = shadow_layer.base_attention
    active_base_attention = active_layer.base_attention

    shadow_layer(prefill_positions, prefill_hidden)
    active_layer(prefill_positions, prefill_hidden)
    assert shadow_base_attention.single_token_forward_call_count == 0
    assert active_base_attention.single_token_forward_call_count == 0

    for decode_positions, decode_hidden in decode_inputs:
        shadow_output = shadow_layer(decode_positions, decode_hidden)
        active_output = active_layer(decode_positions, decode_hidden)
        np.testing.assert_allclose(
            active_output.detach().cpu().numpy(),
            shadow_adapter.last_dotcache_output(0).detach().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
        assert tuple(shadow_output.shape) == (1, 32)

    assert shadow_base_attention.single_token_forward_call_count == len(decode_inputs)
    assert active_base_attention.single_token_forward_call_count == 0
    assert active_adapter.resident_bytes >= 0
