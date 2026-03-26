import numpy as np

from dotcache.attention_runtime import decode_step
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.model_kv_cache import ModelPagedKVCache, default_q_head_to_kv_head


def _encode_pages_for_head(
    values: np.ndarray,
    config: DotCacheConfig,
    *,
    kind: str,
    layer_id: int,
    kv_head_id: int,
) -> list:
    pages = []
    for token_start in range(0, values.shape[0], config.tokens_per_page):
        token_end = min(token_start + config.tokens_per_page, values.shape[0])
        pages.append(
            encode_page(
                values[token_start:token_end],
                config,
                kind=kind,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=token_start,
            )
        )
    return pages


def test_default_q_head_to_kv_head_maps_gqa_groups() -> None:
    mapping = default_q_head_to_kv_head(8, 2)
    np.testing.assert_array_equal(mapping, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64))


def test_model_paged_kv_cache_multihead_decode_matches_quantized_page_reference() -> None:
    rng = np.random.default_rng(301)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="cpu_ref",
    )

    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    outputs = cache.decode_layer(0, queries, mapping)

    expected_outputs = []
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        key_pages = _encode_pages_for_head(layer_keys[kv_head_id], config, kind="K", layer_id=0, kv_head_id=kv_head_id)
        value_pages = _encode_pages_for_head(layer_values[kv_head_id], config, kind="V", layer_id=0, kv_head_id=kv_head_id)
        expected_outputs.append(decode_step(queries[q_head_id], key_pages, value_pages, backend="cpu_ref")[2])

    np.testing.assert_allclose(outputs, np.stack(expected_outputs, axis=0), atol=1e-5, rtol=1e-5)


def test_model_paged_kv_cache_tail_buffer_matches_incremental_reference() -> None:
    rng = np.random.default_rng(302)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    mapping = default_q_head_to_kv_head(4, 2)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    key_history = np.zeros((2, 0, config.head_dim), dtype=np.float32)
    value_history = np.zeros((2, 0, config.head_dim), dtype=np.float32)

    for token_index in range(3):
        key_step = rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)
        value_step = rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)
        cache.append_step(0, key_step, value_step, token_index)
        key_history = np.concatenate([key_history, key_step], axis=1)
        value_history = np.concatenate([value_history, value_step], axis=1)

        outputs = cache.decode_layer(0, queries, mapping)
        expected_outputs = []
        for q_head_id, kv_head_id in enumerate(mapping.tolist()):
            key_pages = _encode_pages_for_head(
                key_history[kv_head_id],
                config,
                kind="K",
                layer_id=0,
                kv_head_id=kv_head_id,
            )
            value_pages = _encode_pages_for_head(
                value_history[kv_head_id],
                config,
                kind="V",
                layer_id=0,
                kv_head_id=kv_head_id,
            )
            expected_outputs.append(decode_step(queries[q_head_id], key_pages, value_pages, backend="cpu_ref")[2])
        np.testing.assert_allclose(outputs, np.stack(expected_outputs, axis=0), atol=1e-5, rtol=1e-5)


def test_model_paged_kv_cache_accepts_batched_prefill_cache_tensors() -> None:
    rng = np.random.default_rng(303)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="cpu_ref",
    )

    layer_keys = rng.normal(size=(1, 2, 6, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(1, 2, 6, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)

    assert cache.layer_sequence_length(0) == 6
    outputs = cache.decode_layer(0, queries, mapping)
    assert outputs.shape == (4, config.head_dim)


def test_model_paged_kv_cache_query_scale_matches_scaled_reference_query() -> None:
    rng = np.random.default_rng(304)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)
    cache.ingest_prefill_cache(0, layer_keys, layer_values)

    scaled_outputs = cache.decode_layer(0, queries, np.array([0, 1]), query_scale=0.5)

    expected_outputs = []
    for q_head_id, kv_head_id in enumerate([0, 1]):
        key_pages = _encode_pages_for_head(layer_keys[kv_head_id], config, kind="K", layer_id=0, kv_head_id=kv_head_id)
        value_pages = _encode_pages_for_head(layer_values[kv_head_id], config, kind="V", layer_id=0, kv_head_id=kv_head_id)
        expected_outputs.append(decode_step(queries[q_head_id] * np.float32(0.5), key_pages, value_pages, backend="cpu_ref")[2])

    np.testing.assert_allclose(scaled_outputs, np.stack(expected_outputs, axis=0), atol=1e-5, rtol=1e-5)
