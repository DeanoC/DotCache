import numpy as np

from dotcache.attention_runtime import decode_step
from dotcache.backends import (
    clear_prepared_chunk_cache,
    configure_prepared_chunk_cache,
    mps_available,
    prepared_chunk_cache_resident_bytes,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.model_kv_cache import ModelPagedKVCache, default_q_head_to_kv_head
from dotcache.tracing import ExecutionTrace


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
        mode = None if token_end - token_start == config.tokens_per_page else "M3"
        pages.append(
            encode_page(
                values[token_start:token_end],
                config,
                kind=kind,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=token_start,
                mode=mode,
                build_runtime_metadata=False,
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


def test_model_paged_kv_cache_reports_page_mode_summary() -> None:
    rng = np.random.default_rng(3041)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M1",
        default_mode_v="M1",
        quant_scheme_k="lut",
        quant_scheme_v="lut",
        m1_fallback_to_m0=True,
        m1_error_threshold=1e-6,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    summary = cache.page_mode_summary()

    assert int(summary["requested_m1_pages"]) == int(summary["total_static_pages"])
    assert int(summary["m1_fallback_pages"]) == int(summary["total_static_pages"])
    assert int(summary["m0_pages"]) == int(summary["total_static_pages"])
    assert int(summary["k_requested_m1_pages"]) == int(summary["k_total_static_pages"])
    assert int(summary["v_requested_m1_pages"]) == int(summary["v_total_static_pages"])
    assert int(summary["k_m1_fallback_pages"]) == int(summary["k_total_static_pages"])
    assert int(summary["v_m1_fallback_pages"]) == int(summary["v_total_static_pages"])
    assert float(summary["m1_trial_error_max"]) > 0.0
    assert float(summary["k_m1_trial_error_max"]) > 0.0
    assert float(summary["v_m1_trial_error_max"]) > 0.0
    assert float(summary["m1_trial_token_p95_error_max"]) > 0.0
    assert float(summary["k_m1_trial_token_p95_error_max"]) > 0.0
    assert float(summary["v_m1_trial_token_p95_error_max"]) > 0.0


def test_model_paged_kv_cache_reports_m2_sidecar_and_prefilter_stats() -> None:
    rng = np.random.default_rng(3042)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M0",
        default_mode_v="M0",
        quant_scheme_k="affine",
        quant_scheme_v="affine",
        m2_prefilter_top_k=1,
        m2_prefilter_min_pages=1,
        m2_sketch_dim_k=4,
    )
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
    outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    summary = cache.page_mode_summary()

    assert outputs.shape == (2, config.head_dim)
    assert int(summary["m2_sidecar_pages"]) == int(summary["k_total_static_pages"])
    assert int(summary["k_m2_sidecar_pages"]) == int(summary["k_total_static_pages"])
    assert int(summary["v_m2_sidecar_pages"]) == 0
    assert int(summary["m2_prefilter_top_k"]) == 1
    assert int(summary["m2_prefilter_min_pages"]) == 1
    assert int(summary["m2_prefilter_invocations"]) == 2
    assert int(summary["m2_prefilter_candidate_pages"]) == 4
    assert int(summary["m2_prefilter_selected_pages"]) == 2


def test_model_paged_kv_cache_persistent_mps_tail_avoids_decode_reupload() -> None:
    if not mps_available():
        return
    rng = np.random.default_rng(305)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    key_step = rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)
    value_step = rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    append_trace = ExecutionTrace()
    cache.append_step(0, key_step, value_step, 0, trace=append_trace)
    decode_trace = ExecutionTrace()
    outputs = cache.decode_layer(0, queries, np.array([0, 1]), trace=decode_trace)

    assert outputs.shape == (2, config.head_dim)
    assert append_trace.host_to_device_bytes > 0
    assert decode_trace.host_to_device_bytes == 0


def test_model_paged_kv_cache_decode_layer_torch_matches_numpy_path() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(306)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, mapping)
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="mps"), mapping)

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-5, rtol=1e-5)


def test_model_paged_kv_cache_append_step_torch_avoids_host_uploads() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(307)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    key_step = torch.from_numpy(rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)).to(device="mps")
    value_step = torch.from_numpy(rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    append_trace = ExecutionTrace()
    cache.append_step_torch(0, key_step, value_step, 0, trace=append_trace)
    decode_trace = ExecutionTrace()
    outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]), trace=decode_trace)

    assert tuple(outputs.shape) == (2, config.head_dim)
    assert append_trace.host_to_device_bytes == 0
    assert decode_trace.host_to_device_bytes == 0


def test_model_paged_kv_cache_ingest_prefill_cache_torch_keeps_remainder_on_device() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(308)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 3, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 3, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    ingest_trace = ExecutionTrace()
    cache.ingest_prefill_cache_torch(0, layer_keys, layer_values, trace=ingest_trace)
    decode_trace = ExecutionTrace()
    outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]), trace=decode_trace)

    assert tuple(outputs.shape) == (2, config.head_dim)
    assert ingest_trace.host_to_device_bytes == 0
    assert decode_trace.host_to_device_bytes == 0


def test_model_paged_kv_cache_ingest_prefill_cache_torch_defers_full_page_prepare() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(309)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 5, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 5, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    ingest_trace = ExecutionTrace()
    cache.ingest_prefill_cache_torch(0, layer_keys, layer_values, trace=ingest_trace)
    decode_trace = ExecutionTrace()
    outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]), trace=decode_trace)

    assert tuple(outputs.shape) == (2, config.head_dim)
    assert ingest_trace.host_to_device_bytes == 0
    assert decode_trace.host_to_device_bytes > 0


def test_model_paged_kv_cache_direct_prefill_pages_count_toward_resident_bytes() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(3091)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        resident_summary = cache.resident_byte_summary()

        assert cache.cache.resident_bytes == 0
        assert cache.resident_bytes == 8 * (64 + 8 + 8)
        assert resident_summary["kv_resident_bytes"] == cache.resident_bytes
        assert resident_summary["prepared_chunk_resident_bytes"] == 0
    finally:
        clear_prepared_chunk_cache()


def test_model_paged_kv_cache_static_chunk_cache_is_reused_across_decodes() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(310)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
    resident_before_decode = cache.resident_bytes
    first_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
    resident_after_first_decode = cache.resident_bytes
    second_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
    resident_after_second_decode = cache.resident_bytes
    third_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
    resident_after_third_decode = cache.resident_bytes

    assert tuple(first_outputs.shape) == (2, config.head_dim)
    assert tuple(second_outputs.shape) == (2, config.head_dim)
    assert tuple(third_outputs.shape) == (2, config.head_dim)
    np.testing.assert_allclose(
        first_outputs.detach().cpu().numpy(),
        second_outputs.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        second_outputs.detach().cpu().numpy(),
        third_outputs.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert resident_after_first_decode > resident_before_decode
    assert resident_after_third_decode == resident_after_second_decode


def test_model_paged_kv_cache_resident_byte_summary_separates_chunk_cache() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(3101)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        before = cache.resident_byte_summary()
        cache.decode_layer_torch(0, queries, np.array([0, 1]))
        after = cache.resident_byte_summary()
    finally:
        clear_prepared_chunk_cache()

    assert before["prepared_chunk_resident_bytes"] == 0
    assert before["resident_bytes"] == before["kv_resident_bytes"]
    assert after["prepared_chunk_resident_bytes"] > 0
    assert after["prepared_chunk_resident_bytes"] <= after["prepared_chunk_cache_budget_bytes"]
    assert after["resident_bytes"] == after["kv_resident_bytes"] + after["prepared_chunk_resident_bytes"]


def test_model_paged_kv_cache_adaptive_chunk_budget_tracks_kv_residency() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(3102)
    config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        prepared_chunk_cache_budget_ratio=0.25,
        prepared_chunk_cache_min_bytes=512,
        prepared_chunk_cache_max_bytes=4096,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(4, config.head_dim)).astype(np.float32)).to(device="mps")

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        cache.decode_layer_torch(0, queries, np.array([0, 0, 1, 1], dtype=np.int64))
        summary = cache.resident_byte_summary()
    finally:
        clear_prepared_chunk_cache()

    expected_budget = min(
        config.prepared_chunk_cache_max_bytes,
        max(config.prepared_chunk_cache_min_bytes, int(summary["kv_resident_bytes"] * config.prepared_chunk_cache_budget_ratio)),
    )
    assert summary["prepared_chunk_cache_budget_bytes"] == expected_budget
    assert summary["prepared_chunk_resident_bytes"] <= expected_budget


def test_model_paged_kv_cache_static_chunk_cache_respects_budget() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(311)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    configure_prepared_chunk_cache(max_resident_bytes=512, min_page_count=1)
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        first_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
        resident_after_first_decode = prepared_chunk_cache_resident_bytes()
        second_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
        resident_after_second_decode = prepared_chunk_cache_resident_bytes()
    finally:
        configure_prepared_chunk_cache(max_resident_bytes=64 * 1024 * 1024, min_page_count=4)
        clear_prepared_chunk_cache()

    np.testing.assert_allclose(
        first_outputs.detach().cpu().numpy(),
        second_outputs.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert resident_after_first_decode <= 512
    assert resident_after_second_decode <= 512


def test_model_paged_kv_cache_static_chunk_cache_can_disable_value_chunks() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(312)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="mps")

    configure_prepared_chunk_cache(cached_kinds=("K",), min_page_count=1)
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        first_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
        resident_after_first_decode = prepared_chunk_cache_resident_bytes()
        second_outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]))
        resident_after_second_decode = prepared_chunk_cache_resident_bytes()
    finally:
        configure_prepared_chunk_cache(cached_kinds=("K", "V"), min_page_count=4)
        clear_prepared_chunk_cache()

    np.testing.assert_allclose(
        first_outputs.detach().cpu().numpy(),
        second_outputs.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert resident_after_first_decode > 0
    assert resident_after_second_decode == resident_after_first_decode
