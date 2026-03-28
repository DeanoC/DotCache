import numpy as np
import pytest

from dotcache.attention_reference import softmax
from dotcache.attention_runtime import (
    decode_step,
    decode_step_with_page_logits,
    mix_page,
    prepare_page,
    prepare_pages,
    score_page,
    score_pages,
)
from dotcache.backends import mps_available
from dotcache.backends.torch_mps import _get_prepared_chunk_mps, prepare_m0_affine_pages_from_tensor_torch
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.page_cache import PreparedPageCache
from dotcache.session_runtime import (
    PagedDecodeSession,
    select_execution_page_indices,
    select_execution_page_pairs,
    sketch_key_page,
)
from dotcache.tracing import ExecutionTrace

requires_mps = pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")


def _encode_paged(values: np.ndarray, config: DotCacheConfig, *, kind: str, mode: str = "M0") -> list:
    pages = []
    for token_start in range(0, values.shape[0], config.tokens_per_page):
        token_end = min(token_start + config.tokens_per_page, values.shape[0])
        pages.append(
            encode_page(
                values[token_start:token_end],
                config,
                kind=kind,
                mode=mode,
                token_start=token_start,
            )
        )
    return pages


def test_auto_backend_falls_back_to_cpu_for_symmetric_page() -> None:
    rng = np.random.default_rng(21)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4, quant_scheme_k="symmetric")
    keys = rng.normal(size=(8, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)
    page = encode_page(keys, config, kind="K", mode="M0", quant_scheme="symmetric")

    trace = ExecutionTrace()
    logits = score_page(query, page, backend="auto", trace=trace)

    assert logits.shape == (8,)
    assert trace.host_to_device_bytes == 0
    assert trace.m0_full_page_materializations == 0


def test_encode_page_stores_runtime_sketch_metadata() -> None:
    rng = np.random.default_rng(19)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, tokens_per_page=64)
    keys = rng.normal(size=(64, config.head_dim)).astype(np.float32)
    page = encode_page(keys, config, kind="K")

    assert page.runtime_page_mean is not None
    assert page.runtime_page_sketch is not None
    assert page.runtime_page_min is not None
    assert page.runtime_page_max is not None
    assert page.runtime_page_mean.shape == (config.head_dim,)
    assert page.runtime_page_sketch.shape == (4, config.head_dim)
    assert page.runtime_page_min.shape == (config.head_dim,)
    assert page.runtime_page_max.shape == (config.head_dim,)


def test_select_execution_page_pairs_keeps_sink_and_recent_pages() -> None:
    rng = np.random.default_rng(20)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    keys = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32)
    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    selected_key_pages, selected_value_pages = select_execution_page_pairs(
        key_pages,
        value_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
    )

    assert [page.header.token_start for page in selected_key_pages] == [0, 256]
    assert [page.header.token_start for page in selected_value_pages] == [0, 256]


def test_select_execution_page_indices_can_admit_relevant_old_pages() -> None:
    rng = np.random.default_rng(201)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    keys = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32) * 0.01
    values = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys[128:192, 0] = 10.0
    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    del value_pages

    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0
    summaries = [np.full(config.head_dim, -1.0, dtype=np.float32) for _ in key_pages]
    summaries[2] = np.concatenate([np.array([10.0], dtype=np.float32), np.zeros(config.head_dim - 1, dtype=np.float32)])

    selected_indices = select_execution_page_indices(
        key_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
        query_slice=query,
        key_page_sketches=[summary[None, :] for summary in summaries],
        relevance_top_k=1,
    )

    assert selected_indices == [0, 2, 4]


def test_sketch_key_page_preserves_chunk_local_signal() -> None:
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, tokens_per_page=64)
    values = np.zeros((64, config.head_dim), dtype=np.float32)
    values[:16, 0] = 8.0
    values[16:, 0] = -2.0
    page = encode_page(values, config, kind="K")

    sketch = sketch_key_page(page, sketch_size=4)

    assert sketch.shape == (4, config.head_dim)
    assert float(sketch[0, 0]) > 6.0
    assert float(sketch[-1, 0]) < -1.0


def test_select_execution_page_indices_prefers_chunk_sketch_match_over_page_mean() -> None:
    rng = np.random.default_rng(202)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    keys = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32) * 0.01
    key_pages = _encode_paged(keys, config, kind="K")
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    mean_like_sketches = [np.full((1, config.head_dim), -1.0, dtype=np.float32) for _ in key_pages]
    chunked_sketches = [np.full((4, config.head_dim), -1.0, dtype=np.float32) for _ in key_pages]
    chunked_sketches[2][0, 0] = 10.0

    selected_with_mean = select_execution_page_indices(
        key_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
        query_slice=query,
        key_page_sketches=mean_like_sketches,
        relevance_top_k=1,
    )
    selected_with_chunked = select_execution_page_indices(
        key_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
        query_slice=query,
        key_page_sketches=chunked_sketches,
        relevance_top_k=1,
    )

    assert selected_with_mean != [0, 2, 4]
    assert selected_with_chunked == [0, 2, 4]


def test_select_execution_page_indices_envelope_admits_spiky_old_page() -> None:
    rng = np.random.default_rng(203)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, tokens_per_page=64)
    keys = rng.normal(size=(5 * config.tokens_per_page, config.head_dim)).astype(np.float32) * 0.01
    key_pages = _encode_paged(keys, config, kind="K")
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    minima = [np.full(config.head_dim, -1.0, dtype=np.float32) for _ in key_pages]
    maxima = [np.full(config.head_dim, -1.0, dtype=np.float32) for _ in key_pages]
    maxima[2][0] = 10.0

    selected_indices = select_execution_page_indices(
        key_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
        query_slice=query,
        key_page_minima=minima,
        key_page_maxima=maxima,
        relevance_top_k=1,
        relevance_mode="envelope",
    )

    assert selected_indices == [0, 2, 4]


@requires_mps
def test_prepare_page_mps_reuses_device_copy() -> None:
    rng = np.random.default_rng(22)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4)
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    page = encode_page(keys, config, kind="K", mode="M0")

    trace = ExecutionTrace()
    prepared = prepare_page(page, backend="torch_mps", trace=trace)
    first_copy_bytes = trace.host_to_device_bytes

    prepared_again = prepare_page(prepared, backend="torch_mps", trace=trace)

    assert prepared_again is prepared
    assert first_copy_bytes > 0
    assert trace.host_to_device_bytes == first_copy_bytes


@requires_mps
def test_prepare_pages_mps_batches_upload_without_widening_affine_metadata() -> None:
    rng = np.random.default_rng(122)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, tokens_per_page=64)
    keys = rng.normal(size=(2 * config.tokens_per_page, config.head_dim)).astype(np.float32)
    key_pages = _encode_paged(keys, config, kind="K")

    trace = ExecutionTrace()
    prepared_pages = prepare_pages(key_pages, backend="torch_mps", trace=trace)

    expected_bytes = sum(page.payload.nbytes + page.scales.nbytes + page.bias.nbytes for page in key_pages)
    assert len(prepared_pages) == len(key_pages)
    assert trace.host_to_device_bytes == expected_bytes
    assert sum(page.host_to_device_nbytes for page in prepared_pages) == expected_bytes


@requires_mps
@pytest.mark.parametrize(("token_count", "head_dim", "bits"), [(8, 48, 4), (64, 128, 4), (64, 128, 3)])
def test_score_page_mps_matches_cpu_reference(token_count: int, head_dim: int, bits: int) -> None:
    rng = np.random.default_rng(token_count + head_dim + bits)
    config = DotCacheConfig(head_dim=head_dim, group_size=32, bits_k=bits, tokens_per_page=64)
    keys = rng.normal(size=(token_count, head_dim)).astype(np.float32)
    query = rng.normal(size=(head_dim,)).astype(np.float32)
    page = encode_page(keys, config, kind="K", mode="M0")

    prepared = prepare_page(page, backend="torch_mps")
    cpu_logits = score_page(query, page, backend="cpu_ref")
    trace = ExecutionTrace()
    mps_logits = score_page(query, prepared, backend="torch_mps", trace=trace)

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    assert trace.payload_bytes_read == page.payload_nbytes
    assert trace.metadata_bytes_read == page.metadata_nbytes
    assert trace.max_temporary_bytes > 0
    assert trace.m0_full_page_materializations == 0
    assert page.full_page_decode_calls == 0


@requires_mps
def test_m1_pages_work_on_mps() -> None:
    rng = np.random.default_rng(2401)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4)
    keys = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    attn = softmax(rng.normal(size=(16,)).astype(np.float32))

    key_page = encode_page(keys, config, kind="K", mode="M1")
    value_page = encode_page(values, config, kind="V", mode="M1")
    prepared_key_page = prepare_page(key_page, backend="torch_mps")
    prepared_value_page = prepare_page(value_page, backend="torch_mps")

    np.testing.assert_allclose(
        score_page(query, prepared_key_page, backend="torch_mps"),
        score_page(query, key_page, backend="cpu_ref"),
        atol=3e-3,
        rtol=3e-3,
    )
    np.testing.assert_allclose(
        mix_page(attn, prepared_value_page, backend="torch_mps"),
        mix_page(attn, value_page, backend="cpu_ref"),
        atol=3e-3,
        rtol=3e-3,
    )


@requires_mps
def test_t3_pages_work_on_mps() -> None:
    rng = np.random.default_rng(2402)
    config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        default_mode_k="T3",
        default_mode_v="T3",
        quant_scheme_k="turbo3",
        quant_scheme_v="turbo3",
    )
    keys = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    attn = softmax(rng.normal(size=(16,)).astype(np.float32))

    key_page = encode_page(keys, config, kind="K", mode="T3")
    value_page = encode_page(values, config, kind="V", mode="T3")
    prepared_key_page = prepare_page(key_page, backend="torch_mps")
    prepared_value_page = prepare_page(value_page, backend="torch_mps")

    np.testing.assert_allclose(
        score_page(query, prepared_key_page, backend="torch_mps"),
        score_page(query, key_page, backend="cpu_ref"),
        atol=3e-3,
        rtol=3e-3,
    )
    np.testing.assert_allclose(
        mix_page(attn, prepared_value_page, backend="torch_mps"),
        mix_page(attn, value_page, backend="cpu_ref"),
        atol=3e-3,
        rtol=3e-3,
    )


@requires_mps
def test_mix_page_mps_matches_cpu_reference() -> None:
    rng = np.random.default_rng(24)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_v=4)
    values = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    attn = softmax(rng.normal(size=(config.tokens_per_page,)).astype(np.float32))
    page = encode_page(values, config, kind="V", mode="M0")

    prepared = prepare_page(page, backend="torch_mps")
    cpu_output = mix_page(attn, page, backend="cpu_ref")
    trace = ExecutionTrace()
    mps_output = mix_page(attn, prepared, backend="torch_mps", trace=trace)

    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-4, rtol=1e-4)
    assert trace.payload_bytes_read == page.payload_nbytes
    assert trace.metadata_bytes_read == page.metadata_nbytes
    assert trace.max_temporary_bytes > 0
    assert trace.m0_full_page_materializations == 0
    assert page.full_page_decode_calls == 0


@requires_mps
def test_decode_step_mps_matches_cpu_reference_across_pages() -> None:
    rng = np.random.default_rng(25)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 160
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_logits, cpu_weights, cpu_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")

    prep_trace = ExecutionTrace()
    prepared_key_pages = prepare_pages(key_pages, backend="torch_mps", trace=prep_trace)
    prepared_value_pages = prepare_pages(value_pages, backend="torch_mps", trace=prep_trace)
    exec_trace = ExecutionTrace()
    mps_logits, mps_weights, mps_output = decode_step(
        query,
        prepared_key_pages,
        prepared_value_pages,
        backend="torch_mps",
        trace=exec_trace,
    )

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-2, rtol=1e-3)
    assert prep_trace.host_to_device_bytes > 0
    assert exec_trace.host_to_device_bytes == 0
    assert exec_trace.m0_full_page_materializations == 0
    assert exec_trace.payload_bytes_read == sum(page.payload_nbytes for page in key_pages + value_pages)


@requires_mps
@pytest.mark.parametrize("bits", [4, 3])
def test_decode_step_mps_matches_cpu_reference_for_uniform_batched_pages(bits: int) -> None:
    rng = np.random.default_rng(125)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=bits, bits_v=bits, tokens_per_page=64)
    context_length = 256
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_logits, cpu_weights, cpu_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")

    prepared_key_pages = prepare_pages(key_pages, backend="torch_mps")
    prepared_value_pages = prepare_pages(value_pages, backend="torch_mps")
    exec_trace = ExecutionTrace()
    mps_logits, mps_weights, mps_output = decode_step(
        query,
        prepared_key_pages,
        prepared_value_pages,
        backend="torch_mps",
        trace=exec_trace,
    )

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-2, rtol=1e-3)
    assert exec_trace.host_to_device_bytes == 0
    assert exec_trace.m0_full_page_materializations == 0
    assert exec_trace.payload_bytes_read == sum(page.payload_nbytes for page in key_pages + value_pages)


@requires_mps
def test_prepared_chunk_cache_builds_fused_codes_for_m0_3bit_pages() -> None:
    rng = np.random.default_rng(2125)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=3, bits_v=3, tokens_per_page=16)
    context_length = 64
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    prepared_key_pages = prepare_pages(key_pages, backend="torch_mps")
    prepared_chunk = _get_prepared_chunk_mps(prepared_key_pages)

    assert prepared_chunk is not None
    assert prepared_chunk.fused_scaled_codes is not None
    assert prepared_chunk.codes_groups is None
    assert prepared_chunk.scales_groups is None
    assert prepared_chunk.bias_groups is not None


@requires_mps
def test_direct_m0_3bit_preparation_from_tensor_matches_cpu_reference() -> None:
    import torch

    rng = np.random.default_rng(2126)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=3, bits_v=3, tokens_per_page=16)
    keys = rng.normal(size=(32, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(32, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    direct_key_pages = prepare_m0_affine_pages_from_tensor_torch(
        torch.from_numpy(keys.reshape(2, 16, config.head_dim)).to(device="mps"),
        config=config,
        kind="K",
        layer_id=0,
        kv_head_id=0,
        token_start=0,
        device_type="mps",
    )
    direct_value_pages = prepare_m0_affine_pages_from_tensor_torch(
        torch.from_numpy(values.reshape(2, 16, config.head_dim)).to(device="mps"),
        config=config,
        kind="V",
        layer_id=0,
        kv_head_id=0,
        token_start=0,
        device_type="mps",
    )

    cpu_key_pages = _encode_paged(keys, config, kind="K")
    cpu_value_pages = _encode_paged(values, config, kind="V")
    cpu_logits, cpu_weights, cpu_output = decode_step(query, cpu_key_pages, cpu_value_pages, backend="cpu_ref")
    mps_logits, mps_weights, mps_output = decode_step(query, direct_key_pages, direct_value_pages, backend="torch_mps")

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-2, rtol=1e-3)


@requires_mps
@pytest.mark.parametrize("escape_dtype", ["float16", "int8"])
def test_m3_pages_work_on_mps(escape_dtype: str) -> None:
    rng = np.random.default_rng(26)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, escape_dtype=escape_dtype)
    keys = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    attn = softmax(rng.normal(size=(16,)).astype(np.float32))

    key_page = encode_page(keys, config, kind="K", mode="M3")
    value_page = encode_page(values, config, kind="V", mode="M3")
    prepared_key_page = prepare_page(key_page, backend="torch_mps")
    prepared_value_page = prepare_page(value_page, backend="torch_mps")

    np.testing.assert_allclose(
        score_page(query, prepared_key_page, backend="torch_mps"),
        score_page(query, key_page, backend="cpu_ref"),
        atol=0.12 if escape_dtype == "int8" else 1e-3,
        rtol=0.12 if escape_dtype == "int8" else 1e-3,
    )
    np.testing.assert_allclose(
        mix_page(attn, prepared_value_page, backend="torch_mps"),
        mix_page(attn, value_page, backend="cpu_ref"),
        atol=0.12 if escape_dtype == "int8" else 1e-3,
        rtol=0.12 if escape_dtype == "int8" else 1e-3,
    )


@requires_mps
def test_prepared_page_cache_reuses_mps_pages_across_decode_steps() -> None:
    rng = np.random.default_rng(27)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 160
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query_a = rng.normal(size=(config.head_dim,)).astype(np.float32)
    query_b = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    cache = PreparedPageCache()

    first_trace = ExecutionTrace()
    first_logits, _, first_output = decode_step(
        query_a,
        key_pages,
        value_pages,
        backend="torch_mps",
        cache=cache,
        trace=first_trace,
    )
    second_trace = ExecutionTrace()
    second_logits, _, second_output = decode_step(
        query_b,
        key_pages,
        value_pages,
        backend="torch_mps",
        cache=cache,
        trace=second_trace,
    )

    ref_logits_a, _, ref_output_a = decode_step(query_a, key_pages, value_pages, backend="cpu_ref")
    ref_logits_b, _, ref_output_b = decode_step(query_b, key_pages, value_pages, backend="cpu_ref")

    np.testing.assert_allclose(first_logits, ref_logits_a, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(first_output, ref_output_a, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(second_logits, ref_logits_b, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(second_output, ref_output_b, atol=1e-4, rtol=1e-4)

    total_pages = len(key_pages) + len(value_pages)
    assert first_trace.prepared_page_cache_misses == total_pages
    assert first_trace.prepared_page_cache_hits == 0
    assert first_trace.host_to_device_bytes > 0
    assert second_trace.prepared_page_cache_hits == total_pages
    assert second_trace.prepared_page_cache_misses == 0
    assert second_trace.host_to_device_bytes == 0
    assert second_trace.cache_resident_bytes == cache.resident_bytes


@requires_mps
def test_paged_decode_session_separates_preload_append_and_decode() -> None:
    rng = np.random.default_rng(127)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 192
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    base_key_pages = _encode_paged(keys[:128], config, kind="K")
    base_value_pages = _encode_paged(values[:128], config, kind="V")
    append_key_pages = _encode_paged(keys[128:], config, kind="K")
    append_value_pages = _encode_paged(values[128:], config, kind="V")
    all_key_pages = base_key_pages + append_key_pages
    all_value_pages = base_value_pages + append_value_pages

    session = PagedDecodeSession(backend="torch_mps", cache=PreparedPageCache())
    preload_trace = ExecutionTrace()
    session.preload(base_key_pages, base_value_pages, trace=preload_trace)
    append_trace = ExecutionTrace()
    session.append(append_key_pages, append_value_pages, trace=append_trace)
    decode_trace = ExecutionTrace()
    logits, weights, output = session.decode(query, trace=decode_trace)

    ref_logits, ref_weights, ref_output = decode_step(query, all_key_pages, all_value_pages, backend="cpu_ref")

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert preload_trace.host_to_device_bytes > 0
    assert append_trace.host_to_device_bytes > 0
    assert decode_trace.host_to_device_bytes == 0
    assert session.page_count == len(all_key_pages)


@requires_mps
def test_paged_decode_session_windowed_decode_matches_selected_page_reference() -> None:
    rng = np.random.default_rng(128)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    selected_key_pages, selected_value_pages = select_execution_page_pairs(
        key_pages,
        value_pages,
        recent_window_tokens=64,
        sink_window_tokens=64,
    )

    session = PagedDecodeSession(
        backend="torch_mps",
        cache=PreparedPageCache(),
        recent_window_tokens=64,
        sink_window_tokens=64,
    )
    session.preload(key_pages, value_pages)
    exec_trace = ExecutionTrace()
    logits, weights, output = session.decode(query, trace=exec_trace)

    ref_logits, ref_weights, ref_output = decode_step(
        query,
        selected_key_pages,
        selected_value_pages,
        backend="cpu_ref",
    )

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert session.active_page_count == len(selected_key_pages)
    assert session.active_token_count == sum(page.header.token_count for page in selected_key_pages)
    assert exec_trace.host_to_device_bytes == 0


@requires_mps
def test_paged_decode_session_hybrid_relevance_matches_selected_page_reference() -> None:
    rng = np.random.default_rng(129)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32) * 0.1
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    keys[128:192, 0] += 6.0
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    session = PagedDecodeSession(
        backend="torch_mps",
        cache=PreparedPageCache(),
        recent_window_tokens=64,
        sink_window_tokens=64,
        relevance_top_k=1,
        relevance_sketch_size=4,
    )
    session.preload(key_pages, value_pages)
    selected_key_pages, selected_value_pages = session.execution_pages(query)
    exec_trace = ExecutionTrace()
    logits, weights, output = session.decode(query, trace=exec_trace)

    ref_logits, ref_weights, ref_output = decode_step(
        query,
        selected_key_pages,
        selected_value_pages,
        backend="cpu_ref",
    )

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert [page.header.token_start for page in selected_key_pages] == [0, 128, 256]
    assert exec_trace.host_to_device_bytes == 0


def test_paged_decode_session_exact_refine_selects_best_old_candidate() -> None:
    rng = np.random.default_rng(132)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32) * 0.05
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    keys[64:128, 0] += 2.5
    keys[128:192, 0] += 6.0
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    session = PagedDecodeSession(
        backend="cpu_ref",
        recent_window_tokens=64,
        sink_window_tokens=64,
        relevance_top_k=2,
        relevance_sketch_size=4,
        exact_refine_top_k=1,
    )
    session.preload(key_pages, value_pages)

    selected_indices = session.execution_indices(query)

    assert [key_pages[index].header.token_start for index in selected_indices] == [0, 128, 256]


def test_paged_decode_session_approximate_old_pages_improves_over_blunt_pruning_on_cpu() -> None:
    rng = np.random.default_rng(130)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32) * 0.05
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    keys[128:192, 0] += 8.0
    values[128:192, 0] += 4.0
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    full_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")[2]

    pruned_session = PagedDecodeSession(
        backend="cpu_ref",
        recent_window_tokens=64,
        sink_window_tokens=64,
    )
    pruned_session.preload(key_pages, value_pages)
    pruned_output = pruned_session.decode(query)[2]

    approx_session = PagedDecodeSession(
        backend="cpu_ref",
        recent_window_tokens=64,
        sink_window_tokens=64,
        approximate_old_pages=True,
    )
    approx_session.preload(key_pages, value_pages)
    approx_output = approx_session.decode(query)[2]

    pruned_error = np.max(np.abs(pruned_output - full_output))
    approx_error = np.max(np.abs(approx_output - full_output))

    assert approx_error < pruned_error


@requires_mps
def test_paged_decode_session_approximate_old_pages_matches_cpu_runtime() -> None:
    rng = np.random.default_rng(131)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_session = PagedDecodeSession(
        backend="cpu_ref",
        recent_window_tokens=64,
        sink_window_tokens=64,
        approximate_old_pages=True,
    )
    cpu_session.preload(key_pages, value_pages)
    cpu_logits, cpu_weights, cpu_output = cpu_session.decode(query)

    mps_session = PagedDecodeSession(
        backend="torch_mps",
        cache=PreparedPageCache(),
        recent_window_tokens=64,
        sink_window_tokens=64,
        approximate_old_pages=True,
    )
    mps_session.preload(key_pages, value_pages)
    exec_trace = ExecutionTrace()
    mps_logits, mps_weights, mps_output = mps_session.decode(query, trace=exec_trace)

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-4, rtol=1e-4)
    assert exec_trace.host_to_device_bytes == 0


@requires_mps
def test_paged_decode_session_exact_refine_matches_selected_page_reference_on_mps() -> None:
    rng = np.random.default_rng(133)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32) * 0.05
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    keys[64:128, 0] += 2.0
    keys[128:192, 0] += 5.5
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    session = PagedDecodeSession(
        backend="torch_mps",
        cache=PreparedPageCache(),
        recent_window_tokens=64,
        sink_window_tokens=64,
        relevance_top_k=2,
        relevance_sketch_size=4,
        exact_refine_top_k=1,
    )
    session.preload(key_pages, value_pages)
    selected_indices = session.execution_indices(query)
    selected_key_pages = [key_pages[index] for index in selected_indices]
    selected_value_pages = [value_pages[index] for index in selected_indices]

    exec_trace = ExecutionTrace()
    logits, weights, output = session.decode(query, trace=exec_trace)
    ref_logits, ref_weights, ref_output = decode_step(
        query,
        selected_key_pages,
        selected_value_pages,
        backend="cpu_ref",
    )

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert [page.header.token_start for page in selected_key_pages] == [0, 128, 256]
    assert exec_trace.host_to_device_bytes == 0


@requires_mps
def test_paged_decode_session_envelope_relevance_matches_selected_page_reference_on_mps() -> None:
    rng = np.random.default_rng(135)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 320
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32) * 0.02
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    keys[128:129, 0] += 7.0
    query = np.zeros(config.head_dim, dtype=np.float32)
    query[0] = 1.0

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    session = PagedDecodeSession(
        backend="torch_mps",
        cache=PreparedPageCache(),
        recent_window_tokens=64,
        sink_window_tokens=64,
        relevance_top_k=1,
        relevance_mode="envelope",
    )
    session.preload(key_pages, value_pages)
    selected_indices = session.execution_indices(query)
    selected_key_pages = [key_pages[index] for index in selected_indices]
    selected_value_pages = [value_pages[index] for index in selected_indices]

    exec_trace = ExecutionTrace()
    logits, weights, output = session.decode(query, trace=exec_trace)
    ref_logits, ref_weights, ref_output = decode_step(
        query,
        selected_key_pages,
        selected_value_pages,
        backend="cpu_ref",
    )

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert [page.header.token_start for page in selected_key_pages] == [0, 128, 256]
    assert exec_trace.host_to_device_bytes == 0


@requires_mps
def test_decode_step_with_page_logits_reuses_precomputed_scores_on_mps() -> None:
    rng = np.random.default_rng(134)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 2 * config.tokens_per_page
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")
    precomputed_logits = score_pages(query, [key_pages[0]], backend="torch_mps")

    exec_trace = ExecutionTrace()
    logits, weights, output = decode_step_with_page_logits(
        query,
        key_pages,
        value_pages,
        page_logits=[precomputed_logits[0], None],
        backend="torch_mps",
        trace=exec_trace,
    )
    ref_logits, ref_weights, ref_output = decode_step(
        query,
        key_pages,
        value_pages,
        backend="cpu_ref",
    )
    baseline_trace = ExecutionTrace()
    decode_step_with_page_logits(
        query,
        key_pages,
        value_pages,
        backend="torch_mps",
        trace=baseline_trace,
    )

    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(weights, ref_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output, ref_output, atol=1e-4, rtol=1e-4)
    assert exec_trace.payload_bytes_read == baseline_trace.payload_bytes_read - key_pages[0].payload_nbytes
    assert exec_trace.metadata_bytes_read == baseline_trace.metadata_bytes_read - key_pages[0].metadata_nbytes


@requires_mps
def test_prepared_page_cache_evicts_oldest_pages_when_capacity_is_capped() -> None:
    rng = np.random.default_rng(28)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, tokens_per_page=64)
    keys_a = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_b = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    page_a = encode_page(keys_a, config, kind="K", token_start=0)
    page_b = encode_page(keys_b, config, kind="K", token_start=config.tokens_per_page)

    warm_trace = ExecutionTrace()
    warm_page = prepare_page(page_a, backend="torch_mps")
    cache = PreparedPageCache(max_resident_bytes=warm_page.resident_nbytes)
    cache.append_page(page_a, trace=warm_trace)

    evict_trace = ExecutionTrace()
    cache.append_page(page_b, trace=evict_trace)

    assert cache.size == 1
    assert evict_trace.prepared_page_cache_evictions == 1
    assert evict_trace.cache_evicted_bytes == warm_page.resident_nbytes
    assert cache.resident_bytes == warm_page.resident_nbytes


@requires_mps
def test_lru_policy_keeps_recently_reused_page_resident() -> None:
    rng = np.random.default_rng(29)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, tokens_per_page=64)
    keys_a = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_b = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_c = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    page_a = encode_page(keys_a, config, kind="K", token_start=0)
    page_b = encode_page(keys_b, config, kind="K", token_start=config.tokens_per_page)
    page_c = encode_page(keys_c, config, kind="K", token_start=2 * config.tokens_per_page)

    sample_prepared = prepare_page(page_a, backend="torch_mps")
    capacity = sample_prepared.resident_nbytes * 2
    cache = PreparedPageCache(max_resident_bytes=capacity, policy="lru")

    cache.append_pages([page_a, page_b])
    hit_trace = ExecutionTrace()
    cache.prepare_page(page_a, trace=hit_trace)
    assert hit_trace.prepared_page_cache_hits == 1

    evict_trace = ExecutionTrace()
    cache.append_page(page_c, trace=evict_trace)

    assert cache.size == 2
    assert evict_trace.prepared_page_cache_evictions == 1

    reuse_a_trace = ExecutionTrace()
    cache.prepare_page(page_a, trace=reuse_a_trace)
    assert reuse_a_trace.prepared_page_cache_hits == 1

    reuse_b_trace = ExecutionTrace()
    cache.prepare_page(page_b, trace=reuse_b_trace)
    assert reuse_b_trace.prepared_page_cache_misses == 1


@requires_mps
def test_pinned_recent_fifo_keeps_newest_page_resident_under_pressure() -> None:
    rng = np.random.default_rng(30)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, tokens_per_page=64)
    keys_a = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_b = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_c = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    page_a = encode_page(keys_a, config, kind="K", token_start=0)
    page_b = encode_page(keys_b, config, kind="K", token_start=config.tokens_per_page)
    page_c = encode_page(keys_c, config, kind="K", token_start=2 * config.tokens_per_page)

    sample_prepared = prepare_page(page_a, backend="torch_mps")
    capacity = sample_prepared.resident_nbytes * 2
    cache = PreparedPageCache(max_resident_bytes=capacity, policy="pinned_recent_fifo", pinned_recent_pages=1)

    cache.append_pages([page_a, page_b])

    evict_trace = ExecutionTrace()
    cache.append_page(page_c, trace=evict_trace)

    assert cache.size == 2
    assert evict_trace.prepared_page_cache_evictions == 1

    recent_trace = ExecutionTrace()
    cache.prepare_page(page_c, trace=recent_trace)
    assert recent_trace.prepared_page_cache_hits == 1

    oldest_trace = ExecutionTrace()
    cache.prepare_page(page_a, trace=oldest_trace)
    assert oldest_trace.prepared_page_cache_misses == 1
