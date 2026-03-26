import numpy as np
import pytest

from dotcache.attention_reference import softmax
from dotcache.attention_runtime import decode_step, mix_page, prepare_page, prepare_pages, score_page
from dotcache.backends import mps_available
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.page_cache import PreparedPageCache
from dotcache.session_runtime import PagedDecodeSession
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
@pytest.mark.parametrize(("token_count", "head_dim"), [(8, 48), (64, 128)])
def test_score_page_mps_matches_cpu_reference(token_count: int, head_dim: int) -> None:
    rng = np.random.default_rng(token_count + head_dim)
    config = DotCacheConfig(head_dim=head_dim, group_size=32, bits_k=4, tokens_per_page=64)
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

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-4, rtol=1e-4)
    assert prep_trace.host_to_device_bytes > 0
    assert exec_trace.host_to_device_bytes == 0
    assert exec_trace.m0_full_page_materializations == 0
    assert exec_trace.payload_bytes_read == sum(page.payload_nbytes for page in key_pages + value_pages)


@requires_mps
def test_decode_step_mps_matches_cpu_reference_for_uniform_batched_pages() -> None:
    rng = np.random.default_rng(125)
    config = DotCacheConfig(head_dim=256, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 192
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

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(mps_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-4, rtol=1e-4)
    assert exec_trace.host_to_device_bytes == 0
    assert exec_trace.m0_full_page_materializations == 0
    assert exec_trace.payload_bytes_read == sum(page.payload_nbytes for page in key_pages + value_pages)


@requires_mps
def test_m3_pages_work_on_mps() -> None:
    rng = np.random.default_rng(26)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4)
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
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        mix_page(attn, prepared_value_page, backend="torch_mps"),
        mix_page(attn, value_page, backend="cpu_ref"),
        atol=1e-3,
        rtol=1e-3,
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
def test_prepared_page_cache_evicts_oldest_pages_when_capacity_is_capped() -> None:
    rng = np.random.default_rng(28)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, tokens_per_page=64)
    keys_a = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    keys_b = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    page_a = encode_page(keys_a, config, kind="K", token_start=0)
    page_b = encode_page(keys_b, config, kind="K", token_start=config.tokens_per_page)

    warm_trace = ExecutionTrace()
    warm_page = prepare_page(page_a, backend="torch_mps")
    cache = PreparedPageCache(max_resident_bytes=warm_page.host_to_device_nbytes)
    cache.append_page(page_a, trace=warm_trace)

    evict_trace = ExecutionTrace()
    cache.append_page(page_b, trace=evict_trace)

    assert cache.size == 1
    assert evict_trace.prepared_page_cache_evictions == 1
    assert evict_trace.cache_evicted_bytes == warm_page.host_to_device_nbytes
    assert cache.resident_bytes == warm_page.host_to_device_nbytes


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
    capacity = sample_prepared.host_to_device_nbytes * 2
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
    capacity = sample_prepared.host_to_device_nbytes * 2
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
