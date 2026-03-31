import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.attention_runtime import decode_step, prepare_page
from dotcache.backends import (
    PreparedPageTorch,
    clear_prepared_chunk_cache,
    cuda_available,
    decode_grouped_multiquery_step_prepared_cuda_tensor,
    decode_grouped_multiquery_step_prepared_cuda_tensor_output_only,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.integrations.llama import LlamaDotCacheModelAdapter, run_llama_generation_harness
from dotcache.model_kv_cache import ModelPagedKVCache, default_q_head_to_kv_head
from dotcache.page_cache import PreparedPageCache
from dotcache.backends.torch_mps import prepare_m0_affine_pages_from_tensor_torch
from dotcache.tracing import ExecutionTrace

LlamaConfig = transformers.LlamaConfig
LlamaForCausalLM = transformers.LlamaForCausalLM

requires_cuda = pytest.mark.skipif(not cuda_available(), reason="torch_cuda is unavailable")


def _encode_paged(values: np.ndarray, config: DotCacheConfig, *, kind: str) -> list:
    pages = []
    for token_start in range(0, values.shape[0], config.tokens_per_page):
        token_end = min(token_start + config.tokens_per_page, values.shape[0])
        pages.append(encode_page(values[token_start:token_end], config, kind=kind, token_start=token_start))
    return pages


@requires_cuda
def test_score_page_cuda_matches_cpu_reference() -> None:
    rng = np.random.default_rng(901)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, tokens_per_page=64)
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    page = encode_page(keys, config, kind="K")

    prepared = prepare_page(page, backend="torch_cuda")
    cpu_logits = decode_step(query, [page], [encode_page(keys, config, kind="V")], backend="cpu_ref")[0]
    cuda_logits = decode_step(query, [prepared], [prepare_page(encode_page(keys, config, kind="V"), backend="torch_cuda")], backend="torch_cuda")[0]

    np.testing.assert_allclose(cuda_logits, cpu_logits, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_decode_step_cuda_matches_cpu_reference() -> None:
    rng = np.random.default_rng(902)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=64)
    context_length = 160
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_logits, cpu_weights, cpu_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")
    cache = PreparedPageCache()
    trace = ExecutionTrace()
    cuda_logits, cuda_weights, cuda_output = decode_step(
        query,
        key_pages,
        value_pages,
        backend="torch_cuda",
        cache=cache,
        trace=trace,
    )

    np.testing.assert_allclose(cuda_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cuda_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cuda_output, cpu_output, atol=1e-4, rtol=1e-4)
    assert trace.host_to_device_bytes > 0


@requires_cuda
@pytest.mark.parametrize("escape_dtype", ["float16", "int8"])
def test_m3_pages_work_on_cuda(escape_dtype: str) -> None:
    rng = np.random.default_rng(9020)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, escape_dtype=escape_dtype)
    keys = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(16, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    attn = np.exp(rng.normal(size=(16,)).astype(np.float32))
    attn /= attn.sum()

    key_page = encode_page(keys, config, kind="K", mode="M3")
    value_page = encode_page(values, config, kind="V", mode="M3")
    prepared_key_page = prepare_page(key_page, backend="torch_cuda")
    prepared_value_page = prepare_page(value_page, backend="torch_cuda")

    np.testing.assert_allclose(
        decode_step(query, [prepared_key_page], [prepared_value_page], backend="torch_cuda")[0],
        decode_step(query, [key_page], [value_page], backend="cpu_ref")[0],
        atol=0.12 if escape_dtype == "int8" else 1e-3,
        rtol=0.12 if escape_dtype == "int8" else 1e-3,
    )
    np.testing.assert_allclose(
        decode_step(query, [prepared_key_page], [prepared_value_page], backend="torch_cuda")[2],
        decode_step(query, [key_page], [value_page], backend="cpu_ref")[2],
        atol=0.12 if escape_dtype == "int8" else 1e-3,
        rtol=0.12 if escape_dtype == "int8" else 1e-3,
    )


@requires_cuda
def test_m0_affine_cuda_preparation_keeps_metadata_in_float32() -> None:
    rng = np.random.default_rng(9021)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    encoded = encode_page(rng.normal(size=(4, config.head_dim)).astype(np.float32), config, kind="K")
    prepared = prepare_page(encoded, backend="torch_cuda")
    assert prepared.scales is not None
    assert prepared.bias is not None
    assert prepared.scales.dtype == torch.float32
    assert prepared.bias.dtype == torch.float32

    direct_pages = prepare_m0_affine_pages_from_tensor_torch(
        torch.from_numpy(rng.normal(size=(1, 4, config.head_dim)).astype(np.float32)).to(device="cuda"),
        config=config,
        kind="K",
        layer_id=0,
        kv_head_id=0,
        token_start=0,
        device_type="cuda",
    )
    assert direct_pages[0].scales is not None
    assert direct_pages[0].bias is not None
    assert direct_pages[0].scales.dtype == torch.float32
    assert direct_pages[0].bias.dtype == torch.float32


@requires_cuda
def test_k4_v3_pages_decode_on_cuda_matches_cpu_reference() -> None:
    rng = np.random.default_rng(9022)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=3, tokens_per_page=16)
    context_length = 32
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_logits, cpu_weights, cpu_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")
    cuda_logits, cuda_weights, cuda_output = decode_step(query, key_pages, value_pages, backend="torch_cuda")

    np.testing.assert_allclose(cuda_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cuda_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cuda_output, cpu_output, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_decode_layer_torch_matches_numpy_path_on_cuda() -> None:
    rng = np.random.default_rng(903)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, mapping)
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), mapping)

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_recent_policy_can_decode_m3_int8_pages_on_cuda() -> None:
    rng = np.random.default_rng(9031)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        key_policy_tier="balanced",
        value_policy_tier="balanced",
        recent_window=1024,
        recent_page_escape_dtype="int8",
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    summary = cache.page_mode_summary()
    numpy_outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), np.array([0, 1], dtype=np.int64))

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=0.12, rtol=0.12)
    assert "K:M3:affine:4:int8" in summary["mode_signature_counts"]
    assert "V:M3:affine:4:int8" in summary["mode_signature_counts"]


@requires_cuda
def test_group_size16_pages_decode_on_cuda_matches_cpu_reference() -> None:
    rng = np.random.default_rng(90315)
    config = DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2)
    context_length = 4
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_pages = _encode_paged(keys, config, kind="K")
    value_pages = _encode_paged(values, config, kind="V")

    cpu_logits, cpu_weights, cpu_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")
    cuda_logits, cuda_weights, cuda_output = decode_step(query, key_pages, value_pages, backend="torch_cuda")

    np.testing.assert_allclose(cuda_logits, cpu_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cuda_weights, cpu_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cuda_output, cpu_output, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_segmented_m2_keys_decode_on_cuda() -> None:
    rng = np.random.default_rng(9032)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M2",
        default_mode_v="M0",
        quant_scheme_k="sketch",
        m2_segment_count_k=2,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), np.array([0, 1], dtype=np.int64))

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_project_m4_keys_decode_on_cuda() -> None:
    rng = np.random.default_rng(9033)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M4",
        default_mode_v="M0",
        quant_scheme_k="project",
        m2_sketch_dim_k=8,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), np.array([0, 1], dtype=np.int64))

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_project_m4_svd_keys_decode_on_cuda() -> None:
    rng = np.random.default_rng(90331)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M4",
        default_mode_v="M0",
        quant_scheme_k="project",
        m2_sketch_dim_k=8,
        m4_project_basis_k="svd",
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), np.array([0, 1], dtype=np.int64))

    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_project_m4_svd_shared_keys_decode_on_cuda() -> None:
    rng = np.random.default_rng(90332)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M4",
        default_mode_v="M0",
        quant_scheme_k="project",
        m2_sketch_dim_k=8,
        m4_project_basis_k="svd_shared",
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(2, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    numpy_outputs = cache.decode_layer(0, queries, np.array([0, 1], dtype=np.int64))
    torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), np.array([0, 1], dtype=np.int64))

    shared_bases = [page.m2_basis for page in cache._state(0, 0).session.key_pages if page.header.mode_default == "M4"]
    assert shared_bases
    assert all(basis is shared_bases[0] for basis in shared_bases)
    np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_model_paged_kv_cache_append_step_torch_avoids_host_uploads_on_cuda() -> None:
    rng = np.random.default_rng(904)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    key_step = torch.from_numpy(rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)).to(device="cuda")
    value_step = torch.from_numpy(rng.normal(size=(2, 1, config.head_dim)).astype(np.float32)).to(device="cuda")
    queries = torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="cuda")

    append_trace = ExecutionTrace()
    cache.append_step_torch(0, key_step, value_step, 0, trace=append_trace)
    decode_trace = ExecutionTrace()
    outputs = cache.decode_layer_torch(0, queries, np.array([0, 1]), trace=decode_trace)

    assert tuple(outputs.shape) == (2, config.head_dim)
    assert append_trace.host_to_device_bytes == 0
    assert decode_trace.host_to_device_bytes == 0


@requires_cuda
def test_model_paged_kv_cache_ingest_prefill_cache_torch_prepares_aligned_m0_pages_on_device() -> None:
    rng = np.random.default_rng(905)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)).to(device="cuda")
    layer_values = torch.from_numpy(rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)).to(device="cuda")

    clear_prepared_chunk_cache()
    try:
        trace = ExecutionTrace()
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values, trace=trace)
        cache.prepare_static_pages(trace=trace)

        for kv_head_id in range(cache.num_key_value_heads):
            state = cache._state(0, kv_head_id)
            assert state.sequence_length == 8
            assert all(isinstance(page, PreparedPageTorch) for page in state.session.key_pages)
            assert all(isinstance(page, PreparedPageTorch) for page in state.session.value_pages)
        assert cache.cache.resident_bytes == 0
        assert cache.resident_bytes == 8 * (64 + 16 + 16)
        assert trace.host_to_device_bytes == 0
    finally:
        clear_prepared_chunk_cache()


@requires_cuda
def test_model_paged_kv_cache_decode_layer_torch_works_with_cached_prepared_chunks_on_cuda() -> None:
    rng = np.random.default_rng(906)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="cuda")
    layer_values = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="cuda")
    queries = torch.from_numpy(rng.normal(size=(4, config.head_dim)).astype(np.float32)).to(device="cuda")

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        outputs = cache.decode_layer_torch(0, queries, np.array([0, 0, 1, 1], dtype=np.int64))
        assert tuple(outputs.shape) == (4, config.head_dim)
        assert cache.resident_bytes > 0
    finally:
        clear_prepared_chunk_cache()


@requires_cuda
def test_model_paged_kv_cache_decode_layer_torch_matches_numpy_path_on_cuda_head_dim64_fused_cache() -> None:
    rng = np.random.default_rng(907)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(4, 2)

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache(0, layer_keys, layer_values)
        numpy_outputs = cache.decode_layer(0, queries, mapping)
        torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), mapping)

        np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)
    finally:
        clear_prepared_chunk_cache()


@requires_cuda
def test_model_paged_kv_cache_decode_layer_torch_matches_numpy_path_on_cuda_head_dim128_grouped_cache() -> None:
    rng = np.random.default_rng(9072)
    config = DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)
    queries = rng.normal(size=(8, config.head_dim)).astype(np.float32)
    mapping = default_q_head_to_kv_head(8, 2)

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache(0, layer_keys, layer_values)
        numpy_outputs = cache.decode_layer(0, queries, mapping)
        torch_outputs = cache.decode_layer_torch(0, torch.from_numpy(queries).to(device="cuda"), mapping)

        np.testing.assert_allclose(torch_outputs.detach().cpu().numpy(), numpy_outputs, atol=1e-4, rtol=1e-4)
    finally:
        clear_prepared_chunk_cache()


def test_grouped_prepared_cuda_output_only_matches_full_decode() -> None:
    rng = np.random.default_rng(9071)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    key_pages_by_group = []
    value_pages_by_group = []
    query_groups = []
    for kv_head_id in range(2):
        keys = rng.normal(size=(16, config.head_dim)).astype(np.float32)
        values = rng.normal(size=(16, config.head_dim)).astype(np.float32)
        key_pages_by_group.append(
            [prepare_page(page, backend="torch_cuda") for page in _encode_paged(keys, config, kind="K")]
        )
        value_pages_by_group.append(
            [prepare_page(page, backend="torch_cuda") for page in _encode_paged(values, config, kind="V")]
        )
        query_groups.append(torch.from_numpy(rng.normal(size=(2, config.head_dim)).astype(np.float32)).to(device="cuda"))

    _, _, full_output = decode_grouped_multiquery_step_prepared_cuda_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
    )
    output_only = decode_grouped_multiquery_step_prepared_cuda_tensor_output_only(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
    )

    np.testing.assert_allclose(output_only.detach().cpu().numpy(), full_output.detach().cpu().numpy(), atol=3e-3, rtol=3e-3)


@requires_cuda
def test_grouped_prepared_cuda_handles_misaligned_key_value_chunk_signatures() -> None:
    rng = np.random.default_rng(9072)
    key_config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        quant_scheme_k="sketch",
    )
    value_config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        quant_scheme_v="lut",
    )
    key_pages_by_group = []
    value_pages_by_group = []
    query_groups = []
    for kv_head_id in range(2):
        keys = rng.normal(size=(8, key_config.head_dim)).astype(np.float32)
        values = rng.normal(size=(8, value_config.head_dim)).astype(np.float32)
        key_pages_by_group.append(
            [
                prepare_page(
                    encode_page(
                        keys[:4],
                        key_config,
                        kind="K",
                        kv_head_id=kv_head_id,
                        token_start=0,
                        mode="M0",
                        quant_scheme="affine",
                    ),
                    backend="torch_cuda",
                ),
                prepare_page(
                    encode_page(
                        keys[4:8],
                        key_config,
                        kind="K",
                        kv_head_id=kv_head_id,
                        token_start=4,
                        mode="M2",
                    ),
                    backend="torch_cuda",
                ),
            ]
        )
        value_pages_by_group.append(
            [
                prepare_page(
                    encode_page(
                        values[:4],
                        value_config,
                        kind="V",
                        kv_head_id=kv_head_id,
                        token_start=0,
                        mode="M1",
                    ),
                    backend="torch_cuda",
                ),
                prepare_page(
                    encode_page(
                        values[4:8],
                        value_config,
                        kind="V",
                        kv_head_id=kv_head_id,
                        token_start=4,
                        mode="M1",
                    ),
                    backend="torch_cuda",
                ),
            ]
        )
        query_groups.append(torch.from_numpy(rng.normal(size=(2, key_config.head_dim)).astype(np.float32)).to(device="cuda"))

    _, _, full_output = decode_grouped_multiquery_step_prepared_cuda_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
    )
    output_only = decode_grouped_multiquery_step_prepared_cuda_tensor_output_only(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
    )

    np.testing.assert_allclose(output_only.detach().cpu().numpy(), full_output.detach().cpu().numpy(), atol=3e-3, rtol=3e-3)


@requires_cuda
def test_grouped_prepared_cuda_respects_distinct_key_and_value_chunk_lengths() -> None:
    rng = np.random.default_rng(90721)
    key_config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        quant_scheme_k="sketch",
    )
    value_config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        quant_scheme_v="lut",
    )
    key_pages_by_group = []
    value_pages_by_group = []
    query_groups = []
    for kv_head_id in range(2):
        keys = rng.normal(size=(8, key_config.head_dim)).astype(np.float32)
        values = rng.normal(size=(8, value_config.head_dim)).astype(np.float32)
        key_pages_by_group.append(
            [
                prepare_page(
                    encode_page(
                        keys[:4],
                        key_config,
                        kind="K",
                        kv_head_id=kv_head_id,
                        token_start=0,
                        mode="M0",
                        quant_scheme="affine",
                    ),
                    backend="torch_cuda",
                ),
                prepare_page(
                    encode_page(
                        keys[4:8],
                        key_config,
                        kind="K",
                        kv_head_id=kv_head_id,
                        token_start=4,
                        mode="M2",
                    ),
                    backend="torch_cuda",
                ),
            ]
        )
        value_pages_by_group.append(
            [
                prepare_page(
                    encode_page(
                        values[:4],
                        value_config,
                        kind="V",
                        kv_head_id=kv_head_id,
                        token_start=0,
                        mode="M1",
                    ),
                    backend="torch_cuda",
                ),
                prepare_page(
                    encode_page(
                        values[4:8],
                        value_config,
                        kind="V",
                        kv_head_id=kv_head_id,
                        token_start=4,
                        mode="M1",
                    ),
                    backend="torch_cuda",
                ),
            ]
        )
        query_groups.append(torch.from_numpy(rng.normal(size=(2, key_config.head_dim)).astype(np.float32)).to(device="cuda"))

    _, _, aligned_output = decode_grouped_multiquery_step_prepared_cuda_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
    )
    _, _, explicit_output = decode_grouped_multiquery_step_prepared_cuda_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        key_chunk_lengths=(1, 1),
        value_chunk_lengths=(2,),
    )

    np.testing.assert_allclose(
        explicit_output.detach().cpu().numpy(),
        aligned_output.detach().cpu().numpy(),
        atol=3e-3,
        rtol=3e-3,
    )


@requires_cuda
def test_model_paged_kv_cache_resident_byte_summary_separates_chunk_cache_on_cuda() -> None:
    rng = np.random.default_rng(908)
    config = DotCacheConfig(head_dim=64, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        backend="torch_cuda",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="cuda")
    layer_values = torch.from_numpy(rng.normal(size=(2, 16, config.head_dim)).astype(np.float32)).to(device="cuda")
    queries = torch.from_numpy(rng.normal(size=(4, config.head_dim)).astype(np.float32)).to(device="cuda")

    clear_prepared_chunk_cache()
    try:
        cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
        before = cache.resident_byte_summary()
        cache.decode_layer_torch(0, queries, np.array([0, 0, 1, 1], dtype=np.int64))
        after = cache.resident_byte_summary()
    finally:
        clear_prepared_chunk_cache()

    assert before["prepared_chunk_resident_bytes"] == 0
    assert before["resident_bytes"] == before["kv_resident_bytes"]
    assert after["prepared_chunk_resident_bytes"] > 0
    assert after["prepared_chunk_resident_bytes"] <= after["prepared_chunk_cache_budget_bytes"]
    assert after["resident_bytes"] == after["kv_resident_bytes"] + after["prepared_chunk_resident_bytes"]


@requires_cuda
def test_llama_generation_harness_runs_on_cuda_tiny_random_model() -> None:
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.to("cuda")
    model.eval()
    dotcache_config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    adapter = LlamaDotCacheModelAdapter(model, dotcache_config, backend="torch_cuda")
    input_ids = torch.tensor([[2, 4, 6, 8]], dtype=torch.long, device="cuda")

    result = run_llama_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=3)

    assert len(result["dotcache_generated_ids"]) == 3
    assert result["resident_bytes"] >= 0
    assert result["resident_bytes"] >= result["kv_resident_bytes"]
    assert result["prepared_chunk_resident_bytes"] <= result["prepared_chunk_cache_budget_bytes"]
    assert result["dotcache_vs_dense_total_resident_bytes_ratio"] >= result["dotcache_vs_dense_kv_bytes_ratio"]
