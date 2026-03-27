import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.attention_runtime import decode_step, prepare_page
from dotcache.backends import PreparedPageTorch, clear_prepared_chunk_cache, cuda_available
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.integrations.llama import LlamaDotCacheModelAdapter, run_llama_generation_harness
from dotcache.model_kv_cache import ModelPagedKVCache, default_q_head_to_kv_head
from dotcache.page_cache import PreparedPageCache
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
        assert cache.resident_bytes == 8 * (64 + 8 + 8)
        assert trace.host_to_device_bytes == 0
    finally:
        clear_prepared_chunk_cache()


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
