import numpy as np
import pytest

from dotcache.attention_runtime import decode_step, prepare_pages
from dotcache.backends import (
    PreparedPageTorch,
    clear_prepared_chunk_cache,
    configure_prepared_chunk_cache,
    mps_available,
    prepared_chunk_cache_resident_bytes,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.model_kv_cache import (
    ModelPagedKVCache,
    _build_prepared_decode_view_layout,
    _grouped_pages_can_batch,
    default_q_head_to_kv_head,
)
from dotcache.selector_baselines import LinearSelectorModel, RUNTIME_SELECTOR_FEATURE_NAMES, save_linear_selector_model
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


def test_model_paged_kv_cache_can_use_learned_page_selector_artifact(tmp_path) -> None:
    rng = np.random.default_rng(30411)
    artifact_path = tmp_path / "linear_selector_model.json"
    feature_dim = len(RUNTIME_SELECTOR_FEATURE_NAMES)
    save_linear_selector_model(
        LinearSelectorModel(
            classes=("M0/affine/4", "M3/affine/4/float16"),
            weight=np.zeros((feature_dim, 2), dtype=np.float32),
            bias=np.asarray([0.0, 1.0], dtype=np.float32),
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=tuple(RUNTIME_SELECTOR_FEATURE_NAMES),
        ),
        artifact_path,
    )
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        learned_page_selector_path=str(artifact_path),
        learned_page_selector_prompt_family="cache",
        learned_page_selector_prompt_variant="locality",
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

    assert cache._states[(0, 0)].session.key_pages[0].header.mode_default == "M3"
    assert cache._states[(0, 0)].session.value_pages[0].header.mode_default == "M3"
    assert bool(summary["learned_page_selector_enabled"]) is True
    assert int(summary["learned_page_selector_invocations"]) == int(summary["total_static_pages"])
    assert int(summary["learned_page_selector_fallbacks"]) == 0
    assert float(summary["learned_page_selector_ms_total"]) >= 0.0
    assert summary["learned_page_selector_invocations_by_stage"] == {"prefill": int(summary["total_static_pages"])}
    assert summary["learned_page_selector_fallbacks_by_stage"] == {}
    assert set(summary["learned_page_selector_ms_total_by_stage"]) == {"prefill"}
    assert summary["learned_page_selector_prediction_counts"] == {"M3/affine/4/float16": int(summary["total_static_pages"])}
    assert summary["learned_page_selector_prediction_counts_by_stage"] == {
        "prefill": {"M3/affine/4/float16": int(summary["total_static_pages"])}
    }


def test_dotcache_config_resolves_specific_mode_overrides() -> None:
    config = DotCacheConfig(
        head_dim=32,
        default_mode_k="M0",
        default_mode_v="M0",
        key_mode_overrides=("layer:0=M1", "layer:0:kv:1=M3"),
        value_mode_overrides=("layer:0=M3",),
    )

    assert config.has_mode_overrides() is True
    assert config.resolve_page_mode(kind="K", layer_id=0, kv_head_id=0) == "M1"
    assert config.resolve_page_mode(kind="K", layer_id=0, kv_head_id=1) == "M3"
    assert config.resolve_page_mode(kind="K", layer_id=1, kv_head_id=0) == "M0"
    assert config.resolve_page_mode(kind="V", layer_id=0, kv_head_id=0) == "M3"
    assert config.resolve_page_mode(kind="V", layer_id=1, kv_head_id=0) == "M0"


def test_dotcache_config_resolves_layer_policy_tiers_and_explicit_candidates() -> None:
    config = DotCacheConfig(
        head_dim=32,
        key_policy_tier="balanced",
        value_policy_tier="strict",
        key_layer_sensitivity=("layer:1=aggressive",),
        value_policy_overrides=("layer:0=M3/affine/4/int8,M1/lut/4,M0/affine/4",),
        recent_page_escape_dtype="int8",
    )

    key_policy_l0 = config.resolve_layer_policy(kind="K", layer_id=0, kv_head_id=0)
    key_policy_l1 = config.resolve_layer_policy(kind="K", layer_id=1, kv_head_id=0)
    value_policy_l0 = config.resolve_layer_policy(kind="V", layer_id=0, kv_head_id=0)

    assert key_policy_l0.sensitivity_tier == "balanced"
    assert key_policy_l1.sensitivity_tier == "aggressive"
    assert len(value_policy_l0.candidates) == 3
    assert value_policy_l0.candidates[0].mode == "M3"
    assert value_policy_l0.candidates[0].escape_dtype == "int8"
    assert value_policy_l0.recent_candidate is not None
    assert value_policy_l0.recent_candidate.escape_dtype == "int8"


def test_dotcache_config_balanced_value_policy_includes_m0_3bit_candidate() -> None:
    config = DotCacheConfig(
        head_dim=32,
        key_policy_tier="strict",
        value_policy_tier="balanced",
    )

    value_policy = config.resolve_layer_policy(kind="V", layer_id=0, kv_head_id=0)

    assert [f"{candidate.mode}:{candidate.bits}" for candidate in value_policy.candidates] == [
        "M0:3",
        "M1:4",
        "M0:4",
    ]


def test_model_paged_kv_cache_applies_key_mode_overrides_per_layer_and_kv_head() -> None:
    rng = np.random.default_rng(30415)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="M0",
        default_mode_v="M0",
        key_mode_overrides=("layer:0=M1", "layer:0:kv:1=M3"),
        quant_scheme_k="lut",
        m1_fallback_to_m0=False,
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

    kv0_page = cache._states[(0, 0)].session.key_pages[0]
    kv1_page = cache._states[(0, 1)].session.key_pages[0]
    assert kv0_page.header.mode_default == "M1"
    assert kv1_page.header.mode_default == "M3"


def test_model_paged_kv_cache_records_policy_metadata_and_fragmentation() -> None:
    rng = np.random.default_rng(30416)
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        key_policy_tier="balanced",
        value_policy_tier="strict",
        key_layer_sensitivity=("layer:0=aggressive",),
        value_policy_overrides=("layer:0=M1/lut/4,M0/affine/4",),
        recent_window=0,
        m1_fallback_to_m0=False,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    layer_keys = rng.normal(loc=0.05, scale=0.01, size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(loc=0.05, scale=0.01, size=(2, 8, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    summary = cache.page_mode_summary()
    first_key_page = cache._states[(0, 0)].session.key_pages[0]
    first_value_page = cache._states[(0, 0)].session.value_pages[0]

    assert first_key_page.header.policy_id.startswith("k_")
    assert first_value_page.header.policy_id.startswith("v_")
    assert first_key_page.header.sensitivity_tier == "aggressive"
    assert first_value_page.header.sensitivity_tier in {"strict", "balanced"}
    assert int(summary["fragmentation_total_buckets"]) >= 1
    assert "policy_tier_counts" in summary
    assert "mode_signature_counts" in summary


def test_model_paged_kv_cache_recent_policy_can_emit_m3_int8_pages() -> None:
    rng = np.random.default_rng(30417)
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
        backend="cpu_ref",
    )
    layer_keys = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)
    layer_values = rng.normal(size=(2, 8, config.head_dim)).astype(np.float32)

    cache.ingest_prefill_cache(0, layer_keys, layer_values)
    summary = cache.page_mode_summary()
    first_key_page = cache._states[(0, 0)].session.key_pages[0]
    first_value_page = cache._states[(0, 0)].session.value_pages[0]

    assert first_key_page.header.mode_default == "M3"
    assert first_value_page.header.mode_default == "M3"
    assert first_key_page.header.escape_dtype == "int8"
    assert first_value_page.header.escape_dtype == "int8"
    assert "K:M3:affine:4:int8" in summary["mode_signature_counts"]
    assert "V:M3:affine:4:int8" in summary["mode_signature_counts"]


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
        assert cache.resident_bytes == 8 * (64 + 16 + 16)
        assert resident_summary["kv_resident_bytes"] == cache.resident_bytes
        assert resident_summary["prepared_chunk_resident_bytes"] == 0
    finally:
        clear_prepared_chunk_cache()


def test_model_paged_kv_cache_ingest_prefill_cache_torch_prepares_aligned_m0_3bit_pages_on_device() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(3092)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=3, bits_v=3, tokens_per_page=4)
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    layer_keys = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")
    layer_values = torch.from_numpy(rng.normal(size=(1, 2, 8, config.head_dim)).astype(np.float32)).to(device="mps")

    trace = ExecutionTrace()
    cache.ingest_prefill_cache_torch(0, layer_keys, layer_values, trace=trace)

    for kv_head_id in range(cache.num_key_value_heads):
        state = cache._state(0, kv_head_id)
        assert state.sequence_length == 8
        assert all(isinstance(page, PreparedPageTorch) for page in state.session.key_pages)
        assert all(isinstance(page, PreparedPageTorch) for page in state.session.value_pages)
    assert trace.host_to_device_bytes == 0


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


def test_model_paged_kv_cache_can_freeze_chunk_budget_sync_during_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mps_available():
        return

    cache = ModelPagedKVCache(
        config=DotCacheConfig(
            head_dim=32,
            group_size=32,
            bits_k=4,
            bits_v=4,
            tokens_per_page=4,
            execution_freeze_chunk_budget_during_decode=True,
        ),
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    budget_compute_calls: list[int] = []
    budget_override_calls: list[int] = []

    def _fake_budget_bytes(*, kv_resident_bytes: int | None = None) -> int:
        del kv_resident_bytes
        budget_compute_calls.append(1)
        return 1234

    def _fake_budget_override(*, max_resident_bytes: int | None) -> None:
        budget_override_calls.append(-1 if max_resident_bytes is None else int(max_resident_bytes))

    monkeypatch.setattr(cache, "_prepared_chunk_cache_budget_bytes", _fake_budget_bytes)
    monkeypatch.setattr("dotcache.model_kv_cache.set_prepared_chunk_cache_budget_override", _fake_budget_override)

    cache._prepared_chunk_cache_budget_dirty = True
    cache._sync_prepared_chunk_cache_budget(freeze_during_decode=True)
    assert budget_compute_calls == [1]
    assert budget_override_calls == [1234]
    assert cache._prepared_chunk_cache_budget_dirty is False

    cache._mark_prepared_chunk_cache_budget_dirty(reason="test_dirty")
    cache._sync_prepared_chunk_cache_budget(freeze_during_decode=True)
    assert budget_compute_calls == [1]
    assert budget_override_calls == [1234]
    assert cache._prepared_chunk_cache_budget_dirty is False

    cache._mark_prepared_chunk_cache_budget_dirty(reason="test_dirty")
    cache._sync_prepared_chunk_cache_budget(freeze_during_decode=False)
    assert budget_compute_calls == [1, 1]
    assert budget_override_calls == [1234, 1234]
    assert cache._prepared_chunk_cache_budget_dirty is False
    summary = cache.chunk_budget_summary()
    assert summary["execution_chunk_budget_sync_invocations"] == 3
    assert summary["execution_chunk_budget_override_calls"] == 2
    assert summary["execution_chunk_budget_freeze_override_calls"] == 1


def test_model_paged_kv_cache_reapplies_unchanged_chunk_budget_outside_freeze_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mps_available():
        return

    cache = ModelPagedKVCache(
        config=DotCacheConfig(
            head_dim=32,
            group_size=32,
            bits_k=4,
            bits_v=4,
            tokens_per_page=4,
        ),
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="torch_mps",
    )
    budget_compute_calls: list[int] = []
    budget_override_calls: list[int] = []

    def _fake_budget_bytes(*, kv_resident_bytes: int | None = None) -> int:
        del kv_resident_bytes
        budget_compute_calls.append(1)
        return 1234

    def _fake_budget_override(*, max_resident_bytes: int | None) -> None:
        budget_override_calls.append(-1 if max_resident_bytes is None else int(max_resident_bytes))

    monkeypatch.setattr(cache, "_prepared_chunk_cache_budget_bytes", _fake_budget_bytes)
    monkeypatch.setattr("dotcache.model_kv_cache.set_prepared_chunk_cache_budget_override", _fake_budget_override)

    cache._prepared_chunk_cache_budget_dirty = True
    cache._sync_prepared_chunk_cache_budget(freeze_during_decode=False)
    assert budget_compute_calls == [1]
    assert budget_override_calls == [1234]
    assert cache._prepared_chunk_cache_budget_dirty is False

    cache._mark_prepared_chunk_cache_budget_dirty(reason="test_dirty")
    cache._sync_prepared_chunk_cache_budget(freeze_during_decode=False)
    assert budget_compute_calls == [1, 1]
    assert budget_override_calls == [1234, 1234]
    assert cache._prepared_chunk_cache_budget_dirty is False
    summary = cache.chunk_budget_summary()
    assert summary["execution_chunk_budget_sync_invocations"] == 2
    assert summary["execution_chunk_budget_override_calls"] == 2
    assert summary["execution_chunk_budget_override_same_budget_calls"] == 1


def test_model_paged_kv_cache_execution_value_escape_caches_prepared_pages() -> None:
    from dotcache.decode_reference import decode_page

    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        execution_value_escape_layers=(0,),
        execution_value_escape_mode="M3",
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    rng = np.random.default_rng(31022)
    dense_values = rng.normal(size=(4, config.head_dim)).astype(np.float32)
    value_page = encode_page(
        dense_values,
        config,
        kind="V",
        layer_id=0,
        kv_head_id=0,
        token_start=0,
        mode="M0",
    )
    cache._maybe_register_execution_value_escape_source(
        value_page,
        dense_values=dense_values,
        escape_mode="M3",
    )

    first = cache._prepare_execution_value_escape_page(value_page, escape_mode="M3")
    second = cache._prepare_execution_value_escape_page(value_page, escape_mode="M3")
    reconstructed = decode_page(first.source_page if isinstance(first, PreparedPageTorch) else first)

    assert first is second
    assert first.header.mode_default == "M3"
    np.testing.assert_allclose(reconstructed, dense_values.astype(np.float32), atol=5e-3, rtol=5e-3)
    summary = cache.execution_value_escape_summary()
    assert summary["execution_value_escape_source_registrations"] == 1
    assert summary["execution_value_escape_prepared_page_builds"] == 1
    assert summary["execution_value_escape_builds"] == 2
    assert summary["execution_value_escape_cache_hits"] == 1


def test_model_paged_kv_cache_execution_value_escape_old_only_skips_sink_and_recent_pages() -> None:
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        execution_value_escape_layers=(0,),
        execution_value_escape_mode="M3",
        execution_value_escape_old_only=True,
        execution_recent_window=4,
        execution_sink_window=4,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    rng = np.random.default_rng(31023)
    key_pages: list = []
    value_pages: list = []
    for page_index in range(5):
        token_start = page_index * config.tokens_per_page
        dense_keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
        dense_values = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
        key_pages.append(
            encode_page(
                dense_keys,
                config,
                kind="K",
                layer_id=0,
                kv_head_id=0,
                token_start=token_start,
                mode="M0",
            )
        )
        value_page = encode_page(
            dense_values,
            config,
            kind="V",
            layer_id=0,
            kv_head_id=0,
            token_start=token_start,
            mode="M0",
        )
        cache._maybe_register_execution_value_escape_source(
            value_page,
            dense_values=dense_values,
            escape_mode="M3",
        )
        value_pages.append(value_page)

    escaped_groups, any_applied = cache._apply_execution_value_escape(
        layer_id=0,
        key_pages_by_group=[key_pages],
        value_pages_by_group=[value_pages],
        context_lengths_by_group=[len(key_pages) * config.tokens_per_page],
    )

    assert any_applied is True
    escaped_pages = list(escaped_groups[0])
    escaped_modes = [page.header.mode_default for page in escaped_pages]
    assert escaped_modes == ["M0", "M3", "M3", "M3", "M0"]
    assert escaped_pages[0] is value_pages[0]
    assert escaped_pages[-1] is value_pages[-1]


def test_model_paged_kv_cache_execution_value_escape_top_k_uses_relevance_ranking() -> None:
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        execution_value_escape_layers=(0,),
        execution_value_escape_mode="M3",
        execution_value_escape_top_k=2,
        execution_relevance_mode="envelope",
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    key_pages: list = []
    value_pages: list = []
    query = np.zeros((config.head_dim,), dtype=np.float32)
    query[:4] = 1.0
    strengths = [0.1, 0.9, 0.2, 0.8]
    for page_index, strength in enumerate(strengths):
        token_start = page_index * config.tokens_per_page
        dense_keys = np.full((config.tokens_per_page, config.head_dim), strength, dtype=np.float32)
        dense_values = np.full((config.tokens_per_page, config.head_dim), page_index + 1.0, dtype=np.float32)
        key_pages.append(
            encode_page(
                dense_keys,
                config,
                kind="K",
                layer_id=0,
                kv_head_id=0,
                token_start=token_start,
                mode="M0",
            )
        )
        value_page = encode_page(
            dense_values,
            config,
            kind="V",
            layer_id=0,
            kv_head_id=0,
            token_start=token_start,
            mode="M0",
        )
        cache._maybe_register_execution_value_escape_source(
            value_page,
            dense_values=dense_values,
            escape_mode="M3",
        )
        value_pages.append(value_page)

    escaped_groups, any_applied = cache._apply_execution_value_escape(
        layer_id=0,
        key_pages_by_group=[key_pages],
        value_pages_by_group=[value_pages],
        representative_queries_by_group=[query],
    )

    assert any_applied is True
    escaped_pages = list(escaped_groups[0])
    escaped_modes = [page.header.mode_default for page in escaped_pages]
    assert escaped_modes == ["M0", "M3", "M0", "M3"]


def test_model_paged_kv_cache_execution_value_escape_prewarm_builds_before_decode() -> None:
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        execution_value_escape_layers=(0,),
        execution_value_escape_mode="M3",
        execution_value_escape_prewarm=True,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    rng = np.random.default_rng(31024)
    dense_values = rng.normal(size=(8, config.head_dim)).astype(np.float32)
    for page_index in range(2):
        token_start = page_index * config.tokens_per_page
        value_page = encode_page(
            dense_values[token_start : token_start + config.tokens_per_page],
            config,
            kind="V",
            layer_id=0,
            kv_head_id=0,
            token_start=token_start,
            mode="M0",
        )
        cache._maybe_register_execution_value_escape_source(
            value_page,
            dense_values=dense_values[token_start : token_start + config.tokens_per_page],
            escape_mode="M3",
        )
        cache._state(0, 0).session.value_pages.append(value_page)

    cache._maybe_prewarm_execution_value_escape_pages(cache._state(0, 0))
    summary = cache.execution_value_escape_summary()

    assert summary["execution_value_escape_prewarm"] is True
    assert summary["execution_value_escape_prewarm_invocations"] == 1
    assert summary["execution_value_escape_prewarm_pages"] == 2
    assert summary["execution_value_escape_prepared_page_builds"] == 2
    assert summary["execution_value_escape_cache_hits"] == 0

    cache._maybe_prewarm_execution_value_escape_pages(cache._state(0, 0))
    summary = cache.execution_value_escape_summary()
    assert summary["execution_value_escape_prewarm_invocations"] == 2
    assert summary["execution_value_escape_prewarm_pages"] == 4
    assert summary["execution_value_escape_prepared_page_builds"] == 2
    assert summary["execution_value_escape_cache_hits"] == 2


def test_model_paged_kv_cache_execution_value_escape_prewarm_respects_min_context() -> None:
    config = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        execution_value_escape_layers=(0,),
        execution_value_escape_mode="M3",
        execution_value_escape_prewarm=True,
        execution_value_escape_prewarm_min_context=16,
    )
    cache = ModelPagedKVCache(
        config=config,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        backend="cpu_ref",
    )
    state = cache._state(0, 0)
    state.sequence_length = 8
    rng = np.random.default_rng(31025)
    for page_index in range(2):
        token_start = page_index * config.tokens_per_page
        dense_values = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
        value_page = encode_page(
            dense_values,
            config,
            kind="V",
            layer_id=0,
            kv_head_id=0,
            token_start=token_start,
            mode="M0",
        )
        cache._maybe_register_execution_value_escape_source(
            value_page,
            dense_values=dense_values,
            escape_mode="M3",
        )
        state.session.value_pages.append(value_page)

    cache._maybe_prewarm_execution_value_escape_pages(state)
    summary = cache.execution_value_escape_summary()
    assert summary["execution_value_escape_prewarm_invocations"] == 0
    assert summary["execution_value_escape_prewarm_pages"] == 0
    assert summary["execution_value_escape_prepared_page_builds"] == 0

    state.sequence_length = 16
    cache._maybe_prewarm_execution_value_escape_pages(state)
    summary = cache.execution_value_escape_summary()
    assert summary["execution_value_escape_prewarm_invocations"] == 1
    assert summary["execution_value_escape_prewarm_pages"] == 2
    assert summary["execution_value_escape_prepared_page_builds"] == 2


def test_model_paged_kv_cache_append_step_torch_keeps_budget_clean_when_tail_residency_is_stable() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(31021)
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
    key_step = torch.from_numpy(rng.normal(size=(1, 2, 1, config.head_dim)).astype(np.float32)).to(device="mps")
    value_step = torch.from_numpy(rng.normal(size=(1, 2, 1, config.head_dim)).astype(np.float32)).to(device="mps")

    cache.ingest_prefill_cache_torch(0, layer_keys, layer_values)
    summary_before_append = cache.resident_byte_summary()
    cache._prepared_chunk_cache_budget_dirty = False

    cache.append_step_torch(0, key_step, value_step, 5)
    summary_after_append = cache.resident_byte_summary()

    assert cache._prepared_chunk_cache_budget_dirty is False
    assert summary_after_append["tail_resident_bytes"] == summary_before_append["tail_resident_bytes"]
    assert summary_after_append["kv_resident_bytes"] == summary_before_append["kv_resident_bytes"]
    chunk_budget_summary = cache.chunk_budget_summary()
    assert chunk_budget_summary["execution_chunk_budget_dirty_reason_counts"] == {
        "ingest_prefill_cache_torch": 1
    }


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


def test_grouped_pages_can_batch_accepts_misaligned_key_value_chunks_on_mps() -> None:
    if not mps_available():
        return
    import torch

    rng = np.random.default_rng(313)
    config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    head_dim = 32
    group_size = 32
    token_count = 4

    key_group0 = prepare_pages(
        [
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=0, token_start=0, mode="M0", quant_scheme="affine"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=0, token_start=4, mode="M0", quant_scheme="affine"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=0, token_start=8, mode="M2", quant_scheme="sketch"),
        ],
        backend="torch_mps",
    )
    key_group1 = prepare_pages(
        [
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=1, token_start=0, mode="M0", quant_scheme="affine"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=1, token_start=4, mode="M2", quant_scheme="sketch"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="K", kv_head_id=1, token_start=8, mode="M2", quant_scheme="sketch"),
        ],
        backend="torch_mps",
    )
    value_group0 = prepare_pages(
        [
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=0, token_start=0, mode="M0", quant_scheme="affine"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=0, token_start=4, mode="M1", quant_scheme="lut"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=0, token_start=8, mode="M1", quant_scheme="lut"),
        ],
        backend="torch_mps",
    )
    value_group1 = prepare_pages(
        [
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=1, token_start=0, mode="M0", quant_scheme="affine"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=1, token_start=4, mode="M1", quant_scheme="lut"),
            encode_page(rng.normal(size=(token_count, head_dim)).astype(np.float32), config, kind="V", kv_head_id=1, token_start=8, mode="M1", quant_scheme="lut"),
        ],
        backend="torch_mps",
    )
    queries = [
        torch.from_numpy(rng.normal(size=(2, head_dim)).astype(np.float32)).to(device="mps"),
        torch.from_numpy(rng.normal(size=(2, head_dim)).astype(np.float32)).to(device="mps"),
    ]

    assert _grouped_pages_can_batch([key_group0, key_group1], [value_group0, value_group1], queries)

    layout = _build_prepared_decode_view_layout(key_group0, value_group0)
    assert layout is not None
    assert layout.key_chunk_lengths == (2, 1)
    assert layout.value_chunk_lengths == (1, 2)
