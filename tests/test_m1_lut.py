import numpy as np

from dotcache.attention_reference import explicit_dequantized_mix, explicit_dequantized_score, mix_page_ref, score_page_ref
from dotcache.config import DotCacheConfig
from dotcache.decode_reference import decode_page
from dotcache.encode import encode_page
from dotcache.modes.m1_lut import quantize_tensor_lut


def test_lut_quantization_shapes_and_error() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(4, 48)).astype(np.float32)
    codes, codebooks, padded_head_dim = quantize_tensor_lut(values, group_size=32, bits=4, segment_count=2)
    decoded = np.zeros((4, padded_head_dim), dtype=np.float32)
    segment_ids = (np.arange(codes.shape[0], dtype=np.int64) * codebooks.shape[1]) // codes.shape[0]
    for token_index in range(codes.shape[0]):
        for group_index in range(codes.shape[1]):
            start = group_index * 32
            end = start + 32
            decoded[token_index, start:end] = codebooks[group_index, segment_ids[token_index]][codes[token_index, group_index]]

    assert codes.shape == (4, 2, 32)
    assert codebooks.shape == (2, 2, 16)
    assert padded_head_dim == 64
    assert np.max(np.abs(values - decoded[:, :48])) < 0.9


def test_m1_page_decodes_and_matches_explicit_reference() -> None:
    rng = np.random.default_rng(8)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4, bits_v=4)
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    values = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)
    attn = rng.normal(size=(6,)).astype(np.float32)
    attn = np.exp(attn - np.max(attn))
    attn = attn / np.sum(attn)

    key_page = encode_page(keys, config, kind="K", mode="M1")
    value_page = encode_page(values, config, kind="V", mode="M1")

    dense_key = decode_page(key_page)
    dense_value = decode_page(value_page)
    assert dense_key.shape == keys.shape
    assert dense_value.shape == values.shape

    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mix_page_ref(attn, value_page), explicit_dequantized_mix(attn, value_page), atol=1e-5, rtol=1e-5)


def test_tanh_preconditioning_preserves_valid_lut_decode() -> None:
    rng = np.random.default_rng(9)
    core = rng.normal(scale=0.5, size=(32, 48)).astype(np.float32)
    outliers = rng.normal(loc=0.0, scale=6.0, size=(32, 48)).astype(np.float32)
    mask = rng.random(size=(32, 48)) < 0.08
    values = np.where(mask, outliers, core).astype(np.float32)

    plain_config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4, default_mode_k="M1", quant_scheme_k="lut")
    conditioned_config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        preconditioner="tanh",
        precondition_strength=2.0,
        m1_fallback_to_m0=False,
    )

    conditioned_page = encode_page(values, conditioned_config, kind="K", mode="M1")
    plain_page = encode_page(values, plain_config, kind="K", mode="M1")

    plain_error = np.mean(np.abs(values - decode_page(plain_page)))
    conditioned_decoded = decode_page(conditioned_page)
    conditioned_error = np.mean(np.abs(values - conditioned_decoded))

    assert conditioned_page.codebooks is not None
    assert conditioned_decoded.shape == values.shape
    assert np.isfinite(conditioned_decoded).all()
    assert np.isfinite(conditioned_error)
    assert conditioned_error > 0.0
    assert plain_error > 0.0


def test_m1_can_fallback_to_m0_for_bad_pages() -> None:
    rng = np.random.default_rng(10)
    values = rng.standard_cauchy(size=(32, 48)).astype(np.float32) * np.float32(3.0)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_fallback_to_m0=True,
        m1_error_threshold=1e-6,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M0"
    assert page.scales is not None
    assert page.codebooks is None


def test_m1_can_stay_enabled_for_reasonable_pages() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(scale=0.4, size=(32, 48)).astype(np.float32)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_fallback_to_m0=True,
        m1_error_threshold=0.8,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M1"
    assert page.codebooks is not None


def test_m1_can_fallback_on_token_error_signal() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(scale=0.2, size=(32, 48)).astype(np.float32)
    values[::2, :24] += np.float32(6.0)
    values[1::2, 24:] -= np.float32(6.0)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_segment_count_k=1,
        m1_fallback_to_m0=True,
        m1_error_threshold=10.0,
        m1_token_p95_error_threshold=1e-6,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M0"
    assert page.trial_token_p95_error is not None
    assert page.trial_token_p95_error > 1e-6
