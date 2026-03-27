import numpy as np

from dotcache.attention_reference import explicit_dequantized_mix, explicit_dequantized_score, mix_page_ref, score_page_ref
from dotcache.config import DotCacheConfig
from dotcache.decode_reference import decode_page
from dotcache.encode import encode_page
from dotcache.modes.turbo3 import TURBO3_CENTROIDS, dequantize_group_turbo3, quantize_tensor_turbo3
from dotcache.session_runtime import sketch_key_page, summarize_value_page


def test_turbo3_quantization_shapes_and_error() -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(size=(4, 48)).astype(np.float32)

    codes, correction, centroids, padded_head_dim = quantize_tensor_turbo3(values, group_size=32)
    decoded = np.zeros((4, padded_head_dim), dtype=np.float32)
    for token_index in range(codes.shape[0]):
        for group_index in range(codes.shape[1]):
            start = group_index * 32
            end = start + 32
            decoded[token_index, start:end] = dequantize_group_turbo3(
                codes[token_index, group_index][None, :],
                correction=correction[token_index, group_index][None],
                centroids=centroids,
            )[0]

    assert codes.shape == (4, 2, 32)
    assert correction.shape == (4, 2)
    assert centroids.shape == (8,)
    assert padded_head_dim == 64
    np.testing.assert_allclose(centroids.astype(np.float32), TURBO3_CENTROIDS, atol=1e-3, rtol=1e-3)
    assert np.max(np.abs(values - decoded[:, :48])) < 3.5


def test_turbo3_page_decodes_and_matches_explicit_reference() -> None:
    rng = np.random.default_rng(14)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        bits_v=4,
        default_mode_k="T3",
        default_mode_v="T3",
        quant_scheme_k="turbo3",
        quant_scheme_v="turbo3",
    )
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    values = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)
    attn = rng.normal(size=(6,)).astype(np.float32)
    attn = np.exp(attn - np.max(attn))
    attn = attn / np.sum(attn)

    key_page = encode_page(keys, config, kind="K", mode="T3")
    value_page = encode_page(values, config, kind="V", mode="T3")

    dense_key = decode_page(key_page)
    dense_value = decode_page(value_page)
    assert dense_key.shape == keys.shape
    assert dense_value.shape == values.shape

    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mix_page_ref(attn, value_page), explicit_dequantized_mix(attn, value_page), atol=1e-5, rtol=1e-5)


def test_turbo3_pages_work_with_session_page_summaries() -> None:
    rng = np.random.default_rng(15)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        default_mode_k="T3",
        default_mode_v="T3",
        quant_scheme_k="turbo3",
        quant_scheme_v="turbo3",
    )
    keys = rng.normal(size=(8, 48)).astype(np.float32)
    values = rng.normal(size=(8, 48)).astype(np.float32)

    key_page = encode_page(keys, config, kind="K", mode="T3")
    value_page = encode_page(values, config, kind="V", mode="T3")

    key_sketch = sketch_key_page(key_page, sketch_size=2)
    value_summary = summarize_value_page(value_page)

    assert key_sketch.shape == (2, 48)
    assert value_summary.shape == (48,)
    assert np.isfinite(key_sketch).all()
    assert np.isfinite(value_summary).all()
