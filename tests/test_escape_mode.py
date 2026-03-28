import numpy as np

from dotcache.attention_reference import explicit_dequantized_score, mix_page_ref, score_page_ref, softmax
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.modes.m3_escape import decode_escape_payload


def test_escape_page_roundtrip() -> None:
    rng = np.random.default_rng(5)
    config = DotCacheConfig(head_dim=32)
    values = rng.normal(size=(3, 32)).astype(np.float32)
    page = encode_page(values, config, kind="K", mode="M3")

    decoded = decode_escape_payload(page.escape_payload, head_dim=32)
    np.testing.assert_allclose(decoded, values, atol=1e-3, rtol=1e-3)


def test_escape_page_int8_roundtrip() -> None:
    rng = np.random.default_rng(15)
    config = DotCacheConfig(head_dim=32, escape_dtype="int8")
    values = rng.normal(size=(5, 32)).astype(np.float32)
    page = encode_page(values, config, kind="K", mode="M3")

    assert page.escape_scales is not None
    decoded = decode_escape_payload(page.escape_payload, head_dim=32, scales=page.escape_scales)
    np.testing.assert_allclose(decoded, values, atol=0.08, rtol=0.08)


def test_escape_page_score_and_mix_behave_like_dense_storage() -> None:
    rng = np.random.default_rng(6)
    config = DotCacheConfig(head_dim=32)
    keys = rng.normal(size=(4, 32)).astype(np.float32)
    values = rng.normal(size=(4, 32)).astype(np.float32)
    query = rng.normal(size=(32,)).astype(np.float32)
    attn = softmax(rng.normal(size=(4,)).astype(np.float32))

    key_page = encode_page(keys, config, kind="K", mode="M3")
    value_page = encode_page(values, config, kind="V", mode="M3")

    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(mix_page_ref(attn, value_page), attn @ values, atol=1e-3, rtol=1e-3)


def test_escape_page_int8_score_and_mix_behave_like_dense_storage() -> None:
    rng = np.random.default_rng(16)
    config = DotCacheConfig(head_dim=32, escape_dtype="int8")
    keys = rng.normal(size=(4, 32)).astype(np.float32)
    values = rng.normal(size=(4, 32)).astype(np.float32)
    query = rng.normal(size=(32,)).astype(np.float32)
    attn = softmax(rng.normal(size=(4,)).astype(np.float32))

    key_page = encode_page(keys, config, kind="K", mode="M3")
    value_page = encode_page(values, config, kind="V", mode="M3")

    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=0.12, rtol=0.12)
    np.testing.assert_allclose(mix_page_ref(attn, value_page), attn @ values, atol=0.12, rtol=0.12)
