import numpy as np
import pytest

from dotcache.attention_reference import explicit_dequantized_score, score_page_ref
from dotcache.attention_runtime import score_page
from dotcache.backends import mps_available
from dotcache.config import DotCacheConfig
from dotcache.decode_reference import decode_page
from dotcache.encode import encode_page
from dotcache.types import EncodedPage


def test_m2_page_decodes_and_matches_explicit_reference() -> None:
    rng = np.random.default_rng(41)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M2",
        quant_scheme_k="sketch",
        m2_sketch_dim_k=8,
    )
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)

    key_page = encode_page(keys, config, kind="K", mode="M2")

    dense_key = decode_page(key_page)
    assert dense_key.shape == keys.shape
    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-5, rtol=1e-5)


def test_m2_value_pages_are_rejected() -> None:
    rng = np.random.default_rng(42)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_v=4)
    values = rng.normal(size=(6, 48)).astype(np.float32)

    with pytest.raises(ValueError, match="M2 is only supported for K pages"):
        encode_page(values, config, kind="V", mode="M2")


@pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")
def test_m2_key_pages_work_on_mps() -> None:
    rng = np.random.default_rng(43)
    config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        default_mode_k="M2",
        quant_scheme_k="sketch",
        m2_sketch_dim_k=8,
    )
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    page: EncodedPage = encode_page(keys, config, kind="K", mode="M2")

    cpu_logits = score_page(query, page, backend="cpu_ref")
    mps_logits = score_page(query, page, backend="torch_mps")

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
