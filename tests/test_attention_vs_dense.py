import numpy as np

from dotcache.attention_reference import explicit_dequantized_attention, run_attention_reference
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def test_attention_reference_matches_explicit_dequantized_attention() -> None:
    rng = np.random.default_rng(4)
    config = DotCacheConfig(head_dim=48, group_size=32)
    keys = rng.normal(size=(8, 48)).astype(np.float32)
    values = rng.normal(size=(8, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)

    key_page = encode_page(keys, config, kind="K", mode="M0")
    value_page = encode_page(values, config, kind="V", mode="M0")

    logits_ref, output_ref = run_attention_reference(query, key_page, value_page)
    logits_dense, output_dense = explicit_dequantized_attention(query, key_page, value_page)

    np.testing.assert_allclose(logits_ref, logits_dense, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_ref, output_dense, atol=1e-5, rtol=1e-5)

