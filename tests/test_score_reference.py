import numpy as np

from dotcache.attention_reference import explicit_dequantized_score, score_page_ref
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def test_score_page_matches_explicit_dequantized_baseline() -> None:
    rng = np.random.default_rng(2)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4)
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)
    page = encode_page(keys, config, kind="K", mode="M0")

    streaming_logits = score_page_ref(query, page)
    assert page.full_page_decode_calls == 0

    baseline_logits = explicit_dequantized_score(query, page)

    np.testing.assert_allclose(streaming_logits, baseline_logits, atol=1e-5, rtol=1e-5)
    assert page.full_page_decode_calls == 1

