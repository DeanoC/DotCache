import numpy as np

from dotcache.attention_reference import explicit_dequantized_mix, mix_page_ref, softmax
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def test_mix_page_matches_explicit_dequantized_baseline() -> None:
    rng = np.random.default_rng(3)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_v=4)
    values = rng.normal(size=(7, 48)).astype(np.float32)
    logits = rng.normal(size=(7,)).astype(np.float32)
    attn = softmax(logits)
    page = encode_page(values, config, kind="V", mode="M0")

    mixed = mix_page_ref(attn, page)
    assert page.full_page_decode_calls == 0

    baseline = explicit_dequantized_mix(attn, page)

    np.testing.assert_allclose(mixed, baseline, atol=1e-5, rtol=1e-5)
    assert page.full_page_decode_calls == 1

