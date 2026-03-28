import numpy as np
import pytest

from dotcache.attention_reference import explicit_dequantized_score, score_page_ref
from dotcache.attention_runtime import score_page
from dotcache.backends import mps_available
from dotcache.config import DotCacheConfig
from dotcache.decode_reference import decode_page
from dotcache.encode import encode_page
from dotcache.types import EncodedPage


def test_m4_page_decodes_and_matches_explicit_reference() -> None:
    rng = np.random.default_rng(441)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M4",
        quant_scheme_k="project",
        m2_sketch_dim_k=8,
    )
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)

    key_page = encode_page(keys, config, kind="K", mode="M4")

    dense_key = decode_page(key_page)
    assert dense_key.shape == keys.shape
    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-5, rtol=1e-5)
    assert key_page.m2_sketch is not None
    assert key_page.m2_mean is not None
    assert key_page.m2_basis is None


def test_m4_value_pages_are_rejected() -> None:
    rng = np.random.default_rng(442)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_v=4)
    values = rng.normal(size=(6, 48)).astype(np.float32)

    with pytest.raises(ValueError, match="M4 is only supported for K pages"):
        encode_page(values, config, kind="V", mode="M4")


def test_config_can_prefer_m4_project_candidates_for_keys() -> None:
    config = DotCacheConfig(head_dim=64, prefer_m4_project_k=True, key_policy_tier="aggressive")
    policy = config.resolve_layer_policy(kind="K", layer_id=0, kv_head_id=0)
    assert [candidate.mode for candidate in policy.candidates] == ["M0", "M4", "M0"]
    assert policy.candidates[1].quant_scheme == "project"


@pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")
def test_m4_key_pages_work_on_mps() -> None:
    rng = np.random.default_rng(443)
    config = DotCacheConfig(
        head_dim=64,
        group_size=32,
        bits_k=4,
        default_mode_k="M4",
        quant_scheme_k="project",
        m2_sketch_dim_k=8,
    )
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    page: EncodedPage = encode_page(keys, config, kind="K", mode="M4")

    cpu_logits = score_page(query, page, backend="cpu_ref")
    mps_logits = score_page(query, page, backend="torch_mps")

    np.testing.assert_allclose(mps_logits, cpu_logits, atol=1e-4, rtol=1e-4)
