from __future__ import annotations

from pathlib import Path

import pytest


def test_cutlass_submodule_version_header_is_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "third_party" / "cutlass" / "include" / "cutlass" / "version.h"
    if not version.exists():
        pytest.skip("CUTLASS submodule is not initialized")
    text = version.read_text()
    assert "#define CUTLASS_MAJOR 4" in text
    assert "#define CUTLASS_MINOR 3" in text
    assert "#define CUTLASS_PATCH 1" in text


def test_cutlass_sm120_probe_if_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 GPU not available")

    from dotcache.backends.cutlass_sm120 import (
        cutlass_sm120_available,
        cutlass_sm120_metadata,
        cutlass_sm120_probe,
    )

    if not cutlass_sm120_available():
        pytest.skip("CUTLASS SM120 extension is not buildable in this environment")

    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = cutlass_sm120_probe(x)
    torch.cuda.synchronize()
    assert torch.equal(x, y)
    assert "cutlass=4.3.1" in cutlass_sm120_metadata()["metadata"]


def test_cutlass_dequant_keys_to_fp16_t_matches_reference() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 GPU not available")

    from dotcache.backends.cutlass_sm120 import (
        cutlass_sm120_available,
        dequant_keys_to_fp16_t,
    )

    if not cutlass_sm120_available():
        pytest.skip("CUTLASS SM120 extension is not buildable in this environment")

    torch.manual_seed(20260427)
    kv_heads, n_blocks, block_size, head_dim = 2, 5, 16, 32
    n_tokens = n_blocks * block_size
    keys_int8 = torch.randint(
        -128, 128, (kv_heads, n_tokens, head_dim), dtype=torch.int8, device="cuda",
    )
    scales = torch.rand(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.02
    zeros = torch.randn(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.01

    got = dequant_keys_to_fp16_t(keys_int8, scales, zeros, block_size=block_size)
    expected = (
        keys_int8.to(torch.float32).reshape(kv_heads, n_blocks, block_size, head_dim)
        * scales.unsqueeze(2)
        + zeros.unsqueeze(2)
    ).reshape(kv_heads, n_tokens, head_dim).transpose(1, 2).contiguous().to(torch.float16)

    assert got.shape == (kv_heads, head_dim, n_tokens)
    torch.testing.assert_close(got, expected)


def test_cutlass_score_backend_falls_back_to_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from dotcache.kernels.fused_score_certify import fused_score_certify_multihead

    torch.manual_seed(20260425)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim = 3, 16, 32
    n_tokens = n_blocks * block_size

    keys_int8 = torch.randint(
        -127, 128, (kv_heads, n_tokens, head_dim), dtype=torch.int8, device="cuda",
    )
    scales = torch.rand(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.02
    zeros = torch.randn(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.01
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")
    corr = torch.ones(kv_heads, n_blocks, dtype=torch.float32, device="cuda")

    base = fused_score_certify_multihead(
        keys_int8, scales, zeros, q, corr, gqa_group,
        block_size=block_size, q_scale=head_dim ** -0.5, block_epsilon=0.0,
    )

    monkeypatch.setenv("DOTCACHE_SCORE_BACKEND", "cutlass_sm120")
    monkeypatch.delenv("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", raising=False)
    fallback = fused_score_certify_multihead(
        keys_int8, scales, zeros, q, corr, gqa_group,
        block_size=block_size, q_scale=head_dim ** -0.5, block_epsilon=0.0,
    )

    for got, expected in zip(fallback, base, strict=True):
        torch.testing.assert_close(got, expected)


def test_cutlass_score_backend_enabled_matches_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 GPU not available")

    from dotcache.backends.cutlass_sm120 import cutlass_sm120_available
    from dotcache.kernels.fused_score_certify import fused_score_certify_multihead

    if not cutlass_sm120_available():
        pytest.skip("CUTLASS SM120 extension is not buildable in this environment")

    torch.manual_seed(20260426)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim = 5, 16, 32
    n_tokens = n_blocks * block_size

    keys_int8 = torch.randint(
        -127, 128, (kv_heads, n_tokens, head_dim), dtype=torch.int8, device="cuda",
    )
    scales = torch.rand(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.02
    zeros = torch.randn(kv_heads, n_blocks, head_dim, dtype=torch.float32, device="cuda") * 0.01
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")
    corr = torch.ones(kv_heads, n_blocks, dtype=torch.float32, device="cuda")

    monkeypatch.setenv("DOTCACHE_SCORE_BACKEND", "triton")
    expected = fused_score_certify_multihead(
        keys_int8, scales, zeros, q, corr, gqa_group,
        block_size=block_size, q_scale=head_dim ** -0.5, block_epsilon=0.0,
    )

    monkeypatch.setenv("DOTCACHE_SCORE_BACKEND", "cutlass_sm120")
    monkeypatch.setenv("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", "1")
    got = fused_score_certify_multihead(
        keys_int8, scales, zeros, q, corr, gqa_group,
        block_size=block_size, q_scale=head_dim ** -0.5, block_epsilon=0.0,
    )

    torch.testing.assert_close(got[0], expected[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got[1], expected[1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got[2], expected[2])
