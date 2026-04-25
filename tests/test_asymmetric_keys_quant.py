"""Step-1 tests: asymmetric INT8 key quantisation (paper §2.3).

Covers:
- TieredKeyCacheLayer dataclass carries keys_zero_points
- from_fp16_cache produces a valid asymmetric encode (z ∈ [-128, 127],
  q ∈ [-128, 127], (q - z) * s reconstructs original within 1 ULP/2)
- Degenerate channel (k_min == k_max) doesn't crash and produces zero error
- Per-channel ranges are random / non-zero-centered (the case symmetric
  quant gets WRONG and this Step exists to fix)
- Triton scorer with (q - z) * s matches the FP32 reference within the
  paper §2.3 bound: max element error ≤ scale/2, accumulated via Cauchy-
  Schwarz across head_dim=128 channels gives ~ ||scale||_2 / 2 per token.

CUDA-gated tests use the existing house style.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


CUDA = torch.cuda.is_available()


def _make_keys_values(
    kv_heads: int = 2, n_blocks: int = 8, block_size: int = 16,
    head_dim: int = 32, d_v: int = 32, seed: int = 0,
    dtype=torch.float32,
):
    torch.manual_seed(seed)
    N = n_blocks * block_size
    keys = torch.randn(kv_heads, N, head_dim, dtype=dtype)
    values = torch.randn(kv_heads, N, d_v, dtype=dtype)
    return keys, values


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestAsymmetricEncode:
    def test_dataclass_has_zero_points_field(self):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        from dataclasses import fields
        names = {f.name for f in fields(TieredKeyCacheLayer)}
        assert "keys_zero_points" in names, (
            "Step 1 must add keys_zero_points to TieredKeyCacheLayer"
        )

    def test_from_fp16_cache_produces_zero_points(self):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        keys, values = _make_keys_values()
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys.cuda(), values.cuda(), block_size=16, max_new_tokens=0,
        )
        assert cache.keys_zero_points is not None
        # Same shape as keys_scale (per-channel-per-block)
        assert cache.keys_zero_points.shape == cache.keys_scale.shape
        # z is the fp-space midpoint (k_min + k_max)/2 — finite and bounded
        # by the data range, but unconstrained vs the integer quant range.
        nb = cache.num_blocks
        z_active = cache.keys_zero_points[:, :nb, :]
        assert torch.all(torch.isfinite(z_active))

    def test_int8_codes_use_full_range(self):
        """Asymmetric range is [-128, 127]; symmetric was [-127, 127].
        Random data should populate the full negative range."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        keys, values = _make_keys_values(seed=42, n_blocks=16)
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys.cuda(), values.cuda(), block_size=16, max_new_tokens=0,
        )
        nt = cache.num_tokens
        active = cache.keys_int8[:, :nt, :]
        # Range must be [-128, 127]; seeing -128 is the asymmetric signature.
        assert int(active.min().item()) == -128, (
            f"Asymmetric encode must reach -128 (got {int(active.min())}). "
            "Symmetric encode would clamp at -127."
        )
        assert int(active.max().item()) == 127

    def test_roundtrip_within_half_ulp_per_channel(self):
        """Per-channel reconstruction error ≤ scale/2 (one half-ULP)."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        keys, values = _make_keys_values(seed=1, n_blocks=4)
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys.cuda(), values.cuda(), block_size=16, max_new_tokens=0,
        )
        nb = cache.num_blocks
        bs = cache.block_size
        nt = cache.num_tokens
        # Dequant (paper §2.3): x_hat = q*s + z
        q_int8 = cache.keys_int8[:, :nt, :].to(torch.float32).reshape(
            cache.kv_heads, nb, bs, cache.head_dim,
        )
        z = cache.keys_zero_points[:, :nb, :].unsqueeze(2)  # broadcast over tokens
        s = cache.keys_scale[:, :nb, :].unsqueeze(2)
        x_hat = (q_int8 * s + z).reshape(cache.kv_heads, nt, cache.head_dim)
        x_orig = keys.cuda().to(torch.float32)
        # Per-channel max error must not exceed scale/2 (with a tiny float
        # tolerance for rounding accumulation).
        s_per_token = cache.keys_scale[:, :nb, :].repeat_interleave(bs, dim=1)
        err = (x_hat - x_orig).abs()
        max_allowed = s_per_token / 2.0 + 1e-6
        violations = (err > max_allowed).float().mean().item()
        assert violations == 0.0, (
            f"{violations:.3%} of elements exceed scale/2 reconstruction bound"
        )

    def test_degenerate_channel_kmin_equals_kmax(self):
        """A constant channel (k_min == k_max) must not crash and dequant to zero error.

        With z = (k_min + k_max) / 2 = constant, q rounds to 0 (the constant
        is exactly at the centre of its zero-width range), and dequant gives
        0 * s + z = z = constant — exact reconstruction regardless of scale.
        """
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        kv, n, hd = 2, 32, 32  # n=32 = 2 blocks × 16
        keys = torch.randn(kv, n, hd)
        # Force channel 0 to a constant
        keys[:, :, 0] = 0.5
        values = torch.randn(kv, n, hd)
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys.cuda(), values.cuda(), block_size=16, max_new_tokens=0,
        )
        nt = cache.num_tokens
        nb = cache.num_blocks
        # Dequant the constant channel and verify it reconstructs to 0.5
        q_int8 = cache.keys_int8[:, :nt, 0].to(torch.float32)  # [kv, n]
        z = cache.keys_zero_points[:, :nb, 0]  # [kv, nb]
        s = cache.keys_scale[:, :nb, 0]
        z_per_t = z.repeat_interleave(16, dim=1)
        s_per_t = s.repeat_interleave(16, dim=1)
        x_hat = q_int8 * s_per_t + z_per_t
        max_err = (x_hat - 0.5).abs().max().item()
        assert max_err < 1e-5, f"degenerate channel reconstruction err {max_err:.3e}"

    def test_non_zero_centered_data_benefits_from_asymmetric(self):
        """Asymmetric quant should beat symmetric on data with mean offset.

        A unit-variance gaussian shifted by +5.0 has all values positive;
        symmetric quant wastes the negative half of the INT8 code range.
        Asymmetric (z = midpoint, s = range/255) uses all 256 codes for the
        actual data span. Per-block-per-channel range of a 16-sample shifted
        gaussian is ~3.0 (typical ±1.5σ spread), so scale ≈ 3/255 ≈ 0.012,
        half-ULP ≈ 0.006, uniform-error RMSE ≈ 0.006/√3 ≈ 0.0034.
        Symmetric on the same data would use scale ≈ 6.5/127 ≈ 0.051 (since
        |max| ≈ 6.5), half-ULP ≈ 0.026, RMSE ≈ 0.015 — about 4× worse.
        """
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        torch.manual_seed(7)
        keys = torch.randn(2, 64, 32) + 5.0  # all positive
        values = torch.randn(2, 64, 32)
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys.cuda(), values.cuda(), block_size=16, max_new_tokens=0,
        )
        nt = cache.num_tokens
        nb = cache.num_blocks
        bs = cache.block_size
        z = cache.keys_zero_points[:, :nb, :].unsqueeze(2)
        s = cache.keys_scale[:, :nb, :].unsqueeze(2)
        q_int8 = cache.keys_int8[:, :nt, :].to(torch.float32).reshape(
            cache.kv_heads, nb, bs, cache.head_dim,
        )
        x_hat = (q_int8 * s + z).reshape(cache.kv_heads, nt, cache.head_dim)
        x_orig = keys.cuda().to(torch.float32)
        rmse = (x_hat - x_orig).pow(2).mean().sqrt().item()
        # Symmetric ceiling ≈ 0.015 RMSE; asymmetric should be well below.
        assert rmse < 0.008, (
            f"asymmetric RMSE {rmse:.4e} exceeds expected (~0.003) — "
            "encode may have regressed to symmetric"
        )


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestAsymmetricKernelParity:
    def test_fused_scorer_matches_fp32_reference(self):
        """The Triton scorer's asymmetric dequant must match a pure-Torch
        FP32 reference within a paper-§2.3-justified tolerance."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        from dotcache.kernels.fused_score_certify import (
            fused_score_certify_multihead,
        )

        kv_heads, n_blocks, bs, head_dim = 2, 8, 16, 32
        N = n_blocks * bs
        torch.manual_seed(11)
        keys = torch.randn(kv_heads, N, head_dim).cuda()
        values = torch.randn(kv_heads, N, head_dim).cuda()
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )

        num_q_heads = kv_heads * 2  # gqa_group=2
        gqa_group = num_q_heads // kv_heads
        q = torch.randn(num_q_heads, head_dim).cuda()
        q_scale = 1.0 / (head_dim ** 0.5)

        nb = cache.num_quantized_blocks
        m_b, S_b, _ = fused_score_certify_multihead(
            K_int8_packed=cache.keys_int8[:, :n_blocks * bs, :],
            K_scale=cache.keys_scale[:, :nb, :],
            K_zero_points=cache.keys_zero_points[:, :nb, :],
            q_all=q,
            correction=cache.correction[:, :nb],
            gqa_group=gqa_group,
            block_size=bs,
            q_scale=q_scale,
            block_epsilon=1e-9,  # don't skip anything
        )

        # FP32 reference: dequant via q*s + z (paper §2.3), then dot with query, per block.
        q_int8 = cache.keys_int8[:, :n_blocks * bs, :].to(torch.float32).reshape(
            kv_heads, nb, bs, head_dim,
        )
        z = cache.keys_zero_points[:, :nb, :].unsqueeze(2)
        s = cache.keys_scale[:, :nb, :].unsqueeze(2)
        keys_dq = (q_int8 * s + z)  # [kv_h, nb, bs, hd]

        ref_m_b = torch.empty(num_q_heads, nb, device=q.device)
        for qh in range(num_q_heads):
            kvh = qh // gqa_group
            for bid in range(nb):
                k_block = keys_dq[kvh, bid]            # [bs, hd]
                scores = (k_block @ q[qh]) * q_scale   # [bs]
                ref_m_b[qh, bid] = scores.max()

        # Threshold: per-token L2 error ≤ ||scale||_2 / 2 (Cauchy-Schwarz on
        # half-ULP per channel). Accumulating over the dot product bounds the
        # logit error tightly. We use atol=5e-3, rtol=1e-3 which is well above
        # the theoretical Triton-vs-Torch FP32 mismatch (~1e-5).
        np.testing.assert_allclose(
            m_b.cpu().numpy(), ref_m_b.cpu().numpy(),
            atol=5e-3, rtol=1e-3,
        )

    def test_attend_int8_matches_fp32_reference(self):
        """The selective-attend INT8 kernel's asymmetric dequant must match FP32 ref."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        from dotcache.kernels.fused_score_certify import (
            fused_score_certify_multihead,
        )
        from dotcache.kernels.selective_attend_triton import (
            selective_attend_multihead_int8,
        )

        kv_heads, n_blocks, bs, head_dim, d_v = 2, 8, 16, 32, 32
        N = n_blocks * bs
        torch.manual_seed(23)
        keys = torch.randn(kv_heads, N, head_dim).cuda()
        values = torch.randn(kv_heads, N, d_v).cuda()
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )

        num_q_heads = kv_heads * 2
        gqa_group = num_q_heads // kv_heads
        q = torch.randn(num_q_heads, head_dim).cuda()
        q_scale = 1.0 / (head_dim ** 0.5)

        nb = cache.num_quantized_blocks
        # Use scorer to produce a skip mask (no skips here)
        skip_mask = torch.zeros(num_q_heads, nb, dtype=torch.int32, device=q.device)

        out = selective_attend_multihead_int8(
            keys_int8=cache.keys_int8[:, :N, :],
            keys_scale=cache.keys_scale[:, :nb, :],
            keys_zero_points=cache.keys_zero_points[:, :nb, :],
            values_fp16=cache.values_fp16[:, :N, :],
            q_all=q,
            skip_mask_i32=skip_mask,
            gqa_group=gqa_group,
            block_size=bs,
            q_scale=q_scale,
        )

        # FP32 reference: full attention with asymmetric-dequanted keys (q*s + z)
        z = cache.keys_zero_points[:, :nb, :].unsqueeze(2)
        s = cache.keys_scale[:, :nb, :].unsqueeze(2)
        q_int8 = cache.keys_int8[:, :N, :].to(torch.float32).reshape(
            kv_heads, nb, bs, head_dim,
        )
        keys_dq = (q_int8 * s + z).reshape(kv_heads, N, head_dim)
        vals_f32 = cache.values_fp16[:, :N, :].to(torch.float32)

        ref = torch.empty(num_q_heads, d_v, device=q.device)
        for qh in range(num_q_heads):
            kvh = qh // gqa_group
            scores = (keys_dq[kvh] @ q[qh]) * q_scale
            w = torch.softmax(scores, dim=0)
            ref[qh] = w @ vals_f32[kvh]

        # Tolerance: per-token error in V is bounded by softmax-weighted V; the
        # softmax dampens gain. atol=5e-3 covers FP32 vs FP64 softmax variance.
        np.testing.assert_allclose(
            out.cpu().numpy(), ref.cpu().numpy(),
            atol=5e-3, rtol=5e-3,
        )
