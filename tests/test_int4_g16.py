"""Step-2 tests: INT4 per-group values default to g=16 (paper §7).

The Apr 17–24 runs all used g=32 because the kernel default was wrong;
this guards against regression. Also exercises the INT4 cache constructor
end-to-end since it had never run a paper bench before.
"""

from __future__ import annotations

import pytest
import torch


CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestInt4G16:
    def test_module_default_is_g16(self):
        """The module-level GROUP_SIZE constant matches paper §7."""
        from dotcache.kernels import int4_group_quantise as m
        assert m.GROUP_SIZE == 16, (
            f"int4_group_quantise.GROUP_SIZE = {m.GROUP_SIZE}; paper §7 requires 16"
        )

    def test_quantise_int4_grouped_block_default_is_g16(self):
        """The quantise_int4_grouped_block(...) default kwarg is 16."""
        import inspect
        from dotcache.kernels.int4_group_quantise import quantise_int4_grouped_block
        sig = inspect.signature(quantise_int4_grouped_block)
        assert sig.parameters["group_size"].default == 16

    def test_create_tiered_cache_int4v_from_model_default_is_g16(self):
        import inspect
        from dotcache.kernels.tiered_kv_cache import create_tiered_cache_int4v_from_model
        sig = inspect.signature(create_tiered_cache_int4v_from_model)
        assert sig.parameters["group_size"].default == 16

    def test_from_fp16_cache_int4v_default_is_g16(self):
        import inspect
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        sig = inspect.signature(TieredKeyCacheLayer.from_fp16_cache_int4v)
        assert sig.parameters["group_size"].default == 16

    def test_int4_cache_layout_with_g16(self):
        """Build an INT4-values cache and verify the per-group metadata
        layout matches paper §8.5: 8 groups for d=128, scales+zeros are FP16.

        Buffers are sized to capacity = N + max_new_tokens (Step 2 added INT4
        decode-append support); the active prefill window is the first N rows.
        """
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        kv_heads, n_blocks, bs, head_dim, d_v = 2, 4, 16, 32, 128
        N = n_blocks * bs
        max_new = 0  # no decode growth → buffer == prefill size
        torch.manual_seed(3)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        values = torch.randn(kv_heads, N, d_v, dtype=torch.float16, device="cuda")

        cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys, values, block_size=bs, group_size=16, max_new_tokens=max_new,
        )

        # group_size stored on cache
        assert cache.values_int4_group_size == 16

        # INT4 packed: [kv_heads, capacity, d_v // 2]
        capacity = N + max_new  # max_new=0 here so capacity == N
        assert cache.values_int4_packed.shape == (kv_heads, capacity, d_v // 2)
        assert cache.values_int4_packed.dtype == torch.uint8

        # Scales / zeros: d_v / group_size = 128 / 16 = 8 groups, sized to capacity
        expected_groups = d_v // 16
        assert expected_groups == 8
        assert cache.values_int4_scales.shape == (kv_heads, capacity, expected_groups)
        assert cache.values_int4_zeros.shape == (kv_heads, capacity, expected_groups)
        assert cache.values_int4_scales.dtype == torch.float16
        assert cache.values_int4_zeros.dtype == torch.float16
        assert cache.values_fp16 is None
        assert cache.values_fp16_gpu is not None
        assert cache.values_fp16_gpu.shape == (kv_heads, capacity, d_v)

    def test_eta_b_is_relative_reconstruction_error(self):
        """Paper v_tol=0.05 is dimensionless; scaling V by a constant should
        scale absolute error but leave η_b approximately unchanged."""
        from dotcache.kernels.int4_group_quantise import quantise_int4_grouped_block

        kv_heads, n_blocks, bs, d_v = 1, 2, 16, 32
        N = n_blocks * bs
        torch.manual_seed(20260424)
        values = torch.randn(kv_heads, N, d_v, dtype=torch.float16, device="cuda")

        small = quantise_int4_grouped_block(values, block_size=bs, group_size=16)
        large = quantise_int4_grouped_block(values * 17.0, block_size=bs, group_size=16)

        torch.testing.assert_close(
            small["error_bounds"],
            large["error_bounds"],
            atol=2e-3,
            rtol=2e-2,
        )
        assert large["abs_error_bounds"].max() > small["abs_error_bounds"].max() * 10

    def test_int4_per_block_error_updates_on_partial_block_append(self):
        """η_b annotation must grow as tokens are appended into a partial block.

        Group quant is per-token (no need to wait for the block), but the
        per-block reconstruction error η_b is maintained over the block's
        tokens. Without an incremental update on append_token, blocks with
        decode-appended tokens keep their initialised η_b (= 0), which makes
        decide_v_format_tight under-estimate the per-block bound and bias
        the Rung-2 decision toward INT4 unfairly.
        """
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        kv_heads, n_blocks, bs, head_dim, d_v = 1, 2, 16, 32, 32
        N = n_blocks * bs
        torch.manual_seed(101)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        values = torch.randn(kv_heads, N, d_v, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys, values, block_size=bs, group_size=16, max_new_tokens=64,
        )

        # The next block (index n_blocks=2) starts empty with η = 0 in buffer.
        next_block_idx = n_blocks
        assert cache.values_int4_errors[0, next_block_idx].item() == 0.0

        # Append a token with deliberately spiky values — high reconstruction
        # error from INT4 → η_b should jump above zero on the very first append.
        spiky = torch.tensor(
            [[100.0 if i % 3 == 0 else -100.0 if i % 3 == 1 else 0.001 for i in range(d_v)]],
            dtype=torch.float16, device="cuda",
        ).unsqueeze(0)  # [1, 1, d_v]
        k_pad = torch.zeros(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
        cache.append_token(k_pad, spiky)
        torch.testing.assert_close(cache.values_fp16_gpu[:, N, :], spiky[:, 0, :])
        eta_after_one = cache.values_int4_errors[0, next_block_idx].item()
        assert eta_after_one > 0.0, (
            "η_b must update on the first appended token; got 0 — "
            "append_token is not propagating to values_int4_errors"
        )

        # Append a second token and verify the running count/sum path updates.
        smooth = torch.randn(kv_heads, 1, d_v, dtype=torch.float16, device="cuda")
        cache.append_token(k_pad, smooth)
        eta_after_two = cache.values_int4_errors[0, next_block_idx].item()
        assert cache.values_int4_error_counts[0, next_block_idx].item() == 2
        expected_mean = (
            cache.values_int4_error_sums[0, next_block_idx]
            / cache.values_int4_error_counts[0, next_block_idx].float()
        ).item()
        assert abs(eta_after_two - expected_mean) < 1e-6, (
            f"η_b must track the running block mean; "
            f"saw {eta_after_two:.6e}, expected {expected_mean:.6e}"
        )

    def test_int4_cache_supports_decode_append(self):
        """Step 2 fix: append_token must quantise to INT4 in-flight so the
        kernel doesn't blow up when N grows beyond the prefill size. This
        was the audit's Risk #3 surprise."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped

        kv_heads, n_blocks, bs, head_dim, d_v = 2, 2, 16, 32, 32
        N = n_blocks * bs  # 32 prefill tokens
        torch.manual_seed(99)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        values = torch.randn(kv_heads, N, d_v, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys, values, block_size=bs, group_size=16, max_new_tokens=64,
        )

        # Append 16 decode tokens with known values
        append_vals = torch.randn(kv_heads, 16, d_v, dtype=torch.float16, device="cuda")
        for t in range(16):
            k_t = torch.randn(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
            v_t = append_vals[:, t:t+1, :]
            cache.append_token(k_t, v_t)

        assert cache.num_tokens == N + 16

        # Dequantise the appended INT4 region for head 0 and verify it
        # reconstructs the original values within INT4 g=16 error.
        appended_packed = cache.values_int4_packed[0, N:N+16, :]
        appended_scales = cache.values_int4_scales[0, N:N+16, :]
        appended_zeros = cache.values_int4_zeros[0, N:N+16, :]
        deq = dequantise_int4_grouped(
            appended_packed, appended_scales, appended_zeros, 16,
        )
        max_err = (deq.float() - append_vals[0].float()).abs().max().item()
        assert max_err < 0.5, (
            f"INT4 decode-append max element error {max_err:.3f} exceeds 0.5 "
            "— append_token may have skipped INT4 quantise"
        )

    def test_g16_reconstruction_tighter_than_g32(self):
        """Smaller groups give tighter per-element reconstruction than larger
        groups, on data with intra-vector dynamic range. This is the reason
        paper §7 picks g=16 not g=32 (paper §8.4 ablation: g=32 'degrades
        catastrophically')."""
        from dotcache.kernels.int4_group_quantise import (
            dequantise_int4_grouped, quantise_int4_grouped,
        )

        # Synthesize a value vector with high intra-dim variance: first half
        # large, second half tiny — mimicking real activation patterns.
        torch.manual_seed(13)
        v = torch.randn(8, 128, dtype=torch.float16, device="cuda")
        v[:, 64:] *= 0.01  # second half is much smaller

        r16 = quantise_int4_grouped(v, group_size=16)
        r32 = quantise_int4_grouped(v, group_size=32)
        d16 = dequantise_int4_grouped(
            r16["data_packed"], r16["scales"], r16["zeros"], 16,
        )
        d32 = dequantise_int4_grouped(
            r32["data_packed"], r32["scales"], r32["zeros"], 32,
        )

        rmse16 = (d16.float() - v.float()).pow(2).mean().sqrt().item()
        rmse32 = (d32.float() - v.float()).pow(2).mean().sqrt().item()
        assert rmse16 < rmse32, (
            f"g=16 RMSE {rmse16:.4e} should be ≤ g=32 RMSE {rmse32:.4e} "
            "on intra-dim-varying data — paper §8.4 ablation result"
        )

    def test_int4_cache_dequant_matches_quantise_grouped(self):
        """Cache's INT4 storage round-trips through dequantise_int4_grouped
        with the configured group_size."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped

        kv_heads, n_blocks, bs, head_dim, d_v = 1, 2, 16, 32, 128
        N = n_blocks * bs
        torch.manual_seed(5)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        values = torch.randn(kv_heads, N, d_v, dtype=torch.float16, device="cuda")

        cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys, values, block_size=bs, group_size=16, max_new_tokens=0,
        )

        # Dequant head 0 (slice to active tokens — buffer has decode growth room)
        deq = dequantise_int4_grouped(
            cache.values_int4_packed[0, :N, :],
            cache.values_int4_scales[0, :N, :],
            cache.values_int4_zeros[0, :N, :],
            16,
        )
        max_err = (deq.float() - values[0].float()).abs().max().item()
        # INT4 quant has ULP = range/15 per group. For unit-variance data,
        # range ≈ 4–5 within a 16-elem group → ULP ≈ 0.3, half-ULP ≈ 0.15.
        assert max_err < 0.5, (
            f"INT4 g=16 max element error {max_err:.3f} exceeds 0.5"
        )
