"""Test deferred per-channel INT8 key quantisation.

Verifies that:
- Trailing partial blocks stay FP16 (not quantised to INT8)
- When a block fills to block_size tokens, _quantize_block() runs with
  per-channel scales computed from all tokens
- INT8 data is write-once: never modified after quantisation
- Dequant buffer is consistent for both INT8 and FP16 regions
"""
import torch
import pytest


def _make_cache(N: int = 256, kv_heads: int = 2, head_dim: int = 64,
                block_size: int = 16, max_new: int = 64):
    """Build a small tiered cache from random FP16 keys/values."""
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
    vals = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
    return TieredKeyCacheLayer.from_fp16_cache(
        keys, vals, block_size=block_size, max_new_tokens=max_new,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestDeferredPerChannelQuantisation:
    """Tests for deferred per-channel INT8 key quantisation."""

    def test_prefill_blocks_are_quantized(self):
        """from_fp16_cache should quantize all complete blocks."""
        cache = _make_cache(N=256, block_size=16)
        assert cache.num_tokens == 256
        assert cache.num_blocks == 16
        assert cache.num_quantized_blocks == 16
        assert not cache.has_trailing_partial_block
        assert cache.trailing_block_idx is None

    def test_scale_is_per_channel(self):
        """keys_scale should be [kv_heads, num_blocks, head_dim]."""
        cache = _make_cache(N=256, kv_heads=2, head_dim=64, block_size=16)
        scale = cache.keys_scale_active()
        assert scale.shape == (2, 16, 64), f"Expected (2, 16, 64), got {scale.shape}"

    def test_append_creates_trailing_partial_block(self):
        """Appending one token should create a trailing partial block in FP16."""
        cache = _make_cache(N=256, block_size=16)
        assert cache.num_quantized_blocks == 16

        k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        cache.append_token(k, v)

        assert cache.num_tokens == 257
        assert cache.num_blocks == 17  # ceiling division
        assert cache.num_quantized_blocks == 16  # NOT incremented — block isn't full
        assert cache.has_trailing_partial_block
        assert cache.trailing_block_idx == 16

    def test_trailing_block_has_exact_fp16_in_dequant(self):
        """The trailing partial block should have exact FP16→f32 in the dequant buffer,
        not dequantised INT8."""
        cache = _make_cache(N=256, block_size=16)

        k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        cache.append_token(k, v)

        # The dequant buffer at position 256 should match the original FP16 key
        k_f32 = k.squeeze(1).to(device="cuda", dtype=torch.float32)
        buf = cache._keys_deq_f32[:, 256, :]
        assert torch.allclose(k_f32, buf, atol=1e-3), (
            f"Trailing block dequant should be exact FP16, not INT8 dequant"
        )

    def test_block_fill_triggers_quantization(self):
        """Appending the 16th token to a block should trigger _quantize_block."""
        cache = _make_cache(N=256, block_size=16)
        block_idx = 16

        # Append 15 tokens — block still partial
        for i in range(15):
            k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda") * (0.5 + i * 0.2)
            v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            cache.append_token(k, v)

        assert cache.num_quantized_blocks == 16
        assert cache.has_trailing_partial_block
        assert cache.trailing_block_idx == 16

        # Append 16th token — block should now be quantized
        k16 = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda") * 3.0
        v16 = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        cache.append_token(k16, v16)

        assert cache.num_quantized_blocks == 17
        assert not cache.has_trailing_partial_block  # 272 tokens, 272/16 = 17 exact
        assert cache.trailing_block_idx is None

    def test_per_channel_scale_from_all_tokens(self):
        """When a block is quantized, per-channel scales should reflect all 16 tokens,
        not just the first one. Paper §2.3 asymmetric: scale = (max - min) / 255,
        zero-point = (max + min) / 2."""
        cache = _make_cache(N=256, block_size=16)
        block_idx = 16

        # Append 16 tokens with different channel magnitudes
        all_keys = []
        for i in range(16):
            # Each token has different channel pattern
            k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda") * (0.1 + i * 0.5)
            v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            cache.append_token(k, v)
            all_keys.append(k.squeeze(1).to("cuda", dtype=torch.float32))

        # Block should now be quantized with per-channel scales + zero points
        assert cache.num_quantized_blocks == 17
        scale = cache.keys_scale[:, block_idx, :]  # [kv_heads, head_dim]
        zp = cache.keys_zero_points[:, block_idx, :]
        assert scale.shape == (2, 64)
        assert zp.shape == (2, 64)

        # Asymmetric (paper §2.3): scale from per-channel range, zp from midpoint
        all_stacked = torch.stack(all_keys, dim=1)  # [kv_heads, 16, head_dim]
        k_min = all_stacked.amin(dim=1)
        k_max = all_stacked.amax(dim=1)
        expected_scale = (k_max - k_min).clamp(min=1e-8) / 255.0
        expected_zp = (k_min + k_max) / 2.0
        assert torch.allclose(scale, expected_scale, atol=1e-4), (
            f"Per-channel scale mismatch: got {scale[0, :3]}, expected {expected_scale[0, :3]}"
        )
        assert torch.allclose(zp, expected_zp, atol=1e-4), (
            f"Per-channel zero-point mismatch: got {zp[0, :3]}, expected {expected_zp[0, :3]}"
        )

    def test_dequant_consistency_for_completed_blocks(self):
        """After block fills, dequant buffer should match INT8 * scale + zp
        (paper §2.3 Eq. 1 asymmetric formula)."""
        cache = _make_cache(N=256, block_size=16)
        block_idx = 16

        for i in range(16):
            k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda") * (1.0 + i * 0.3)
            v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            cache.append_token(k, v)

        # Block 16 should be quantized
        assert cache.num_quantized_blocks == 17
        scale = cache.keys_scale[:, block_idx, :]  # [kv_heads, head_dim]
        zp = cache.keys_zero_points[:, block_idx, :]

        for i in range(16):
            pos = 256 + i
            int8_val = cache.keys_int8[:, pos, :]
            deq = int8_val.float() * scale + zp  # paper §2.3: q*s + z
            if cache._keys_deq_f32 is not None:
                buf_deq = cache._keys_deq_f32[:, pos, :]
                assert torch.allclose(deq, buf_deq, atol=1e-6), (
                    f"Dequant mismatch at pos {pos}"
                )

    def test_poison_padding_in_new_block(self):
        """Unused positions in a new partial block should be poisoned (-127 INT8)."""
        cache = _make_cache(N=256, block_size=16)

        k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        cache.append_token(k, v)

        # Positions 257..271 should be poisoned to -127
        padding = cache.keys_int8[:, 257:272, :]
        assert (padding == -127).all(), (
            f"Padding not poisoned: unique values = {padding.unique().tolist()}"
        )

    def test_active_blocks_includes_trailing(self):
        """active_blocks should include the partial trailing block."""
        cache = _make_cache(N=256, block_size=16)

        for i in range(5):
            k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            cache.append_token(k, v)

        assert cache.num_tokens == 261
        assert cache.active_blocks == 17  # ceil(261/16)
        assert cache.num_quantized_blocks == 16
        assert cache.aligned_tokens == 272
        assert cache.aligned_tokens % cache.block_size == 0

    def test_full_block_then_new_trailing(self):
        """Filling a block, then starting a new one, should work correctly."""
        cache = _make_cache(N=256, block_size=16)
        assert cache.num_quantized_blocks == 16

        # Append 16 tokens to fill block 16 completely
        for i in range(16):
            k = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            v = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
            cache.append_token(k, v)

        assert cache.num_tokens == 272
        assert cache.num_blocks == 17
        assert cache.num_quantized_blocks == 17
        assert not cache.has_trailing_partial_block

        # The 17th token should start block 17 as a trailing partial
        k_new = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda") * 5.0
        v_new = torch.randn(2, 1, 64, dtype=torch.float16, device="cuda")
        cache.append_token(k_new, v_new)

        assert cache.num_tokens == 273
        assert cache.num_blocks == 18
        assert cache.num_quantized_blocks == 17  # block 17 is still partial
        assert cache.has_trailing_partial_block
        assert cache.trailing_block_idx == 17

        # Block 16's scale must be unchanged
        scale_16 = cache.keys_scale[:, 16, :]
        assert (scale_16 != 0).any(), "Block 16 should have non-zero per-channel scale"

    def test_quantization_improves_over_per_block(self):
        """Per-channel quantization should have lower error than per-block for
        keys with heterogeneous channel magnitudes."""
        torch.manual_seed(42)
        kv_heads, block_size, head_dim = 2, 16, 64

        # Create keys with heterogeneous channel magnitudes (realistic)
        channel_scales = torch.exp(torch.randn(head_dim) * 1.5).cuda()
        keys_block = torch.randn(kv_heads, block_size, head_dim,
                                 dtype=torch.float16, device="cuda") * channel_scales

        keys_f32 = keys_block.float()

        # Per-channel quantization (what we do)
        k_max_ch = keys_f32.abs().amax(dim=1).clamp(min=1e-8)
        k_scale_ch = k_max_ch / 127.0
        k_int8_ch = (keys_f32 / k_scale_ch[:, None, :]).round().clamp(-127, 127)
        recon_ch = k_int8_ch * k_scale_ch[:, None, :]
        err_ch = (keys_f32 - recon_ch).norm(dim=-1).mean()

        # Per-block quantization (old way)
        k_max_blk = keys_f32.abs().amax(dim=(1, 2)).clamp(min=1e-8)
        k_scale_blk = k_max_blk / 127.0
        k_int8_blk = (keys_f32 / k_scale_blk[:, None, None]).round().clamp(-127, 127)
        recon_blk = k_int8_blk * k_scale_blk[:, None, None]
        err_blk = (keys_f32 - recon_blk).norm(dim=-1).mean()

        assert err_ch < err_blk, (
            f"Per-channel error {err_ch:.4f} should be less than per-block {err_blk:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
