"""Unit tests for the adaptive K* selector (paper §3.3)."""
from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestAdaptiveTopK:
    def _make_m_S(self, logits_per_block: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct (m_b, S_b) consistent with a block containing one token per entry.
        With one token per block: m_b = score, S_b = 1.
        """
        m = torch.tensor(logits_per_block, dtype=torch.float32, device="cuda")
        S = torch.ones_like(m)
        return m, S

    def test_single_head_concentrated_picks_minimum(self):
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        # One block has ~all the mass — selector should bottom-out at k_min.
        m, S = self._make_m_S([[10.0, 0.0, 0.0, 0.0, 0.0]])
        mask, k, tail, tau_actual = compute_adaptive_topk_mask(m, S, tau_cov=0.995, k_min=2, k_max=10)
        assert k.shape == (1,)
        assert int(k[0]) == 2  # k_min floor
        assert mask.sum().item() == 2
        # Top block (index 0) must be in the mask
        assert mask[0, 0].item() is True
        # Coverage at k_min=2 should already exceed 0.995
        assert tau_actual[0].item() >= 0.995

    def test_diffuse_head_expands_up_to_k_max(self):
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        # Uniform block masses over 20 blocks: cumulative 0.05 per block.
        # tau_cov=0.995 would want k=20; clamp to k_max=8.
        m = torch.zeros(1, 20, device="cuda")
        S = torch.ones_like(m)
        mask, k, tail, tau_actual = compute_adaptive_topk_mask(m, S, tau_cov=0.995, k_min=2, k_max=8)
        assert int(k[0]) == 8
        assert mask.sum().item() == 8
        # Tail mass must be positive (cap hit)
        assert tail[0].item() > 0.0

    def test_tail_mass_matches_cumsum(self):
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        m, S = self._make_m_S([[5.0, 4.0, 3.0, 2.0, 1.0]])
        mask, k, tail, tau_actual = compute_adaptive_topk_mask(m, S, tau_cov=0.9, k_min=1, k_max=5)
        # Computed tail should equal 1 - cumulative coverage at K*.
        assert math.isclose(tail[0].item(), 1.0 - tau_actual[0].item(), abs_tol=1e-6)

    def test_per_head_independent(self):
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        # Head 0: concentrated. Head 1: diffuse.
        m = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0, 0.0],   # one dominant block
                [0.0, 0.0, 0.0, 0.0, 0.0],     # uniform
            ],
            dtype=torch.float32, device="cuda",
        )
        S = torch.ones_like(m)
        mask, k, tail, _ = compute_adaptive_topk_mask(m, S, tau_cov=0.995, k_min=1, k_max=5)
        assert int(k[0]) == 1
        assert int(k[1]) == 5  # hits k_max (can't go higher than num_blocks)
        # Masks are disjoint per head in their structure
        assert mask[0].sum().item() == 1
        assert mask[1].sum().item() == 5

    def test_empty_blocks_returns_empty(self):
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        m = torch.empty(3, 0, device="cuda")
        S = torch.empty(3, 0, device="cuda")
        mask, k, tail, tau = compute_adaptive_topk_mask(m, S, tau_cov=0.995, k_min=2, k_max=10)
        assert mask.shape == (3, 0)
        assert k.shape == (3,)
        assert torch.all(k == 0)

    def test_k_min_clamped_to_num_blocks(self):
        """k_min should be silently reduced when num_blocks < k_min."""
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        m = torch.zeros(1, 3, device="cuda")
        S = torch.ones_like(m)
        mask, k, _, _ = compute_adaptive_topk_mask(m, S, tau_cov=0.1, k_min=5, k_max=10)
        # num_blocks=3; cannot select 5. Must be clamped to 3.
        assert int(k[0]) == 3
        assert mask.sum().item() == 3

    def test_no_cpu_sync_path(self):
        """Function output tensors must be on the same device as inputs."""
        from dotcache.kernels.certified_attention import compute_adaptive_topk_mask
        m = torch.randn(4, 16, device="cuda")
        S = torch.rand(4, 16, device="cuda").clamp(min=0.1)
        mask, k, tail, tau = compute_adaptive_topk_mask(m, S, tau_cov=0.995, k_min=2, k_max=10)
        for t in (mask, k, tail, tau):
            assert t.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCertifiedAttentionLayerAdaptive:
    def _make_cache(self, N=128, kv_heads=2, head_dim=32, block_size=16):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        torch.manual_seed(0)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        vals = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        return TieredKeyCacheLayer.from_fp16_cache(
            keys, vals, block_size=block_size, max_new_tokens=0,
        )

    def test_stats_include_adaptive_fields_when_enabled(self):
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        _, stats = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, tau_cov=0.995, k_min=2, k_max=8,
        )
        assert "k_star_mean" in stats
        assert "tau_cov_actual_mean" in stats
        assert "tail_mass_int8_est_mean" in stats
        assert 2 <= stats["k_star_min"] <= stats["k_star_max"] <= 8
        assert 0.0 <= stats["tau_cov_actual_mean"] <= 1.0

    def test_adaptive_disabled_leaves_existing_behavior(self):
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        out_off, stats_off = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, tau_cov=None,
        )
        out_none, stats_none = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, tau_cov=0.0,
        )
        # tau_cov None or 0 → no adaptive selection
        assert torch.allclose(out_off, out_none, atol=0.0, rtol=0.0)
        assert "k_star_mean" not in stats_off
        assert "k_star_mean" not in stats_none
