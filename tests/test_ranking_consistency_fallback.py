"""Unit tests for the Rung-3 ranking-consistency fallback detection.

Covers:
- detect_ranking_disagreement: pure-tensor correctness
- compute_fp16_block_scores: matches a reference per-head dense computation
- certified_attention_layer: telemetry schema under ranking_fallback=True

The actual per-head fallback action (output equals dense on trigger) is
validated in the next commit once Rung 3 rewrites skip_mask on disagree.
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestDetectRankingDisagreement:
    def test_identical_rankings_no_disagree(self):
        from dotcache.kernels.certified_attention import detect_ranking_disagreement

        int8 = torch.tensor([[5.0, 3.0, 1.0, 7.0]], device="cuda")
        fp16 = torch.tensor([[5.1, 3.0, 0.9, 7.2]], device="cuda")  # same argsort
        mask = detect_ranking_disagreement(int8, fp16, r=1)
        assert mask.shape == (1,)
        assert mask.dtype == torch.bool
        assert not mask.any().item()

        mask3 = detect_ranking_disagreement(int8, fp16, r=3)
        assert not mask3.any().item()

    def test_swapped_top1_flags_r1(self):
        from dotcache.kernels.certified_attention import detect_ranking_disagreement

        int8 = torch.tensor([[5.0, 3.0, 1.0, 7.0]], device="cuda")  # top-1 = idx 3
        fp16 = torch.tensor([[8.0, 3.0, 1.0, 7.0]], device="cuda")  # top-1 = idx 0
        assert detect_ranking_disagreement(int8, fp16, r=1).item() is True
        assert detect_ranking_disagreement(int8, fp16, r=3).item() is True

    def test_top1_agrees_but_top3_differs(self):
        from dotcache.kernels.certified_attention import detect_ranking_disagreement

        # argsort desc INT8: [3, 0, 1, 2]; FP16: [3, 0, 2, 1] — top-1 same, top-3 differs
        int8 = torch.tensor([[5.0, 3.0, 1.0, 7.0]], device="cuda")
        fp16 = torch.tensor([[5.0, 2.0, 3.0, 7.0]], device="cuda")
        assert detect_ranking_disagreement(int8, fp16, r=1).item() is False
        assert detect_ranking_disagreement(int8, fp16, r=3).item() is True

    def test_empty_input_returns_no_disagreement(self):
        from dotcache.kernels.certified_attention import detect_ranking_disagreement

        int8 = torch.empty(4, 0, device="cuda")
        fp16 = torch.empty(4, 0, device="cuda")
        mask = detect_ranking_disagreement(int8, fp16, r=1)
        assert mask.shape == (4,)
        assert not mask.any().item()

    def test_r_clamped_to_k(self):
        """If r > K, compare only the K available positions — don't crash."""
        from dotcache.kernels.certified_attention import detect_ranking_disagreement

        int8 = torch.tensor([[5.0, 3.0]], device="cuda")
        fp16 = torch.tensor([[3.0, 5.0]], device="cuda")  # swap
        # r=5 > K=2 should still flag the swap
        assert detect_ranking_disagreement(int8, fp16, r=5).item() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestComputeFp16BlockScores:
    def _make_cache(self, N=64, kv_heads=2, head_dim=32, block_size=16):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        torch.manual_seed(0)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        vals = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, vals, block_size=block_size, max_new_tokens=0,
        )
        return cache

    def test_matches_reference_per_head(self):
        from dotcache.kernels.certified_attention import compute_fp16_block_scores

        cache = self._make_cache(N=64, kv_heads=2, head_dim=32, block_size=16)
        num_q_heads = 4  # gqa_group = 2
        gqa_group = num_q_heads // cache.keys_fp16_cpu.shape[0]
        q_scale = 1.0 / (32 ** 0.5)
        q_all = torch.randn(num_q_heads, 32, dtype=torch.float16, device="cuda")

        # Pick two blocks per head (blocks 0 and 2 across all heads).
        K = 2
        block_indices = torch.tensor(
            [[0, 2]] * num_q_heads, dtype=torch.long, device="cuda",
        )

        got = compute_fp16_block_scores(
            cache, q_all, block_indices, cache.num_quantized_blocks,
            gqa_group, q_scale,
        )
        assert got.shape == (num_q_heads, K)

        # Reference: per head, per block, max over block tokens of (q · k) * scale.
        keys_ref = cache.keys_fp16_gpu if cache.keys_fp16_gpu is not None else cache.keys_fp16_cpu
        keys_ref = keys_ref.to(device="cuda", dtype=torch.float32)
        q_f32 = q_all.float()
        expected = torch.zeros_like(got)
        bs = cache.block_size
        for h in range(num_q_heads):
            kv_h = h // gqa_group
            for j in range(K):
                b = int(block_indices[h, j])
                start = b * bs
                end = min(start + bs, cache.num_tokens)
                logits = (keys_ref[kv_h, start:end] @ q_f32[h]) * q_scale
                expected[h, j] = logits.max()

        assert torch.allclose(got, expected, atol=1e-2, rtol=1e-2), \
            f"max abs diff {(got - expected).abs().max().item()}"

    def test_empty_k_returns_empty(self):
        from dotcache.kernels.certified_attention import compute_fp16_block_scores

        cache = self._make_cache(N=32, kv_heads=1, head_dim=16, block_size=16)
        q_all = torch.randn(2, 16, dtype=torch.float16, device="cuda")
        block_indices = torch.empty(2, 0, dtype=torch.long, device="cuda")
        got = compute_fp16_block_scores(
            cache, q_all, block_indices, cache.num_quantized_blocks,
            gqa_group=2, q_scale=1.0,
        )
        assert got.shape == (2, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCertifiedAttentionLayerTelemetry:
    def _make_cache(self, N=128, kv_heads=2, head_dim=32, block_size=16):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        torch.manual_seed(0)
        keys = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        vals = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
        return TieredKeyCacheLayer.from_fp16_cache(
            keys, vals, block_size=block_size, max_new_tokens=0,
        )

    def test_schema_when_disabled(self):
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        _, stats = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=False,
        )
        # When feature is off, ranking fields must not leak into the schema.
        assert "ranking_heads_total" not in stats
        assert "ranking_disagree_r1" not in stats

    def test_schema_when_enabled_populates_counts(self):
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        _, stats = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=True, ranking_r=1,
        )
        assert stats["ranking_heads_total"] == 4
        # r1/r3 are ints in [0, num_q_heads]
        assert 0 <= stats["ranking_disagree_r1"] <= 4
        assert 0 <= stats["ranking_disagree_r3"] <= 4
        # commit-2 doesn't act on disagreement yet
        assert stats["ranking_fallback_triggered"] == 0
        assert stats["ranking_fallback_mode"] == "full"
        assert stats["ranking_r"] == 1
        assert stats["ranking_k"] >= 1
        # score_gap / s_fp16_top1 present when ranking ran
        assert "s_int8_top1_mean" in stats
        assert "s_fp16_top1_mean" in stats

    def test_measure_mode_leaves_output_identical(self):
        """mode='measure' must detect disagreement without touching the output."""
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        out_off, _ = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=False,
        )
        out_measure, stats = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=True, ranking_r=1,
            ranking_fallback_mode="measure",
        )
        assert torch.allclose(out_off, out_measure, atol=0.0, rtol=0.0), \
            "measure mode must not alter the output tensor"
        assert stats["ranking_fallback_triggered"] == 0

    def test_full_mode_only_changes_disagreeing_heads(self):
        """mode='full' must replace output[h] only for heads where detection fired."""
        from dotcache.kernels.certified_attention import (
            certified_attention_layer,
            detect_ranking_disagreement,
            compute_fp16_block_scores,
        )
        cache = self._make_cache()
        q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")

        out_off, _ = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=False,
        )
        out_full, stats = certified_attention_layer(
            cache, q_all, gqa_group=2,
            collect_stats=True, ranking_fallback=True, ranking_r=1,
            ranking_fallback_mode="full",
        )

        # Determine which heads should have been replaced by re-running detection
        # directly on the same inputs. certified_attention_layer is deterministic
        # here so the disagree mask reproduces.
        from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
        n_qblocks = cache.num_quantized_blocks
        bs = cache.block_size
        q_scale = 1.0 / (32 ** 0.5)
        m_b, _, _ = fused_score_certify_multihead(
            K_int8_packed=cache.keys_int8[:, :n_qblocks * bs, :],
            K_scale=cache.keys_scale[:, :n_qblocks, :],
            q_all=q_all,
            correction=cache.correction[:, :n_qblocks],
            gqa_group=2,
            block_size=bs,
            q_scale=q_scale,
            block_epsilon=0.001,
        )
        K = max(1, stats["ranking_k"])
        top_idx = m_b[:, :n_qblocks].topk(K, dim=1).indices
        top_int8 = m_b[:, :n_qblocks].gather(1, top_idx)
        top_fp16 = compute_fp16_block_scores(
            cache, q_all, top_idx, n_qblocks, gqa_group=2, q_scale=q_scale,
        )
        disagree = detect_ranking_disagreement(top_int8, top_fp16, r=1)

        # Non-disagreeing heads stay bitwise-identical
        agree = ~disagree
        if agree.any():
            assert torch.allclose(
                out_off[agree], out_full[agree], atol=0.0, rtol=0.0,
            ), "non-disagreeing heads must keep their Phase-2 output"

        # Trigger count matches the disagree mask
        assert stats["ranking_fallback_triggered"] == int(disagree.sum().item())

    def test_full_mode_matches_dense_on_triggered_heads(self):
        """When fallback fires for a head, output[h] must equal FP16 dense attention."""
        from dotcache.kernels.certified_attention import certified_attention_layer
        cache = self._make_cache(N=512, kv_heads=2, head_dim=32, block_size=16)
        q_scale = 1.0 / (32 ** 0.5)

        # Search seeds until we find an input that actually triggers disagreement
        # (a larger context + larger K than the other tests makes this near-certain).
        q_all = None
        out_full = None
        stats = None
        for seed in range(64):
            torch.manual_seed(1000 + seed)
            q_all = torch.randn(4, 32, dtype=torch.float16, device="cuda")
            out_full, stats = certified_attention_layer(
                cache, q_all, gqa_group=2, q_scale=q_scale,
                collect_stats=True, ranking_fallback=True, ranking_r=3,
                ranking_fallback_mode="full",
            )
            if stats["ranking_fallback_triggered"] > 0:
                break
        assert stats["ranking_fallback_triggered"] > 0, (
            "Detection never fired across 64 seeds — the detection path is likely broken"
        )

        # Reference: full FP16 dense attention for every head.
        nt = cache.num_tokens
        keys = (cache.keys_fp16_gpu if cache.keys_fp16_gpu is not None
                else cache.keys_fp16_cpu.to("cuda"))[:, :nt, :].float()
        values = cache.get_values_f32()[:, :nt, :]
        ref = torch.zeros_like(out_full)
        for h in range(q_all.shape[0]):
            kv = h // 2
            logits = (keys[kv] @ q_all[h].float()) * q_scale
            w = torch.softmax(logits, dim=0)
            ref[h] = (w.unsqueeze(1) * values[kv]).sum(dim=0)

        # The test's strongest claim: on triggered heads, certified output
        # equals the dense reference to near-float precision. Use cosine
        # since the Phase-2 path keeps some intermediates in float32 while
        # the reference accumulates differently.
        # Re-derive the disagree mask deterministically.
        from dotcache.kernels.certified_attention import (
            detect_ranking_disagreement, compute_fp16_block_scores,
        )
        from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
        n_qblocks = cache.num_quantized_blocks
        bs = cache.block_size
        m_b, _, _ = fused_score_certify_multihead(
            K_int8_packed=cache.keys_int8[:, :n_qblocks * bs, :],
            K_scale=cache.keys_scale[:, :n_qblocks, :],
            q_all=q_all,
            correction=cache.correction[:, :n_qblocks],
            gqa_group=2,
            block_size=bs,
            q_scale=q_scale,
            block_epsilon=0.001,
        )
        K = max(1, stats["ranking_k"])
        top_idx = m_b[:, :n_qblocks].topk(K, dim=1).indices
        top_int8 = m_b[:, :n_qblocks].gather(1, top_idx)
        top_fp16 = compute_fp16_block_scores(
            cache, q_all, top_idx, n_qblocks, gqa_group=2, q_scale=q_scale,
        )
        disagree = detect_ranking_disagreement(top_int8, top_fp16, r=3)

        disagree_heads = torch.nonzero(disagree, as_tuple=True)[0]
        assert disagree_heads.numel() > 0
        for h in disagree_heads.tolist():
            cos = torch.nn.functional.cosine_similarity(
                out_full[h].unsqueeze(0), ref[h].unsqueeze(0), dim=1,
            ).item()
            assert cos > 0.9999, (
                f"head {h}: fallback output should equal FP16 dense (cos={cos:.6f})"
            )
