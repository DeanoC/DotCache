"""Step-3 tests: Eq. 30 boundary verification (paper §6.1, §8.6).

The audit found that the §8.6 "zero boundary triggers across all benchmark
runs" claim was VACUOUS — no boundary check existed in code. This step's
core deliverable is a live check, and the forced-trigger test below is a
hard merge gate: without it, an under-implemented check would silently
report zero triggers and reproduce the original vacuity.

Inputs to the boundary check (paper §6.1, eq:boundary_check):
    ℓ_b^int8 + Δ > ℓ^fp16_(r)
For each tail block b not in the promoted top-K set, the upper bound on
its FP16 log-mass (= INT8 log-mass + Δ) must NOT exceed the r-th highest
FP16 log-mass among promoted blocks.
"""

from __future__ import annotations

import math

import pytest
import torch


CUDA = torch.cuda.is_available()


def _make_cache_with_known_logits(
    *, n_blocks: int = 8, block_size: int = 16, head_dim: int = 32, kv_heads: int = 1,
):
    """Build a TieredKeyCacheLayer where we control the per-block max-logit
    of head 0 by injecting handcrafted FP16 keys before quantisation.

    Each block's keys are constant along the token dim with a chosen "spike"
    value at channel 0 — the dot product with q = e_0 then yields a known
    per-block logit equal to the spike value (× q_scale).
    """
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
    N = n_blocks * block_size
    keys = torch.zeros(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
    # Per-block logit "spikes" at channel 0; later overridden by caller.
    values = torch.randn(kv_heads, N, head_dim, dtype=torch.float16, device="cuda")
    return keys, values


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestEq30BoundaryCheck:
    def _run_certified(self, cache, q, *, ranking_r=1):
        """Wrap certified_attention_layer with paper §7 settings + boundary check."""
        from dotcache.kernels.certified_attention import certified_attention_layer
        return certified_attention_layer(
            cache, q, gqa_group=1,
            v_tolerance=0.5,
            collect_stats=True,
            ranking_fallback=True,
            ranking_r=ranking_r,
            ranking_fallback_mode="full",
            tau_cov=None,  # disable adaptive K* for predictability
            top_k_fp16_keys=2,
            score_consistency_check=False,  # isolate boundary path
        )

    def test_no_trigger_when_promoted_set_dominates(self):
        """When the promoted top-K blocks have logits >> any tail block's
        upper bound, the boundary check must not fire."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        torch.manual_seed(0)
        kv, n_blocks, bs, hd = 1, 8, 16, 32
        N = n_blocks * bs
        keys = torch.zeros(kv, N, hd, dtype=torch.float16, device="cuda")
        # Block 0 → big spike; block 1 → medium; rest → tiny.
        keys[:, 0:bs, 0] = 10.0     # block 0
        keys[:, bs:2*bs, 0] = 5.0   # block 1
        keys[:, 2*bs:, 0] = 0.01    # tail blocks: tiny
        values = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )
        q = torch.zeros(1, hd, dtype=torch.float16, device="cuda")
        q[0, 0] = 1.0  # query selects channel 0
        _, stats = self._run_certified(cache, q, ranking_r=1)
        assert stats["boundary_check_triggered_heads"] == 0, stats
        assert stats["boundary_check_fired"] is False, stats

    def test_forced_trigger_when_blocks_are_near_equal_with_large_delta(self):
        """HARD MERGE GATE — proves the check is live, not vacuous.

        Construction: high-variance random keys + q=ones saturates the per-
        channel Δ bound (Σ_c |q_c|·s_c is large). With many blocks of similar
        mass, the INT8 ranking inevitably places a block in the tail whose
        INT8 log-mass + Δ exceeds the r=1-th promoted FP16 log-mass. Across
        many random seeds at least one MUST trigger if the check is live;
        zero triggers across all seeds means the check is silently absent
        (the original §8.6 vacuity bug).
        """
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        kv, n_blocks, bs, hd = 1, 16, 16, 32
        N = n_blocks * bs
        triggers_seen = 0
        n_seeds = 16
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # std=2 → per-channel data range ~ ±6 → per-block scale ~ 12/255
            #   → Δ = (1/(2√d)) · d · 0.047 ≈ 0.27  (sizable)
            keys = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda") * 2.0
            values = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
            cache = TieredKeyCacheLayer.from_fp16_cache(
                keys, values, block_size=bs, max_new_tokens=0,
            )
            q = torch.ones(1, hd, dtype=torch.float16, device="cuda")  # saturates Δ
            _, stats = self._run_certified(cache, q, ranking_r=1)
            if stats["boundary_check_triggered_heads"] > 0:
                triggers_seen += 1
        assert triggers_seen >= 1, (
            f"Boundary check fired on 0/{n_seeds} random seeds with q=ones, "
            f"std=2 keys, n_blocks={n_blocks}, K=4. Either the check is "
            f"silently absent (the original §8.6 vacuity bug) or Δ is "
            f"under-computed."
        )

    def test_zero_delta_corner_case_no_false_trigger(self):
        """When Δ ≈ 0 (constant per-channel scales → tiny bound), the upper
        bound is just ℓ_b^int8 itself. A tail block with INT8 log-mass
        strictly less than the FP16 promoted r-th must NOT trigger."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
        torch.manual_seed(2)
        kv, n_blocks, bs, hd = 1, 4, 16, 32
        N = n_blocks * bs
        # Use uniform tiny values so per-channel scales are tiny → Δ ≈ 0.
        keys = torch.full((kv, N, hd), 1e-3, dtype=torch.float16, device="cuda")
        keys[:, 0:bs, 0] = 1.0    # block 0 promoted
        keys[:, bs:2*bs, 0] = 0.5  # tail
        keys[:, 2*bs:, 0] = 0.01   # other tail
        values = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )
        q = torch.zeros(1, hd, dtype=torch.float16, device="cuda")
        q[0, 0] = 1.0
        _, stats = self._run_certified(cache, q, ranking_r=1)
        # With clear separation (1.0 vs 0.5), no tail block should trigger.
        assert stats["boundary_check_triggered_heads"] == 0, stats

    def test_aggregator_carries_boundary_step_layer_counts(self):
        """The llama.py aggregator must propagate the boundary check fields
        the same way it does for rung1..rung4."""
        from dotcache.integrations.llama import CertifiedAttentionState

        # Synthesise per-layer step records and exercise aggregation directly.
        s = CertifiedAttentionState(
            tiered_caches={}, v_tolerance=0.05,
        )
        # Minimum fields the existing aggregator expects, plus the new
        # boundary-check fields this step adds.
        _base = {"total_blocks": 100, "skipped_blocks": 0, "skip_rate": 0.0}
        s.step_stats = [
            {**_base, "layer": 0, "boundary_check_fired": False,
             "boundary_check_triggered_heads": 0},
            {**_base, "layer": 1, "boundary_check_fired": True,
             "boundary_check_triggered_heads": 3},
            {**_base, "layer": 2, "boundary_check_fired": True,
             "boundary_check_triggered_heads": 1},
        ]
        agg = s.aggregate_step_stats()
        assert agg["boundary_check_fired"] is True
        assert agg["boundary_check_fired_layers"] == 2
        assert agg["boundary_check_triggered_heads_total"] == 4
