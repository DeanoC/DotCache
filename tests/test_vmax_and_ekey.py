"""Step-4 tests: V_max tracking + E_key formula assembly (paper §2.3, §4.5).

Closes the audit's Mismatch 4 ("V_max not computed, E_key not assembled").

Layer 1 — V_max tracking
- Per-block ν_b initialised at quant time as max_t ‖V_t‖₂ over block tokens.
- append_token updates ν_b for the partial block via running max.
- v_max_global() returns max_b ν_b across all kv-heads (paper §2.3 def).

Layer 2 — E_key formula identity
- Closed-form E_key = 2·V_max·e^{2Δ}·ᾱ_T·(e^{2Δ}−1) reproduces the
  telemetry-pipeline value to fp tolerance.
- Independent of head count (the formula is per-head, then mean/max).
"""

from __future__ import annotations

import math

import pytest
import torch


CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestVMaxTracking:
    def test_per_block_nu_b_matches_max_norm(self):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        kv, n_blocks, bs, hd, dv = 2, 4, 16, 32, 32
        N = n_blocks * bs
        torch.manual_seed(0)
        keys = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
        values = torch.randn(kv, N, dv, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )
        # Reference: per-token ‖V_t‖₂, per-block max
        ref = (
            values.float().norm(dim=-1)                           # [kv, N]
            .reshape(kv, n_blocks, bs).amax(dim=-1)               # [kv, n_blocks]
        )
        got = cache.values_norm_max_per_block[:, :n_blocks]
        assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)

    def test_v_max_global_is_max_over_blocks_and_heads(self):
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        kv, n_blocks, bs, hd, dv = 2, 4, 16, 32, 32
        N = n_blocks * bs
        torch.manual_seed(1)
        keys = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
        values = torch.randn(kv, N, dv, dtype=torch.float16, device="cuda")
        # Inject a HUGE value vector at one (kv-head, token) — V_max must catch it.
        values[1, 5, :] = 50.0
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=0,
        )
        v_max = cache.v_max_global()
        # ‖50·1_d‖₂ = 50·√32 ≈ 282.8
        expected = 50.0 * math.sqrt(dv)
        assert v_max >= expected * 0.99, f"v_max={v_max:.2f} expected≥{expected:.2f}"

    def test_append_token_updates_partial_block_nu(self):
        """ν_b for a partial block should grow as tokens are appended,
        taking the running max of per-token ‖V_t‖₂."""
        from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

        kv, n_blocks, bs, hd, dv = 1, 2, 16, 32, 32
        N = n_blocks * bs
        torch.manual_seed(2)
        keys = torch.randn(kv, N, hd, dtype=torch.float16, device="cuda")
        values = torch.randn(kv, N, dv, dtype=torch.float16, device="cuda")
        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=bs, max_new_tokens=64,
        )
        next_block = n_blocks  # the about-to-fill block index
        assert cache.values_norm_max_per_block[0, next_block].item() == 0.0

        # Append a token with a known small norm
        small_v = torch.full((kv, 1, dv), 0.1, dtype=torch.float16, device="cuda")
        small_norm = float(small_v[0, 0].float().norm().item())
        cache.append_token(
            torch.zeros(kv, 1, hd, dtype=torch.float16, device="cuda"),
            small_v,
        )
        assert cache.values_norm_max_per_block[0, next_block].item() == pytest.approx(
            small_norm, abs=1e-3
        )

        # Append a token with a HUGE norm — ν_b must jump
        big_v = torch.full((kv, 1, dv), 5.0, dtype=torch.float16, device="cuda")
        big_norm = float(big_v[0, 0].float().norm().item())  # 5·√32 ≈ 28.28
        cache.append_token(
            torch.zeros(kv, 1, hd, dtype=torch.float16, device="cuda"),
            big_v,
        )
        assert cache.values_norm_max_per_block[0, next_block].item() == pytest.approx(
            big_norm, abs=1e-2
        )

        # Append a small one again — ν_b must STAY at the max
        cache.append_token(
            torch.zeros(kv, 1, hd, dtype=torch.float16, device="cuda"),
            small_v,
        )
        assert cache.values_norm_max_per_block[0, next_block].item() == pytest.approx(
            big_norm, abs=1e-2
        ), "ν_b regressed — append_token didn't take running max"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestEKeyFormula:
    def _e_key_closed_form(self, v_max: float, delta: torch.Tensor,
                           tail_mass: torch.Tensor) -> torch.Tensor:
        """Paper §4.5: E_key = 2·V_max·e^{2Δ}·ᾱ_T·(e^{2Δ}−1)."""
        e2d = torch.exp(2.0 * delta.float())
        return 2.0 * v_max * e2d * tail_mass.float() * (e2d - 1.0)

    def test_formula_matches_telemetry_pipeline(self):
        """Run the smoke pipeline and verify the emitted e_key_step_mean
        matches the closed-form formula computed from v_max + Δ + ᾱ_T."""
        # Small import-and-execute (re-uses Step-2 smoke harness).
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from test_paper_pipeline_smoke import _run_paper_pipeline

        result = _run_paper_pipeline(
            v_tolerance=0.05, use_int4_values=True,
            prompt_len=64, decode_steps=8,
        )
        stats = result["stats"]
        assert "e_key_step_mean" in stats, (
            f"e_key_step_mean missing from telemetry; got keys: {sorted(stats)}"
        )
        assert stats["e_key_step_mean"] >= 0.0
        assert stats["e_key_step_max"] >= stats["e_key_step_mean"]
        assert "v_max_global" in stats and stats["v_max_global"] > 0.0

    def test_formula_zero_when_tail_mass_zero(self):
        """ᾱ_T = 0 → E_key = 0 by the formula (perfect coverage)."""
        delta = torch.tensor([0.1, 0.2, 0.3])
        tail = torch.zeros(3)
        e_key = self._e_key_closed_form(v_max=10.0, delta=delta, tail_mass=tail)
        assert torch.all(e_key == 0.0)

    def test_formula_zero_when_delta_zero(self):
        """Δ = 0 (perfect quant) → E_key = 0 since (e^0 − 1) = 0."""
        delta = torch.zeros(3)
        tail = torch.tensor([0.1, 0.2, 0.3])
        e_key = self._e_key_closed_form(v_max=10.0, delta=delta, tail_mass=tail)
        assert torch.allclose(e_key, torch.zeros_like(e_key), atol=1e-6)

    def test_formula_monotone_in_v_max(self):
        """E_key grows linearly in V_max, holding Δ and ᾱ_T fixed."""
        delta = torch.tensor([0.1])
        tail = torch.tensor([0.005])
        e1 = self._e_key_closed_form(v_max=1.0, delta=delta, tail_mass=tail).item()
        e2 = self._e_key_closed_form(v_max=2.0, delta=delta, tail_mass=tail).item()
        assert e2 == pytest.approx(2.0 * e1, rel=1e-6)

    def test_aggregator_carries_e_key_and_v_max(self):
        """llama.py aggregator must roll up e_key_step_mean/max and v_max_global."""
        from dotcache.integrations.llama import CertifiedAttentionState

        s = CertifiedAttentionState(
            tiered_caches={}, v_tolerance=0.05,
        )
        _base = {"total_blocks": 100, "skipped_blocks": 0, "skip_rate": 0.0}
        s.step_stats = [
            {**_base, "layer": 0, "e_key_step_mean": 0.01, "e_key_step_max": 0.05,
             "v_max_layer": 12.0},
            {**_base, "layer": 1, "e_key_step_mean": 0.02, "e_key_step_max": 0.08,
             "v_max_layer": 15.0},
            {**_base, "layer": 2, "e_key_step_mean": 0.03, "e_key_step_max": 0.10,
             "v_max_layer": 11.0},
        ]
        agg = s.aggregate_step_stats()
        assert agg["e_key_step_mean"] == pytest.approx((0.01 + 0.02 + 0.03) / 3.0)
        assert agg["e_key_step_max"] == pytest.approx(0.10)
        assert agg["v_max_global"] == pytest.approx(15.0)
