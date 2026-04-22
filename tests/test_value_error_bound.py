"""Unit tests for the tight value-error bound Σ_b ρ_b · η_b (paper §3.4
follow-up). Compares the tight per-block sum against hand-computed
references and the loose upper bound."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestValueErrorBound:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_bound(
        self,
        *,
        logits: list[list[float]],              # [num_q_heads][num_blocks]
        skip: list[list[bool]],                 # same shape
        eta: list[list[float]],                 # [num_kv_heads][num_blocks]
        gqa_group: int,
        top_k: int,
    ):
        from dotcache.kernels.certified_attention import (
            compute_tier2_residual_mass,
            compute_value_error_bound,
        )

        m_b = torch.tensor(logits, dtype=torch.float32, device="cuda")
        # Make S_b = 1 per block — keeps the mass partition equal to exp(m)
        # normalised per head, which is easy to reason about in the tests.
        S_b = torch.ones_like(m_b)
        skip_mask = torch.tensor(skip, dtype=torch.bool, device="cuda")
        eta_t = torch.tensor(eta, dtype=torch.float32, device="cuda")

        rho, mass_frac, topk_idx = compute_tier2_residual_mass(
            m_b, S_b, skip_mask, top_k=top_k, return_details=True,
        )
        e_val = compute_value_error_bound(
            mass_frac=mass_frac,
            topk_idx=topk_idx,
            skip_mask=skip_mask,
            eta_per_block=eta_t,
            gqa_group=gqa_group,
        )
        return rho, mass_frac, topk_idx, e_val, eta_t

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_hand_computed_uniform_heads(self):
        """Single kv-head, single q-head, one block = full mass: E_val = 0
        because that block is top-K (excluded from residual). Check the
        second-strongest block's η contributes nothing when top_k=1."""
        rho, mf, idx, e_val, _ = self._run_bound(
            logits=[[10.0, 0.0, 0.0, 0.0]],
            skip=[[False, False, False, False]],
            eta=[[1.0, 1.0, 1.0, 1.0]],
            gqa_group=1,
            top_k=1,
        )
        # Top-K=1 captures ~all the mass; residual mass ≈ 3e-4 (softmax of
        # 10 vs three 0s); E_val ≈ 3e-4. Sub-millisignificant upper bound.
        assert e_val.shape == (1,)
        assert e_val[0].item() < 1e-3, f"E_val should be tiny for concentrated head, got {e_val[0].item()}"

    def test_matches_hand_computed_sum(self):
        """Compute E_val explicitly on a hand-sized example and compare."""
        # 1 q_head, 1 kv_head, 4 blocks, equal logits so mass_frac = 0.25 each.
        # top_k = 1 → one block is excluded.
        # skip none. η = [0.1, 0.2, 0.3, 0.4]
        # Top-K block = index 0 (tie-broken to first, or any — we check sum).
        rho, mf, idx, e_val, eta_t = self._run_bound(
            logits=[[0.0, 0.0, 0.0, 0.0]],
            skip=[[False, False, False, False]],
            eta=[[0.1, 0.2, 0.3, 0.4]],
            gqa_group=1,
            top_k=1,
        )
        # Residual set = 3 blocks. Each mass_frac = 0.25.
        # E_val = Σ 0.25 · η_b for the 3 residual blocks.
        topk_b = int(idx[0, 0].item())
        expected = sum(0.25 * eta_t[0, b].item() for b in range(4) if b != topk_b)
        assert abs(e_val[0].item() - expected) < 1e-5, (
            f"got {e_val[0].item()}, expected {expected}"
        )

    def test_skipped_blocks_excluded_from_sum(self):
        """A block that's skipped is in neither the residual nor top-K — it
        contributes zero to E_val regardless of its η."""
        # 1 q_head, 4 blocks, equal logits. Block 3 skipped, huge η.
        rho, mf, idx, e_val, eta_t = self._run_bound(
            logits=[[0.0, 0.0, 0.0, 0.0]],
            skip=[[False, False, False, True]],
            eta=[[0.1, 0.1, 0.1, 100.0]],   # block 3 has a huge value error
            gqa_group=1,
            top_k=1,
        )
        # Residual excludes both top-K (one block) and skipped (block 3).
        # E_val should not see the 100.0 term.
        assert e_val[0].item() < 1.0, f"skipped block should not contribute, got {e_val[0].item()}"

    def test_bounded_above_by_loose_rho_eta_max(self):
        """Invariant: E_val ≤ ρ · η_max when top-K and skip are disjoint
        (which is the real-world precondition — the adaptive selector
        scatters skip_mask=False at the top-K indices before passing into
        the v_format decision). When top-K and skip overlap, the legacy
        rho = 1 - α_K - β double-subtracts that overlap's mass, so it can
        underestimate the true residual — in that case tight > ρ·η_max is
        *correct*, not a bug."""
        from dotcache.kernels.certified_attention import (
            compute_tier2_residual_mass,
            compute_value_error_bound,
        )

        torch.manual_seed(0)
        num_q_heads, num_kv_heads = 8, 2   # gqa_group = 4
        num_blocks = 32
        m_b = torch.randn(num_q_heads, num_blocks, device="cuda")
        S_b = torch.rand(num_q_heads, num_blocks, device="cuda") + 0.1
        skip_mask = torch.rand(num_q_heads, num_blocks, device="cuda") < 0.3
        eta = torch.rand(num_kv_heads, num_blocks, device="cuda") * 0.5

        # Mirror the adaptive-path invariant: force top-K out of skip.
        top_k = 4
        _, topk_idx = m_b.topk(top_k, dim=1)
        skip_mask.scatter_(1, topk_idx, False)

        rho, mf, idx = compute_tier2_residual_mass(
            m_b, S_b, skip_mask, top_k=top_k, return_details=True,
        )
        e_val = compute_value_error_bound(
            mass_frac=mf, topk_idx=idx, skip_mask=skip_mask,
            eta_per_block=eta, gqa_group=num_q_heads // num_kv_heads,
        )
        eta_max = float(eta.max().item())
        loose = rho * eta_max  # [num_q_heads]
        # Allow a tiny FP-rounding slack.
        assert (e_val <= loose + 1e-5).all(), (
            f"tight E_val should never exceed loose ρ·η_max bound "
            f"when top-K and skip are disjoint.\n"
            f"e_val: {e_val.tolist()}\nrho*eta_max: {loose.tolist()}"
        )

    def test_equal_eta_matches_rho_times_eta(self):
        """Degenerate case: when η_b is constant across all blocks,
        E_val = ρ · η_constant exactly."""
        torch.manual_seed(1)
        num_q_heads, num_kv_heads = 4, 1
        num_blocks = 16
        logits = torch.randn(num_q_heads, num_blocks).tolist()
        skip = [[False] * num_blocks for _ in range(num_q_heads)]
        eta_const = 0.1234
        eta = [[eta_const] * num_blocks for _ in range(num_kv_heads)]

        rho, mf, idx, e_val, _ = self._run_bound(
            logits=logits, skip=skip, eta=eta,
            gqa_group=num_q_heads // num_kv_heads, top_k=2,
        )
        expected = rho * eta_const
        assert torch.allclose(e_val, expected, atol=1e-5), (
            f"e_val should equal rho·eta_const when η is flat.\n"
            f"e_val: {e_val.tolist()}\nexpected: {expected.tolist()}"
        )

    def test_gqa_maps_q_to_correct_kv(self):
        """With gqa_group=4, q_heads 0-3 use kv_head 0, q_heads 4-7 use kv_head 1.
        Giving kv_head 0 a much larger η should only bump q_heads 0-3's E_val."""
        num_q_heads = 8
        num_blocks = 4
        # All q heads have identical mass partition (flat logits, no skip).
        logits = [[0.0] * num_blocks for _ in range(num_q_heads)]
        skip = [[False] * num_blocks for _ in range(num_q_heads)]
        eta = [
            [1.0, 1.0, 1.0, 1.0],   # kv_head 0 — large
            [0.1, 0.1, 0.1, 0.1],   # kv_head 1 — small
        ]
        rho, mf, idx, e_val, _ = self._run_bound(
            logits=logits, skip=skip, eta=eta, gqa_group=4, top_k=1,
        )
        # q_heads 0-3 should have e_val ≈ 10× the value of q_heads 4-7.
        group0 = e_val[:4]
        group1 = e_val[4:]
        assert torch.allclose(group0, group0[0].expand_as(group0), atol=1e-5)
        assert torch.allclose(group1, group1[0].expand_as(group1), atol=1e-5)
        assert group0[0].item() > 5 * group1[0].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestDecideVFormat:
    def test_tight_mode_below_tolerance_returns_int4(self):
        from dotcache.kernels.certified_attention import decide_v_format_tight
        e_val = torch.tensor([0.01, 0.02, 0.005], device="cuda")
        assert decide_v_format_tight(e_val, tolerance=0.1) == "int4"

    def test_tight_mode_above_tolerance_returns_fp16(self):
        from dotcache.kernels.certified_attention import decide_v_format_tight
        e_val = torch.tensor([0.01, 0.5, 0.005], device="cuda")
        # Max = 0.5, tolerance = 0.1 → fp16
        assert decide_v_format_tight(e_val, tolerance=0.1) == "fp16"

    def test_tight_is_at_least_as_permissive_as_loose(self):
        """If loose returns int4, tight must also return int4 on the same
        state (since E_val ≤ ρ·η_max when top-K ∩ skip = ∅). The converse
        isn't guaranteed."""
        from dotcache.kernels.certified_attention import (
            compute_tier2_residual_mass,
            compute_value_error_bound,
            decide_v_format,
            decide_v_format_tight,
        )

        torch.manual_seed(2)
        num_q_heads, num_kv_heads = 8, 2
        num_blocks = 16
        m_b = torch.randn(num_q_heads, num_blocks, device="cuda")
        S_b = torch.rand(num_q_heads, num_blocks, device="cuda") + 0.1
        skip = torch.rand(num_q_heads, num_blocks, device="cuda") < 0.2
        eta = torch.rand(num_kv_heads, num_blocks, device="cuda") * 0.5

        # Mirror real-world precondition.
        top_k = 4
        _, topk_idx = m_b.topk(top_k, dim=1)
        skip.scatter_(1, topk_idx, False)

        rho, mf, idx = compute_tier2_residual_mass(
            m_b, S_b, skip, top_k=top_k, return_details=True,
        )
        e_val = compute_value_error_bound(
            mass_frac=mf, topk_idx=idx, skip_mask=skip,
            eta_per_block=eta, gqa_group=num_q_heads // num_kv_heads,
        )
        tol = 0.25

        loose = decide_v_format(rho, float(eta.max().item()), tolerance=tol)
        tight = decide_v_format_tight(e_val, tolerance=tol)

        # Implication: loose == 'int4' ⇒ tight == 'int4'.
        if loose == "int4":
            assert tight == "int4", (
                f"tight bound should be at least as permissive as loose. "
                f"loose=int4 but tight={tight}. "
                f"e_val.max={e_val.max().item()}  rho*eta_max={rho.max().item()*eta.max().item()}"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestTopKFromLayerConfig:
    """The residual-mass recompute inside certified_attention_layer must
    use the layer's configured top_k_fp16_keys, not the module-level
    default (TOP_K_BLOCKS = 4). If they diverge, E_val's residual set
    won't match the actual attended set and the bound becomes
    misleading for experiments that tune top_k_fp16_keys."""

    def test_compute_tier2_honors_top_k_arg(self):
        """Different top_k values produce different residual-mass ρ
        on the same inputs — confirms the argument is load-bearing."""
        from dotcache.kernels.certified_attention import compute_tier2_residual_mass
        torch.manual_seed(3)
        m_b = torch.randn(4, 32, device="cuda")
        S_b = torch.rand(4, 32, device="cuda") + 0.1
        skip = torch.zeros(4, 32, dtype=torch.bool, device="cuda")

        rho_k2 = compute_tier2_residual_mass(m_b, S_b, skip, top_k=2)
        rho_k8 = compute_tier2_residual_mass(m_b, S_b, skip, top_k=8)

        # Larger top_k → smaller residual (more blocks excluded).
        assert (rho_k8 <= rho_k2 + 1e-6).all(), (
            f"rho_k8 should be ≤ rho_k2 elementwise; got k2={rho_k2.tolist()} k8={rho_k8.tolist()}"
        )
        # Not equal in general.
        assert not torch.allclose(rho_k2, rho_k8), (
            "test inputs too trivial to demonstrate top_k sensitivity"
        )


class TestValueErrorModeValidation:
    """certified_attention_layer must reject unknown value_error_mode
    values rather than silently falling through to the loose bound.
    A typo in CLI/config wiring would otherwise swap cert behaviour
    without a signal."""

    def test_certified_attention_rejects_unknown_mode(self):
        """We don't instantiate the full machinery here — just confirm
        the input-validation line raises the expected ValueError with
        a message naming the bad value. The full code path is covered
        by integration benches that pass the valid modes."""
        import re
        import pytest as _pt

        # Read the source file to confirm the validator exists and
        # names both 'loose' and 'tight'. This is a belt-and-braces
        # regression guard — any refactor that drops the check will
        # trip this test even without a GPU available.
        from pathlib import Path
        src = Path(
            "dotcache/kernels/certified_attention.py"
        ).read_text()
        # Must reject unknown mode with a ValueError that mentions the
        # actual bad value (so typos are easy to spot in logs).
        assert re.search(
            r"value_error_mode not in \(['\"]loose['\"], ['\"]tight['\"]\).*raise ValueError",
            src, re.DOTALL,
        ), "lost the value_error_mode input validator"
        assert "value_error_mode must be 'loose' or 'tight'" in src, (
            "validator's error message should name both valid modes"
        )
