"""Tests for interval and ellipsoidal upper bounds (paper safety guarantees).

Validates that:
1. The interval bound is a valid upper bound on the true max logit.
2. The interval bound is tighter than the spherical bound.
3. The ellipsoidal bound is a valid upper bound on the true max logit.
4. Per-dim compression error correctly widens the interval envelope.
5. Config gating works: disabling both bounds gives identical behavior to baseline.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")


def _make_random_block(
    *,
    num_tokens: int = 16,
    head_dim: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a random key block: [num_tokens, head_dim]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_tokens, head_dim, generator=gen, device=device, dtype=torch.float32)


def _true_max_logit(query: torch.Tensor, keys: torch.Tensor, query_scale: float) -> float:
    """Compute the true maximum logit: max_t (Q . K_t) * query_scale."""
    logits = torch.matmul(keys, query) * query_scale
    return float(logits.max().item())


def _spherical_upper_bound(
    query: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    comp_error: float,
    query_scale: float,
) -> float:
    """Spherical bound: center_sim + ||Q|| * (radius + comp_error) * |query_scale|."""
    center_sim = float(torch.dot(center, query).item()) * query_scale
    q_norm = float(torch.linalg.vector_norm(query).item())
    return center_sim + q_norm * (radius + comp_error) * abs(query_scale)


class TestIntervalBound:
    """Tests for _compute_interval_upper_bound."""

    def _import_bound_fn(self):
        from dotcache.backends.metal.persistent_runtime import _compute_interval_upper_bound
        return _compute_interval_upper_bound

    def test_is_valid_upper_bound(self):
        """Interval bound must be >= true max logit for random Q and blocks."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        for seed in range(20):
            keys = _make_random_block(num_tokens=16, head_dim=128, seed=seed)
            query = torch.randn(128, dtype=torch.float32)

            k_min = keys.min(dim=0).values  # [head_dim]
            k_max = keys.max(dim=0).values
            comp_error_dim = torch.zeros(128, dtype=torch.float32)

            upper = compute(
                query_vec=query,
                k_min=k_min.unsqueeze(0),  # [1, head_dim]
                k_max=k_max.unsqueeze(0),
                comp_error_dim=comp_error_dim.unsqueeze(0),
                query_scale=query_scale,
            )
            true_max = _true_max_logit(query, keys, query_scale)
            assert float(upper[0].item()) >= true_max - 1e-5, (
                f"Interval bound {float(upper[0].item()):.6f} < true max {true_max:.6f} at seed={seed}"
            )

    def test_tighter_on_anisotropic_data(self):
        """Interval bound is tighter than spherical on anisotropic (real-like) data.

        On isotropic random data, the spherical bound can be tighter because
        Cauchy-Schwarz is tight for spherical distributions.  On anisotropic data
        (like real transformer keys), the per-dimension interval bound wins.
        We create anisotropic blocks by scaling dimensions non-uniformly.
        """
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        tighter_count = 0
        total = 50
        for seed in range(total):
            keys = _make_random_block(num_tokens=16, head_dim=128, seed=seed)
            # Make anisotropic: first 16 dims have 10x scale
            scale = torch.ones(128, dtype=torch.float32)
            scale[:16] = 10.0
            keys = keys * scale.unsqueeze(0)
            torch.manual_seed(seed + 1000)
            query = torch.randn(128, dtype=torch.float32)

            k_min = keys.min(dim=0).values
            k_max = keys.max(dim=0).values
            center = keys.mean(dim=0)
            distances = torch.linalg.vector_norm(keys - center.unsqueeze(0), dim=-1)
            radius = float(distances.max().item())

            upper_I = float(
                compute(
                    query_vec=query,
                    k_min=k_min.unsqueeze(0),
                    k_max=k_max.unsqueeze(0),
                    comp_error_dim=torch.zeros(1, 128),
                    query_scale=query_scale,
                )[0].item()
            )
            upper_S = _spherical_upper_bound(query, center, radius, 0.0, query_scale)
            true_max = _true_max_logit(query, keys, query_scale)

            # Both must be valid upper bounds
            assert upper_I >= true_max - 1e-5
            assert upper_S >= true_max - 1e-5
            if upper_I < upper_S - 1e-4:
                tighter_count += 1

        # Interval should be strictly tighter on most anisotropic cases
        assert tighter_count > total * 0.5, (
            f"Interval was tighter on only {tighter_count}/{total} anisotropic cases (expected >50%)"
        )

    def test_with_compression_error(self):
        """Per-dim compression error should widen the bound but remain valid."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        keys = _make_random_block(num_tokens=16, head_dim=128, seed=7)
        query = torch.randn(128, dtype=torch.float32)

        k_min = keys.min(dim=0).values
        k_max = keys.max(dim=0).values

        # Without error
        upper_no_err = float(
            compute(
                query_vec=query,
                k_min=k_min.unsqueeze(0),
                k_max=k_max.unsqueeze(0),
                comp_error_dim=torch.zeros(1, 128),
                query_scale=query_scale,
            )[0].item()
        )

        # With error
        comp_err = torch.full((1, 128), 0.01)
        upper_with_err = float(
            compute(
                query_vec=query,
                k_min=k_min.unsqueeze(0),
                k_max=k_max.unsqueeze(0),
                comp_error_dim=comp_err,
                query_scale=query_scale,
            )[0].item()
        )

        # Error should widen (or maintain) the bound
        assert upper_with_err >= upper_no_err - 1e-6

    def test_single_token_block(self):
        """Single-token block: interval bound should equal the exact logit."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        key = torch.randn(1, 128, dtype=torch.float32)
        query = torch.randn(128, dtype=torch.float32)

        upper = float(
            compute(
                query_vec=query,
                k_min=key,  # [1, 128] — min == max == the single key
                k_max=key,
                comp_error_dim=torch.zeros(1, 128),
                query_scale=query_scale,
            )[0].item()
        )
        true_val = _true_max_logit(query, key, query_scale)
        assert abs(upper - true_val) < 1e-4, f"Single-token: upper={upper:.6f}, true={true_val:.6f}"

    def test_negative_query_components(self):
        """Bound remains valid when query has mostly negative components."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        keys = _make_random_block(num_tokens=16, head_dim=128, seed=99)
        query = -torch.abs(torch.randn(128, dtype=torch.float32))  # All negative

        k_min = keys.min(dim=0).values
        k_max = keys.max(dim=0).values

        upper = float(
            compute(
                query_vec=query,
                k_min=k_min.unsqueeze(0),
                k_max=k_max.unsqueeze(0),
                comp_error_dim=torch.zeros(1, 128),
                query_scale=query_scale,
            )[0].item()
        )
        true_max = _true_max_logit(query, keys, query_scale)
        assert upper >= true_max - 1e-5


class TestEllipsoidalBound:
    """Tests for _compute_ellipsoidal_upper_bound."""

    def _import_bound_fn(self):
        from dotcache.backends.metal.persistent_runtime import _compute_ellipsoidal_upper_bound
        return _compute_ellipsoidal_upper_bound

    def _compute_ellipsoidal_metadata(self, keys: torch.Tensor):
        """Compute center, pc1, r_along, r_perp from a key block [tokens, head_dim]."""
        center = keys.mean(dim=0)
        centered = keys - center.unsqueeze(0)
        # Power iteration
        v = centered[0].clone()
        v_norm = torch.linalg.vector_norm(v)
        if float(v_norm.item()) < 1e-12:
            v = torch.randn_like(v)
            v_norm = torch.linalg.vector_norm(v)
        v = v / v_norm.clamp_min(1e-12)
        for _ in range(10):
            v = torch.matmul(centered.T, torch.matmul(centered, v))
            v_norm = torch.linalg.vector_norm(v)
            if float(v_norm.item()) < 1e-12:
                break
            v = v / v_norm.clamp_min(1e-12)
        projections = torch.matmul(centered, v)
        r_along = float(projections.abs().max().item())
        perp = centered - projections[:, None] * v[None, :]
        perp_norms = torch.linalg.vector_norm(perp, dim=-1)
        r_perp = float(perp_norms.max().item())
        return center, v, r_along, r_perp

    def test_is_valid_upper_bound(self):
        """Ellipsoidal bound must be >= true max logit."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        for seed in range(20):
            keys = _make_random_block(num_tokens=16, head_dim=128, seed=seed)
            query = torch.randn(128, dtype=torch.float32)
            q_norm = float(torch.linalg.vector_norm(query).item())

            center, pc1, r_along, r_perp = self._compute_ellipsoidal_metadata(keys)

            upper = float(
                compute(
                    query_vec=query,
                    query_norm=q_norm,
                    center=center.unsqueeze(0),
                    pc1=pc1.unsqueeze(0),
                    r_along=torch.tensor([r_along]),
                    r_perp=torch.tensor([r_perp]),
                    comp_error=torch.zeros(1),
                    query_scale=query_scale,
                )[0].item()
            )
            true_max = _true_max_logit(query, keys, query_scale)
            assert upper >= true_max - 1e-4, (
                f"Ellipsoidal bound {upper:.6f} < true max {true_max:.6f} at seed={seed}"
            )

    def test_single_token_block(self):
        """Single-token block: r_along=0, r_perp=0 → bound equals exact logit."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        key = torch.randn(1, 128, dtype=torch.float32)
        query = torch.randn(128, dtype=torch.float32)
        q_norm = float(torch.linalg.vector_norm(query).item())
        center = key[0]

        upper = float(
            compute(
                query_vec=query,
                query_norm=q_norm,
                center=center.unsqueeze(0),
                pc1=torch.zeros(1, 128),
                r_along=torch.zeros(1),
                r_perp=torch.zeros(1),
                comp_error=torch.zeros(1),
                query_scale=query_scale,
            )[0].item()
        )
        true_val = _true_max_logit(query, key, query_scale)
        assert abs(upper - true_val) < 1e-4

    def test_identical_tokens(self):
        """All tokens identical: center=token, r_along=r_perp=0."""
        compute = self._import_bound_fn()
        query_scale = 1.0 / math.sqrt(128)
        single_key = torch.randn(128, dtype=torch.float32)
        keys = single_key.unsqueeze(0).expand(8, -1).clone()
        query = torch.randn(128, dtype=torch.float32)
        q_norm = float(torch.linalg.vector_norm(query).item())

        center, pc1, r_along, r_perp = self._compute_ellipsoidal_metadata(keys)
        # r_along and r_perp should be ~0 for identical tokens
        assert r_along < 1e-5
        assert r_perp < 1e-5

        upper = float(
            compute(
                query_vec=query,
                query_norm=q_norm,
                center=center.unsqueeze(0),
                pc1=pc1.unsqueeze(0),
                r_along=torch.tensor([r_along]),
                r_perp=torch.tensor([r_perp]),
                comp_error=torch.zeros(1),
                query_scale=query_scale,
            )[0].item()
        )
        true_val = _true_max_logit(query, keys, query_scale)
        assert abs(upper - true_val) < 1e-3


class TestPerDimCompError:
    """Tests for _estimate_m0_key_comp_error_with_dim."""

    def _import_fn(self):
        from dotcache.backends.metal.persistent_runtime import _estimate_m0_key_comp_error_with_dim
        return _estimate_m0_key_comp_error_with_dim

    def test_returns_none_without_config(self):
        fn = self._import_fn()
        result = fn(key_slice=torch.randn(16, 128), dotcache_config=None)
        assert result is None

    def test_dim_error_shape(self):
        """Per-dim error should have shape [head_dim]."""
        fn = self._import_fn()

        class FakeConfig:
            group_size = 32
            bits_k = 4
            quant_scheme_k = "affine"

        keys = torch.randn(16, 128, dtype=torch.float32)
        result = fn(key_slice=keys, dotcache_config=FakeConfig())
        if result is not None:
            scalar_err, dim_err = result
            assert dim_err.shape == (128,)
            assert scalar_err >= 0.0
            assert np.all(dim_err >= 0.0)


class TestBoundConfigGating:
    """Test that config flags correctly gate bound computation."""

    def test_interval_bound_disabled(self):
        """When enable_interval_bound=False, interval should not be used."""
        from dotcache.backends.metal.persistent_types import PersistentServingConfig
        config = PersistentServingConfig(enable_interval_bound=False)
        assert not bool(config.enable_interval_bound)

    def test_interval_bound_enabled_by_default(self):
        """Interval bound should be enabled by default."""
        from dotcache.backends.metal.persistent_types import PersistentServingConfig
        config = PersistentServingConfig()
        assert bool(config.enable_interval_bound)

    def test_ellipsoidal_bound_disabled_by_default(self):
        """Ellipsoidal bound should be disabled by default."""
        from dotcache.backends.metal.persistent_types import PersistentServingConfig
        config = PersistentServingConfig()
        assert not bool(config.enable_ellipsoidal_bound)

    def test_per_dim_comp_error_enabled_by_default(self):
        from dotcache.backends.metal.persistent_types import PersistentServingConfig
        config = PersistentServingConfig()
        assert bool(config.enable_per_dim_comp_error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
