import numpy as np

from dotcache.planner import (
    LayerPolicy,
    PageModeSpec,
    choose_page_mode,
    make_tier_candidates,
    observe_page,
)


def test_observe_page_reports_basic_stats() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, -3.0]], dtype=np.float32)
    stats = observe_page(values)
    assert stats.token_count == 2
    assert stats.abs_max == 3.0
    assert stats.rms > 0.0
    assert stats.channel_range_mean > 0.0


def test_choose_page_mode_forces_recent_pages_to_escape() -> None:
    policy = make_tier_candidates(
        kind="K",
        sensitivity_tier="balanced",
        default_bits=4,
        default_quant_scheme="affine",
        default_mode="M0",
        recent_window=128,
    )
    mode = choose_page_mode(0, "K", 12, observe_page(np.ones((4, 8), dtype=np.float32)), layer_policy=policy)
    assert mode.mode == "M3"
    assert mode.fallback_reason == "recent_window"
    assert mode.age_bucket == "recent"


def test_choose_page_mode_balanced_k_prefers_cheaper_candidate_when_stats_are_safe() -> None:
    policy = make_tier_candidates(
        kind="K",
        sensitivity_tier="balanced",
        default_bits=4,
        default_quant_scheme="affine",
        default_mode="M0",
        recent_window=0,
    )
    values = np.full((8, 32), 0.05, dtype=np.float32)
    mode = choose_page_mode(0, "K", 256, observe_page(values), layer_policy=policy)
    assert mode.mode in {"M0", "M2"}
    assert mode.bits in {2, 4}
    assert mode.sensitivity_tier == "balanced"


def test_choose_page_mode_strict_policy_excludes_aggressive_candidates() -> None:
    policy = make_tier_candidates(
        kind="K",
        sensitivity_tier="strict",
        default_bits=4,
        default_quant_scheme="affine",
        default_mode="M0",
        recent_window=0,
    )
    mode = choose_page_mode(0, "K", 256, observe_page(np.ones((8, 32), dtype=np.float32)), layer_policy=policy)
    assert mode.mode == "M0"
    assert mode.bits == 4


def test_choose_page_mode_falls_back_deterministically() -> None:
    policy = LayerPolicy(
        policy_id="test",
        sensitivity_tier="balanced",
        kind="V",
        recent_window=0,
        outlier_fraction_threshold=0.0,
        abs_max_threshold=1.0,
        channel_range_threshold=0.1,
        candidates=(
            PageModeSpec(mode="M0", bits=2, quant_scheme="affine"),
            PageModeSpec(mode="M1", bits=4, quant_scheme="lut"),
            PageModeSpec(mode="M0", bits=4, quant_scheme="affine"),
        ),
    )
    values = np.linspace(-10.0, 10.0, num=8 * 32, dtype=np.float32).reshape(8, 32)
    mode = choose_page_mode(0, "V", 256, observe_page(values), layer_policy=policy)
    assert mode.mode == "M0"
    assert mode.bits == 4
    assert "m0_stats" in mode.fallback_reason or "m1_stats" in mode.fallback_reason
