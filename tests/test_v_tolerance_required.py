"""Step-0 plumbing tests: v_tolerance is required at kernel + state level.

These guard against the regression where the Apr 17–24 paper benches
silently used v_tolerance=0.5 because the DOTCACHE_V_TOL env var was
referenced but never read by any kernel. After the port, every entry
point must either pass v_tolerance explicitly or fail loudly.

CPU-only — no CUDA needed.
"""

import pytest


def test_certified_attention_state_rejects_missing_v_tolerance():
    """CertifiedAttentionState raises ValueError when v_tolerance is omitted."""
    from dotcache.integrations.llama import CertifiedAttentionState
    with pytest.raises(ValueError, match="v_tolerance"):
        CertifiedAttentionState(tiered_caches={}, layer_epsilons={})


def test_certified_attention_state_accepts_explicit_v_tolerance():
    """Construction succeeds when v_tolerance is passed."""
    from dotcache.integrations.llama import CertifiedAttentionState
    s = CertifiedAttentionState(
        tiered_caches={}, layer_epsilons={}, v_tolerance=0.05,
    )
    assert s.v_tolerance == 0.05


def test_score_consistency_check_defaults_true_on_state():
    """Paper §7 specifies score_consistency_check enabled."""
    from dotcache.integrations.llama import CertifiedAttentionState
    s = CertifiedAttentionState(
        tiered_caches={}, layer_epsilons={}, v_tolerance=0.5,
    )
    assert s.score_consistency_check is True


def test_certified_attention_layer_signature_requires_v_tolerance():
    """The kernel function signature has v_tolerance as keyword-only with no default.

    This is a static check via inspect — calling the function with real
    tensors needs CUDA, but we can verify the signature shape on CPU.
    """
    import inspect
    from dotcache.kernels.certified_attention import certified_attention_layer
    sig = inspect.signature(certified_attention_layer)
    v_tol = sig.parameters.get("v_tolerance")
    assert v_tol is not None, "certified_attention_layer must declare v_tolerance"
    assert v_tol.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"v_tolerance must be keyword-only (got {v_tol.kind}). "
        "This guards against positional-call accidents."
    )
    assert v_tol.default is inspect.Parameter.empty, (
        "v_tolerance must have NO default value — silent defaulting to 0.5 "
        "is exactly the bug this guard prevents. See "
        "docs/paper_code_audit_20260424.md."
    )


def test_score_consistency_check_kernel_default_true():
    """Paper §7: kernel default for score_consistency_check is True."""
    import inspect
    from dotcache.kernels.certified_attention import certified_attention_layer
    sig = inspect.signature(certified_attention_layer)
    scc = sig.parameters.get("score_consistency_check")
    assert scc is not None
    assert scc.default is True, (
        f"score_consistency_check kernel default is {scc.default!r}; "
        "paper §7 requires True"
    )
