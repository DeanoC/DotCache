from dotcache.execution_profiles import ExecutionProfile, resolve_execution_profile


def test_resolve_m4_envelope_auto_profile_at_4k() -> None:
    profile = resolve_execution_profile("m4_envelope_auto", context_length=4096)

    assert profile == ExecutionProfile(
        sink_window_tokens=256,
        recent_window_tokens=1024,
        relevance_top_k=4,
    )


def test_resolve_fixed_m4_envelope_profiles() -> None:
    fast = resolve_execution_profile("m4_envelope_fast", context_length=4096)
    balanced = resolve_execution_profile("m4_envelope_balanced", context_length=16384)

    assert fast == ExecutionProfile(
        sink_window_tokens=256,
        recent_window_tokens=1024,
        relevance_top_k=2,
    )
    assert balanced == ExecutionProfile(
        sink_window_tokens=256,
        recent_window_tokens=1024,
        relevance_top_k=4,
    )


def test_resolve_m4_envelope_auto_profile_scales_for_8k_and_16k() -> None:
    profile_8k = resolve_execution_profile("m4_envelope_auto", context_length=8192)
    profile_16k = resolve_execution_profile("m4_envelope_auto", context_length=16384)

    assert profile_8k == ExecutionProfile(
        sink_window_tokens=256,
        recent_window_tokens=2048,
        relevance_top_k=4,
    )
    assert profile_16k == ExecutionProfile(
        sink_window_tokens=256,
        recent_window_tokens=4096,
        relevance_top_k=8,
    )
