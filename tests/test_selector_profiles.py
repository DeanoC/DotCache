from dotcache.selector_profiles import resolve_learned_page_selector_profile


def test_quality_profile_keeps_unbiased_defaults() -> None:
    resolved = resolve_learned_page_selector_profile(
        profile="quality",
        model_id="Qwen/Qwen3.5-9B",
        target_candidate="M0/affine/4",
        logit_offset=7.0,
    )

    assert resolved.profile == "quality"
    assert resolved.target_candidate == "M3/affine/4/float16"
    assert resolved.logit_offset == 0.0


def test_systems_profile_biases_qwen_but_not_llama() -> None:
    qwen = resolve_learned_page_selector_profile(
        profile="systems",
        model_id="Qwen/Qwen3.5-9B",
        target_candidate="M0/affine/4",
        logit_offset=0.0,
    )
    llama = resolve_learned_page_selector_profile(
        profile="systems",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        target_candidate="M0/affine/4",
        logit_offset=0.0,
    )

    assert qwen.target_candidate == "M3/affine/4/float16"
    assert qwen.logit_offset == 2.0
    assert llama.target_candidate == "M3/affine/4/float16"
    assert llama.logit_offset == 0.0


def test_manual_profile_preserves_explicit_target_and_offset() -> None:
    resolved = resolve_learned_page_selector_profile(
        profile="manual",
        model_id="Qwen/Qwen3.5-0.8B",
        target_candidate="M0/affine/4",
        logit_offset=-1.5,
    )

    assert resolved.profile == "manual"
    assert resolved.target_candidate == "M0/affine/4"
    assert resolved.logit_offset == -1.5
