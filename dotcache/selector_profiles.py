from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectorProfileResolution:
    profile: str
    target_candidate: str
    logit_offset: float


def resolve_learned_page_selector_profile(
    *,
    profile: str,
    model_id: str | None,
    target_candidate: str,
    logit_offset: float,
) -> SelectorProfileResolution:
    profile_name = str(profile or "quality")
    if profile_name == "manual":
        return SelectorProfileResolution(
            profile="manual",
            target_candidate=str(target_candidate),
            logit_offset=float(logit_offset),
        )

    resolved_target = "M3/affine/4/float16"
    resolved_offset = 0.0
    model_token = str(model_id or "").lower()

    if profile_name == "systems":
        if "qwen3.5" in model_token or "qwen/qwen3.5" in model_token:
            resolved_offset = 2.0
    elif profile_name != "quality":
        raise ValueError("profile must be one of: quality, systems, manual")

    return SelectorProfileResolution(
        profile=profile_name,
        target_candidate=resolved_target,
        logit_offset=float(resolved_offset),
    )
