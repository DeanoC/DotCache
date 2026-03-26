from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    sink_window_tokens: int
    recent_window_tokens: int
    relevance_top_k: int
    relevance_mode: str = "envelope"
    relevance_sketch_size: int = 1
    exact_refine_top_k: int = 0
    approximate_old_pages: bool = False


def resolve_execution_profile(name: str | None, *, context_length: int) -> ExecutionProfile | None:
    if name is None or name == "none":
        return None
    if name == "m4_envelope_auto":
        if context_length <= 4_096:
            return ExecutionProfile(
                sink_window_tokens=256,
                recent_window_tokens=1_024,
                relevance_top_k=4,
            )
        if context_length <= 8_192:
            return ExecutionProfile(
                sink_window_tokens=256,
                recent_window_tokens=2_048,
                relevance_top_k=4,
            )
        return ExecutionProfile(
            sink_window_tokens=256,
            recent_window_tokens=4_096,
            relevance_top_k=8,
        )
    raise ValueError(f"unknown execution profile: {name}")
