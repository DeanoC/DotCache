from __future__ import annotations


def choose_mode(
    layer: int,
    head: int,
    token_age: int,
    stats: dict[str, float | bool] | None = None,
    *,
    recent_window: int = 128,
    error_threshold: float | None = None,
) -> str:
    del layer
    del head

    if token_age < recent_window:
        return "M3"

    if stats is None:
        return "M0"

    if bool(stats.get("force_escape", False)):
        return "M3"

    quant_error = float(stats.get("quant_error", 0.0))
    if error_threshold is not None and quant_error > error_threshold:
        return "M3"

    return "M0"

