from dotcache.planner import choose_mode


def test_recent_tokens_choose_escape_mode() -> None:
    assert choose_mode(0, 0, token_age=12, recent_window=128) == "M3"


def test_old_tokens_default_to_m0() -> None:
    assert choose_mode(0, 0, token_age=256, recent_window=128) == "M0"


def test_error_threshold_can_force_escape_mode() -> None:
    stats = {"quant_error": 0.12}
    assert choose_mode(0, 0, token_age=256, stats=stats, recent_window=128, error_threshold=0.05) == "M3"

