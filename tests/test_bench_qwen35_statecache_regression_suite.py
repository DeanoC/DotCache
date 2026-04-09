from __future__ import annotations

import pytest

from benchmarks import bench_qwen35_statecache_regression_suite as suite


def test_parse_case_accepts_prefix_eval_pair() -> None:
    assert suite._parse_case("128:16") == (128, 16)


def test_parse_case_rejects_invalid_value() -> None:
    with pytest.raises(Exception):
        suite._parse_case("128")


def test_regression_suite_parse_args_supports_scope_and_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_statecache_regression_suite.py",
            "--cases",
            "64:8",
            "128:16",
            "--statecache-scopes",
            "recurrent_only",
            "conv_plus_recurrent",
            "--localization-scopes",
            "conv_plus_recurrent",
            "--conv-bits",
            "4",
            "--conv-layer-bit-overrides",
            "0:8",
            "--conv-mode-override",
            "0:M3",
        ],
    )
    args = suite.parse_args()
    assert args.cases == ["64:8", "128:16"]
    assert args.statecache_scopes == ["recurrent_only", "conv_plus_recurrent"]
    assert args.localization_scopes == ["conv_plus_recurrent"]
    assert args.conv_bits == 4
    assert args.conv_layer_bit_overrides == ["0:8"]
    assert args.conv_mode_override == ["0:M3"]


def test_statecache_loss_summary_exposes_key_bytes() -> None:
    summary = suite._statecache_loss_summary(
        {
            "deltanet_statecache_scope": "conv_plus_recurrent",
            "deltanet_statecache_teacher_forced_loss": 0.1,
            "deltanet_statecache_teacher_forced_perplexity": 1.1,
            "deltanet_statecache_teacher_forced_target_match_rate": 1.0,
            "teacher_forced_loss_delta": 0.01,
            "teacher_forced_perplexity_ratio": 1.01,
            "deltanet_statecache_decode_ms_per_step": 12.5,
            "deltanet_conv_state_bytes": 100,
            "deltanet_recurrent_state_bytes": 200,
            "deltanet_statecache_conv_state_bytes": 80,
            "deltanet_statecache_recurrent_state_bytes": 120,
            "deltanet_statecache_fixed_resident_bytes": 200,
            "deltanet_statecache_effective_conv_compression_ratio": 1.25,
            "deltanet_statecache_effective_recurrent_compression_ratio": 1.66,
            "deltanet_statecache_effective_fixed_resident_compression_ratio": 1.5,
            "deltanet_statecache_recurrent_mode_overrides": {"0": "M3"},
            "deltanet_statecache_conv_mode_overrides": {"1": "M3"},
        }
    )
    assert summary["scope"] == "conv_plus_recurrent"
    assert summary["dense_conv_state_bytes"] == 100
    assert summary["statecache_recurrent_state_bytes"] == 120
    assert summary["recurrent_mode_overrides"] == {"0": "M3"}
    assert summary["conv_mode_overrides"] == {"1": "M3"}


def test_localization_error_keeps_loss_summary() -> None:
    mode_summary = {
        "status": "ok",
        "loss": {"scope": "conv_plus_recurrent", "teacher_forced_loss": 0.1},
    }

    updated = suite._record_statecache_localization_error(
        mode_summary=mode_summary,
        case_id="128:16",
        scope="conv_plus_recurrent",
        exc=RuntimeError("localization blew up"),
    )

    assert updated["loss"] == mode_summary["loss"]
    assert updated["localization"]["status"] == "error"
    assert updated["localization"]["stage"] == "localization"
    assert updated["localization"]["scope"] == "conv_plus_recurrent"
    assert updated["localization"]["case_id"] == "128:16"
