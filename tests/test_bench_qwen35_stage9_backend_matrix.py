from __future__ import annotations

from benchmarks.bench_qwen35_stage9_backend_matrix import _render_markdown, _winner_rows


def test_winner_rows_selects_lowest_bias_per_backend_and_corpus() -> None:
    rows = [
        {"corpus": "large", "backend": "mps", "policy": "real_mixed", "bias_ms_per_step": 10.0},
        {"corpus": "large", "backend": "mps", "policy": "non_m0", "bias_ms_per_step": 12.0},
        {"corpus": "large", "backend": "mps", "policy": "conservative", "bias_ms_per_step": 15.0},
        {"corpus": "large", "backend": "cuda", "policy": "real_mixed", "bias_ms_per_step": 6.0},
        {"corpus": "large", "backend": "cuda", "policy": "non_m0", "bias_ms_per_step": 5.0},
        {"corpus": "large", "backend": "cuda", "policy": "conservative", "bias_ms_per_step": 7.0},
        {"corpus": "broad", "backend": "mps", "policy": "real_mixed", "bias_ms_per_step": 11.0},
        {"corpus": "broad", "backend": "mps", "policy": "non_m0", "bias_ms_per_step": 13.0},
        {"corpus": "broad", "backend": "mps", "policy": "conservative", "bias_ms_per_step": 14.0},
        {"corpus": "broad", "backend": "cuda", "policy": "real_mixed", "bias_ms_per_step": 8.0},
        {"corpus": "broad", "backend": "cuda", "policy": "non_m0", "bias_ms_per_step": 9.0},
        {"corpus": "broad", "backend": "cuda", "policy": "conservative", "bias_ms_per_step": 10.0},
        {"corpus": "external", "backend": "mps", "policy": "real_mixed", "bias_ms_per_step": 4.0},
        {"corpus": "external", "backend": "mps", "policy": "non_m0", "bias_ms_per_step": 5.0},
        {"corpus": "external", "backend": "mps", "policy": "conservative", "bias_ms_per_step": 6.0},
        {"corpus": "external", "backend": "cuda", "policy": "real_mixed", "bias_ms_per_step": 3.0},
        {"corpus": "external", "backend": "cuda", "policy": "non_m0", "bias_ms_per_step": 2.0},
        {"corpus": "external", "backend": "cuda", "policy": "conservative", "bias_ms_per_step": 4.0},
    ]

    winners = _winner_rows(rows)

    assert winners == [
        {"corpus": "large", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 10.0},
        {"corpus": "large", "backend": "cuda", "winner_policy": "non_m0", "winner_bias_ms_per_step": 5.0},
        {"corpus": "broad", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 11.0},
        {"corpus": "broad", "backend": "cuda", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 8.0},
        {"corpus": "external", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 4.0},
        {"corpus": "external", "backend": "cuda", "winner_policy": "non_m0", "winner_bias_ms_per_step": 2.0},
    ]


def test_render_markdown_includes_winner_table_and_matrix_rows() -> None:
    payload = {
        "winner_rows": [
            {"corpus": "large", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 10.0},
            {"corpus": "large", "backend": "cuda", "winner_policy": "non_m0", "winner_bias_ms_per_step": 5.0},
            {"corpus": "broad", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 11.0},
            {"corpus": "broad", "backend": "cuda", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 8.0},
            {"corpus": "external", "backend": "mps", "winner_policy": "real_mixed", "winner_bias_ms_per_step": 4.0},
            {"corpus": "external", "backend": "cuda", "winner_policy": "non_m0", "winner_bias_ms_per_step": 2.0},
        ],
        "rows": [
            {
                "corpus": "large",
                "backend": "mps",
                "policy": "real_mixed",
                "bias_ms_per_step": 10.0,
                "hand_ms_per_step": 20.0,
                "bias_vs_hand_exact_match_rate": 1.0,
                "bias_beats_hand_tuned_latency_rate": 1.0,
            }
        ],
    }

    markdown = _render_markdown(payload)

    assert "| large | real_mixed `10.00` | non_m0 `5.00` |" in markdown
    assert "| large | mps | real_mixed | `10.00` | `20.00` | `1.000` | `1.000` |" in markdown
