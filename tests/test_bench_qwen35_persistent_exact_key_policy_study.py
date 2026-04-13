from __future__ import annotations

from benchmarks.bench_qwen35_persistent_exact_key_policy_study import (
    _policy_length_le_1536_else_024,
    _render_markdown,
    _score_policy,
)


def test_score_policy_summarizes_overall_and_per_corpus() -> None:
    rows = [
        {
            "corpus": "external",
            "prompt_length": 1024,
            "baseline_ms_per_step": 10.0,
            "threshold_ms_per_step": {0.2: 9.0, 0.24: 11.0},
        },
        {
            "corpus": "large",
            "prompt_length": 2048,
            "baseline_ms_per_step": 20.0,
            "threshold_ms_per_step": {0.2: 18.0, 0.24: 19.0},
        },
    ]

    summary = _score_policy(
        rows,
        name="test",
        description="desc",
        chooser=lambda row: "0.20" if int(row["prompt_length"]) <= 1536 else "baseline",
    )

    assert summary["overall_avg_ms_per_step"] == 14.5
    assert summary["per_corpus_avg_ms_per_step"] == {"external": 9.0, "large": 20.0}
    assert summary["chosen_count_by_policy_value"] == {"0.20": 1, "baseline": 1}


def test_length_policy_switches_after_1536() -> None:
    assert _policy_length_le_1536_else_024({"prompt_length": 1536}) == "0.20"
    assert _policy_length_le_1536_else_024({"prompt_length": 2048}) == "0.24"


def test_render_markdown_includes_ranked_policy_summary() -> None:
    markdown = _render_markdown(
        payload={
            "policies": [
                {
                    "name": "layer15_always_020",
                    "description": "Always set layer 15 to 0.20.",
                    "overall_avg_ms_per_step": 12.34,
                    "per_corpus_avg_ms_per_step": {"external": 10.0},
                    "chosen_count_by_policy_value": {"0.20": 3},
                }
            ]
        }
    )

    assert "layer15_always_020" in markdown
    assert "12.3400" in markdown
