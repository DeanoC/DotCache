from __future__ import annotations

from benchmarks.bench_qwen35_persistent_exact_key_live_policy_compare import (
    _policy_layer15_code_or_len_ge_1800_024,
    _policy_layer15_len_ge_1800_024,
    _render_markdown,
    _summarize_policy,
)


def test_length_policy_uses_override_only_for_long_prompts() -> None:
    assert _policy_layer15_len_ge_1800_024({"prompt_length": 1792}) is None
    assert _policy_layer15_len_ge_1800_024({"prompt_length": 1800}) == {15: 0.24}


def test_code_or_length_policy_uses_override_for_python_files() -> None:
    assert _policy_layer15_code_or_len_ge_1800_024({"prompt_length": 1024, "prompt_file_path": "foo.py"}) == {15: 0.24}
    assert _policy_layer15_code_or_len_ge_1800_024({"prompt_length": 1024, "prompt_file_path": "foo.md"}) is None


def test_summarize_policy_aggregates_manifest_means_and_exact_match() -> None:
    rows = [
        {
            "manifest_path": "a.json",
            "case_tag": "one",
            "decode_ms_per_step": 10.0,
            "generated_ids": [1, 2],
        },
        {
            "manifest_path": "a.json",
            "case_tag": "two",
            "decode_ms_per_step": 14.0,
            "generated_ids": [3, 4],
        },
        {
            "manifest_path": "b.json",
            "case_tag": "three",
            "decode_ms_per_step": 6.0,
            "generated_ids": [5, 6],
        },
    ]
    summary = _summarize_policy(
        name="test",
        description="desc",
        rows=rows,
        baseline_by_manifest_case={
            ("a.json", "one"): [1, 2],
            ("a.json", "two"): [0, 0],
            ("b.json", "three"): [5, 6],
        },
    )

    assert summary["overall_avg_ms_per_step"] == 10.0
    assert summary["exact_match_rate_vs_baseline"] == 2 / 3
    assert summary["per_manifest_avg_ms_per_step"] == {"a.json": 12.0, "b.json": 6.0}


def test_render_markdown_lists_ranked_policies() -> None:
    text = _render_markdown(
        {
            "policies": [
                {
                    "name": "baseline",
                    "description": "desc",
                    "overall_avg_ms_per_step": 12.34,
                    "exact_match_rate_vs_baseline": 1.0,
                    "per_manifest_avg_ms_per_step": {"a.json": 12.34},
                }
            ]
        }
    )

    assert "baseline" in text
    assert "12.3400" in text
    assert "1.000" in text
