from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_attention_subset_dotcache_longbench_qa import (
    clean_longbench_generated_text,
    normalize_answer,
    qa_f1_score,
    score_longbench_answers,
)
from benchmarks.bench_qwen35_attention_subset_dotcache_needle import _run_case


def test_normalize_answer_removes_articles_and_punctuation() -> None:
    assert normalize_answer("The Miller v. California.") == "miller v california"


def test_qa_f1_score_matches_full_overlap() -> None:
    assert qa_f1_score("Vice Admiral", "Vice Admiral.") == 1.0


def test_score_longbench_answers_tracks_exact_and_best_f1() -> None:
    score = score_longbench_answers(
        "extension of the NetVLAD, adds Ghost clusters along with the NetVLAD clusters and more detail",
        [
            "extension of the NetVLAD, adds Ghost clusters along with the NetVLAD clusters",
            "something else",
        ],
    )
    assert score["longbench_answer_exact_match"] is False
    assert float(score["longbench_qa_f1_max"]) > 0.8
    assert score["longbench_best_matching_answer"] == "extension of the NetVLAD, adds Ghost clusters along with the NetVLAD clusters"


def test_clean_longbench_generated_text_removes_chat_artifacts() -> None:
    raw = "Miller v. California\n</think>\n\nMiller v. California.\nassistant\n<think>\n\n</think>\n\nMiller v. California."
    cleaned = clean_longbench_generated_text(raw)
    assert "assistant" not in cleaned
    assert "<think>" not in cleaned
    assert "Miller v. California" in cleaned


def test_run_case_uses_scorer_diagnostic_when_requested() -> None:
    class _FakeHarness:
        def __init__(self) -> None:
            self.called = None

        def run_attention_subset_dotcache_serving_scorer_diagnostic(self, **_: object) -> dict[str, object]:
            self.called = "scorer"
            return {"runtime_mode": "scorer"}

    harness = _FakeHarness()
    args = SimpleNamespace(
        scorer_diagnostic=True,
        recall_analysis=False,
        quality_check=False,
        max_new_tokens=4,
        profile_backend=False,
        trace_python_allocations=False,
    )

    result = _run_case(harness, input_ids=None, attention_mask=None, args=args)

    assert harness.called == "scorer"
    assert result["runtime_mode"] == "scorer"
