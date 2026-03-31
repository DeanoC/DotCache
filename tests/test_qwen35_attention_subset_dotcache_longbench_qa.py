from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_attention_subset_dotcache_longbench_qa import (
    normalize_answer,
    qa_f1_score,
    score_longbench_answers,
)


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
