from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_attention_subset_dotcache_needle import (
    build_needle_prompt_inputs,
    normalize_needle_text,
    score_needle_answer,
)


class FakeTokenizer:
    bos_token_id = 1

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {self.bos_token_id: "<bos>"}
        self._next_id = 2

    def _encode_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = [token for token in text.split() if token]
        return {"input_ids": [self._encode_token(token) for token in tokens]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id == self.bos_token_id:
                continue
            tokens.append(self._id_to_token[token_id])
        return " ".join(tokens)


def test_build_needle_prompt_inputs_hits_exact_length() -> None:
    tokenizer = FakeTokenizer()
    build = build_needle_prompt_inputs(
        tokenizer,
        device=torch.device("cpu"),
        prompt_length=40,
        needle_position_fraction=0.5,
        haystack_unit="filler words only",
        needle_key="secret code",
        needle_value="amber-42",
        needle_template="The {needle_key} is {needle_value}.",
        question_template="What is the {needle_key}? Answer:",
    )

    assert build.input_ids.shape == (1, 40)
    assert build.attention_mask.shape == (1, 40)
    assert build.needle_token_start < build.needle_token_end < build.question_token_start
    assert build.filler_before_tokens + build.filler_after_tokens > 0
    assert 0.0 <= build.needle_position_fraction_actual <= 1.0


def test_normalize_needle_text_collapses_spacing_and_punctuation() -> None:
    assert normalize_needle_text("  Crimson-Velvet-472. \n") == "crimson-velvet-472"


def test_score_needle_answer_accepts_exact_and_prefix_matches() -> None:
    exact = score_needle_answer("crimson-velvet-472\n", "crimson-velvet-472")
    assert exact["needle_answer_exact_match"] is True
    assert exact["needle_answer_correct"] is True

    prefix = score_needle_answer("crimson-velvet-472 is the passphrase", "crimson-velvet-472")
    assert prefix["needle_answer_exact_match"] is False
    assert prefix["needle_answer_prefix_match"] is True
    assert prefix["needle_answer_correct"] is True
