from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_llama_task_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("run_llama_task_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_instruction_strips_trailing_chat_artifact() -> None:
    result = MODULE._score_instruction("STATUS: READY\nCOLOR: BLUEuser", "STATUS: READY\nCOLOR: BLUE")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_text_cleaned"] == "STATUS: READY\nCOLOR: BLUE"


def test_score_instruction_tracks_answer_success_after_think_block_but_requires_exact_contract() -> None:
    generated = "<think>plan</think>\nSTATUS: READY\nCOLOR: BLUE\nDone."
    result = MODULE._score_instruction(generated, "STATUS: READY\nCOLOR: BLUE")
    assert result["task_answer_success"] is True
    assert result["task_metric_value"] == 0.0


def test_score_reasoning_extracts_final_integer_after_think_text() -> None:
    generated = "<think>\nThinking Process:\n1. Add 17 and 26 to get 43.\n2. Subtract 9 to get 34.\n3. Add 14 to get 48.\n</think>\nFINAL: 48"
    result = MODULE._score_reasoning(generated, "48")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_value"] == "48"


def test_score_reasoning_tracks_answer_success_before_prompt_echo_but_requires_contract() -> None:
    generated = "48\nCompute 17 + 26 - 9 + 14.\nFINAL:"
    result = MODULE._score_reasoning(generated, "48")
    assert result["task_answer_success"] is True
    assert result["task_metric_value"] == 0.0
    assert result["task_generated_value"] == "48"


def test_score_retrieval_tracks_answer_success_after_think_block_but_requires_exact_contract() -> None:
    generated = "<think>searching</think>\nRIVER-58142."
    result = MODULE._score_retrieval(generated, "RIVER-58142")
    assert result["task_answer_success"] is True
    assert result["task_metric_value"] == 0.0
    assert result["task_generated_text_cleaned"] == "RIVER-58142."


def test_dense_result_helpers_choose_dense_fields() -> None:
    record = {
        "dense_generated_ids": [1, 2, 3],
        "dotcache_generated_ids": [4, 5],
        "dense_text": "dense answer",
        "dotcache_text": "dotcache answer",
        "dense_decode_ms_per_step": 12.0,
        "decode_ms_per_step": 34.0,
    }

    assert MODULE._result_generated_ids(record, profile="dense") == [1, 2, 3]
    assert MODULE._result_generated_text(record, profile="dense") == "dense answer"
    assert MODULE._record_decode_ms_per_step(record, profile="dense") == 12.0
    assert MODULE._result_generated_ids(record, profile="quality") == [4, 5]
    assert MODULE._result_generated_text(record, profile="quality") == "dotcache answer"
    assert MODULE._record_decode_ms_per_step(record, profile="quality") == 34.0
