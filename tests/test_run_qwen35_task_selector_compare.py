from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen35_task_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("run_qwen35_task_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_instruction_strips_trailing_chat_artifact() -> None:
    result = MODULE._score_instruction("STATUS: READY\nCOLOR: BLUEuser", "STATUS: READY\nCOLOR: BLUE")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_text_cleaned"] == "STATUS: READY\nCOLOR: BLUE"


def test_score_reasoning_extracts_final_integer_after_think_text() -> None:
    generated = "<think>\nThinking Process:\n1. Add 17 and 26 to get 43.\n2. Subtract 9 to get 34.\n3. Add 14 to get 48.\n</think>\nFINAL: 48"
    result = MODULE._score_reasoning(generated, "48")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_value"] == "48"
