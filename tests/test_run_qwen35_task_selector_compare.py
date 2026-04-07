from __future__ import annotations

import importlib.util
from types import SimpleNamespace
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


def test_score_instruction_strips_unmatched_think_tag_artifacts() -> None:
    generated = "</think>\nSTATUS: READY\nCOLOR: BLUE"
    result = MODULE._score_instruction(generated, "STATUS: READY\nCOLOR: BLUE")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_text_cleaned"] == "STATUS: READY\nCOLOR: BLUE"


def test_score_retrieval_strips_unmatched_think_tag_artifacts() -> None:
    generated = "RIVER-58142\n</think>"
    result = MODULE._score_retrieval(generated, "RIVER-58142")
    assert result["task_metric_value"] == 1.0
    assert result["task_generated_text_cleaned"] == "RIVER-58142"


def test_apply_selector_task_context_sets_runtime_prompt_family_and_variant() -> None:
    config = SimpleNamespace(
        learned_page_selector_prompt_family="cache",
        learned_page_selector_prompt_variant="locality",
    )
    harness = SimpleNamespace(
        adapter=SimpleNamespace(
            dotcache_config=config,
            model_kv_cache=SimpleNamespace(config=config),
        )
    )

    MODULE._apply_selector_task_context(
        harness,
        profile="quality",
        task_family="reasoning",
        task_variant="arithmetic",
    )
    assert config.learned_page_selector_prompt_family == "reasoning"
    assert config.learned_page_selector_prompt_variant == "arithmetic"

    MODULE._apply_selector_task_context(
        harness,
        profile="exact",
        task_family="reasoning",
        task_variant="arithmetic",
    )
    assert config.learned_page_selector_prompt_family is None
    assert config.learned_page_selector_prompt_variant is None


def test_run_quality_case_forwards_stop_sequences() -> None:
    captured: dict[str, object] = {}

    class _FakeHarness:
        def run_attention_subset_dotcache_serving_quality(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    result = MODULE._run_quality_case(
        _FakeHarness(),
        input_ids="ids",
        attention_mask="mask",
        decode_steps=7,
        stop_sequences=("FINAL: 48",),
    )

    assert result == {"ok": True}
    assert captured["decode_steps"] == 7
    assert captured["stop_sequences"] == ("FINAL: 48",)


def test_run_quality_case_uses_dense_harness_generate_greedy() -> None:
    captured: dict[str, object] = {}

    class _FakeDenseHarness:
        def generate_greedy(self, **kwargs):
            captured.update(kwargs)
            return {"mode": "dense"}

    result = MODULE._run_quality_case(
        _FakeDenseHarness(),
        input_ids="ids",
        attention_mask="mask",
        decode_steps=9,
        stop_sequences=("RIVER-58142",),
    )

    assert result == {"mode": "dense"}
    assert captured["max_new_tokens"] == 9
    assert captured["stop_sequences"] == ("RIVER-58142",)
