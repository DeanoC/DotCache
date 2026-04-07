from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen35_longbench_pack.py"
SPEC = importlib.util.spec_from_file_location("run_qwen35_longbench_pack", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_output_paths_adds_shard_suffix_for_parallel_runs(tmp_path: Path) -> None:
    paths = MODULE.build_output_paths(
        output_dir=tmp_path,
        model_id="Qwen/Qwen3.5-9B",
        pack="original_suite",
        prompt_shard_count=4,
        prompt_shard_index=2,
    )

    assert paths["jsonl"].name == "qwen3p5-9b_longbench_original_suite.shard02-of-04.jsonl"
    assert paths["markdown"].name == "longbench_selector_compare.shard02-of-04.md"
    assert paths["workbook_json"].name == "longbench_failure_workbook.shard02-of-04.json"


def test_build_output_paths_keeps_unsharded_names_for_single_run(tmp_path: Path) -> None:
    paths = MODULE.build_output_paths(
        output_dir=tmp_path,
        model_id="Qwen/Qwen3.5-9B",
        pack="original_suite",
        prompt_shard_count=1,
        prompt_shard_index=0,
    )

    assert paths["jsonl"].name == "qwen3p5-9b_longbench_original_suite.jsonl"
    assert paths["markdown"].name == "longbench_selector_compare.md"


def test_main_propagates_no_quality_check_to_compare_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def _fake_run(command: list[str], **_: object) -> None:
        commands.append(list(command))

    monkeypatch.setattr(MODULE.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        MODULE,
        "parse_args",
        lambda: MODULE.argparse.Namespace(
            model_id="Qwen/Qwen3.5-9B",
            backend="torch_cuda",
            device="cuda",
            torch_dtype="float16",
            selector_artifact="/tmp/selector.json",
            pack="stratified_16",
            prompt_pack=None,
            comparison_cases=None,
            comparison_case_preset="paper_headline",
            prompt_shard_count=1,
            prompt_shard_index=0,
            max_prompt_tokens=[4096],
            warmup_runs=0,
            measured_runs=1,
            timeout_seconds=2400,
            profile_backend=False,
            trace_python_allocations=False,
            quality_check=False,
            skip_report=True,
            output_dir=str(tmp_path),
        ),
    )

    result = MODULE.main()

    assert result == 0
    assert len(commands) == 1
    assert "--no-quality-check" in commands[0]
    assert "--quality-check" not in commands[0]


def test_resolve_pack_label_uses_manifest_name_and_phase(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lb21.json"
    manifest_path.write_text(
        '{"manifest_name": "LB21-16", "phase": "smoke", "prompt_specs": []}\n',
        encoding="utf-8",
    )

    args = MODULE.argparse.Namespace(
        pack="original_suite",
        prompt_pack=str(manifest_path),
    )

    assert MODULE.resolve_pack_label(args) == "lb21_16_smoke"
