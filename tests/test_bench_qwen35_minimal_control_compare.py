from __future__ import annotations

import json

from benchmarks.bench_qwen35_minimal_control_compare import (
    DEFAULT_MAIN_RUNTIME,
    DEFAULT_MODEL,
    RunSpec,
    _can_resume_existing_run,
    env_for_run,
    _run_config_signature,
    build_matrix,
    build_report,
    command_for_run,
)


def test_build_matrix_locks_default_compare_run_count() -> None:
    specs = build_matrix()
    assert len(specs) == 6
    assert {spec.lane for spec in specs} == {
        "minimal_control",
        "minimal_megakernel",
        "main_dense_control",
    }
    assert {spec.prompt_token_count for spec in specs} == {2048, 8192}


def test_build_matrix_adds_luce_lane_when_requested() -> None:
    specs = build_matrix(luce_repo="/tmp/luce-megakernel")
    assert len(specs) == 8
    assert "luce_external_megakernel" in {spec.lane for spec in specs}


def test_command_for_run_emits_lane_specific_examples(tmp_path) -> None:
    specs = build_matrix(model_id=DEFAULT_MODEL, contexts=(2048,), max_new_tokens=4)
    minimal = next(spec for spec in specs if spec.lane == "minimal_control")
    megakernel = next(spec for spec in specs if spec.lane == "minimal_megakernel")
    main = next(spec for spec in specs if spec.lane == "main_dense_control")

    minimal_command = command_for_run(
        minimal,
        out_prefix=tmp_path / "minimal",
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
        luce_repo=None,
    )
    assert minimal_command[0].endswith("cargo")
    assert minimal_command[1:7] == [
        "run",
        "--features",
        "qwen35-minimal-cuda",
        "--example",
        "hf_qwen35_minimal_bench",
        "--",
    ]
    assert "--prompt-token-target" in minimal_command
    assert env_for_run(minimal) == {}

    assert env_for_run(megakernel) == {"CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL": "1"}

    main_command = command_for_run(
        main,
        out_prefix=tmp_path / "main",
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
        luce_repo=None,
    )
    assert main_command[0].endswith("cargo")
    assert main_command[1:7] == [
        "run",
        "--features",
        "candle-cuda",
        "--example",
        "hf_bench",
        "--",
    ]
    assert DEFAULT_MAIN_RUNTIME in main_command
    assert "--sync-stage-profile" in main_command


def test_command_for_run_emits_luce_external_wrapper(tmp_path) -> None:
    spec = RunSpec(
        model_id=DEFAULT_MODEL,
        prompt_token_count=2048,
        lane="luce_external_megakernel",
        device="cuda:0",
        warmup_runs=0,
        max_new_tokens=4,
        prompt_text="Cache locality matters for fast decoding.",
        dtype="f16",
    )
    command = command_for_run(
        spec,
        out_prefix=tmp_path / "luce",
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
        luce_repo="/tmp/luce-megakernel",
    )
    assert command[0].endswith("python") or command[0].endswith("python3")
    assert "bench_qwen35_luce_external.py" in command[1]
    assert "--luce-repo" in command


def test_build_report_compares_minimal_to_main() -> None:
    records = [
        {
            "run_id": "main",
            "model_id": DEFAULT_MODEL,
            "prompt_token_count": 2048,
            "lane": "main_dense_control",
            "status": "completed",
            "summary_metrics": {
                "prefill_millis": 1000.0,
                "decode_millis": 500.0,
                "total_millis": 1500.0,
                "total_tokens_per_second": 40.0,
            },
        },
        {
            "run_id": "minimal",
            "model_id": DEFAULT_MODEL,
            "prompt_token_count": 2048,
            "lane": "minimal_control",
            "status": "completed",
            "summary_metrics": {
                "prefill_millis": 900.0,
                "decode_millis": 600.0,
                "total_millis": 1500.0,
                "total_tokens_per_second": 44.0,
            },
        },
        {
            "run_id": "megakernel",
            "model_id": DEFAULT_MODEL,
            "prompt_token_count": 2048,
            "lane": "minimal_megakernel",
            "status": "completed",
            "summary_metrics": {
                "prefill_millis": 850.0,
                "decode_millis": 550.0,
                "total_millis": 1400.0,
                "total_tokens_per_second": 46.0,
                "full_prefill_megakernel_requested": True,
            },
        },
        {
            "run_id": "luce",
            "model_id": DEFAULT_MODEL,
            "prompt_token_count": 2048,
            "lane": "luce_external_megakernel",
            "status": "completed",
            "summary_metrics": {
                "prefill_millis": 800.0,
                "decode_millis": 400.0,
                "total_millis": 1200.0,
                "total_tokens_per_second": 50.0,
            },
        },
    ]

    report = build_report(records)
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["main_dense_control"]["run_id"] == "main"
    assert group["minimal_control"]["run_id"] == "minimal"
    assert group["minimal_megakernel"]["run_id"] == "megakernel"
    assert group["luce_external_megakernel"]["run_id"] == "luce"
    assert group["comparisons"]["minimal_control_vs_main"]["delta_prefill_millis"] == -100.0
    assert group["comparisons"]["minimal_control_vs_main"]["decode_millis_ratio"] == 1.2
    assert (
        group["comparisons"]["minimal_control_vs_main"]["total_tokens_per_second_ratio"] == 1.1
    )
    assert group["comparisons"]["minimal_megakernel_vs_main"]["delta_total_millis"] == -100.0
    assert group["comparisons"]["minimal_megakernel_vs_minimal_control"]["delta_total_millis"] == -100.0
    assert group["comparisons"]["luce_external_megakernel_vs_main"]["delta_total_millis"] == -300.0


def test_build_report_tolerates_partial_groups() -> None:
    report = build_report(
        [
            {
                "run_id": "main",
                "model_id": DEFAULT_MODEL,
                "prompt_token_count": 64,
                "lane": "main_dense_control",
                "status": "completed",
                "summary_metrics": {
                    "prefill_millis": 100.0,
                    "decode_millis": 20.0,
                    "total_millis": 120.0,
                    "total_tokens_per_second": 10.0,
                },
            }
        ]
    )
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["comparisons"]["minimal_control_vs_main"]["delta_total_millis"] is None


def test_resume_requires_matching_run_configuration(tmp_path) -> None:
    spec = RunSpec(
        model_id=DEFAULT_MODEL,
        prompt_token_count=2048,
        lane="minimal_control",
        device="cuda:0",
        warmup_runs=0,
        max_new_tokens=4,
        prompt_text="Cache locality matters for fast decoding.",
        dtype="f16",
    )
    signature = _run_config_signature(
        spec,
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
        luce_repo=None,
    )
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(signature), encoding="utf-8")
    assert _can_resume_existing_run(config_path, signature)

    changed = _run_config_signature(
        RunSpec(**{**spec.__dict__, "max_new_tokens": 8}),
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
        luce_repo=None,
    )
    assert not _can_resume_existing_run(config_path, changed)
