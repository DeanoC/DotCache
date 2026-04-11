from __future__ import annotations

import json

from benchmarks.bench_qwen35_minimal_control_compare import (
    DEFAULT_MAIN_RUNTIME,
    DEFAULT_MODEL,
    RunSpec,
    _can_resume_existing_run,
    _run_config_signature,
    build_matrix,
    build_report,
    command_for_run,
)


def test_build_matrix_locks_default_compare_run_count() -> None:
    specs = build_matrix()
    assert len(specs) == 4
    assert {spec.lane for spec in specs} == {"minimal_control", "main_dense_control"}
    assert {spec.prompt_token_count for spec in specs} == {2048, 8192}


def test_command_for_run_emits_lane_specific_examples(tmp_path) -> None:
    specs = build_matrix(model_id=DEFAULT_MODEL, contexts=(2048,), max_new_tokens=4)
    minimal = next(spec for spec in specs if spec.lane == "minimal_control")
    main = next(spec for spec in specs if spec.lane == "main_dense_control")

    minimal_command = command_for_run(
        minimal,
        out_prefix=tmp_path / "minimal",
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
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

    main_command = command_for_run(
        main,
        out_prefix=tmp_path / "main",
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
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
    ]

    report = build_report(records)
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["main_dense_control"]["run_id"] == "main"
    assert group["minimal_control"]["run_id"] == "minimal"
    assert group["comparison"]["delta_prefill_millis_vs_main"] == -100.0
    assert group["comparison"]["decode_millis_ratio_vs_main"] == 1.2
    assert group["comparison"]["total_tokens_per_second_ratio_vs_main"] == 1.1


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
    )
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(signature), encoding="utf-8")
    assert _can_resume_existing_run(config_path, signature)

    changed = _run_config_signature(
        RunSpec(**{**spec.__dict__, "max_new_tokens": 8}),
        minimal_cargo_features="qwen35-minimal-cuda",
        main_cargo_features="candle-cuda",
    )
    assert not _can_resume_existing_run(config_path, changed)
