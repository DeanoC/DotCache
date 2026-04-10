from __future__ import annotations

import json

from benchmarks.bench_qwen35_page_mode_compare import (
    DEFAULT_ATTENTION_PATH,
    DEFAULT_RUNTIME,
    WorkloadShape,
    _can_resume_existing_run,
    _run_config_signature,
    RunSpec,
    build_matrix,
    build_report,
    command_for_run,
    parse_page_mode_variant,
)


def test_parse_page_mode_variant_supports_exact_and_asymmetric_forms() -> None:
    exact = parse_page_mode_variant("exact")
    assert exact.label == "exact"
    assert exact.key_page_mode == "exact"
    assert exact.value_page_mode == "exact"

    symmetric = parse_page_mode_variant("M3/affine/4/int8")
    assert symmetric.label == "m3-affine-4-int8"
    assert symmetric.key_page_mode == "M3/affine/4/int8"
    assert symmetric.value_page_mode == "M3/affine/4/int8"

    asymmetric = parse_page_mode_variant("m2k=M2/affine/4|exact")
    assert asymmetric.label == "m2k"
    assert asymmetric.key_page_mode == "M2/affine/4"
    assert asymmetric.value_page_mode == "exact"


def test_build_matrix_locks_default_page_mode_compare_run_count() -> None:
    specs = build_matrix()
    assert len(specs) == 12

    bench_specs = [spec for spec in specs if spec.kind == "bench"]
    workload_specs = [spec for spec in specs if spec.kind == "workload"]
    assert len(bench_specs) == 8
    assert len(workload_specs) == 4
    assert {spec.prompt_token_count for spec in bench_specs} == {2048, 8192}
    assert {spec.prompt_token_count for spec in workload_specs} == {2048}
    assert {spec.resident_page_budget for spec in specs} == {32, 128}
    assert {spec.page_mode_variant.label for spec in specs} == {"exact", "m3-affine-4-int8"}


def test_command_for_workload_run_emits_page_mode_and_runtime_flags(tmp_path) -> None:
    workload_spec = build_matrix(
        model_id="Qwen/Qwen3.5-0.8B",
        dtype="f16",
        page_modes=("exact", "M3/affine/4/int8"),
        workload_shape=WorkloadShape(total_sessions=4, wave_size=2, decode_rounds_per_wave=1, max_new_tokens=4),
    )[-1]

    command = command_for_run(workload_spec, out_prefix=tmp_path / "run", cargo_features="candle-cuda")
    assert command[:7] == ["cargo", "run", "--features", "candle-cuda", "--example", "hf_workload_bench", "--"]
    assert DEFAULT_RUNTIME in command
    assert DEFAULT_ATTENTION_PATH in command
    assert "--resident-page-budget" in command
    assert "--default-key-page-mode" in command
    assert "M3/affine/4/int8" in command
    assert "--default-value-page-mode" in command
    assert "--shared-prompt-token-target" in command


def test_build_report_compares_variants_to_exact_baseline() -> None:
    records = [
        {
            "run_id": "exact",
            "model_id": "Qwen/Qwen3.5-0.8B",
            "kind": "bench",
            "prompt_token_count": 2048,
            "resident_page_budget": 32,
            "page_mode_label": "exact",
            "default_key_page_mode": "exact",
            "default_value_page_mode": "exact",
            "status": "completed",
            "workload_shape": None,
            "summary_metrics": {
                "total_millis": 1000.0,
                "total_tokens_per_second": 20.0,
                "spilled_bytes": 200,
            },
        },
        {
            "run_id": "m3",
            "model_id": "Qwen/Qwen3.5-0.8B",
            "kind": "bench",
            "prompt_token_count": 2048,
            "resident_page_budget": 32,
            "page_mode_label": "m3-affine-4-int8",
            "default_key_page_mode": "M3/affine/4/int8",
            "default_value_page_mode": "M3/affine/4/int8",
            "status": "completed",
            "workload_shape": None,
            "summary_metrics": {
                "total_millis": 900.0,
                "total_tokens_per_second": 25.0,
                "spilled_bytes": 100,
            },
        },
    ]

    report = build_report(records)
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["exact_baseline"]["run_id"] == "exact"
    row = group["variants"][0]
    assert row["delta_total_millis_vs_exact"] == -100.0
    assert row["total_millis_ratio_vs_exact"] == 0.9
    assert row["delta_total_tokens_per_second_vs_exact"] == 5.0
    assert row["spilled_bytes_ratio_vs_exact"] == 0.5
    assert group["best_variant_by_total_millis"]["run_id"] == "m3"


def test_resume_requires_matching_run_configuration(tmp_path) -> None:
    spec = RunSpec(
        model_id="Qwen/Qwen3.5-0.8B",
        kind="bench",
        prompt_token_count=2048,
        resident_page_budget=32,
        dtype="f16",
        device="cuda:0",
        warmup_runs=0,
        max_new_tokens=4,
        prompt_text="Cache locality matters for fast decoding.",
        page_mode_variant=parse_page_mode_variant("M3/affine/4/int8"),
        workload_shape=None,
    )
    signature = _run_config_signature(spec, cargo_features="candle-cuda")
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(signature), encoding="utf-8")
    assert _can_resume_existing_run(config_path, signature)

    changed = _run_config_signature(
        RunSpec(**{**spec.__dict__, "resident_page_budget": 128}),
        cargo_features="candle-cuda",
    )
    assert not _can_resume_existing_run(config_path, changed)
