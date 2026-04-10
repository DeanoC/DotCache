from __future__ import annotations

from benchmarks.bench_qwen35_paged_dense_matrix import (
    DEFAULT_ATTENTION_PATH,
    DEFAULT_RUNTIME_DENSE,
    DEFAULT_RUNTIME_PAGED,
    WorkloadShape,
    build_matrix,
    build_report,
    command_for_run,
)


def test_build_matrix_locks_default_run_count_and_contexts() -> None:
    specs = build_matrix()
    assert len(specs) == 24

    bench_specs = [spec for spec in specs if spec.kind == "bench"]
    workload_specs = [spec for spec in specs if spec.kind == "workload"]

    assert len(bench_specs) == 16
    assert len(workload_specs) == 8
    assert {spec.model_id for spec in specs} == {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"}
    assert {spec.prompt_token_count for spec in bench_specs} == {8192, 32768}
    assert {spec.prompt_token_count for spec in workload_specs} == {8192}

    dense_specs = [spec for spec in specs if spec.runtime_mode == DEFAULT_RUNTIME_DENSE]
    paged_specs = [spec for spec in specs if spec.runtime_mode == DEFAULT_RUNTIME_PAGED]
    assert len(dense_specs) == 6
    assert len(paged_specs) == 18
    assert all(spec.attention_path is None for spec in dense_specs)
    assert {spec.attention_path for spec in paged_specs} == {DEFAULT_ATTENTION_PATH}
    assert {spec.resident_page_budget for spec in paged_specs} == {32, 128, 512}


def test_command_for_workload_run_emits_expected_runtime_flags(tmp_path) -> None:
    workload_spec = build_matrix(
        models=("Qwen/Qwen3.5-9B",),
        dtype="bf16",
        workload_shape=WorkloadShape(total_sessions=4, wave_size=2, decode_rounds_per_wave=1, max_new_tokens=4),
    )[-1]

    command = command_for_run(
        workload_spec,
        out_prefix=tmp_path / "run",
        cargo_features="candle-cuda",
    )

    assert command[:7] == ["cargo", "run", "--features", "candle-cuda", "--example", "hf_workload_bench", "--"]
    assert "--device" in command
    assert "cuda:0" in command
    assert "--dtype" in command
    assert "bf16" in command
    assert "--runtime-mode" in command
    assert DEFAULT_RUNTIME_PAGED in command
    assert "--attention-path" in command
    assert DEFAULT_ATTENTION_PATH in command
    assert "--resident-page-budget" in command
    assert "512" in command
    assert "--shared-prompt-token-target" in command
    assert "8192" in command


def test_build_report_computes_dense_deltas_and_best_paged_variants() -> None:
    records = [
        {
            "run_id": "dense",
            "model_id": "Qwen/Qwen3.5-9B",
            "kind": "bench",
            "prompt_token_count": 8192,
            "runtime_mode": DEFAULT_RUNTIME_DENSE,
            "attention_path": None,
            "resident_page_budget": None,
            "dtype": "bf16",
            "status": "completed",
            "workload_shape": None,
            "summary_metrics": {"total_millis": 1000.0, "total_tokens_per_second": 20.0},
        },
        {
            "run_id": "paged-32",
            "model_id": "Qwen/Qwen3.5-9B",
            "kind": "bench",
            "prompt_token_count": 8192,
            "runtime_mode": DEFAULT_RUNTIME_PAGED,
            "attention_path": DEFAULT_ATTENTION_PATH,
            "resident_page_budget": 32,
            "dtype": "bf16",
            "status": "completed",
            "workload_shape": None,
            "summary_metrics": {"total_millis": 900.0, "total_tokens_per_second": 22.0},
        },
        {
            "run_id": "paged-128",
            "model_id": "Qwen/Qwen3.5-9B",
            "kind": "bench",
            "prompt_token_count": 8192,
            "runtime_mode": DEFAULT_RUNTIME_PAGED,
            "attention_path": DEFAULT_ATTENTION_PATH,
            "resident_page_budget": 128,
            "dtype": "bf16",
            "status": "completed",
            "workload_shape": None,
            "summary_metrics": {"total_millis": 1100.0, "total_tokens_per_second": 25.0},
        },
    ]

    report = build_report(records)
    assert report["group_count"] == 1
    group = report["groups"][0]

    assert group["dense_baseline"]["run_id"] == "dense"
    assert group["best_paged_by_total_millis"]["run_id"] == "paged-32"
    assert group["best_paged_by_total_tokens_per_second"]["run_id"] == "paged-128"

    paged_rows = {row["run_id"]: row for row in group["paged_variants"]}
    assert paged_rows["paged-32"]["delta_total_millis_vs_dense"] == -100.0
    assert paged_rows["paged-32"]["total_millis_ratio_vs_dense"] == 0.9
    assert paged_rows["paged-32"]["delta_total_tokens_per_second_vs_dense"] == 2.0
    assert paged_rows["paged-32"]["total_tokens_per_second_ratio_vs_dense"] == 1.1
