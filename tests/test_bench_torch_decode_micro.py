from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_bench_torch_decode_micro_emits_m0_execution_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "bench_torch_decode_micro.py"),
            "--device",
            "cpu",
            "--head-dim",
            "64",
            "--num-key-value-heads",
            "1",
            "--query-count",
            "1",
            "--prompt-length",
            "32",
            "--tokens-per-page",
            "16",
            "--warmup-iters",
            "0",
            "--bench-iters",
            "1",
            "--output-format",
            "json",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["benchmark"] == "torch_decode_micro"
    assert payload["mode"] == "m0_execution"
    assert payload["page_count"] == 2
    assert payload["selected_token_count"] == 32
    assert payload["direct_m0_variant"] == "fused_two_group64"
    assert payload["blockwise_qdq_exact_combined_ms"] >= payload["direct_m0_combined_ms"]
    assert payload["direct_vs_cached_mix_max_abs_error"] < 1e-3


def test_bench_torch_decode_micro_emits_runtime_shaped_breakdown() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "bench_torch_decode_micro.py"),
            "--device",
            "cpu",
            "--head-dim",
            "64",
            "--num-key-value-heads",
            "1",
            "--query-count",
            "1",
            "--tokens-per-page",
            "16",
            "--prompt-length",
            "32",
            "--runtime-shaped-profile",
            "qwen35_bias_large",
            "--warmup-iters",
            "0",
            "--bench-iters",
            "1",
            "--output-format",
            "json",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    breakdown = payload["runtime_score_breakdown"]

    assert payload["runtime_shaped_profile"] == "qwen35_bias_large"
    assert breakdown["enabled"] is True
    assert breakdown["m0_pages"] == 24
    assert breakdown["m3_pages"] == 24
    assert breakdown["m0_selected_tokens"] == 384
    assert breakdown["m3_selected_tokens"] == 384
    assert breakdown["total_selected_tokens"] == 768
    assert breakdown["combined_ms"] >= 0.0
    assert breakdown["exact_m3_score_ms"] >= 0.0
    assert breakdown["exact_m3_flat_score_ms"] >= 0.0
    assert breakdown["exact_m3_transposed_score_ms"] >= 0.0
    assert breakdown["direct_m0_transposed_score_ms"] >= 0.0
    assert breakdown["exact_m3_flat_vs_page_score_max_abs_error"] >= 0.0
    assert breakdown["exact_m3_transposed_vs_page_score_max_abs_error"] >= 0.0
    assert breakdown["direct_vs_dense_score_max_abs_error"] >= 0.0
    assert breakdown["direct_m0_transposed_vs_dense_score_max_abs_error"] >= 0.0
    assert breakdown["combined_vs_dense_mix_max_abs_error"] >= 0.0
