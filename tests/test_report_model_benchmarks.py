from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_model_benchmarks_supports_qwen_compare_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T01:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "qwen2_compare",
                                "backend": "torch_mps",
                                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                                "default_mode_k": "M0",
                                "quant_scheme_k": "affine",
                                "default_mode_v": "M0",
                                "quant_scheme_v": "affine",
                                "tokens_per_page": 256,
                                "torch_dtype": "float16",
                                "prompt_length": 2048,
                                "dense_decode_ms_per_step": 1000.0,
                                "decode_ms_per_step": 2500.0,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.25,
                                "greedy_token_agreement_rate": 1.0,
                                "prefill_cache_ingest_ms": 3200.0,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T02:00:00+00:00",
                        "benchmark": "qwen2_compare",
                        "backend": "torch_cuda",
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "default_mode_k": "M0",
                        "quant_scheme_k": "affine",
                        "default_mode_v": "M0",
                        "quant_scheme_v": "affine",
                        "tokens_per_page": 256,
                        "torch_dtype": "float16",
                        "prompt_length": 2048,
                        "dense_decode_ms_per_step": 180.0,
                        "decode_ms_per_step": 900.0,
                        "dotcache_vs_dense_kv_bytes_ratio": 0.19,
                        "greedy_token_agreement_rate": 1.0,
                        "prefill_cache_ingest_ms": 780.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_model_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "qwen2_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "Qwen/Qwen2.5-7B-Instruct" in stdout
    assert "2500.00" in stdout
    assert "900.00" in stdout
    assert "0.36" in stdout
    assert "ok" in stdout


def test_report_model_benchmarks_keeps_structured_error_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T01:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "qwen2_compare",
                                "backend": "torch_cuda",
                                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                                "default_mode_k": "M0",
                                "quant_scheme_k": "affine",
                                "default_mode_v": "M0",
                                "quant_scheme_v": "affine",
                                "tokens_per_page": 256,
                                "torch_dtype": "float16",
                                "prompt_length": 4096,
                                "status": "error",
                                "error_type": "RuntimeError",
                                "error_message": "CUDA out of memory",
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_model_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "qwen2_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "Qwen/Qwen2.5-7B-Instruct" in stdout
    assert "error" in stdout
    assert "RuntimeError: CUDA out of memory" in stdout


def test_report_model_benchmarks_skips_nul_padded_corrupt_lines(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_bytes(
        (
            json.dumps(
                {
                    "recorded_at": "2026-03-27T01:00:00+00:00",
                    "records": [
                        {
                            "benchmark": "qwen2_compare",
                            "backend": "torch_cuda",
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "prompt_length": 1024,
                            "dense_decode_ms_per_step": 45.0,
                            "decode_ms_per_step": 263.0,
                            "dotcache_vs_dense_kv_bytes_ratio": 0.28,
                            "greedy_token_agreement_rate": 0.25,
                            "prefill_cache_ingest_ms": 327.0,
                        }
                    ],
                }
            ).encode("utf-8")
            + b"\n"
            + (b"\x00" * 128)
            + b"\n"
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_model_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "qwen2_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "Qwen/Qwen2.5-7B-Instruct" in stdout
    assert "263.00" in stdout
