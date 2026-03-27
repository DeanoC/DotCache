from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_llama_benchmarks_aligns_backends(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T01:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "llama_loss",
                                "backend": "torch_mps",
                                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "default_mode_k": "M0",
                                "quant_scheme_k": "affine",
                                "default_mode_v": "M0",
                                "quant_scheme_v": "affine",
                                "tokens_per_page": 256,
                                "torch_dtype": "float16",
                                "sequence_length": 320,
                                "prefix_length": 288,
                                "eval_steps": 32,
                                "dense_decode_ms_per_step": 1000.0,
                                "dotcache_decode_ms_per_step": 4000.0,
                                "teacher_forced_loss_delta": -0.001,
                                "teacher_forced_token_agreement_rate": 1.0,
                                "prefill_ms": 4490.0,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T02:00:00+00:00",
                        "benchmark": "llama_loss",
                        "backend": "torch_cuda",
                        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "default_mode_k": "M0",
                        "quant_scheme_k": "affine",
                        "default_mode_v": "M0",
                        "quant_scheme_v": "affine",
                        "tokens_per_page": 256,
                        "torch_dtype": "float16",
                        "sequence_length": 320,
                        "prefix_length": 288,
                        "eval_steps": 32,
                        "dense_decode_ms_per_step": 36.2,
                        "dotcache_decode_ms_per_step": 145.5,
                        "teacher_forced_loss_delta": 0.0,
                        "teacher_forced_token_agreement_rate": 1.0,
                        "prefill_ms": 1140.9,
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
            "scripts/report_llama_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "llama_loss",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in stdout
    assert "4000.00" in stdout
    assert "145.50" in stdout
    assert "0.04" in stdout


def test_report_llama_benchmarks_keeps_missing_backend_cases(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "recorded_at": "2026-03-27T01:00:00+00:00",
                "records": [
                    {
                        "benchmark": "llama_compare",
                        "backend": "torch_mps",
                        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "default_mode_k": "M0",
                        "quant_scheme_k": "affine",
                        "default_mode_v": "M0",
                        "quant_scheme_v": "affine",
                        "tokens_per_page": 256,
                        "torch_dtype": "float16",
                        "prompt_length": 289,
                        "prompt_mode": "target_prompt_length",
                        "repeat_count": None,
                        "requested_prompt_length": 289,
                        "dense_decode_ms_per_step": 44.3,
                        "decode_ms_per_step": 254.4,
                        "dotcache_vs_dense_kv_bytes_ratio": 0.58,
                        "greedy_token_agreement_rate": 1.0,
                        "prefill_cache_ingest_ms": 95.7,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_llama_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "llama_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "254.40" in stdout
    assert "44.30" in stdout
    assert " | - | - | " in stdout


def test_report_llama_benchmarks_normalizes_missing_default_modes(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T01:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "llama_compare",
                                "backend": "torch_mps",
                                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "prompt_length": 289,
                                "dense_decode_ms_per_step": 44.3,
                                "decode_ms_per_step": 254.4,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.58,
                                "greedy_token_agreement_rate": 1.0,
                                "prefill_cache_ingest_ms": 95.7,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T02:00:00+00:00",
                        "benchmark": "llama_compare",
                        "backend": "torch_cuda",
                        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "default_mode_k": "M0",
                        "default_mode_v": "M0",
                        "quant_scheme_k": "affine",
                        "quant_scheme_v": "affine",
                        "tokens_per_page": 256,
                        "torch_dtype": "float16",
                        "prompt_length": 289,
                        "dense_decode_ms_per_step": 22.31,
                        "decode_ms_per_step": 132.15,
                        "dotcache_vs_dense_kv_bytes_ratio": 0.58,
                        "greedy_token_agreement_rate": 1.0,
                        "prefill_cache_ingest_ms": 87.28,
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
            "scripts/report_llama_benchmarks.py",
            "--input",
            str(history_path),
            "--benchmark",
            "llama_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    lines = [line for line in completed.stdout.splitlines() if line.startswith("| TinyLlama/")]
    assert len(lines) == 1
    assert "254.40" in lines[0]
    assert "132.15" in lines[0]
