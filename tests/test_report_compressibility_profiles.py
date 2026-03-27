from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_compressibility_profiles_classifies_selective_models(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T10:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "qwen2_compare",
                                "backend": "torch_cuda",
                                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                                "prompt_length": 2048,
                                "default_mode_k": "M0",
                                "default_mode_v": "M0",
                                "k_m3_pages": 0,
                                "k_total_static_pages": 576,
                                "kv_resident_bytes": 33030144,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.21843,
                                "decode_ms_per_step": 292.03,
                                "greedy_token_agreement_rate": 0.75,
                            },
                            {
                                "benchmark": "qwen2_compare",
                                "backend": "torch_cuda",
                                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                                "prompt_length": 2048,
                                "default_mode_k": "M0",
                                "default_mode_v": "M0",
                                "key_mode_overrides": ["layer:0=M3", "layer:27:kv:1=M3"],
                                "k_m3_pages": 24,
                                "k_total_static_pages": 576,
                                "kv_resident_bytes": 34111488,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.22558,
                                "decode_ms_per_step": 281.68,
                                "greedy_token_agreement_rate": 1.0,
                            },
                            {
                                "benchmark": "qwen2_compare",
                                "backend": "torch_cuda",
                                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                                "prompt_length": 2048,
                                "default_mode_k": "M3",
                                "default_mode_v": "M0",
                                "k_m3_pages": 576,
                                "k_total_static_pages": 576,
                                "kv_resident_bytes": 58982400,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.39005,
                                "decode_ms_per_step": 186.94,
                                "greedy_token_agreement_rate": 1.0,
                            },
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
            "scripts/report_compressibility_profiles.py",
            "--input",
            str(history_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "benefits from selective exact K" in stdout
    assert "4.17%" in stdout
    assert "0.578x" in stdout
    assert "selective K*=layer:0=M3,layer:27:kv:1=M3" in stdout


def test_report_compressibility_profiles_classifies_tolerant_models(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "recorded_at": "2026-03-27T11:00:00+00:00",
                        "records": [
                            {
                                "benchmark": "llama_compare",
                                "backend": "torch_cuda",
                                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "prompt_length": 1024,
                                "default_mode_k": "M0",
                                "default_mode_v": "M0",
                                "k_m3_pages": 0,
                                "k_total_static_pages": 256,
                                "kv_resident_bytes": 20000000,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.25,
                                "decode_ms_per_step": 250.0,
                                "greedy_token_agreement_rate": 1.0,
                            },
                            {
                                "benchmark": "llama_compare",
                                "backend": "torch_cuda",
                                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "prompt_length": 1024,
                                "default_mode_k": "M3",
                                "default_mode_v": "M0",
                                "k_m3_pages": 256,
                                "k_total_static_pages": 256,
                                "kv_resident_bytes": 32000000,
                                "dotcache_vs_dense_kv_bytes_ratio": 0.4,
                                "decode_ms_per_step": 150.0,
                                "greedy_token_agreement_rate": 1.0,
                            },
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
            "scripts/report_compressibility_profiles.py",
            "--input",
            str(history_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "tolerates all-M0" in stdout
    assert "| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1024 | tolerates all-M0 | all M0 | 0.00% | 1.000 | 0.625x | 0.250x | 4.00 | ok |" in stdout


def test_report_compressibility_profiles_leaves_exact_only_rows_unknown(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "recorded_at": "2026-03-27T12:00:00+00:00",
                "records": [
                    {
                        "benchmark": "qwen2_compare",
                        "backend": "torch_cuda",
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_length": 448,
                        "default_mode_k": "M3",
                        "default_mode_v": "M0",
                        "k_m3_pages": 112,
                        "k_total_static_pages": 112,
                        "kv_resident_bytes": 24313856,
                        "dotcache_vs_dense_kv_bytes_ratio": 0.47007,
                        "decode_ms_per_step": 145.09,
                        "greedy_token_agreement_rate": 1.0,
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
            "scripts/report_compressibility_profiles.py",
            "--input",
            str(history_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "| Qwen/Qwen2.5-7B-Instruct | 448 | unknown | exact K | 100.00% | 1.000 | - | 0.470x | 6.89 | ok |" in stdout
