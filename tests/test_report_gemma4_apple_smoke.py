from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_gemma4_apple_smoke_renders_markdown_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "smoke_runner.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "model_id": "google/gemma-4-E2B",
                "prompt": "Cache locality on Apple Silicon is",
                "elapsed_s": 110.67,
                "timeout_seconds": 180,
                "returncode": 0,
                "probe_record": {
                    "runtime_device": "mps",
                    "runtime_torch_dtype": "float16",
                    "greedy_token_agreement_rate": 1.0,
                    "teacher_forced_logit_max_abs_error": 0.0,
                    "resident_bytes": 132096,
                    "kv_resident_bytes": 132096,
                    "m0_pages": 12,
                    "m3_pages": 18,
                    "dense_text": " a",
                    "dotcache_text": " a",
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_gemma4_apple_smoke.py",
            "--input",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = completed.stdout
    assert "# Gemma 4 Apple Smoke" in stdout
    assert "completed" in stdout
    assert "mps / float16" in stdout
    assert "1.000" in stdout
