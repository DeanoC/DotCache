import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "report_qwen35_page_mix_sensitivity",
        repo_root / "scripts" / "report_qwen35_page_mix_sensitivity.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_page_mix_sensitivity_report_summarizes_variant_rows(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "selector_logit_sweep_manifest.json"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "target_candidate": "M3/affine/4/float16",
                "variants": [
                    {"variant": "offset_md1d00", "logit_offset": -1.0, "artifact_path": "a.json"},
                    {"variant": "offset_pd0d00", "logit_offset": 0.0, "artifact_path": "b.json"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = {
        "offset_md1d00": {
            "prompt_mode": "exact_length",
            "prompt_length": 1024,
            "measurement_kind": "aggregate",
            "dotcache_decode_ms_per_step": 70.0,
            "dotcache_decode_ms_per_step_p95": 72.0,
            "resident_bytes": 1024 * 1024 * 10,
            "kv_resident_bytes": 1024 * 1024 * 8,
            "mode_signature_counts": {"K:M0:affine:4": 6, "V:M3:affine:4:float16": 4},
            "learned_page_selector_invocations": 100,
            "learned_page_selector_ms_total": 2.5,
            "decode_backend_trace": {"score_ms_total": 40.0, "mix_ms_total": 20.0, "payload_bytes_read": 1024 * 1024 * 12},
            "decode_steps": 4,
        },
        "offset_pd0d00": {
            "prompt_mode": "exact_length",
            "prompt_length": 1024,
            "measurement_kind": "aggregate",
            "dotcache_decode_ms_per_step": 60.0,
            "dotcache_decode_ms_per_step_p95": 61.0,
            "resident_bytes": 1024 * 1024 * 11,
            "kv_resident_bytes": 1024 * 1024 * 9,
            "mode_signature_counts": {"K:M0:affine:4": 2, "V:M3:affine:4:float16": 8},
            "learned_page_selector_invocations": 100,
            "learned_page_selector_ms_total": 2.6,
            "decode_backend_trace": {"score_ms_total": 36.0, "mix_ms_total": 16.0, "payload_bytes_read": 1024 * 1024 * 16},
            "decode_steps": 4,
        },
    }
    for variant, row in rows.items():
        (results_dir / f"{variant}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = module._build_report(
        type(
            "Args",
            (),
            {
                "manifest": str(manifest_path),
                "results_dir": str(results_dir),
            },
        )()
    )

    assert report["target_candidate"] == "M3/affine/4/float16"
    slower = report["variants"][0]["rows"][0]
    faster = report["variants"][1]["rows"][0]
    assert slower["decode_vs_best_ratio"] == 70.0 / 60.0
    assert faster["decode_vs_best_ratio"] == 1.0

    markdown = module._render_markdown(report)
    assert "Decode / best" in markdown
    assert "| -1.00 | 1024 | 70.00 | 72.00 | 1.167 |" in markdown
    assert "| 0.00 | 1024 | 60.00 | 61.00 | 1.000 |" in markdown
