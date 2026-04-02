import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "report_qwen35_backend_truth",
        repo_root / "scripts" / "report_qwen35_backend_truth.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_prefers_aggregate_rows_and_exposes_decode_p95(tmp_path: Path) -> None:
    module = _load_module()

    def write_rows(path: Path, variant: str, decode_value: float, decode_p95: float) -> None:
        rows = [
            {
                "prompt_mode": "exact_length",
                "prompt_length": 1024,
                "measurement_kind": "trial",
                "measurement_index": 0,
                "dotcache_decode_ms_per_step": 999.0,
                "dotcache_prefill_ms": 10.0,
                "resident_bytes": 1024,
                "kv_resident_bytes": 512,
                "mode_signature_counts": {"K:M0:affine:4": 4, "V:M3:affine:4:float16": 2},
                "execution_shortlist_selected_pages": 0,
                "execution_shortlist_total_pages": 0,
                "learned_page_selector_enabled": variant == "learned",
                "learned_page_selector_invocations": 8 if variant == "learned" else 0,
                "learned_page_selector_ms_total": 0.8 if variant == "learned" else 0.0,
                "decode_backend_trace": {"score_ms_total": 40.0, "mix_ms_total": 20.0},
                "decode_steps": 4,
            },
            {
                "prompt_mode": "exact_length",
                "prompt_length": 1024,
                "measurement_kind": "aggregate",
                "warmup_runs": 1,
                "measured_runs": 3,
                "dotcache_decode_ms_per_step": decode_value,
                "dotcache_decode_ms_per_step_p95": decode_p95,
                "dotcache_prefill_ms": 11.0,
                "resident_bytes": 2048,
                "kv_resident_bytes": 1024,
                "mode_signature_counts": {"K:M0:affine:4": 4, "V:M3:affine:4:float16": 2},
                "execution_shortlist_selected_pages": 0,
                "execution_shortlist_total_pages": 0,
                "learned_page_selector_enabled": variant == "learned",
                "learned_page_selector_invocations": 8 if variant == "learned" else 0,
                "learned_page_selector_ms_total": 0.8 if variant == "learned" else 0.0,
                "decode_backend_trace": {"score_ms_total": 44.0, "mix_ms_total": 24.0},
                "decode_steps": 4,
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    exact = tmp_path / "exact.jsonl"
    shortlist = tmp_path / "shortlist.jsonl"
    learned = tmp_path / "learned.jsonl"
    write_rows(exact, "exact", 120.0, 130.0)
    write_rows(shortlist, "shortlist", 100.0, 110.0)
    write_rows(learned, "learned", 80.0, 90.0)

    report = module._build_report(
        type(
            "Args",
            (),
            {
                "exact": str(exact),
                "shortlist": str(shortlist),
                "learned": str(learned),
                "learned_k_only": None,
                "learned_v_only": None,
            },
        )()
    )
    comparison = report["comparisons"][0]
    learned_row = comparison["variants"]["learned_selector"]
    assert learned_row["decode_ms_per_step"] == 80.0
    assert learned_row["decode_ms_per_step_p95"] == 90.0

    markdown = module._render_markdown(report)
    assert "Decode p95" in markdown
    assert "| 1024 | learned_selector | 80.00 | 90.00 |" in markdown


def test_report_can_include_k_only_and_v_only_learned_variants(tmp_path: Path) -> None:
    module = _load_module()

    def write_rows(path: Path, decode_value: float) -> None:
        row = {
            "prompt_mode": "exact_length",
            "prompt_length": 2048,
            "measurement_kind": "aggregate",
            "warmup_runs": 1,
            "measured_runs": 3,
            "dotcache_decode_ms_per_step": decode_value,
            "dotcache_decode_ms_per_step_p95": decode_value + 1.0,
            "dotcache_prefill_ms": 10.0,
            "resident_bytes": 4096,
            "kv_resident_bytes": 2048,
            "mode_signature_counts": {"K:M0:affine:4": 4, "V:M3:affine:4:float16": 2},
            "execution_shortlist_selected_pages": 0,
            "execution_shortlist_total_pages": 0,
            "learned_page_selector_enabled": True,
            "learned_page_selector_invocations": 8,
            "learned_page_selector_ms_total": 0.8,
            "decode_backend_trace": {"score_ms_total": 44.0, "mix_ms_total": 24.0},
            "decode_steps": 4,
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    exact = tmp_path / "exact.jsonl"
    shortlist = tmp_path / "shortlist.jsonl"
    learned = tmp_path / "learned.jsonl"
    learned_k_only = tmp_path / "learned_k_only.jsonl"
    learned_v_only = tmp_path / "learned_v_only.jsonl"
    write_rows(exact, 160.0)
    write_rows(shortlist, 120.0)
    write_rows(learned, 100.0)
    write_rows(learned_k_only, 110.0)
    write_rows(learned_v_only, 105.0)

    report = module._build_report(
        type(
            "Args",
            (),
            {
                "exact": str(exact),
                "shortlist": str(shortlist),
                "learned": str(learned),
                "learned_k_only": str(learned_k_only),
                "learned_v_only": str(learned_v_only),
            },
        )()
    )

    comparison = report["comparisons"][0]
    assert comparison["speedups"]["learned_selector"]["vs_exact"] == 1.6
    assert comparison["speedups"]["learned_selector_k_only"]["vs_shortlist"] == 120.0 / 110.0
    assert comparison["speedups"]["learned_selector_v_only"]["vs_exact"] == 160.0 / 105.0

    markdown = module._render_markdown(report)
    assert "| 2048 | learned_selector | 1.60 | 1.20 |" in markdown
    assert "| 2048 | learned_selector_k_only | 1.45 | 1.09 |" in markdown
    assert "| 2048 | learned_selector_v_only | 1.52 | 1.14 |" in markdown
