import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "report_qwen35_selector_quality_compare",
        repo_root / "scripts" / "report_qwen35_selector_quality_compare.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_quality_compare_report_renders_tradeoff_rows(tmp_path: Path) -> None:
    module = _load_module()

    def write_row(path: Path, *, decode: float, token_agreement: float, rmse: float, profile: str, offset: float) -> None:
        path.write_text(
            (
                "{"
                f"\"prompt_mode\":\"exact_length\",\"prompt_length\":1024,\"measurement_kind\":\"aggregate\","
                f"\"dotcache_prefill_ms\":100.0,\"dotcache_decode_ms_per_step\":{decode},"
                f"\"dotcache_decode_ms_per_step_p95\":{decode + 1.0},"
                f"\"teacher_forced_token_agreement_rate\":{token_agreement},"
                f"\"teacher_forced_logit_rmse\":{rmse},"
                "\"teacher_forced_logit_max_abs_error\":1.0,"
                "\"replay_context_max_abs_error\":0.1,"
                "\"replay_output_max_abs_error\":0.2,"
                "\"m3_pages\":90,\"total_static_pages\":100,"
                f"\"learned_page_selector_profile\":\"{profile}\","
                f"\"learned_page_selector_logit_offset\":{offset}"
                "}\n"
            ),
            encoding="utf-8",
        )

    exact = tmp_path / "exact.jsonl"
    quality = tmp_path / "quality.jsonl"
    systems = tmp_path / "systems.jsonl"
    write_row(exact, decode=120.0, token_agreement=1.0, rmse=0.01, profile="quality", offset=0.0)
    write_row(quality, decode=100.0, token_agreement=0.99, rmse=0.02, profile="quality", offset=0.0)
    write_row(systems, decode=80.0, token_agreement=0.97, rmse=0.05, profile="systems", offset=2.0)

    report = module._build_report(
        type(
            "Args",
            (),
            {
                "exact": str(exact),
                "quality": str(quality),
                "systems": str(systems),
            },
        )()
    )

    comparison = report["comparisons"][0]
    assert comparison["quality_vs_exact_decode_speedup"] == 1.2
    assert comparison["systems_vs_exact_decode_speedup"] == 1.5
    assert comparison["systems_vs_quality_decode_speedup"] == 1.25
    assert comparison["systems_minus_quality_token_agreement"] == -0.020000000000000018

    markdown = module._render_markdown(report)
    assert "Systems vs Quality speedup" in markdown
    assert "| 1024 | systems | 80.000 | 81.000 | 100.000 | 0.970 | 0.050 |" in markdown
