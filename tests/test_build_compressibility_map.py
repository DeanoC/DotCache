from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "build_compressibility_map.py"
    spec = importlib.util.spec_from_file_location("build_compressibility_map", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_compressibility_map_classifies_tolerant_model(monkeypatch) -> None:
    module = _load_module()

    def fake_build_policy_suggestions(args):
        return {
            "candidate_policies": [
                {"label": "all_m0", "estimated_k_exact_fraction": 0.0},
                {"label": "exact_k", "estimated_k_exact_fraction": 1.0},
            ],
            "validated_recommended_policy": None,
            "recommended_policy": None,
        }

    def fake_validate(args, *, label, targets):
        if label == "all_m0":
            return {
                "label": label,
                "command": "cmd all_m0",
                "status": "ok",
                "agreement": 1.0,
                "decode_ms_per_step": 120.0,
                "kv_vs_dense": 0.22,
                "k_exact_fraction": 0.0,
            }
        return {
            "label": label,
            "command": f"cmd {label}",
            "status": "ok",
            "agreement": 1.0,
            "decode_ms_per_step": 150.0,
            "kv_vs_dense": 0.40,
            "k_exact_fraction": 1.0,
        }

    monkeypatch.setattr(module.suggest, "build_policy_suggestions", fake_build_policy_suggestions)
    monkeypatch.setattr(module.suggest, "_validate_candidate_policy", fake_validate)

    args = module.parse_args(["--spec", "llama|TinyLlama/TinyLlama-1.1B-Chat-v1.0|2048", "--format", "json"])
    report = module.build_compressibility_map(args)

    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["classification"] == "tolerates all-M0"
    assert row["selected_policy"] == "all M0"
    assert row["k_exact_fraction"] == 0.0
    assert abs(row["kv_vs_exact_k"] - 0.55) < 1e-9


def test_build_compressibility_map_classifies_selective_model(monkeypatch) -> None:
    module = _load_module()

    def fake_build_policy_suggestions(args):
        return {
            "candidate_policies": [
                {"label": "all_m0", "estimated_k_exact_fraction": 0.0},
                {"label": "exact_k", "estimated_k_exact_fraction": 1.0},
            ],
            "validated_recommended_policy": {
                "label": "layer:0=M3",
                "estimated_k_exact_fraction": 0.0357,
                "validation": {
                    "command": "cmd selective",
                    "status": "ok",
                    "agreement": 1.0,
                    "decode_ms_per_step": 200.0,
                    "kv_vs_dense": 0.23,
                    "k_exact_fraction": 0.0357,
                },
            },
            "recommended_policy": {"label": "layer:0=M3", "estimated_k_exact_fraction": 0.0357},
        }

    def fake_validate(args, *, label, targets):
        if label == "all_m0":
            return {
                "label": label,
                "command": "cmd all_m0",
                "status": "ok",
                "agreement": 0.5,
                "decode_ms_per_step": 180.0,
                "kv_vs_dense": 0.21,
                "k_exact_fraction": 0.0,
            }
        if label == "layer:0=M3":
            return {
                "label": label,
                "command": "cmd selective",
                "status": "ok",
                "agreement": 1.0,
                "decode_ms_per_step": 200.0,
                "kv_vs_dense": 0.23,
                "k_exact_fraction": 0.0357,
            }
        return {
            "label": label,
            "command": "cmd exact_k",
            "status": "ok",
            "agreement": 1.0,
            "decode_ms_per_step": 240.0,
            "kv_vs_dense": 0.39,
            "k_exact_fraction": 1.0,
        }

    monkeypatch.setattr(module.suggest, "build_policy_suggestions", fake_build_policy_suggestions)
    monkeypatch.setattr(module.suggest, "_validate_candidate_policy", fake_validate)

    args = module.parse_args(["--spec", "qwen2|Qwen/Qwen2.5-1.5B-Instruct|2048", "--format", "markdown"])
    report = module.build_compressibility_map(args)
    rendered = module.render_markdown(report)

    row = report["rows"][0]
    assert row["classification"] == "benefits from selective exact K"
    assert row["selected_policy"] == "layer:0=M3"
    assert abs(row["k_exact_fraction"] - 0.0357) < 1e-9
    assert abs(row["kv_vs_exact_k"] - (0.23 / 0.39)) < 1e-9
    assert "benefits from selective exact K" in rendered
    assert "cmd selective" in rendered
