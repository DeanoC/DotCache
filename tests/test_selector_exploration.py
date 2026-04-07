from __future__ import annotations

import json
from pathlib import Path

from dotcache.selector_exploration import (
    _apply_pareto_membership,
    list_selector_exploration_strategies,
    resolve_selector_exploration_config,
    run_selector_exploration_lab,
)


def _write_tiny_selector_split_suite(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    split_root = root / "tiny_split"
    train_dir = split_root / "train"
    test_dir = split_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_labels: list[str] = []
    train_selector: list[str] = []
    train_candidate: list[str] = []
    test_labels: list[str] = []
    test_selector: list[str] = []
    test_candidate: list[str] = []

    for stage, query_present, target_candidate, safe_candidates in (
        ("prefill", False, "M0/affine/4", {"M0/affine/4": 740, "M3/affine/4/float16": 1448}),
        ("decode", True, "M3/affine/4/float16", {"M3/affine/4/float16": 1450}),
    ):
        for kind in ("K", "V"):
            for replica in range(2):
                prompt_family = "reasoning" if stage == "decode" else "cache"
                prompt_variant = "logic" if stage == "decode" else "locality"
                prompt_length = 1024 if stage == "decode" else 512
                trace_path = str(tmp_path / f"family-{prompt_family}_prompt{prompt_length:04d}_{stage}_{kind}_{replica}.npz")
                m0_total_bytes = 740 if kind == "K" else 744
                m3_total_bytes = 1448 if kind == "K" else 1450
                candidate_byte_map = {
                    "M0/affine/4": m0_total_bytes,
                    "M3/affine/4/float16": m3_total_bytes,
                }
                candidate_labels = [
                    {
                        "candidate": "M0/affine/4",
                        "safe": "M0/affine/4" in safe_candidates,
                        "total_bytes": m0_total_bytes,
                    },
                    {
                        "candidate": "M3/affine/4/float16",
                        "safe": "M3/affine/4/float16" in safe_candidates,
                        "total_bytes": m3_total_bytes,
                    },
                ]
                label_payload = json.dumps(
                    {
                        "trace_path": trace_path,
                        "stage": stage,
                        "prompt_family": prompt_family,
                        "prompt_variant": prompt_variant,
                        "source": "unit-test",
                        "kind": kind,
                        "layer_id": 3 if stage == "prefill" else 23,
                        "kv_head_id": replica,
                        "token_start": replica * 16,
                        "token_age": 24 if stage == "prefill" else 0,
                        "token_count": 2,
                        "head_dim": 256,
                        "query_present": query_present,
                        "cheapest_safe_candidate": target_candidate,
                        "safe_candidates": list(safe_candidates.keys()),
                        "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                        "candidate_labels": candidate_labels,
                        "trace_stats": {
                            "rms": 0.5 if stage == "prefill" else 1.2,
                            "abs_max": 2.0 if stage == "prefill" else 6.0,
                            "channel_range_mean": 0.3 if kind == "V" else 0.9,
                            "outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                        },
                    },
                    sort_keys=True,
                )
                selector_payload = json.dumps(
                    {
                        "trace_path": trace_path,
                        "source": "unit-test",
                        "stage": stage,
                        "prompt_family": prompt_family,
                        "prompt_variant": prompt_variant,
                        "kind": kind,
                        "layer_id": 3 if stage == "prefill" else 23,
                        "layer_fraction": 0.13 if stage == "prefill" else 1.0,
                        "kv_head_id": replica,
                        "kv_head_fraction": float(replica),
                        "token_start": replica * 16,
                        "token_age": 24 if stage == "prefill" else 0,
                        "token_count": 2,
                        "head_dim": 256,
                        "query_present": query_present,
                        "safe_candidate_count": len(safe_candidates),
                        "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                        "target_candidate": target_candidate,
                        "target_present": True,
                        "trace_rms": 0.5 if stage == "prefill" else 1.2,
                        "trace_abs_max": 2.0 if stage == "prefill" else 6.0,
                        "trace_channel_range_mean": 0.3 if kind == "V" else 0.9,
                        "trace_outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                        "age_per_token": 12.0 if stage == "prefill" else 0.0,
                    },
                    sort_keys=True,
                )
                candidate_payloads = [
                    json.dumps(
                        {
                            "trace_path": trace_path,
                            "source": "unit-test",
                            "stage": stage,
                            "prompt_family": prompt_family,
                            "prompt_variant": prompt_variant,
                            "kind": kind,
                            "layer_id": 3 if stage == "prefill" else 23,
                            "layer_fraction": 0.13 if stage == "prefill" else 1.0,
                            "kv_head_id": replica,
                            "kv_head_fraction": float(replica),
                            "token_start": replica * 16,
                            "token_age": 24 if stage == "prefill" else 0,
                            "token_count": 2,
                            "head_dim": 256,
                            "query_present": query_present,
                            "safe_candidate_count": len(safe_candidates),
                            "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                            "target_candidate": target_candidate,
                            "target_present": True,
                            "trace_rms": 0.5 if stage == "prefill" else 1.2,
                            "trace_abs_max": 2.0 if stage == "prefill" else 6.0,
                            "trace_channel_range_mean": 0.3 if kind == "V" else 0.9,
                            "trace_outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                            "age_per_token": 12.0 if stage == "prefill" else 0.0,
                            "candidate": candidate_label["candidate"],
                            "candidate_mode": "M0" if candidate_label["candidate"] == "M0/affine/4" else "M3",
                            "candidate_bits": 4,
                            "candidate_quant_scheme": "affine",
                            "candidate_total_bytes": candidate_label["total_bytes"],
                            "candidate_payload_bytes": 256 if candidate_label["candidate"] == "M0/affine/4" else 1024,
                            "candidate_metadata_bytes": candidate_label["total_bytes"] - (256 if candidate_label["candidate"] == "M0/affine/4" else 1024),
                            "candidate_has_escape_dtype": candidate_label["candidate"] == "M3/affine/4/float16",
                            "candidate_safe": candidate_label["safe"],
                            "candidate_is_target": candidate_label["candidate"] == target_candidate,
                            "candidate_bytes_over_best_safe": candidate_label["total_bytes"] - min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                        },
                        sort_keys=True,
                    )
                    for candidate_label in candidate_labels
                ]
                destination_labels = train_labels if replica == 0 else test_labels
                destination_selector = train_selector if replica == 0 else test_selector
                destination_candidate = train_candidate if replica == 0 else test_candidate
                destination_labels.append(label_payload)
                destination_selector.append(selector_payload)
                destination_candidate.extend(candidate_payloads)

    (train_dir / "labels.jsonl").write_text("\n".join(train_labels) + "\n", encoding="utf-8")
    (train_dir / "selector_dataset.jsonl").write_text("\n".join(train_selector) + "\n", encoding="utf-8")
    (train_dir / "selector_candidate_dataset.jsonl").write_text("\n".join(train_candidate) + "\n", encoding="utf-8")
    (test_dir / "labels.jsonl").write_text("\n".join(test_labels) + "\n", encoding="utf-8")
    (test_dir / "selector_dataset.jsonl").write_text("\n".join(test_selector) + "\n", encoding="utf-8")
    (test_dir / "selector_candidate_dataset.jsonl").write_text("\n".join(test_candidate) + "\n", encoding="utf-8")
    (split_root / "split_summary.json").write_text(
        json.dumps(
            {
                "split_name": "tiny_split",
                "holdout_prompt_families": ["reasoning"],
                "holdout_prompt_variants": [],
                "holdout_layers": [],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _write_dense_control_report(tmp_path: Path) -> Path:
    rows = [
        {
            "measurement_kind": "aggregate",
            "selector_profile": "dense",
            "task_family": "cache",
            "prompt_length": 512,
            "task_success": True,
        },
        {
            "measurement_kind": "aggregate",
            "selector_profile": "dense",
            "task_family": "reasoning",
            "prompt_length": 1024,
            "task_success": False,
        },
    ]
    path = tmp_path / "dense_control.jsonl"
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_selector_exploration_registry_includes_planned_strategy_families() -> None:
    strategies = list_selector_exploration_strategies()

    assert "candidate_target_linear" in strategies
    assert "candidate_target_mlp" in strategies
    assert "candidate_target_gbdt" in strategies
    assert "linear_softmax_distilled_mlp_teacher" in strategies
    assert "linear_softmax_compression_equal_tradeoff" in strategies
    assert strategies["candidate_safe_router"].supported_calibration_modes == ("global", "per_prompt_family")


def test_selector_exploration_pareto_marks_non_dominated_rows() -> None:
    class _Row:
        def __init__(self, strategy_id: str, metrics: dict[str, float]) -> None:
            self.strategy_id = strategy_id
            self.status = "ok"
            self.aggregate_metrics = metrics
            self.pareto_optimal = False

    rows = [
        _Row(
            "a",
            {
                "min_family_safe_prediction_rate": 1.0,
                "min_family_target_accuracy": 0.9,
                "mean_predicted_total_bytes": 4000.0,
                "mean_safe_bytes_regret": 100.0,
            },
        ),
        _Row(
            "b",
            {
                "min_family_safe_prediction_rate": 0.9,
                "min_family_target_accuracy": 0.8,
                "mean_predicted_total_bytes": 4500.0,
                "mean_safe_bytes_regret": 120.0,
            },
        ),
    ]

    _apply_pareto_membership(rows, report_axes=("min_family_safe_prediction_rate", "min_family_target_accuracy", "mean_predicted_total_bytes", "mean_safe_bytes_regret"))

    assert rows[0].pareto_optimal is True
    assert rows[1].pareto_optimal is False


def test_selector_exploration_lab_writes_reports_predictions_and_promotability(tmp_path) -> None:
    suite_root = _write_tiny_selector_split_suite(tmp_path)
    smoke_script = tmp_path / "fake_smoke.sh"
    smoke_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$1\"\n"
        "printf '{\"status\": \"ok\"}\\n' > \"$1/task_selector_compare.json\"\n"
        "printf '# smoke\\n' > \"$1/task_selector_compare.md\"\n",
        encoding="utf-8",
    )
    smoke_script.chmod(0o755)
    config = resolve_selector_exploration_config(
        {
            "suite_root": str(suite_root),
            "feature_set_id": "runtime_safe",
            "strategies": [
                {"strategy_id": "static_rule"},
                {"strategy_id": "linear_softmax"},
                {"strategy_id": "linear_softmax_compression_equal_tradeoff"},
                {"strategy_id": "linear_softmax_distilled_mlp_teacher", "params": {"teacher_epochs": 20}},
                {"strategy_id": "candidate_safe_router", "calibration_mode": "per_prompt_family"},
                {"strategy_id": "candidate_target_linear", "calibration_mode": "global"},
                {"strategy_id": "candidate_target_mlp", "calibration_mode": "global", "params": {"epochs": 20}},
            ],
            "serving_smoke": {
                "enabled": True,
                "command_template": ["bash", str(smoke_script), "{output_dir}", "{artifact_path}"],
            },
        }
    )

    payload = run_selector_exploration_lab(config=config, output_dir=tmp_path / "output")

    assert Path(payload["json_path"]).exists()
    assert Path(payload["markdown_path"]).exists()
    strategy_rows = {row["strategy_id"]: row for row in payload["strategies"]}
    assert set(strategy_rows.keys()) == {
        "static_rule",
        "linear_softmax",
        "linear_softmax_compression_equal_tradeoff",
        "linear_softmax_distilled_mlp_teacher",
        "candidate_safe_router",
        "candidate_target_linear",
        "candidate_target_mlp",
    }
    assert strategy_rows["linear_softmax"]["runtime_compatible"] is True
    assert strategy_rows["linear_softmax_distilled_mlp_teacher"]["runtime_compatible"] is True
    assert strategy_rows["linear_softmax_distilled_mlp_teacher"]["artifact_path"] is not None
    assert strategy_rows["candidate_target_linear"]["artifact_path"] is not None
    assert strategy_rows["candidate_target_mlp"]["research_model_path"] is not None
    assert any(bool(row["pareto_optimal"]) for row in payload["strategies"])
    assert any(bool(row["promotable"]) for row in payload["strategies"])

    prediction_path = Path(strategy_rows["candidate_target_linear"]["prediction_path"])
    prediction_row = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    assert {
        "split_name",
        "trace_path",
        "predicted_candidate",
        "oracle_target_candidate",
        "prompt_family",
        "prompt_variant",
    }.issubset(prediction_row.keys())


def test_selector_exploration_lab_marks_research_extended_rows_non_promotable(tmp_path) -> None:
    suite_root = _write_tiny_selector_split_suite(tmp_path)
    config = resolve_selector_exploration_config(
        {
            "suite_root": str(suite_root),
            "feature_set_id": "research_extended",
            "strategies": [{"strategy_id": "candidate_target_linear", "calibration_mode": "per_prompt_family"}],
            "serving_smoke": {"enabled": False},
        }
    )

    payload = run_selector_exploration_lab(config=config, output_dir=tmp_path / "output_research")
    row = payload["strategies"][0]

    assert row["feature_set_id"] == "research_extended"
    assert row["runtime_compatible"] is False
    assert row["artifact_path"] is None
    assert row["promotable"] is False


def test_selector_exploration_lab_can_apply_dense_control_weighting(tmp_path) -> None:
    suite_root = _write_tiny_selector_split_suite(tmp_path)
    dense_control_path = _write_dense_control_report(tmp_path)
    config = resolve_selector_exploration_config(
        {
            "suite_root": str(suite_root),
            "feature_set_id": "runtime_safe",
            "dense_control": {
                "enabled": True,
                "report_path": str(dense_control_path),
                "correct_example_weight": 2.0,
                "incorrect_example_weight": 0.5,
            },
            "strategies": [
                {
                    "strategy_id": "linear_softmax_compression_weighted",
                    "result_id": "linear_softmax_compression_weighted_dense_control",
                    "params": {"dense_control_weighting": True},
                }
            ],
            "serving_smoke": {"enabled": False},
        }
    )

    payload = run_selector_exploration_lab(config=config, output_dir=tmp_path / "output_dense_control")
    row = payload["strategies"][0]

    assert row["strategy_id"] == "linear_softmax_compression_weighted_dense_control"
    assert row["artifact_path"] is not None
    assert row["model_summary"]["dense_control_weighting"] is True
    summary = row["model_summary"]["dense_control_summary"]
    assert summary["enabled"] is True
    assert summary["matched_examples"] > 0
    assert summary["matched_correct_examples"] > 0
    assert summary["matched_incorrect_examples"] > 0


def test_selector_exploration_lab_allows_per_strategy_calibration_overrides(tmp_path) -> None:
    suite_root = _write_tiny_selector_split_suite(tmp_path)
    config = resolve_selector_exploration_config(
        {
            "suite_root": str(suite_root),
            "feature_set_id": "runtime_safe",
            "strategies": [
                {
                    "strategy_id": "linear_softmax_compression_calibrated",
                    "result_id": "linear_softmax_compression_calibrated_floor_override",
                    "params": {
                        "calibration_min_target_accuracy": 0.5,
                        "calibration_min_safe_prediction_rate": 0.5,
                    },
                }
            ],
            "serving_smoke": {"enabled": False},
        }
    )

    payload = run_selector_exploration_lab(config=config, output_dir=tmp_path / "output_calibration_override")
    row = payload["strategies"][0]

    assert row["strategy_id"] == "linear_softmax_compression_calibrated_floor_override"
    calibration = row["model_summary"]["calibration"]
    assert float(calibration["min_target_accuracy"]) == 0.5
    assert float(calibration["min_safe_prediction_rate"]) == 0.5
