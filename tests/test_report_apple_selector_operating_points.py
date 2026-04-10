from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_apple_selector_operating_points.py"
SPEC = importlib.util.spec_from_file_location("report_apple_selector_operating_points", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_combines_offline_and_live_operating_point_metrics(tmp_path: Path) -> None:
    exploration_results = tmp_path / "selector_exploration_results.json"
    exploration_results.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "linear_softmax_compression_weighted",
                        "pareto_optimal": True,
                        "aggregate_metrics": {
                            "min_family_safe_prediction_rate": 1.0,
                            "min_family_target_accuracy": 1.0,
                            "mean_predicted_total_bytes": 3489.1,
                            "mean_safe_bytes_regret": 0.0,
                        },
                        "model_summary": {},
                    },
                    {
                        "strategy_id": "linear_softmax_compression_floor_0p85",
                        "pareto_optimal": True,
                        "aggregate_metrics": {
                            "min_family_safe_prediction_rate": 0.94,
                            "min_family_target_accuracy": 0.901,
                            "mean_predicted_total_bytes": 3419.3,
                            "mean_safe_bytes_regret": 84.0,
                        },
                        "model_summary": {
                            "calibration": {
                                "min_target_accuracy": 0.85,
                                "min_safe_prediction_rate": 0.85,
                            }
                        },
                    },
                ]
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "task_compare_floor_0p85"
    run_dir.mkdir()
    (run_dir / "task_selector_compare.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_name": "retrieval_passkey",
                        "prompt_length": 512,
                        "quality_success": 1.0,
                        "quality_matches_dense_output": 1.0,
                        "quality_decode_ms_per_step": 700.0,
                        "systems_success": 1.0,
                        "systems_matches_dense_output": 1.0,
                        "systems_decode_ms_per_step": 400.0,
                    },
                    {
                        "task_name": "reasoning_arithmetic",
                        "prompt_length": 1024,
                        "quality_success": 0.0,
                        "quality_matches_dense_output": 1.0,
                        "quality_decode_ms_per_step": 1200.0,
                        "systems_success": 0.0,
                        "systems_matches_dense_output": 1.0,
                        "systems_decode_ms_per_step": 800.0,
                    },
                ]
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "qwen35_0p8b_task_selector_compare.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "measurement_kind": "aggregate",
                        "selector_profile": "quality",
                        "task_name": "retrieval_passkey",
                        "prompt_length_requested": 512,
                        "v_m0_pages": 180,
                        "v_m3_pages": 204,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "measurement_kind": "aggregate",
                        "selector_profile": "quality",
                        "task_name": "reasoning_arithmetic",
                        "prompt_length_requested": 1024,
                        "v_m0_pages": 608,
                        "v_m3_pages": 544,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "measurement_kind": "aggregate",
                        "selector_profile": "systems",
                        "task_name": "retrieval_passkey",
                        "prompt_length_requested": 512,
                        "v_m0_pages": 120,
                        "v_m3_pages": 264,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "measurement_kind": "aggregate",
                        "selector_profile": "systems",
                        "task_name": "reasoning_arithmetic",
                        "prompt_length_requested": 1024,
                        "v_m0_pages": 252,
                        "v_m3_pages": 516,
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload, markdown = MODULE.build_report(
        exploration_rows=MODULE._load_exploration_rows(exploration_results),
        task_runs={"linear_softmax_compression_floor_0p85": run_dir},
        title="Apple Selector Operating Points",
    )

    row = payload["operating_points"][0]
    assert row["strategy_id"] == "linear_softmax_compression_floor_0p85"
    assert row["floor"] == "0.85"
    assert row["quality"]["success_rate"] == 0.5
    assert row["quality"]["mean_v_m0_pages"] == 394.0
    assert row["systems"]["mean_v_m3_pages"] == 390.0
    assert "offline_bytes" in markdown
    assert "quality_m0" in markdown
    assert "3419.3" in markdown
