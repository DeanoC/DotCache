from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_apple_selector_paper_table.py"
SPEC = importlib.util.spec_from_file_location("report_apple_selector_paper_table", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_artifact(path: Path, *, bias: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bias": bias, "weights": []}, sort_keys=True) + "\n", encoding="utf-8")


def test_build_report_reuses_live_metrics_for_matching_artifact_cluster(tmp_path: Path) -> None:
    weighted_artifact = tmp_path / "weighted" / "selector_artifact.json"
    floor_artifact = tmp_path / "floor" / "selector_artifact.json"
    alias_artifact = tmp_path / "alias" / "selector_artifact.json"
    _write_artifact(weighted_artifact, bias=[-0.4, -0.5])
    _write_artifact(floor_artifact, bias=[-2.0, -3.2])
    _write_artifact(alias_artifact, bias=[-2.0, -3.2])

    exploration_results = tmp_path / "selector_exploration_results.json"
    exploration_results.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "linear_softmax_compression_weighted",
                        "artifact_path": str(weighted_artifact),
                        "aggregate_metrics": {
                            "mean_predicted_total_bytes": 3489.1,
                            "min_family_target_accuracy": 1.0,
                        },
                        "model_summary": {},
                    },
                    {
                        "strategy_id": "linear_softmax_compression_floor_0p85",
                        "artifact_path": str(floor_artifact),
                        "aggregate_metrics": {
                            "mean_predicted_total_bytes": 3419.3,
                            "min_family_target_accuracy": 0.901,
                        },
                        "model_summary": {
                            "calibration": {
                                "min_target_accuracy": 0.85,
                                "min_safe_prediction_rate": 0.85,
                            }
                        },
                    },
                    {
                        "strategy_id": "linear_softmax_compression_floor_0p80",
                        "artifact_path": str(alias_artifact),
                        "aggregate_metrics": {
                            "mean_predicted_total_bytes": 3101.2,
                            "min_family_target_accuracy": 0.861,
                        },
                        "model_summary": {
                            "calibration": {
                                "min_target_accuracy": 0.80,
                                "min_safe_prediction_rate": 0.80,
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
                    },
                    {
                        "task_name": "reasoning_arithmetic",
                        "prompt_length": 1024,
                        "quality_success": 0.0,
                        "quality_matches_dense_output": 1.0,
                        "quality_decode_ms_per_step": 1100.0,
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
                        "resident_bytes": 8 * 1024 * 1024,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "measurement_kind": "aggregate",
                        "selector_profile": "quality",
                        "task_name": "reasoning_arithmetic",
                        "prompt_length_requested": 1024,
                        "resident_bytes": 10 * 1024 * 1024,
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
        include_strategy_ids=[
            "linear_softmax_compression_weighted",
            "linear_softmax_compression_floor_0p80",
            "linear_softmax_compression_floor_0p85",
        ],
        profiles=["quality"],
        title="Apple Selector Sweep Table",
    )

    rows = {row["strategy_id"]: row for row in payload["rows"]}
    assert rows["linear_softmax_compression_floor_0p80"]["live_source_strategy_id"] == "linear_softmax_compression_floor_0p85"
    assert rows["linear_softmax_compression_floor_0p80"]["profiles"]["quality"]["error_rate"] == 0.5
    assert rows["linear_softmax_compression_floor_0p85"]["profiles"]["quality"]["resident_mib"] == 9.0
    assert "live_error_rate" in markdown
    assert "0.500" in markdown
    assert "9.00" in markdown
