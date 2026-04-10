#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Apple selector operating points across offline and live runs.")
    parser.add_argument("--exploration-results", required=True, help="Path to selector_exploration_results.json.")
    parser.add_argument(
        "--task-run",
        action="append",
        default=[],
        help="Mapping in the form strategy_id=path/to/task_compare_dir_or_jsonl_or_json.",
    )
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Apple Selector Operating Points")
    return parser.parse_args()


def _parse_task_run_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"invalid --task-run mapping {value!r}; expected strategy_id=path")
    strategy_id, raw_path = value.split("=", 1)
    return strategy_id.strip(), Path(raw_path.strip())


def _resolve_run_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        json_path = path / "task_selector_compare.json"
        jsonl_path = path / "qwen35_0p8b_task_selector_compare.jsonl"
    elif path.suffix == ".json":
        json_path = path
        jsonl_path = path.with_name("qwen35_0p8b_task_selector_compare.jsonl")
    elif path.suffix == ".jsonl":
        jsonl_path = path
        json_path = path.with_name("task_selector_compare.json")
    else:
        raise SystemExit(f"unsupported task run path: {path}")
    if not json_path.exists():
        raise SystemExit(f"missing task report json: {json_path}")
    if not jsonl_path.exists():
        raise SystemExit(f"missing task report jsonl: {jsonl_path}")
    return json_path, jsonl_path


def _fmt_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _load_task_profile_metrics(run_path: Path) -> dict[str, dict[str, float | int | None]]:
    json_path, jsonl_path = _resolve_run_paths(run_path)
    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report_rows = list(report_payload.get("rows", []))
    raw_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregate_rows = [row for row in raw_rows if row.get("measurement_kind") == "aggregate"]
    by_profile_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        profile = str(row.get("selector_profile", ""))
        if profile:
            by_profile_rows[profile].append(row)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for profile in ("quality", "systems"):
        live_rows = by_profile_rows.get(profile, [])
        if not live_rows:
            continue
        success_values = [row.get(f"{profile}_success") for row in report_rows if row.get(f"{profile}_success") is not None]
        dense_match_values = [
            row.get(f"{profile}_matches_dense_output")
            for row in report_rows
            if row.get(f"{profile}_matches_dense_output") is not None
        ]
        decode_values = [
            row.get(f"{profile}_decode_ms_per_step")
            for row in report_rows
            if row.get(f"{profile}_decode_ms_per_step") is not None
        ]
        metrics[profile] = {
            "task_count": len(live_rows),
            "success_rate": None if not success_values else float(sum(float(v) for v in success_values) / len(success_values)),
            "dense_match_rate": (
                None if not dense_match_values else float(sum(float(v) for v in dense_match_values) / len(dense_match_values))
            ),
            "decode_ms_per_step": None if not decode_values else float(sum(float(v) for v in decode_values) / len(decode_values)),
            "mean_v_m0_pages": float(sum(float(row.get("v_m0_pages") or 0.0) for row in live_rows) / len(live_rows)),
            "mean_v_m3_pages": float(sum(float(row.get("v_m3_pages") or 0.0) for row in live_rows) / len(live_rows)),
        }
    return metrics


def _load_exploration_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["strategy_id"]): row
        for row in payload.get("strategies", [])
    }


def _extract_floor_label(strategy_row: dict[str, Any]) -> str:
    calibration = (strategy_row.get("model_summary") or {}).get("calibration")
    if not calibration:
        return "-"
    min_target = calibration.get("min_target_accuracy")
    min_safe = calibration.get("min_safe_prediction_rate")
    if min_target is None and min_safe is None:
        return "-"
    if min_target == min_safe:
        return _fmt_float(min_target, digits=2)
    return f"{_fmt_float(min_target, digits=2)}/{_fmt_float(min_safe, digits=2)}"


def build_report(
    *,
    exploration_rows: dict[str, dict[str, Any]],
    task_runs: dict[str, Path],
    title: str,
) -> tuple[dict[str, Any], str]:
    operating_rows: list[dict[str, Any]] = []
    for strategy_id, run_path in task_runs.items():
        exploration_row = exploration_rows.get(strategy_id)
        if exploration_row is None:
            raise SystemExit(f"strategy_id {strategy_id!r} not found in exploration results")
        live_metrics = _load_task_profile_metrics(run_path)
        aggregate = exploration_row.get("aggregate_metrics") or {}
        row = {
            "strategy_id": strategy_id,
            "floor": _extract_floor_label(exploration_row),
            "pareto_optimal": bool(exploration_row.get("pareto_optimal")),
            "min_family_safe_prediction_rate": aggregate.get("min_family_safe_prediction_rate"),
            "min_family_target_accuracy": aggregate.get("min_family_target_accuracy"),
            "mean_predicted_total_bytes": aggregate.get("mean_predicted_total_bytes"),
            "mean_safe_bytes_regret": aggregate.get("mean_safe_bytes_regret"),
            "quality": live_metrics.get("quality"),
            "systems": live_metrics.get("systems"),
        }
        operating_rows.append(row)

    operating_rows.sort(
        key=lambda row: (
            0 if row["strategy_id"] == "linear_softmax_compression_weighted" else 1,
            -(float(row["min_family_target_accuracy"]) if row["min_family_target_accuracy"] is not None else -1.0),
            float(row["mean_predicted_total_bytes"]) if row["mean_predicted_total_bytes"] is not None else float("inf"),
            row["strategy_id"],
        )
    )

    table = [[
        "strategy_id",
        "floor",
        "pareto",
        "offline_safe",
        "offline_target",
        "offline_bytes",
        "quality_success",
        "quality_dense_match",
        "quality_ms",
        "quality_m0",
        "quality_m3",
        "systems_success",
        "systems_dense_match",
        "systems_ms",
        "systems_m0",
        "systems_m3",
    ]]
    for row in operating_rows:
        quality = row.get("quality") or {}
        systems = row.get("systems") or {}
        table.append(
            [
                str(row["strategy_id"]),
                str(row["floor"]),
                "yes" if row["pareto_optimal"] else "no",
                _fmt_float(row["min_family_safe_prediction_rate"]),
                _fmt_float(row["min_family_target_accuracy"]),
                _fmt_float(row["mean_predicted_total_bytes"], digits=1),
                _fmt_float(quality.get("success_rate")),
                _fmt_float(quality.get("dense_match_rate")),
                _fmt_float(quality.get("decode_ms_per_step"), digits=1),
                _fmt_float(quality.get("mean_v_m0_pages"), digits=1),
                _fmt_float(quality.get("mean_v_m3_pages"), digits=1),
                _fmt_float(systems.get("success_rate")),
                _fmt_float(systems.get("dense_match_rate")),
                _fmt_float(systems.get("decode_ms_per_step"), digits=1),
                _fmt_float(systems.get("mean_v_m0_pages"), digits=1),
                _fmt_float(systems.get("mean_v_m3_pages"), digits=1),
            ]
        )

    markdown = "\n".join(
        [
            f"# {title}",
            "",
            "| " + " | ".join(table[0]) + " |",
            "| " + " | ".join("---" for _ in table[0]) + " |",
            *("| " + " | ".join(row) + " |" for row in table[1:]),
        ]
    )
    return {"operating_points": operating_rows}, markdown


def main() -> int:
    args = parse_args()
    exploration_rows = _load_exploration_rows(Path(args.exploration_results))
    task_runs = dict(_parse_task_run_mapping(value) for value in args.task_run)
    payload, markdown = build_report(
        exploration_rows=exploration_rows,
        task_runs=task_runs,
        title=str(args.title),
    )
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
