#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILE_ORDER = ("quality", "systems")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-oriented Apple selector sweep table.")
    parser.add_argument("--exploration-results", required=True, help="Path to selector_exploration_results.json.")
    parser.add_argument(
        "--task-run",
        action="append",
        default=[],
        help="Mapping in the form strategy_id=path/to/task_compare_dir_or_jsonl_or_json.",
    )
    parser.add_argument(
        "--include-strategy-id",
        action="append",
        default=[],
        help="Optional strategy ids to include; defaults to all sweep strategies sorted by floor.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(PROFILE_ORDER),
        default=list(PROFILE_ORDER),
        help="Profiles to include in the paper table.",
    )
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Apple Selector Sweep Table")
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


def _load_exploration_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["strategy_id"]): row for row in payload.get("strategies", [])}


def _artifact_digest(strategy_row: dict[str, Any]) -> str | None:
    artifact_path = strategy_row.get("artifact_path")
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _extract_floor_value(strategy_row: dict[str, Any]) -> float | None:
    calibration = (strategy_row.get("model_summary") or {}).get("calibration")
    if calibration:
        min_target = calibration.get("min_target_accuracy")
        min_safe = calibration.get("min_safe_prediction_rate")
        if min_target is not None and min_safe is not None and float(min_target) == float(min_safe):
            return float(min_target)
    strategy_id = str(strategy_row.get("strategy_id", ""))
    if "weighted" in strategy_id:
        return None
    return None


def _floor_label(strategy_row: dict[str, Any]) -> str:
    floor_value = _extract_floor_value(strategy_row)
    if floor_value is None:
        return "weighted"
    return f"{floor_value:.2f}"


def _load_live_metrics(run_path: Path) -> dict[str, dict[str, float | int | None]]:
    json_path, jsonl_path = _resolve_run_paths(run_path)
    report_rows = json.loads(json_path.read_text(encoding="utf-8")).get("rows", [])
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
    for profile in PROFILE_ORDER:
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
        resident_values = [row.get("resident_bytes") for row in live_rows if row.get("resident_bytes") is not None]
        metrics[profile] = {
            "task_count": len(live_rows),
            "success_rate": None if not success_values else float(sum(float(v) for v in success_values) / len(success_values)),
            "error_rate": None if not success_values else float(1.0 - (sum(float(v) for v in success_values) / len(success_values))),
            "dense_match_rate": None if not dense_match_values else float(sum(float(v) for v in dense_match_values) / len(dense_match_values)),
            "dense_error_rate": None if not dense_match_values else float(1.0 - (sum(float(v) for v in dense_match_values) / len(dense_match_values))),
            "decode_ms_per_step": None if not decode_values else float(sum(float(v) for v in decode_values) / len(decode_values)),
            "resident_bytes": None if not resident_values else float(sum(float(v) for v in resident_values) / len(resident_values)),
            "resident_mib": None if not resident_values else float(sum(float(v) for v in resident_values) / len(resident_values) / (1024.0 * 1024.0)),
        }
    return metrics


def _strategy_sort_key(strategy_row: dict[str, Any]) -> tuple[int, float, str]:
    floor_value = _extract_floor_value(strategy_row)
    if floor_value is None:
        return (0, -1.0, str(strategy_row.get("strategy_id", "")))
    return (1, floor_value, str(strategy_row.get("strategy_id", "")))


def build_report(
    *,
    exploration_rows: dict[str, dict[str, Any]],
    task_runs: dict[str, Path],
    include_strategy_ids: list[str] | None,
    profiles: list[str],
    title: str,
) -> tuple[dict[str, Any], str]:
    selected_rows = (
        [exploration_rows[strategy_id] for strategy_id in include_strategy_ids]
        if include_strategy_ids
        else sorted(exploration_rows.values(), key=_strategy_sort_key)
    )

    live_by_strategy: dict[str, dict[str, dict[str, float | int | None]]] = {}
    live_by_digest: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for strategy_id, run_path in task_runs.items():
        metrics = _load_live_metrics(run_path)
        live_by_strategy[strategy_id] = metrics
        strategy_row = exploration_rows.get(strategy_id)
        if strategy_row is not None:
            digest = _artifact_digest(strategy_row)
            if digest:
                live_by_digest[digest] = metrics

    rows: list[dict[str, Any]] = []
    for strategy_row in selected_rows:
        strategy_id = str(strategy_row["strategy_id"])
        aggregate = strategy_row.get("aggregate_metrics") or {}
        digest = _artifact_digest(strategy_row)
        live_metrics = live_by_strategy.get(strategy_id)
        live_source = strategy_id if live_metrics is not None else None
        if live_metrics is None and digest is not None:
            live_metrics = live_by_digest.get(digest)
            if live_metrics is not None:
                for source_strategy_id, source_metrics in live_by_strategy.items():
                    source_digest = _artifact_digest(exploration_rows[source_strategy_id])
                    if source_digest == digest and source_metrics is live_metrics:
                        live_source = source_strategy_id
                        break
        rows.append(
            {
                "strategy_id": strategy_id,
                "floor": _floor_label(strategy_row),
                "artifact_cluster": digest,
                "live_source_strategy_id": live_source,
                "offline_predicted_bytes": aggregate.get("mean_predicted_total_bytes"),
                "offline_error_rate": None if aggregate.get("min_family_target_accuracy") is None else float(1.0 - float(aggregate["min_family_target_accuracy"])),
                "profiles": live_metrics or {},
            }
        )

    sections = [f"# {title}"]
    for profile in profiles:
        table = [[
            "floor",
            "strategy_id",
            "cluster",
            "live_source",
            "offline_bytes",
            "offline_error_rate",
            "live_error_rate",
            "live_dense_error_rate",
            "live_speed_ms",
            "live_resident_mib",
        ]]
        for row in rows:
            metrics = row["profiles"].get(profile, {})
            table.append(
                [
                    str(row["floor"]),
                    str(row["strategy_id"]),
                    "-" if row["artifact_cluster"] is None else str(row["artifact_cluster"]),
                    "-" if row["live_source_strategy_id"] is None else str(row["live_source_strategy_id"]),
                    _fmt_float(row["offline_predicted_bytes"], digits=1),
                    _fmt_float(row["offline_error_rate"]),
                    _fmt_float(metrics.get("error_rate")),
                    _fmt_float(metrics.get("dense_error_rate")),
                    _fmt_float(metrics.get("decode_ms_per_step"), digits=1),
                    _fmt_float(metrics.get("resident_mib"), digits=2),
                ]
            )
        sections.extend(
            [
                "",
                f"## {profile.capitalize()}",
                "",
                "| " + " | ".join(table[0]) + " |",
                "| " + " | ".join("---" for _ in table[0]) + " |",
                *("| " + " | ".join(row) + " |" for row in table[1:]),
            ]
        )

    payload = {"rows": rows, "profiles": list(profiles)}
    return payload, "\n".join(sections)


def main() -> int:
    args = parse_args()
    exploration_rows = _load_exploration_rows(Path(args.exploration_results))
    task_runs = dict(_parse_task_run_mapping(value) for value in args.task_run)
    payload, markdown = build_report(
        exploration_rows=exploration_rows,
        task_runs=task_runs,
        include_strategy_ids=list(args.include_strategy_id),
        profiles=list(args.profiles),
        title=str(args.title),
    )
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
