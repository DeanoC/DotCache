#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TASK_PROFILES = ("dense", "exact", "quality", "systems")
LONGBENCH_CASES = ("dense", "exact", "quality", "systems")
DEFAULT_PACKS = ("task_compare", "longbench_mini", "longbench_lb21_16_smoke_20260406")
PRIMARY_PACK = "longbench_lb21_16_smoke_20260406"
PRIMARY_PROFILE = "systems"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the CUDA selector operating-point frontier for Qwen3.5-9B."
    )
    parser.add_argument("--base-dir", required=True, help="Root results dir containing per-point subdirectories.")
    parser.add_argument("--exploration-results", required=True, help="Path to selector_exploration_results.json.")
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Mapping in the form point_dir_name=strategy_id.",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Pack names to summarize. Defaults to task_compare,longbench_mini,longbench_lb21_16_smoke_20260406.",
    )
    parser.add_argument(
        "--acceptable-dense-gap",
        type=float,
        default=0.02,
        help="Maximum allowed dense-match drop from the matched-quality point.",
    )
    parser.add_argument(
        "--acceptable-conditioned-accuracy-gap",
        type=float,
        default=0.02,
        help="Maximum allowed conditioned-accuracy drop from the matched-quality point.",
    )
    parser.add_argument(
        "--acceptable-score-gap",
        type=float,
        default=0.02,
        help="Maximum allowed official-score drop from the matched-quality point.",
    )
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Qwen3.5-9B CUDA Selector Operating Points")
    return parser.parse_args()


def _parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit(f"invalid --point mapping {value!r}; expected point_dir_name=strategy_id")
    point_id, strategy_id = value.split("=", 1)
    return point_id.strip(), strategy_id.strip()


def _fmt_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _load_exploration_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {str(row["strategy_id"]): row for row in payload.get("strategies", [])}


def _task_paths(pack_dir: Path) -> tuple[Path, Path]:
    return pack_dir / "task_selector_compare.json", pack_dir / "qwen35_9b_task_selector_compare.jsonl"


def _longbench_paths(pack_dir: Path) -> tuple[Path, Path]:
    json_path = pack_dir / "longbench_selector_compare.json"
    matches = sorted(pack_dir.glob("qwen3p5-9b_longbench_*.jsonl"))
    if len(matches) != 1:
        raise SystemExit(f"expected one longbench jsonl in {pack_dir}, found {len(matches)}")
    return json_path, matches[0]


def _task_group_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["task_name"]), int(row["prompt_length_requested"])


def _task_generated_cleaned(row: dict[str, Any]) -> str:
    cleaned = row.get("task_generated_text_cleaned")
    if cleaned is not None:
        return str(cleaned).strip()
    return str(row.get("task_generated_text") or "").strip()


def _task_decode_ms(row: dict[str, Any]) -> float | None:
    value = row.get("task_decode_ms_per_step")
    if value is None:
        value = row.get("dotcache_decode_ms_per_step")
    if value is None:
        value = row.get("dense_decode_ms_per_step")
    if value is None:
        return None
    return float(value)


def _task_effective_bytes_per_token(row: dict[str, Any]) -> float | None:
    resident = row.get("resident_bytes")
    prompt_length = row.get("prompt_length_requested")
    if resident is None or prompt_length in (None, 0):
        return None
    return float(resident) / float(prompt_length)


def _task_profile_metrics(raw_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate_rows = [row for row in raw_rows if row.get("measurement_kind") == "aggregate"]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate_rows:
        profile = str(row.get("selector_profile") or "")
        if profile:
            grouped[_task_group_key(row)][profile] = row

    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_match_rates: dict[str, list[float]] = defaultdict(list)
    dense_conditioned_success: dict[str, list[float]] = defaultdict(list)
    dense_match_rates: dict[str, list[float]] = defaultdict(list)
    for per_group in grouped.values():
        dense_row = per_group.get("dense")
        exact_row = per_group.get("exact")
        for profile, row in per_group.items():
            by_profile[profile].append(row)
            if dense_row is not None:
                dense_match_rates[profile].append(float(_task_generated_cleaned(row) == _task_generated_cleaned(dense_row)))
                if float(dense_row.get("task_metric_value", 0.0) or 0.0) >= 0.5:
                    dense_conditioned_success[profile].append(float(row.get("task_metric_value", 0.0) or 0.0))
            if exact_row is not None:
                exact_match_rates[profile].append(float(_task_generated_cleaned(row) == _task_generated_cleaned(exact_row)))

    output: dict[str, dict[str, Any]] = {}
    for profile in TASK_PROFILES:
        rows = by_profile.get(profile, [])
        if not rows:
            continue
        decode_values = [_task_decode_ms(row) for row in rows if _task_decode_ms(row) is not None]
        resident_values = [float(row["resident_bytes"]) for row in rows if row.get("resident_bytes") is not None]
        effective_values = [
            _task_effective_bytes_per_token(row)
            for row in rows
            if _task_effective_bytes_per_token(row) is not None
        ]
        output[profile] = {
            "n_rows": len(rows),
            "task_success_rate": float(mean(float(row.get("task_metric_value", 0.0) or 0.0) for row in rows)),
            "dense_match_rate": None if not dense_match_rates.get(profile) else float(mean(dense_match_rates[profile])),
            "accuracy_when_dense_correct": (
                None
                if not dense_conditioned_success.get(profile)
                else float(mean(dense_conditioned_success[profile]))
            ),
            "error_rate_vs_exact": (
                None
                if not exact_match_rates.get(profile)
                else float(1.0 - mean(exact_match_rates[profile]))
            ),
            "decode_ms_per_step": None if not decode_values else float(mean(decode_values)),
            "resident_bytes": None if not resident_values else float(mean(resident_values)),
            "effective_bytes_per_token": None if not effective_values else float(mean(effective_values)),
            "mean_v_m0_pages": float(mean(float(row.get("v_m0_pages") or 0.0) for row in rows)),
            "mean_v_m3_pages": float(mean(float(row.get("v_m3_pages") or 0.0) for row in rows)),
            "fit_status": "fit",
        }
    return output


def _longbench_prompt_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["comparison_max_prompt_tokens"]), str(row["evaluation_prompt_id"])


def _longbench_group_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["comparison_max_prompt_tokens"]),
        str(row["evaluation_prompt_id"]),
        str(row["comparison_case"]),
    )


def _longbench_generated_cleaned(row: dict[str, Any]) -> str:
    cleaned = row.get("comparison_generated_text_cleaned")
    if cleaned is not None:
        return str(cleaned).strip()
    cleaned = row.get("longbench_generated_text_cleaned")
    if cleaned is not None:
        return str(cleaned).strip()
    return ""


def _longbench_exact_match(row: dict[str, Any]) -> float:
    if row.get("comparison_answer_exact_match_cleaned") is not None:
        return float(1.0 if row.get("comparison_answer_exact_match_cleaned") else 0.0)
    return float(1.0 if row.get("longbench_answer_exact_match_cleaned") else 0.0)


def _longbench_official_score(row: dict[str, Any]) -> float | None:
    if row.get("comparison_official_score") is not None:
        return float(row["comparison_official_score"])
    if str(row.get("comparison_case") or "") == "dense":
        return None
    if row.get("longbench_official_score") is not None:
        return float(row["longbench_official_score"])
    return None


def _longbench_decode_ms(row: dict[str, Any]) -> float | None:
    if row.get("comparison_decode_ms_per_step") is not None:
        return float(row["comparison_decode_ms_per_step"])
    if row.get("dotcache_decode_ms_per_step") is not None:
        return float(row["dotcache_decode_ms_per_step"])
    return None


def _longbench_effective_bytes_per_token(row: dict[str, Any]) -> float | None:
    if row.get("effective_bytes_per_token") is not None:
        return float(row["effective_bytes_per_token"])
    resident = row.get("resident_bytes")
    prompt_length = row.get("prompt_length")
    if resident is None or prompt_length in (None, 0):
        return None
    return float(resident) / float(prompt_length)


def _longbench_profile_metrics(raw_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate_rows = [row for row in raw_rows if row.get("measurement_kind") == "aggregate"]
    dense_by_prompt = {
        _longbench_prompt_key(row): row
        for row in aggregate_rows
        if str(row.get("comparison_case") or "") == "dense"
    }
    exact_by_prompt = {
        _longbench_prompt_key(row): row
        for row in aggregate_rows
        if str(row.get("comparison_case") or "") == "exact"
    }

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dense_match_rates: dict[str, list[float]] = defaultdict(list)
    conditioned_exact_match: dict[str, list[float]] = defaultdict(list)
    exact_match_rates: dict[str, list[float]] = defaultdict(list)
    for row in aggregate_rows:
        case = str(row.get("comparison_case") or "")
        if not case:
            continue
        by_case[case].append(row)
        dense_row = dense_by_prompt.get(_longbench_prompt_key(row))
        exact_row = exact_by_prompt.get(_longbench_prompt_key(row))
        if dense_row is not None:
            dense_match_rates[case].append(float(_longbench_generated_cleaned(row) == _longbench_generated_cleaned(dense_row)))
            if _longbench_exact_match(dense_row) >= 0.5:
                conditioned_exact_match[case].append(_longbench_exact_match(row))
        if exact_row is not None:
            exact_match_rates[case].append(float(_longbench_generated_cleaned(row) == _longbench_generated_cleaned(exact_row)))

    output: dict[str, dict[str, Any]] = {}
    for case in LONGBENCH_CASES:
        rows = by_case.get(case, [])
        if not rows:
            continue
        decode_values = [_longbench_decode_ms(row) for row in rows if _longbench_decode_ms(row) is not None]
        resident_values = [float(row["resident_bytes"]) for row in rows if row.get("resident_bytes") is not None]
        effective_values = [
            _longbench_effective_bytes_per_token(row)
            for row in rows
            if _longbench_effective_bytes_per_token(row) is not None
        ]
        official_scores = [_longbench_official_score(row) for row in rows if _longbench_official_score(row) is not None]
        output[case] = {
            "n_rows": len(rows),
            "official_score": None if not official_scores else float(mean(official_scores)),
            "dense_match_rate": None if not dense_match_rates.get(case) else float(mean(dense_match_rates[case])),
            "accuracy_when_dense_correct": (
                None
                if not conditioned_exact_match.get(case)
                else float(mean(conditioned_exact_match[case]))
            ),
            "error_rate_vs_exact": (
                None
                if not exact_match_rates.get(case)
                else float(1.0 - mean(exact_match_rates[case]))
            ),
            "decode_ms_per_step": None if not decode_values else float(mean(decode_values)),
            "resident_bytes": None if not resident_values else float(mean(resident_values)),
            "effective_bytes_per_token": None if not effective_values else float(mean(effective_values)),
            "mean_v_m0_pages": float(mean(float(row.get("v_m0_pages") or 0.0) for row in rows)),
            "mean_v_m3_pages": float(mean(float(row.get("v_m3_pages") or 0.0) for row in rows)),
            "fit_status": "fit",
        }
    return output


def _extract_floor_label(strategy_row: dict[str, Any]) -> str:
    calibration = (strategy_row.get("model_summary") or {}).get("calibration")
    if not calibration:
        return "weighted"
    min_target = calibration.get("min_target_accuracy")
    min_safe = calibration.get("min_safe_prediction_rate")
    if min_target is None or min_safe is None:
        return "weighted"
    if float(min_target) == float(min_safe):
        return f"{float(min_target):.2f}"
    return f"{float(min_target):.2f}/{float(min_safe):.2f}"


def _resolve_pack_metrics(pack_name: str, pack_dir: Path) -> dict[str, Any]:
    if pack_name == "task_compare":
        _json_path, jsonl_path = _task_paths(pack_dir)
        raw_rows = _load_jsonl(jsonl_path)
        return {"profiles": _task_profile_metrics(raw_rows)}
    _json_path, jsonl_path = _longbench_paths(pack_dir)
    raw_rows = _load_jsonl(jsonl_path)
    return {"profiles": _longbench_profile_metrics(raw_rows)}


def _pack_status(pack_name: str, pack_dir: Path) -> dict[str, Any]:
    if not pack_dir.exists():
        return {"state": "missing", "detail": "pack directory missing", "shard_line_counts": {}}

    if pack_name == "task_compare":
        json_path, jsonl_path = _task_paths(pack_dir)
        if json_path.exists() and jsonl_path.exists():
            return {"state": "complete", "detail": "task compare artifacts present", "shard_line_counts": {}}
        return {"state": "incomplete", "detail": "task compare outputs missing", "shard_line_counts": {}}

    merged_matches = sorted(pack_dir.glob("qwen3p5-9b_longbench_*.jsonl"))
    compare_json = pack_dir / "longbench_selector_compare.json"
    workbook_json = pack_dir / "longbench_failure_workbook.json"
    if len(merged_matches) == 1 and compare_json.exists() and workbook_json.exists():
        return {"state": "complete", "detail": "merged longbench outputs present", "shard_line_counts": {}}

    shard_dir = pack_dir / "shards"
    shard_line_counts: dict[str, int] = {}
    if shard_dir.exists():
        for shard_path in sorted(shard_dir.glob("shard_*.jsonl")):
            shard_line_counts[shard_path.name] = sum(1 for _ in shard_path.open("r", encoding="utf-8"))
    if shard_line_counts:
        detail = ", ".join(f"{name}={count}" for name, count in shard_line_counts.items())
        return {"state": "partial", "detail": detail, "shard_line_counts": shard_line_counts}
    return {"state": "missing", "detail": "longbench outputs missing", "shard_line_counts": {}}


def _primary_metrics(point_payload: dict[str, Any], pack_name: str, profile_name: str) -> dict[str, Any]:
    return ((point_payload.get("packs") or {}).get(pack_name) or {}).get("profiles", {}).get(profile_name, {})


def _primary_status(point_payload: dict[str, Any], pack_name: str) -> str:
    return str(((point_payload.get("packs") or {}).get(pack_name) or {}).get("status", {}).get("state") or "missing")


def _point_sort_key(point_payload: dict[str, Any]) -> tuple[float, float, float, str]:
    primary = _primary_metrics(point_payload, PRIMARY_PACK, PRIMARY_PROFILE)
    dense_match = float(primary.get("dense_match_rate") or -1.0)
    conditioned = float(primary.get("accuracy_when_dense_correct") or -1.0)
    official = float(primary.get("official_score") or -1.0)
    point_id = str(point_payload.get("point_id") or "")
    return (-dense_match, -conditioned, -official, point_id)


def _choose_matched_quality(points: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [point for point in points if _primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE)]
    if not valid:
        raise SystemExit("no valid primary-pack metrics available for matched-quality selection")
    return min(valid, key=_point_sort_key)


def _is_acceptable(
    point_payload: dict[str, Any],
    matched_payload: dict[str, Any],
    *,
    dense_gap: float,
    conditioned_gap: float,
    score_gap: float,
) -> bool:
    point_primary = _primary_metrics(point_payload, PRIMARY_PACK, PRIMARY_PROFILE)
    matched_primary = _primary_metrics(matched_payload, PRIMARY_PACK, PRIMARY_PROFILE)
    if not point_primary or not matched_primary:
        return False
    point_dense = point_primary.get("dense_match_rate")
    point_conditioned = point_primary.get("accuracy_when_dense_correct")
    point_score = point_primary.get("official_score")
    matched_dense = matched_primary.get("dense_match_rate")
    matched_conditioned = matched_primary.get("accuracy_when_dense_correct")
    matched_score = matched_primary.get("official_score")
    if point_dense is None or point_conditioned is None or point_score is None:
        return False
    if matched_dense is None or matched_conditioned is None or matched_score is None:
        return False
    return (
        float(point_dense) >= float(matched_dense) - dense_gap
        and float(point_conditioned) >= float(matched_conditioned) - conditioned_gap
        and float(point_score) >= float(matched_score) - score_gap
    )


def _select_smallest_memory(points: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        points,
        key=lambda point: (
            float(_primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE).get("effective_bytes_per_token") or float("inf")),
            float(_primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE).get("resident_bytes") or float("inf")),
            point.get("point_id", ""),
        ),
    )


def _select_fastest(points: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        points,
        key=lambda point: (
            float(_primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE).get("decode_ms_per_step") or float("inf")),
            float(_primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE).get("effective_bytes_per_token") or float("inf")),
            point.get("point_id", ""),
        ),
    )


def _build_pack_table(points: list[dict[str, Any]], pack_name: str, profile_name: str) -> str:
    table = [[
        "point",
        "floor",
        "status",
        "dense_match_rate",
        "accuracy_when_dense_correct",
        "score",
        "error_vs_exact",
        "decode_ms",
        "eff_bytes/token",
        "resident_mib",
        "v_m0",
        "v_m3",
        "fit",
    ]]
    for point in points:
        metrics = ((point.get("packs") or {}).get(pack_name) or {}).get("profiles", {}).get(profile_name, {})
        resident_mib = (
            None if metrics.get("resident_bytes") is None else float(metrics["resident_bytes"]) / (1024.0 * 1024.0)
        )
        score = metrics.get("task_success_rate")
        if score is None:
            score = metrics.get("official_score")
        table.append(
            [
                str(point["point_id"]),
                str(point["floor"]),
                str(((point.get("packs") or {}).get(pack_name) or {}).get("status", {}).get("state") or "-"),
                _fmt_float(metrics.get("dense_match_rate")),
                _fmt_float(metrics.get("accuracy_when_dense_correct")),
                _fmt_float(score),
                _fmt_float(metrics.get("error_rate_vs_exact")),
                _fmt_float(metrics.get("decode_ms_per_step"), digits=1),
                _fmt_float(metrics.get("effective_bytes_per_token"), digits=1),
                _fmt_float(resident_mib, digits=1),
                _fmt_float(metrics.get("mean_v_m0_pages"), digits=1),
                _fmt_float(metrics.get("mean_v_m3_pages"), digits=1),
                str(metrics.get("fit_status") or "-"),
            ]
        )
    return _markdown_table(table)


def build_report(
    *,
    title: str,
    points: list[dict[str, Any]],
    packs: tuple[str, ...],
    acceptable_dense_gap: float,
    acceptable_conditioned_gap: float,
    acceptable_score_gap: float,
) -> tuple[dict[str, Any], str]:
    ordered_points = sorted(points, key=lambda point: str(point["point_id"]))
    incomplete_points = [
        point["point_id"]
        for point in ordered_points
        if _primary_status(point, PRIMARY_PACK) != "complete"
    ]
    matched_quality = _choose_matched_quality(ordered_points)
    acceptable_points = [
        point
        for point in ordered_points
        if _is_acceptable(
            point,
            matched_quality,
            dense_gap=acceptable_dense_gap,
            conditioned_gap=acceptable_conditioned_gap,
            score_gap=acceptable_score_gap,
        )
    ]
    if not acceptable_points:
        acceptable_points = [matched_quality]
    smallest_memory = _select_smallest_memory(acceptable_points)
    fastest = _select_fastest(acceptable_points)
    recommendation = fastest

    payload = {
        "primary_pack": PRIMARY_PACK,
        "primary_profile": PRIMARY_PROFILE,
        "incomplete_points": incomplete_points,
        "matched_quality_point": matched_quality["point_id"],
        "smallest_memory_acceptable_point": smallest_memory["point_id"],
        "fastest_acceptable_point": fastest["point_id"],
        "recommended_default_point": recommendation["point_id"],
        "acceptable_points": [point["point_id"] for point in acceptable_points],
        "points": ordered_points,
    }

    summary_table = [[
        "selection",
        "point",
        "floor",
        "dense_match_rate",
        "accuracy_when_dense_correct",
        "official_score",
        "decode_ms",
        "eff_bytes/token",
    ]]
    for label, point in (
        ("matched_quality", matched_quality),
        ("smallest_memory_acceptable", smallest_memory),
        ("fastest_acceptable", fastest),
        ("recommended_default", recommendation),
    ):
        primary = _primary_metrics(point, PRIMARY_PACK, PRIMARY_PROFILE)
        summary_table.append(
            [
                label,
                str(point["point_id"]),
                str(point["floor"]),
                _fmt_float(primary.get("dense_match_rate")),
                _fmt_float(primary.get("accuracy_when_dense_correct")),
                _fmt_float(primary.get("official_score")),
                _fmt_float(primary.get("decode_ms_per_step"), digits=1),
                _fmt_float(primary.get("effective_bytes_per_token"), digits=1),
            ]
        )

    lines = [
        f"# {title}",
        "",
        (
            "This is a partial operating-points summary. Any point or pack marked `partial` or `missing` "
            "did not finish and is reported without final metrics."
        ),
        "",
        "## Recommendation",
        "",
        f"- Matched-quality point: `{matched_quality['point_id']}`",
        f"- Smallest-memory acceptable point: `{smallest_memory['point_id']}`",
        f"- Fastest acceptable point: `{fastest['point_id']}`",
        f"- Recommended CUDA default: `{recommendation['point_id']}`",
        "",
        _markdown_table(summary_table),
        "",
        "## Completion Status",
        "",
    ]

    status_table = [[
        "point",
        "pack",
        "status",
        "detail",
    ]]
    for point in ordered_points:
        for pack_name in packs:
            status = ((point.get("packs") or {}).get(pack_name) or {}).get("status", {})
            status_table.append(
                [
                    str(point["point_id"]),
                    pack_name,
                    str(status.get("state") or "missing"),
                    str(status.get("detail") or "-"),
                ]
            )
    lines.extend(
        [
            _markdown_table(status_table),
            "",
        "## Distinct Operating Points",
        "",
        ]
    )

    distinct_table = [[
        "point",
        "strategy_id",
        "floor",
        "offline_safe",
        "offline_target",
        "offline_bytes",
    ]]
    for point in ordered_points:
        aggregate = point.get("offline_metrics") or {}
        distinct_table.append(
            [
                str(point["point_id"]),
                str(point["strategy_id"]),
                str(point["floor"]),
                _fmt_float(aggregate.get("min_family_safe_prediction_rate")),
                _fmt_float(aggregate.get("min_family_target_accuracy")),
                _fmt_float(aggregate.get("mean_predicted_total_bytes"), digits=1),
            ]
        )
    lines.extend([_markdown_table(distinct_table)])

    for pack_name in packs:
        lines.extend(
            [
                "",
                f"## {pack_name} Systems",
                "",
                _build_pack_table(ordered_points, pack_name, "systems"),
                "",
                f"## {pack_name} Quality",
                "",
                _build_pack_table(ordered_points, pack_name, "quality"),
            ]
        )

    recommendation_primary = _primary_metrics(recommendation, PRIMARY_PACK, PRIMARY_PROFILE)
    lines.extend(
        [
            "",
            "## Paper Note",
            "",
            (
                "Dense preservation remains the control objective. The recommended CUDA default is "
                f"`{recommendation['point_id']}` because it stays inside the acceptable dense-preservation envelope "
                "on the stronger LongBench smoke pack while minimizing serving latency."
            ),
            (
                "If the paper wants the cleanest matched-quality row, use "
                f"`{matched_quality['point_id']}`. If the paper wants the clearest memory story, use "
                f"`{smallest_memory['point_id']}` alongside the recommended default."
            ),
            (
                "Primary recommendation metrics on the stronger smoke pack: dense_match_rate="
                f"{_fmt_float(recommendation_primary.get('dense_match_rate'))}, "
                "accuracy_when_dense_correct="
                f"{_fmt_float(recommendation_primary.get('accuracy_when_dense_correct'))}, "
                "official_score="
                f"{_fmt_float(recommendation_primary.get('official_score'))}, "
                "decode_ms="
                f"{_fmt_float(recommendation_primary.get('decode_ms_per_step'), digits=1)}, "
                "effective_bytes_per_token="
                f"{_fmt_float(recommendation_primary.get('effective_bytes_per_token'), digits=1)}."
            ),
        ]
    )

    return payload, "\n".join(lines)


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)
    exploration_rows = _load_exploration_rows(Path(args.exploration_results))
    packs = tuple(args.pack or DEFAULT_PACKS)
    point_mappings = [_parse_mapping(value) for value in args.point]
    if not point_mappings:
        raise SystemExit("at least one --point mapping is required")

    points: list[dict[str, Any]] = []
    for point_id, strategy_id in point_mappings:
        strategy_row = exploration_rows.get(strategy_id)
        if strategy_row is None:
            raise SystemExit(f"strategy_id {strategy_id!r} not found in exploration results")
        point_dir = base_dir / point_id
        pack_payloads: dict[str, Any] = {}
        for pack_name in packs:
            pack_dir = point_dir / pack_name
            status = _pack_status(pack_name, pack_dir)
            pack_payload = {"status": status, "profiles": {}}
            if status["state"] == "complete":
                pack_payload.update(_resolve_pack_metrics(pack_name, pack_dir))
            pack_payloads[pack_name] = pack_payload
        points.append(
            {
                "point_id": point_id,
                "strategy_id": strategy_id,
                "floor": _extract_floor_label(strategy_row),
                "artifact_path": strategy_row.get("artifact_path"),
                "offline_metrics": strategy_row.get("aggregate_metrics") or {},
                "packs": pack_payloads,
            }
        )

    payload, markdown = build_report(
        title=str(args.title),
        points=points,
        packs=packs,
        acceptable_dense_gap=float(args.acceptable_dense_gap),
        acceptable_conditioned_gap=float(args.acceptable_conditioned_accuracy_gap),
        acceptable_score_gap=float(args.acceptable_score_gap),
    )
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
