#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ABS_THRESHOLDS = (1e-2, 1e-1, 5e-1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize persistent full-attention replay compare sweeps.")
    parser.add_argument("--compare-inputs", nargs="+", required=True, help="Paths to compare JSON payloads.")
    parser.add_argument("--markdown-output", required=True, help="Path to write markdown summary.")
    parser.add_argument("--json-output", required=True, help="Path to write distilled JSON summary.")
    parser.add_argument("--title", default="Qwen3.5 Persistent Full-Attention Replay Sweep")
    return parser.parse_args()


def _fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[list[str]]) -> str:
    def escape(cell: str) -> str:
        return str(cell).replace("|", "\\|")

    header = [escape(cell) for cell in rows[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(escape(cell) for cell in row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _config_label(config: dict[str, Any]) -> str:
    if not bool(config.get("enable_priority", False)):
        return "full_coverage"
    label = (
        "priority"
        f"_recent{int(config.get('recent_block_count', 0))}"
        f"_topk{int(config.get('optional_top_k', 0))}"
        f"_sink{int(config.get('sink_block_count', 0))}"
        f"_explore{int(config.get('exploration_blocks_per_region', 0))}"
    )
    history_mode = str(config.get("history_mode", "none"))
    if history_mode != "none":
        label += f"_hist{history_mode}"
    diversity_weight = float(config.get("optional_diversity_weight", 0.0))
    diversity_radius = int(config.get("optional_diversity_radius", 0))
    if diversity_weight > 0.0 and diversity_radius > 0:
        label += f"_div{diversity_weight:g}_r{diversity_radius}"
        if bool(config.get("optional_diversity_requires_history", False)):
            label += "_histgate"
        min_history = int(config.get("optional_diversity_min_history_count", 0))
        max_history = config.get("optional_diversity_max_history_count")
        if min_history > 0 or max_history is not None:
            if max_history is None:
                label += f"_h{min_history}plus"
            else:
                label += f"_h{min_history}to{int(max_history)}"
    return label


def _relative_snapshot_path(path: str) -> str:
    p = Path(path)
    if len(p.parts) >= 2:
        return str(Path(*p.parts[-2:]))
    return str(p)


def _threshold_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"abs_le_{threshold:g}": int(sum(1 for record in records if float(record["max_abs_error"]) <= threshold))
        for threshold in ABS_THRESHOLDS
    }


def _build_case_row(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    config = dict(payload["config"])
    summary = dict(payload["summary"])
    records = list(payload["records"])
    counts = _threshold_counts(records)
    return {
        "label": _config_label(config),
        "source_json": str(source_path),
        "config": config,
        "summary": summary,
        "threshold_counts": counts,
        "worst_records": [
            {
                "snapshot_path": _relative_snapshot_path(str(record["snapshot_path"])),
                "max_abs_error": float(record["max_abs_error"]),
                "max_rel_error": float(record["max_rel_error"]),
                "selected_token_count": int(record["selected_token_count"]),
                "full_token_count": int(record["full_token_count"]),
            }
            for record in sorted(records, key=lambda item: float(item["max_abs_error"]), reverse=True)[:5]
        ],
    }


def _build_report(compare_inputs: list[Path]) -> dict[str, Any]:
    rows = []
    for path in compare_inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_build_case_row(payload, path))

    rows.sort(key=lambda row: (row["label"] != "full_coverage", float(row["summary"]["max_abs_error"])))

    priority_rows = [row for row in rows if row["label"] != "full_coverage"]
    best_priority = (
        min(
            priority_rows,
            key=lambda row: (
                float(row["summary"]["max_abs_error"]),
                float(row["summary"].get("avg_max_abs_error", float("inf"))),
                float(row["summary"]["avg_selected_token_count"]),
            ),
        )
        if priority_rows
        else None
    )
    best_compression = min(priority_rows, key=lambda row: float(row["summary"]["avg_selected_token_count"])) if priority_rows else None

    return {
        "cases": rows,
        "best_priority_candidate": best_priority,
        "most_aggressive_candidate": best_compression,
    }


def _build_markdown(title: str, report: dict[str, Any]) -> str:
    cases = list(report["cases"])
    best_priority = report.get("best_priority_candidate")
    best_compression = report.get("most_aggressive_candidate")

    lines = [f"# {title}", ""]

    if best_priority is not None:
        summary = best_priority["summary"]
        lines.append(
            "Best priority candidate by worst-case error:"
            f" `{best_priority['label']}` keeps {_fmt(summary['avg_selected_fraction'], 3)} of tokens"
            f" ({_fmt(summary['avg_selected_token_count'], 1)} / {_fmt(summary['avg_full_token_count'], 1)})"
            f" with max abs error {_fmt(summary['max_abs_error'], 4)}."
        )
        lines.append("")

    if best_compression is not None:
        summary = best_compression["summary"]
        lines.append(
            "Most aggressive candidate:"
            f" `{best_compression['label']}` keeps {_fmt(summary['avg_selected_fraction'], 3)} of tokens"
            f" ({_fmt(summary['avg_selected_token_count'], 1)} / {_fmt(summary['avg_full_token_count'], 1)})"
            f" with max abs error {_fmt(summary['max_abs_error'], 4)}."
        )
        lines.append("")

    table = [["Config", "Sel Tokens", "Sel Frac", "Max Abs", "abs<=0.1", "abs<=0.5", "abs<=1.0"]]
    for case in cases:
        summary = case["summary"]
        counts = case["threshold_counts"]
        snapshot_count = int(summary["snapshot_count"])
        table.append(
            [
                str(case["label"]),
                _fmt(summary["avg_selected_token_count"], 1),
                _fmt(summary["avg_selected_fraction"], 3),
                _fmt(summary["max_abs_error"], 4),
                f"{counts['abs_le_0.1']}/{snapshot_count}",
                f"{counts['abs_le_0.5']}/{snapshot_count}",
                f"{counts['abs_le_1']}/{snapshot_count}",
            ]
        )
    lines.append("## Config Sweep")
    lines.append("")
    lines.append(_markdown_table(table))
    lines.append("")

    if best_priority is not None:
        lines.append(f"## Worst Slices For `{best_priority['label']}`")
        lines.append("")
        worst_rows = [["Snapshot", "Sel Tokens", "Max Abs", "Max Rel"]]
        for record in best_priority["worst_records"]:
            worst_rows.append(
                [
                    str(record["snapshot_path"]),
                    str(record["selected_token_count"]),
                    _fmt(record["max_abs_error"], 6),
                    _fmt(record["max_rel_error"], 3),
                ]
            )
        lines.append(_markdown_table(worst_rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    compare_inputs = [Path(path).resolve() for path in args.compare_inputs]
    report = _build_report(compare_inputs)
    markdown = _build_markdown(str(args.title), report)

    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown, encoding="utf-8")

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
