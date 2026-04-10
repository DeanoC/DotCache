#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.persistent_predictor import (
    build_persistent_shortlist_policy,
    evaluate_persistent_shortlist_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bucketed runtime-style policy table from persistent shortlist recommendations and replay-evaluate it.")
    parser.add_argument("--recommendations-json", required=True)
    parser.add_argument("--compare-inputs", nargs="*", default=[])
    parser.add_argument("--recommender-summary-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-by", default="layer_id,step_bucket")
    parser.add_argument("--abs-threshold", type=float, default=None)
    return parser.parse_args()


def _load_compare_records(compare_inputs: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path_str in compare_inputs:
        path = Path(path_str).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            enriched = dict(record)
            enriched["source_compare_json"] = str(path)
            records.append(enriched)
    return records


def _render_markdown(summary: dict[str, object]) -> str:
    evaluation = dict(summary["evaluation_summary"])
    lines = [
        "# Persistent Shortlist Policy",
        "",
        f"- group by: {', '.join(summary['group_by'])}",
        f"- policy groups: {int(summary['policy_group_count'])}",
        f"- compare inputs: {int(summary['compare_input_count'])}",
        f"- target abs threshold: {float(summary['abs_threshold']):.4f}",
        "",
        "## Replay Evaluation",
        "",
        f"- top-1 accuracy: {float(evaluation['top1_accuracy']):.3f}",
        f"- chosen-safe rate: {float(evaluation['chosen_safe_rate']):.3f}",
        f"- avg selected tokens: {float(evaluation['avg_selected_token_count']):.1f}",
        f"- oracle avg selected tokens: {float(evaluation['avg_oracle_selected_token_count']):.1f}",
        f"- fallback rate: {float(evaluation['fallback_rate']):.3f}",
        f"- missing bucket rate: {float(evaluation['missing_bucket_rate']):.3f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    recommender_summary = None
    compare_inputs = list(args.compare_inputs)
    abs_threshold = args.abs_threshold
    if args.recommender_summary_json:
        recommender_summary = json.loads(Path(args.recommender_summary_json).read_text(encoding="utf-8"))
        if not compare_inputs:
            compare_inputs = list(recommender_summary.get("compare_inputs", []))
        if abs_threshold is None:
            abs_threshold = float(recommender_summary.get("abs_threshold", 0.05))
    if abs_threshold is None:
        abs_threshold = 0.05

    recommendations_payload = json.loads(Path(args.recommendations_json).read_text(encoding="utf-8"))
    compare_records = _load_compare_records(compare_inputs)
    group_by = tuple(part.strip() for part in args.group_by.split(",") if part.strip())
    policy = build_persistent_shortlist_policy(recommendations_payload, group_by=group_by)
    evaluation = evaluate_persistent_shortlist_policy(policy, compare_records, abs_threshold=float(abs_threshold))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "persistent_shortlist_policy.json"
    evaluation_path = output_dir / "persistent_shortlist_policy_evaluation.json"
    summary_path = output_dir / "persistent_shortlist_policy_summary.json"
    markdown_path = output_dir / "persistent_shortlist_policy_summary.md"
    policy_path.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    evaluation_path.write_text(json.dumps(evaluation, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    summary = {
        "recommendations_json": str(Path(args.recommendations_json).resolve()),
        "recommender_summary_json": None if args.recommender_summary_json is None else str(Path(args.recommender_summary_json).resolve()),
        "compare_input_count": int(len(compare_inputs)),
        "compare_inputs": [str(Path(path).resolve()) for path in compare_inputs],
        "group_by": list(group_by),
        "policy_group_count": int(policy.get("group_count", 0)),
        "abs_threshold": float(abs_threshold),
        "policy_path": str(policy_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_summary": evaluation["summary"],
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(policy_path)
    print(evaluation_path)
    print(summary_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
