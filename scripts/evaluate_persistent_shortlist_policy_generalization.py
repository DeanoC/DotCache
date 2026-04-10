#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.persistent_predictor import (
    build_persistent_shortlist_policy,
    evaluate_persistent_shortlist_policy,
)


def prompt_family_from_snapshot_path(snapshot_path: str) -> str:
    return Path(snapshot_path).resolve().parent.name


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


def _group_by_key(group_by: list[str]) -> str:
    return "__".join(group_by)


def _aggregate_evaluation_summaries(summaries: list[dict[str, Any]]) -> dict[str, float]:
    if not summaries:
        return {
            "snapshot_group_count": 0,
            "top1_accuracy": 0.0,
            "chosen_safe_rate": 0.0,
            "avg_selected_token_count": 0.0,
            "avg_oracle_selected_token_count": 0.0,
            "fallback_rate": 0.0,
            "missing_bucket_rate": 0.0,
        }
    total = sum(int(item.get("snapshot_group_count", 0)) for item in summaries)
    if total <= 0:
        total = len(summaries)
    weighted = {}
    for key in (
        "top1_accuracy",
        "chosen_safe_rate",
        "avg_selected_token_count",
        "avg_oracle_selected_token_count",
        "fallback_rate",
        "missing_bucket_rate",
    ):
        weighted[key] = float(
            sum(float(item.get(key, 0.0)) * max(int(item.get("snapshot_group_count", 0)), 1) for item in summaries)
            / total
        )
    weighted["snapshot_group_count"] = int(total)
    return weighted


def evaluate_persistent_shortlist_policy_generalization(
    recommendations_payload: dict[str, Any],
    compare_records: list[dict[str, object]],
    *,
    group_bys: list[list[str]],
    abs_threshold: float,
) -> dict[str, Any]:
    recommendations = list(recommendations_payload.get("recommendations", []))
    families = sorted(
        {
            prompt_family_from_snapshot_path(str(item.get("snapshot_path", "")))
            for item in recommendations
            if item.get("snapshot_path")
        }
    )
    results: list[dict[str, Any]] = []
    for group_by in group_bys:
        heldout_runs: list[dict[str, Any]] = []
        for holdout_family in families:
            train_recommendations = [
                item
                for item in recommendations
                if prompt_family_from_snapshot_path(str(item.get("snapshot_path", ""))) != holdout_family
            ]
            heldout_records = [
                record
                for record in compare_records
                if prompt_family_from_snapshot_path(str(record.get("snapshot_path", ""))) == holdout_family
            ]
            policy = build_persistent_shortlist_policy(
                {"recommendations": train_recommendations},
                group_by=tuple(group_by),
            )
            evaluation = evaluate_persistent_shortlist_policy(
                policy,
                heldout_records,
                abs_threshold=float(abs_threshold),
            )
            heldout_runs.append(
                {
                    "holdout_prompt_family": holdout_family,
                    "train_snapshot_count": int(len(train_recommendations)),
                    "test_record_count": int(len(heldout_records)),
                    "policy": policy,
                    "evaluation": evaluation,
                }
            )
        results.append(
            {
                "group_by": list(group_by),
                "group_key": _group_by_key(group_by),
                "aggregate_summary": _aggregate_evaluation_summaries(
                    [run["evaluation"]["summary"] for run in heldout_runs]
                ),
                "heldout_runs": heldout_runs,
            }
        )
    return {
        "prompt_families": families,
        "result_count": int(len(results)),
        "results": results,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Persistent Shortlist Policy Generalization",
        "",
        f"- prompt families: {', '.join(summary['prompt_families'])}",
        f"- abs threshold: {float(summary['abs_threshold']):.4f}",
        "",
    ]
    for result in summary["results"]:
        aggregate = result["aggregate_summary"]
        lines.extend(
            [
                f"## {', '.join(result['group_by'])}",
                "",
                f"- policy groups (mean across held-out runs): {float(result['avg_policy_group_count']):.1f}",
                f"- top-1 accuracy: {float(aggregate['top1_accuracy']):.3f}",
                f"- chosen-safe rate: {float(aggregate['chosen_safe_rate']):.3f}",
                f"- avg selected tokens: {float(aggregate['avg_selected_token_count']):.1f}",
                f"- oracle avg selected tokens: {float(aggregate['avg_oracle_selected_token_count']):.1f}",
                f"- fallback rate: {float(aggregate['fallback_rate']):.3f}",
                f"- missing bucket rate: {float(aggregate['missing_bucket_rate']):.3f}",
                "",
            ]
        )
        if float(aggregate["missing_bucket_rate"]) > 0.0:
            lines.extend(
                [
                    "Note: non-zero missing-bucket rate means some or all held-out results are driven by fallback, not by policy matches.",
                    "",
                ]
            )
        lines.extend(
            [
                "| Holdout family | Groups | Top-1 | Safe rate | Fallback | Missing |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in result["heldout_runs"]:
            evaluation = run["evaluation_summary"]
            lines.append(
                "| {family} | {groups} | {top1:.3f} | {safe:.3f} | {fallback:.3f} | {missing:.3f} |".format(
                    family=run["holdout_prompt_family"],
                    groups=int(run["policy_group_count"]),
                    top1=float(evaluation["top1_accuracy"]),
                    safe=float(evaluation["chosen_safe_rate"]),
                    fallback=float(evaluation["fallback_rate"]),
                    missing=float(evaluation["missing_bucket_rate"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-prompt-family-out evaluation for persistent shortlist runtime policies."
    )
    parser.add_argument("--recommendations-json", required=True)
    parser.add_argument("--compare-inputs", nargs="*", default=[])
    parser.add_argument("--recommender-summary-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help="Comma-separated policy bucket fields. May be provided multiple times.",
    )
    parser.add_argument("--abs-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    group_bys = [
        [part.strip() for part in spec.split(",") if part.strip()]
        for spec in (args.group_by or [
            "layer_id,kv_head_id,step_bucket",
            "layer_id,kv_head_id,prompt_family,step_bucket",
        ])
    ]

    recommendations_payload = json.loads(Path(args.recommendations_json).read_text(encoding="utf-8"))
    compare_records = _load_compare_records(compare_inputs)
    payload = evaluate_persistent_shortlist_policy_generalization(
        recommendations_payload,
        compare_records,
        group_bys=group_bys,
        abs_threshold=float(abs_threshold),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_results: list[dict[str, Any]] = []
    for result in payload["results"]:
        group_dir = details_dir / result["group_key"]
        group_dir.mkdir(parents=True, exist_ok=True)
        policy_counts: list[int] = []
        heldout_summaries: list[dict[str, Any]] = []
        for run in result["heldout_runs"]:
            family_slug = run["holdout_prompt_family"]
            policy_path = group_dir / f"{family_slug}_policy.json"
            evaluation_path = group_dir / f"{family_slug}_evaluation.json"
            policy_path.write_text(json.dumps(run["policy"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
            evaluation_path.write_text(json.dumps(run["evaluation"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
            policy_counts.append(int(run["policy"]["group_count"]))
            heldout_summaries.append(
                {
                    "holdout_prompt_family": family_slug,
                    "policy_group_count": int(run["policy"]["group_count"]),
                    "policy_path": str(policy_path),
                    "evaluation_path": str(evaluation_path),
                    "evaluation_summary": run["evaluation"]["summary"],
                }
            )
        summary_results.append(
            {
                "group_by": list(result["group_by"]),
                "group_key": str(result["group_key"]),
                "avg_policy_group_count": float(sum(policy_counts) / max(len(policy_counts), 1)),
                "aggregate_summary": result["aggregate_summary"],
                "heldout_runs": heldout_summaries,
            }
        )

    summary = {
        "recommendations_json": str(Path(args.recommendations_json).resolve()),
        "recommender_summary_json": None if args.recommender_summary_json is None else str(Path(args.recommender_summary_json).resolve()),
        "compare_input_count": int(len(compare_inputs)),
        "compare_inputs": [str(Path(path).resolve()) for path in compare_inputs],
        "prompt_families": payload["prompt_families"],
        "abs_threshold": float(abs_threshold),
        "results": summary_results,
    }
    summary_json_path = output_dir / "persistent_shortlist_policy_generalization_summary.json"
    summary_md_path = output_dir / "persistent_shortlist_policy_generalization_summary.md"
    summary_json_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(summary_json_path)
    print(summary_md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
