#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.persistent_predictor import (
    PERSISTENT_PREDICTOR_FEATURE_NAMES,
    evaluate_persistent_residual_predictor,
    recommend_safe_then_cheapest_configs,
    save_persistent_residual_predictor_model,
    split_predictor_records,
    train_persistent_residual_predictor,
)

SHORTLIST_PROFILES: dict[str, tuple[str, ...]] = {
    "current_longdecode_best": (
        "baseline_none_longdecode.json",
        "history_ema05_longdecode.json",
        "history_ema05_diversity_gated_longdecode.json",
        "history_ema05_diversity_sched12_longdecode.json",
        "history_ema05_div_sched12_anchor0_longdecode.json",
        "history_ema05_div_sched12_anchor4_trigger_p025_u0_longdecode.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and apply a shortlist-safe-then-cheapest offline recommender.")
    parser.add_argument("--compare-inputs", nargs="*", default=[])
    parser.add_argument("--compare-glob", default=None)
    parser.add_argument("--compare-dir", default=None)
    parser.add_argument("--shortlist-profile", choices=sorted(SHORTLIST_PROFILES.keys()), default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--abs-threshold", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    resolved_paths = [Path(path).resolve() for path in args.compare_inputs]
    if args.compare_glob:
        resolved_paths.extend(Path(path).resolve() for path in glob.glob(args.compare_glob))
    if args.shortlist_profile:
        if not args.compare_dir:
            raise SystemExit("--compare-dir is required with --shortlist-profile")
        compare_dir = Path(args.compare_dir).resolve()
        for filename in SHORTLIST_PROFILES[args.shortlist_profile]:
            resolved_paths.append((compare_dir / filename).resolve())
    unique_paths = sorted(set(resolved_paths))
    if not unique_paths:
        raise SystemExit("no compare inputs resolved")
    return unique_paths


def _load_records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            enriched = dict(record)
            enriched["source_compare_json"] = str(path)
            records.append(enriched)
    return records


def _render_markdown(summary: dict[str, object]) -> str:
    train_binary = dict(summary["train_binary_metrics"])
    test_binary = dict(summary["test_binary_metrics"])
    test_reco = dict(summary["test_recommendation_summary"])
    full_reco = dict(summary["full_recommendation_summary"])
    lines = [
        "# Persistent Shortlist Recommender",
        "",
        f"- profile: {summary.get('shortlist_profile') or 'custom'}",
        f"- compare inputs: {int(summary['compare_input_count'])}",
        f"- total records: {int(summary['record_count'])}",
        f"- train records: {int(summary['train_record_count'])}",
        f"- test records: {int(summary['test_record_count'])}",
        f"- target abs threshold: {float(summary['abs_threshold']):.4f}",
        f"- decision threshold: {float(summary['decision_threshold']):.3f}",
        "",
        "## Binary Safety Model",
        "",
        f"- train accuracy: {float(train_binary['accuracy']):.3f}",
        f"- test accuracy: {float(test_binary['accuracy']):.3f}",
        f"- train f1: {float(train_binary['f1']):.3f}",
        f"- test f1: {float(test_binary['f1']):.3f}",
        "",
        "## Recommendation Quality",
        "",
        f"- test top-1 accuracy: {float(test_reco['top1_accuracy']):.3f}",
        f"- test chosen-safe rate: {float(test_reco['chosen_safe_rate']):.3f}",
        f"- test avg selected tokens: {float(test_reco['avg_selected_token_count']):.1f}",
        f"- test oracle avg selected tokens: {float(test_reco['avg_oracle_selected_token_count']):.1f}",
        "",
        "## Full Recommendation Pass",
        "",
        f"- full top-1 agreement vs oracle: {float(full_reco['top1_accuracy']):.3f}",
        f"- full chosen-safe rate: {float(full_reco['chosen_safe_rate']):.3f}",
        f"- full avg selected tokens: {float(full_reco['avg_selected_token_count']):.1f}",
        "",
        "## Inputs",
        "",
    ]
    for path in summary["compare_inputs"]:
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_paths = _resolve_input_paths(args)
    records = _load_records(input_paths)
    split = split_predictor_records(records, test_fraction=float(args.test_fraction))

    eval_model = train_persistent_residual_predictor(
        split["train_records"],
        abs_threshold=float(args.abs_threshold),
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )
    final_model = train_persistent_residual_predictor(
        records,
        abs_threshold=float(args.abs_threshold),
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )

    train_binary = evaluate_persistent_residual_predictor(eval_model, split["train_records"])
    test_binary = evaluate_persistent_residual_predictor(eval_model, split["test_records"])
    test_recommendations = recommend_safe_then_cheapest_configs(
        eval_model,
        split["test_records"],
        abs_threshold=float(args.abs_threshold),
    )
    full_recommendations = recommend_safe_then_cheapest_configs(
        final_model,
        records,
        abs_threshold=float(args.abs_threshold),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "persistent_shortlist_recommender_model.json"
    summary_path = output_dir / "persistent_shortlist_recommender_summary.json"
    markdown_path = output_dir / "persistent_shortlist_recommender_summary.md"
    test_recommendations_path = output_dir / "persistent_shortlist_test_recommendations.json"
    full_recommendations_path = output_dir / "persistent_shortlist_full_recommendations.json"
    save_persistent_residual_predictor_model(final_model, model_path)
    test_recommendations_path.write_text(json.dumps(test_recommendations, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    full_recommendations_path.write_text(json.dumps(full_recommendations, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    summary = {
        "artifact_path": str(model_path),
        "shortlist_profile": args.shortlist_profile,
        "compare_input_count": int(len(input_paths)),
        "compare_inputs": [str(path) for path in input_paths],
        "record_count": int(len(records)),
        "train_record_count": int(len(split["train_records"])),
        "test_record_count": int(len(split["test_records"])),
        "abs_threshold": float(args.abs_threshold),
        "decision_threshold": float(eval_model.decision_threshold),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "l2": float(args.l2),
        "feature_names": list(eval_model.feature_names),
        "train_binary_metrics": train_binary,
        "test_binary_metrics": test_binary,
        "test_recommendation_summary": test_recommendations["summary"],
        "full_recommendation_summary": full_recommendations["summary"],
        "test_recommendations_path": str(test_recommendations_path),
        "full_recommendations_path": str(full_recommendations_path),
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(model_path)
    print(summary_path)
    print(markdown_path)
    print(test_recommendations_path)
    print(full_recommendations_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
