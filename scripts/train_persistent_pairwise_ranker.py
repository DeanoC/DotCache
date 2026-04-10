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
    evaluate_persistent_pairwise_ranker,
    save_persistent_pairwise_ranker_model,
    split_predictor_records,
    train_persistent_pairwise_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline pairwise config ranker on persistent replay compare outputs.")
    parser.add_argument("--compare-inputs", nargs="*", default=[])
    parser.add_argument("--compare-glob", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def _load_records(compare_inputs: list[str], compare_glob: str | None) -> tuple[list[dict[str, object]], list[str]]:
    resolved_paths = [Path(path).resolve() for path in compare_inputs]
    if compare_glob:
        resolved_paths.extend(Path(path).resolve() for path in glob.glob(compare_glob))
    unique_paths = sorted(set(resolved_paths))
    records: list[dict[str, object]] = []
    for path in unique_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            enriched = dict(record)
            enriched["source_compare_json"] = str(path)
            records.append(enriched)
    return records, [str(path) for path in unique_paths]


def _render_markdown(summary: dict[str, object]) -> str:
    train_metrics = dict(summary["train_metrics"])
    test_metrics = dict(summary["test_metrics"])
    lines = [
        "# Persistent Pairwise Ranker",
        "",
        f"- compare inputs: {int(summary['compare_input_count'])}",
        f"- total records: {int(summary['record_count'])}",
        f"- train records: {int(summary['train_record_count'])}",
        f"- test records: {int(summary['test_record_count'])}",
        "",
        "## Train",
        "",
        f"- pair accuracy: {float(train_metrics['pair_accuracy']):.3f}",
        f"- top-1 accuracy: {float(train_metrics['top1_accuracy']):.3f}",
        f"- pair count: {int(train_metrics['pair_count'])}",
        f"- snapshot groups: {int(train_metrics['snapshot_group_count'])}",
        "",
        "## Test",
        "",
        f"- pair accuracy: {float(test_metrics['pair_accuracy']):.3f}",
        f"- top-1 accuracy: {float(test_metrics['top1_accuracy']):.3f}",
        f"- pair count: {int(test_metrics['pair_count'])}",
        f"- snapshot groups: {int(test_metrics['snapshot_group_count'])}",
        "",
        "## Features",
        "",
    ]
    for feature_name in summary["feature_names"]:
        lines.append(f"- `{feature_name}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    records, input_paths = _load_records(args.compare_inputs, args.compare_glob)
    if not records:
        raise SystemExit("no compare records found")
    split = split_predictor_records(records, test_fraction=float(args.test_fraction))
    model = train_persistent_pairwise_ranker(
        split["train_records"],
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )
    train_metrics = evaluate_persistent_pairwise_ranker(model, split["train_records"])
    test_metrics = evaluate_persistent_pairwise_ranker(model, split["test_records"])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "persistent_pairwise_ranker_model.json"
    summary_path = output_dir / "persistent_pairwise_ranker_summary.json"
    markdown_path = output_dir / "persistent_pairwise_ranker_summary.md"
    save_persistent_pairwise_ranker_model(model, artifact_path)

    summary = {
        "artifact_path": str(artifact_path),
        "compare_input_count": int(len(input_paths)),
        "compare_inputs": input_paths,
        "record_count": int(len(records)),
        "train_record_count": int(len(split["train_records"])),
        "test_record_count": int(len(split["test_records"])),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "l2": float(args.l2),
        "feature_names": list(model.feature_names),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(artifact_path)
    print(summary_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
