#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotcache.page_oracle import load_oracle_dataset_split_manifest
from dotcache.selector_baselines import (
    discover_selector_split_dirs,
    run_selector_fixed_split_batch_bakeoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selector training/eval across a directory of frozen split bundles.")
    parser.add_argument("--split-root", default=None, help="Directory containing one or more frozen selector split bundles.")
    parser.add_argument("--split-manifest", default=None, help="Optional split manifest JSON emitted by materialize_page_selector_dataset_split.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where aggregate and per-split outputs should be written.")
    parser.add_argument("--linear-steps", type=int, default=400)
    parser.add_argument("--linear-learning-rate", type=float, default=0.2)
    parser.add_argument("--linear-l2", type=float, default=1e-3)
    return parser.parse_args()


def _write_prediction_block(handle, results: dict[str, dict], *, split_name: str) -> None:
    for baseline_name, summary in results.items():
        for prediction in summary.get("predictions", []):
            handle.write(
                json.dumps(
                    {
                        "split_name": split_name,
                        "baseline": baseline_name,
                        **prediction,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    args = parse_args()
    if (args.split_root is None) == (args.split_manifest is None):
        raise SystemExit("provide exactly one of --split-root or --split-manifest")
    manifest_payload = None
    if args.split_manifest is not None:
        manifest_payload = load_oracle_dataset_split_manifest(args.split_manifest)
        split_dirs = [Path(payload["split_dir"]) for payload in manifest_payload.get("splits", [])]
        if not split_dirs:
            raise SystemExit(f"no frozen split bundles listed in {args.split_manifest}")
    else:
        split_dirs = discover_selector_split_dirs(args.split_root)
        if not split_dirs:
            raise SystemExit(f"no frozen split bundles found under {args.split_root}")

    payload = run_selector_fixed_split_batch_bakeoff(
        split_dirs=split_dirs,
        linear_steps=args.linear_steps,
        linear_learning_rate=args.linear_learning_rate,
        linear_l2=args.linear_l2,
    )
    if manifest_payload is not None:
        payload["split_manifest"] = manifest_payload

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_split_dir = output_dir / "per_split"
    per_split_dir.mkdir(parents=True, exist_ok=True)

    for split_payload in payload["splits"]:
        split_name = str(split_payload["split_name"])
        split_output_dir = per_split_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        (split_output_dir / "selector_baseline_summary.json").write_text(
            json.dumps(split_payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (split_output_dir / "selector_baseline_summary.md").write_text(
            str(split_payload["summary_markdown"]) + "\n",
            encoding="utf-8",
        )

    summary_path = output_dir / "selector_split_batch_summary.json"
    markdown_path = output_dir / "selector_split_batch_summary.md"
    predictions_path = output_dir / "selector_split_batch_predictions.jsonl"
    summary_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_sections = [
        str(payload["summary_markdown"]),
        "",
        "## Aggregate",
        str(payload["aggregate_markdown"]),
    ]
    markdown_path.write_text("\n".join(markdown_sections) + "\n", encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for split_payload in payload["splits"]:
            _write_prediction_block(handle, split_payload["results"], split_name=str(split_payload["split_name"]))

    print(summary_path)
    print(markdown_path)
    print(predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
