#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dotcache.page_oracle import materialize_oracle_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze train/test page-selector datasets from an existing oracle label bundle.")
    parser.add_argument("--input-dir", required=True, help="Directory containing labels.jsonl, selector_dataset.jsonl, and optional selector_candidate_dataset.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory where train/test split bundles should be written.")
    parser.add_argument("--split-name", default="heldout_split")
    parser.add_argument("--manifest-path", default=None, help="Optional manifest JSON path to append/update with this split entry.")
    parser.add_argument("--annotation", action="append", default=[], help="Optional key=value annotation to store in the split manifest.")
    parser.add_argument("--holdout-prompt-family", action="append", default=[])
    parser.add_argument("--holdout-prompt-variant", action="append", default=[])
    parser.add_argument("--holdout-layer", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    candidate_dataset_path = input_dir / "selector_candidate_dataset.jsonl"
    annotations: dict[str, str] = {}
    for token in args.annotation:
        if "=" not in token:
            raise SystemExit(f"--annotation must be key=value, got: {token}")
        key, value = token.split("=", 1)
        annotations[key.strip()] = value.strip()
    summary = materialize_oracle_dataset_split(
        labels_path=input_dir / "labels.jsonl",
        selector_dataset_path=input_dir / "selector_dataset.jsonl",
        selector_candidate_dataset_path=candidate_dataset_path if candidate_dataset_path.exists() else None,
        output_dir=args.output_dir,
        holdout_prompt_families=tuple(args.holdout_prompt_family) if args.holdout_prompt_family else None,
        holdout_prompt_variants=tuple(args.holdout_prompt_variant) if args.holdout_prompt_variant else None,
        holdout_layers=tuple(args.holdout_layer) if args.holdout_layer else None,
        split_name=args.split_name,
        manifest_path=args.manifest_path,
        annotations=annotations or None,
    )
    print(Path(args.output_dir) / "split_summary.json")
    print(summary.train_label_count)
    print(summary.test_label_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
