#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotcache.page_oracle import materialize_oracle_dataset_split_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize an ordered suite of frozen page-selector splits from a suite config JSON.")
    parser.add_argument("--input-dir", required=True, help="Directory containing labels.jsonl, selector_dataset.jsonl, and optional selector_candidate_dataset.jsonl.")
    parser.add_argument("--output-root", required=True, help="Directory where the split suite should be written.")
    parser.add_argument("--suite-config", required=True, help="Path to a JSON file with suite_name and splits[].")
    parser.add_argument("--manifest-path", default=None, help="Optional manifest JSON path. Defaults to <output-root>/split_manifest.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    suite_config = json.loads(Path(args.suite_config).read_text(encoding="utf-8"))
    candidate_dataset_path = input_dir / "selector_candidate_dataset.jsonl"
    manifest_path = Path(args.manifest_path) if args.manifest_path is not None else Path(args.output_root) / "split_manifest.json"

    result = materialize_oracle_dataset_split_suite(
        labels_path=input_dir / "labels.jsonl",
        selector_dataset_path=input_dir / "selector_dataset.jsonl",
        selector_candidate_dataset_path=candidate_dataset_path if candidate_dataset_path.exists() else None,
        output_root=args.output_root,
        suite_specs=suite_config.get("splits", []),
        suite_name=str(suite_config.get("suite_name", "selector_split_suite")),
        manifest_path=manifest_path,
        overwrite_manifest=True,
    )

    print(Path(args.output_root) / "split_suite_summary.json")
    print(manifest_path)
    print(result.split_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
