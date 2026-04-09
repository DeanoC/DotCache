from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.integrations.qwen35 import (
    PersistentServingConfig,
    run_qwen35_persistent_full_attention_snapshot_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare persistent full-attention selected-block exact decode against full exact decode.")
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--enable-priority", action="store_true")
    parser.add_argument("--sink-block-count", type=int, default=1)
    parser.add_argument("--recent-block-count", type=int, default=1)
    parser.add_argument("--exploration-blocks-per-region", type=int, default=1)
    parser.add_argument("--optional-top-k", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _resolve_snapshot_paths(*, snapshot_paths: list[str], snapshot_glob: str | None, manifest_path: str | None) -> list[Path]:
    resolved: set[Path] = set(Path(path).resolve() for path in snapshot_paths)
    if snapshot_glob:
        resolved.update(Path(path).resolve() for path in glob.glob(snapshot_glob))
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for record in manifest.get("snapshot_records", []):
            resolved.add(Path(record["paged_attention_snapshot_path"]).resolve())
    return sorted(resolved)


def _build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "snapshot_count": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "avg_selected_block_count": 0.0,
            "avg_selected_token_count": 0.0,
            "avg_full_block_count": 0.0,
            "avg_full_token_count": 0.0,
        }

    def _avg(field: str) -> float:
        return float(sum(float(record[field]) for record in records) / len(records))

    return {
        "snapshot_count": int(len(records)),
        "max_abs_error": float(max(float(record["max_abs_error"]) for record in records)),
        "max_rel_error": float(max(float(record["max_rel_error"]) for record in records)),
        "avg_selected_block_count": _avg("selected_block_count"),
        "avg_selected_token_count": _avg("selected_token_count"),
        "avg_full_block_count": _avg("full_block_count"),
        "avg_full_token_count": _avg("full_token_count"),
        "avg_selected_fraction": float(
            sum(float(record["selected_token_count"]) / max(float(record["full_token_count"]), 1.0) for record in records)
            / len(records)
        ),
    }


def main() -> None:
    args = parse_args()
    snapshot_paths = _resolve_snapshot_paths(
        snapshot_paths=[str(path) for path in args.snapshot_paths],
        snapshot_glob=args.snapshot_glob,
        manifest_path=args.manifest_path,
    )
    config = PersistentServingConfig(
        block_size=int(args.block_size),
        enable_priority=bool(args.enable_priority),
        full_attention_sink_block_count=int(args.sink_block_count),
        full_attention_recent_block_count=int(args.recent_block_count),
        full_attention_exploration_blocks_per_region=int(args.exploration_blocks_per_region),
        full_attention_optional_top_k=int(args.optional_top_k),
    )
    records = [
        run_qwen35_persistent_full_attention_snapshot_comparison(
            path,
            persistent_serving_config=config,
        )
        | {"snapshot_path": str(path)}
        for path in snapshot_paths
    ]
    payload = {
        "config": {
            "block_size": int(config.block_size),
            "enable_priority": bool(config.enable_priority),
            "sink_block_count": int(config.full_attention_sink_block_count),
            "recent_block_count": int(config.full_attention_recent_block_count),
            "exploration_blocks_per_region": int(config.full_attention_exploration_blocks_per_region),
            "optional_top_k": int(config.full_attention_optional_top_k),
        },
        "records": records,
        "summary": _build_summary(records),
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
