#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.integrations.qwen35 import (
    PersistentServingConfig,
    debug_qwen35_persistent_full_attention_snapshot_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug persistent full-attention block selection for one replay snapshot.")
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--enable-priority", action="store_true")
    parser.add_argument("--sink-block-count", type=int, default=1)
    parser.add_argument("--recent-block-count", type=int, default=1)
    parser.add_argument("--mandatory-recent-block-count", type=int, default=None)
    parser.add_argument("--exploration-blocks-per-region", type=int, default=1)
    parser.add_argument("--optional-top-k", type=int, default=0)
    parser.add_argument("--disable-upper-bound-ranking", action="store_true")
    parser.add_argument("--upper-bound-quota", type=int, default=0)
    parser.add_argument("--far-quota", type=int, default=0)
    parser.add_argument("--mid-quota", type=int, default=0)
    parser.add_argument("--near-quota", type=int, default=0)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--diversity-radius", type=int, default=0)
    parser.add_argument("--diversity-requires-history", action="store_true")
    parser.add_argument("--priority-prev-weight", type=float, default=1.0)
    parser.add_argument("--priority-recency-weight", type=float, default=0.05)
    parser.add_argument("--priority-recency-decay-blocks", type=float, default=32.0)
    parser.add_argument("--priority-value-norm-weight", type=float, default=0.05)
    parser.add_argument("--prev-attention-transform", default="sqrt", choices=["identity", "sqrt", "log1p"])
    parser.add_argument("--prev-attention-neighbor-blend", type=float, default=0.2)
    parser.add_argument("--prev-attention-smoothing-passes", type=int, default=1)
    parser.add_argument("--prev-attention-floor", type=float, default=1e-6)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _build_markdown(debug_payload: dict[str, object]) -> str:
    def table(rows: list[list[str]]) -> str:
        header = rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows[1:])
        return "\n".join(lines)

    lines = ["# Persistent Full-Attention Snapshot Debug", ""]
    lines.append(
        f"Selected {debug_payload['selected_block_count']} / {debug_payload['full_block_count']} blocks"
        f" ({debug_payload['selected_token_count']} tokens)."
    )
    lines.append("")
    lines.append("## Prior")
    lines.append("")
    prior_rows = [["Page", "Raw Prev", "Shaped Prev"]]
    for row in debug_payload["top_shaped_prev_attention_pages"]:
        prior_rows.append(
            [
                str(row["page_id"]),
                f"{float(row['raw_prev_attention']):.6f}",
                f"{float(row['shaped_prev_attention']):.6f}",
            ]
        )
    lines.append(table(prior_rows))
    lines.append("")
    for title, key in (
        ("Top Omitted By Priority", "top_omitted_by_priority"),
        ("Top Omitted By Upper Bound", "top_omitted_by_upper_bound"),
        ("Lowest Kept By Priority", "lowest_kept_by_priority"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        rows = [["Block", "Tok Start", "Tok Count", "Region", "Priority", "Upper", "Prev", "Flags"]]
        for row in debug_payload[key]:
            flags = ",".join(
                name
                for name in ("mandatory", "soft_recent", "exploration", "optional")
                if bool(row[name])
            )
            rows.append(
                [
                    str(row["block_id"]),
                    str(row["token_start"]),
                    str(row["token_count"]),
                    str(row["region_id"]),
                    f"{float(row['priority_score']):.4f}",
                    f"{float(row['upper_bound']):.4f}",
                    f"{float(row['prev_attention_ema']):.6f}",
                    flags or "-",
                ]
            )
        lines.append(table(rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    config = PersistentServingConfig(
        block_size=int(args.block_size),
        enable_priority=bool(args.enable_priority),
        full_attention_sink_block_count=int(args.sink_block_count),
        full_attention_recent_block_count=int(args.recent_block_count),
        full_attention_mandatory_recent_block_count=(
            None if args.mandatory_recent_block_count is None else int(args.mandatory_recent_block_count)
        ),
        full_attention_exploration_blocks_per_region=int(args.exploration_blocks_per_region),
        full_attention_optional_top_k=int(args.optional_top_k),
        full_attention_optional_use_upper_bounds_first=not bool(args.disable_upper_bound_ranking),
        full_attention_optional_upper_bound_quota=int(args.upper_bound_quota),
        full_attention_optional_far_quota=int(args.far_quota),
        full_attention_optional_mid_quota=int(args.mid_quota),
        full_attention_optional_near_quota=int(args.near_quota),
        full_attention_optional_diversity_weight=float(args.diversity_weight),
        full_attention_optional_diversity_radius=int(args.diversity_radius),
        full_attention_optional_diversity_requires_history=bool(args.diversity_requires_history),
        full_attention_priority_prev_attention_weight=float(args.priority_prev_weight),
        full_attention_priority_recency_weight=float(args.priority_recency_weight),
        full_attention_priority_recency_decay_blocks=float(args.priority_recency_decay_blocks),
        full_attention_priority_value_norm_weight=float(args.priority_value_norm_weight),
    )
    payload = debug_qwen35_persistent_full_attention_snapshot_selection(
        args.snapshot_path,
        persistent_serving_config=config,
        prev_attention_transform=str(args.prev_attention_transform),
        prev_attention_neighbor_blend=float(args.prev_attention_neighbor_blend),
        prev_attention_smoothing_passes=int(args.prev_attention_smoothing_passes),
        prev_attention_floor=float(args.prev_attention_floor),
        max_rows=int(args.max_rows),
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _build_markdown(payload)
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
