from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
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
    parser.add_argument("--mandatory-recent-block-count", type=int, default=None)
    parser.add_argument("--exploration-blocks-per-region", type=int, default=1)
    parser.add_argument("--optional-top-k", type=int, default=0)
    parser.add_argument("--disable-upper-bound-ranking", action="store_true")
    parser.add_argument("--upper-bound-quota", type=int, default=0)
    parser.add_argument("--far-anchor-quota", type=int, default=0)
    parser.add_argument("--far-anchor-priority-margin", type=float, default=0.0)
    parser.add_argument("--far-anchor-upper-bound-margin", type=float, default=0.0)
    parser.add_argument("--far-quota", type=int, default=0)
    parser.add_argument("--mid-quota", type=int, default=0)
    parser.add_argument("--near-quota", type=int, default=0)
    parser.add_argument("--bootstrap-far-anchor-quota", type=int, default=None)
    parser.add_argument("--bootstrap-far-quota", type=int, default=None)
    parser.add_argument("--bootstrap-mid-quota", type=int, default=None)
    parser.add_argument("--bootstrap-near-quota", type=int, default=None)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--diversity-radius", type=int, default=0)
    parser.add_argument("--diversity-requires-history", action="store_true")
    parser.add_argument("--diversity-min-history-count", type=int, default=0)
    parser.add_argument("--diversity-max-history-count", type=int, default=None)
    parser.add_argument("--priority-prev-weight", type=float, default=1.0)
    parser.add_argument("--priority-recency-weight", type=float, default=0.05)
    parser.add_argument("--priority-recency-decay-blocks", type=float, default=32.0)
    parser.add_argument("--priority-value-norm-weight", type=float, default=0.05)
    parser.add_argument("--prev-attention-transform", default="sqrt", choices=["identity", "sqrt", "log1p"])
    parser.add_argument("--prev-attention-neighbor-blend", type=float, default=0.2)
    parser.add_argument("--prev-attention-smoothing-passes", type=int, default=1)
    parser.add_argument("--prev-attention-floor", type=float, default=1e-6)
    parser.add_argument("--history-mode", default="none", choices=["none", "mean", "ema"])
    parser.add_argument("--history-decay", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _resolve_snapshot_records(
    *,
    snapshot_paths: list[str],
    snapshot_glob: str | None,
    manifest_path: str | None,
) -> list[dict[str, object]]:
    resolved: dict[Path, dict[str, object]] = {}
    if snapshot_glob:
        snapshot_paths = [*snapshot_paths, *glob.glob(snapshot_glob)]
    for path in snapshot_paths:
        resolved[Path(path).resolve()] = {"snapshot_path": str(Path(path).resolve())}
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if "snapshot_records" in manifest:
            for record in manifest.get("snapshot_records", []):
                path = Path(record["paged_attention_snapshot_path"]).resolve()
                resolved[path] = {
                    "snapshot_path": str(path),
                    "step_index": int(record.get("paged_attention_snapshot_step_index", 0)),
                    "layer_id": int(record.get("paged_attention_snapshot_layer_id", 0)),
                    "kv_head_id": int(record.get("paged_attention_snapshot_kv_head_id", 0)),
                    "case_tag": str(path.parent.name),
                }
        else:
            for case_record in manifest.get("records", []):
                case_tag = str(case_record.get("case_tag", Path(case_record.get("output_dir", "")).name))
                child_manifest_path = case_record.get("paged_attention_snapshot_corpus_manifest_path")
                if not child_manifest_path:
                    continue
                child_manifest = json.loads(Path(child_manifest_path).read_text(encoding="utf-8"))
                for record in child_manifest.get("snapshot_records", []):
                    path = Path(record["paged_attention_snapshot_path"]).resolve()
                    resolved[path] = {
                        "snapshot_path": str(path),
                        "step_index": int(record.get("paged_attention_snapshot_step_index", 0)),
                        "layer_id": int(record.get("paged_attention_snapshot_layer_id", 0)),
                        "kv_head_id": int(record.get("paged_attention_snapshot_kv_head_id", 0)),
                        "case_tag": case_tag,
                    }
    return sorted(
        resolved.values(),
        key=lambda record: (
            str(record.get("case_tag", "")),
            int(record.get("layer_id", 0)),
            int(record.get("kv_head_id", 0)),
            int(record.get("step_index", 0)),
            str(record["snapshot_path"]),
        ),
    )


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
        "avg_max_abs_error": _avg("max_abs_error"),
        "avg_max_rel_error": _avg("max_rel_error"),
        "avg_selected_block_count": _avg("selected_block_count"),
        "avg_selected_token_count": _avg("selected_token_count"),
        "avg_full_block_count": _avg("full_block_count"),
        "avg_full_token_count": _avg("full_token_count"),
        "avg_selected_fraction": float(
            sum(float(record["selected_token_count"]) / max(float(record["full_token_count"]), 1.0) for record in records)
            / len(records)
        ),
    }


def _build_sequence_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    if not records or "case_tag" not in records[0]:
        return {}
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["case_tag"]), int(record["layer_id"]), int(record["kv_head_id"]))].append(record)
    by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    terminal_records: list[dict[str, object]] = []
    for group_records in grouped.values():
        ordered = sorted(group_records, key=lambda record: int(record["step_index"]))
        for record in ordered:
            by_step[int(record["step_index"])].append(record)
        terminal_records.append(ordered[-1])

    def _step_summary(items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "count": int(len(items)),
            "max_abs_error": float(max(float(item["max_abs_error"]) for item in items)),
            "avg_max_abs_error": float(sum(float(item["max_abs_error"]) for item in items) / len(items)),
            "avg_selected_fraction": float(
                sum(float(item["selected_token_count"]) / max(float(item["full_token_count"]), 1.0) for item in items) / len(items)
            ),
            "avg_history_snapshot_count": float(sum(float(item.get("history_snapshot_count", 0)) for item in items) / len(items)),
        }

    return {
        "sequence_count": int(len(grouped)),
        "terminal": _step_summary(terminal_records),
        "by_step_index": {
            str(step_index): _step_summary(items)
            for step_index, items in sorted(by_step.items())
        },
    }


def main() -> None:
    args = parse_args()
    snapshot_records = _resolve_snapshot_records(
        snapshot_paths=[str(path) for path in args.snapshot_paths],
        snapshot_glob=args.snapshot_glob,
        manifest_path=args.manifest_path,
    )
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
        full_attention_optional_far_anchor_quota=int(args.far_anchor_quota),
        full_attention_optional_far_anchor_priority_margin=float(args.far_anchor_priority_margin),
        full_attention_optional_far_anchor_upper_bound_margin=float(args.far_anchor_upper_bound_margin),
        full_attention_optional_far_quota=int(args.far_quota),
        full_attention_optional_mid_quota=int(args.mid_quota),
        full_attention_optional_near_quota=int(args.near_quota),
        full_attention_optional_bootstrap_far_anchor_quota=(
            None if args.bootstrap_far_anchor_quota is None else int(args.bootstrap_far_anchor_quota)
        ),
        full_attention_optional_bootstrap_far_quota=(
            None if args.bootstrap_far_quota is None else int(args.bootstrap_far_quota)
        ),
        full_attention_optional_bootstrap_mid_quota=(
            None if args.bootstrap_mid_quota is None else int(args.bootstrap_mid_quota)
        ),
        full_attention_optional_bootstrap_near_quota=(
            None if args.bootstrap_near_quota is None else int(args.bootstrap_near_quota)
        ),
        full_attention_optional_diversity_weight=float(args.diversity_weight),
        full_attention_optional_diversity_radius=int(args.diversity_radius),
        full_attention_optional_diversity_requires_history=bool(args.diversity_requires_history),
        full_attention_optional_diversity_min_history_count=int(args.diversity_min_history_count),
        full_attention_optional_diversity_max_history_count=(
            None if args.diversity_max_history_count is None else int(args.diversity_max_history_count)
        ),
        full_attention_priority_prev_attention_weight=float(args.priority_prev_weight),
        full_attention_priority_recency_weight=float(args.priority_recency_weight),
        full_attention_priority_recency_decay_blocks=float(args.priority_recency_decay_blocks),
        full_attention_priority_value_norm_weight=float(args.priority_value_norm_weight),
    )
    grouped_history: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    records = []
    for snapshot_record in snapshot_records:
        case_tag = str(snapshot_record.get("case_tag", Path(str(snapshot_record["snapshot_path"])).parent.name))
        layer_id = int(snapshot_record.get("layer_id", 0))
        kv_head_id = int(snapshot_record.get("kv_head_id", 0))
        group_key = (case_tag, layer_id, kv_head_id)
        history_records = grouped_history[group_key] if str(args.history_mode) != "none" else []
        result = run_qwen35_persistent_full_attention_snapshot_comparison(
            snapshot_record["snapshot_path"],
            persistent_serving_config=config,
            history_snapshots_or_paths=[item["snapshot_path"] for item in history_records],
            history_mode=str(args.history_mode),
            history_decay=float(args.history_decay),
            prev_attention_transform=str(args.prev_attention_transform),
            prev_attention_neighbor_blend=float(args.prev_attention_neighbor_blend),
            prev_attention_smoothing_passes=int(args.prev_attention_smoothing_passes),
            prev_attention_floor=float(args.prev_attention_floor),
        )
        enriched = result | {
            "snapshot_path": str(snapshot_record["snapshot_path"]),
            "case_tag": case_tag,
            "layer_id": layer_id,
            "kv_head_id": kv_head_id,
            "step_index": int(snapshot_record.get("step_index", 0)),
        }
        records.append(enriched)
        grouped_history[group_key].append(snapshot_record)
    payload = {
        "config": {
            "block_size": int(config.block_size),
            "enable_priority": bool(config.enable_priority),
            "sink_block_count": int(config.full_attention_sink_block_count),
            "recent_block_count": int(config.full_attention_recent_block_count),
            "mandatory_recent_block_count": (
                None
                if config.full_attention_mandatory_recent_block_count is None
                else int(config.full_attention_mandatory_recent_block_count)
            ),
            "exploration_blocks_per_region": int(config.full_attention_exploration_blocks_per_region),
            "optional_top_k": int(config.full_attention_optional_top_k),
            "optional_use_upper_bounds_first": bool(config.full_attention_optional_use_upper_bounds_first),
            "optional_upper_bound_quota": int(config.full_attention_optional_upper_bound_quota),
            "optional_far_anchor_quota": int(config.full_attention_optional_far_anchor_quota),
            "optional_far_anchor_priority_margin": float(
                config.full_attention_optional_far_anchor_priority_margin
            ),
            "optional_far_anchor_upper_bound_margin": float(
                config.full_attention_optional_far_anchor_upper_bound_margin
            ),
            "optional_far_quota": int(config.full_attention_optional_far_quota),
            "optional_mid_quota": int(config.full_attention_optional_mid_quota),
            "optional_near_quota": int(config.full_attention_optional_near_quota),
            "optional_bootstrap_far_anchor_quota": (
                None
                if config.full_attention_optional_bootstrap_far_anchor_quota is None
                else int(config.full_attention_optional_bootstrap_far_anchor_quota)
            ),
            "optional_bootstrap_far_quota": (
                None
                if config.full_attention_optional_bootstrap_far_quota is None
                else int(config.full_attention_optional_bootstrap_far_quota)
            ),
            "optional_bootstrap_mid_quota": (
                None
                if config.full_attention_optional_bootstrap_mid_quota is None
                else int(config.full_attention_optional_bootstrap_mid_quota)
            ),
            "optional_bootstrap_near_quota": (
                None
                if config.full_attention_optional_bootstrap_near_quota is None
                else int(config.full_attention_optional_bootstrap_near_quota)
            ),
            "optional_diversity_weight": float(config.full_attention_optional_diversity_weight),
            "optional_diversity_radius": int(config.full_attention_optional_diversity_radius),
            "optional_diversity_requires_history": bool(config.full_attention_optional_diversity_requires_history),
            "optional_diversity_min_history_count": int(config.full_attention_optional_diversity_min_history_count),
            "optional_diversity_max_history_count": (
                None
                if config.full_attention_optional_diversity_max_history_count is None
                else int(config.full_attention_optional_diversity_max_history_count)
            ),
            "priority_prev_weight": float(config.full_attention_priority_prev_attention_weight),
            "priority_recency_weight": float(config.full_attention_priority_recency_weight),
            "priority_recency_decay_blocks": float(config.full_attention_priority_recency_decay_blocks),
            "priority_value_norm_weight": float(config.full_attention_priority_value_norm_weight),
            "prev_attention_transform": str(args.prev_attention_transform),
            "prev_attention_neighbor_blend": float(args.prev_attention_neighbor_blend),
            "prev_attention_smoothing_passes": int(args.prev_attention_smoothing_passes),
            "prev_attention_floor": float(args.prev_attention_floor),
            "history_mode": str(args.history_mode),
            "history_decay": float(args.history_decay),
        },
        "records": records,
        "summary": _build_summary(records),
        "sequence_summary": _build_sequence_metrics(records),
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
