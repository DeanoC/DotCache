from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import _resolve_snapshot_records
from dotcache.integrations.qwen35 import (
    PersistentServingConfig,
    run_qwen35_persistent_full_attention_snapshot_oracle_ordering_compare,
)


_VARIANT_CHOICES = (
    "runtime",
    "oracle_mass",
    "oracle_value",
    "oracle_stop",
    "dynamic_oracle_mass",
    "dynamic_oracle_value",
    "dynamic_oracle_stop",
    "dynamic_metadata_mass",
    "dynamic_metadata_value",
    "dynamic_metadata_stop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact oracle ordering frontiers for conservative certified-streaming snapshots."
    )
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--variants", nargs="*", default=list(_VARIANT_CHOICES), choices=list(_VARIANT_CHOICES))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--full-attention-check-interval", type=int, default=16)
    parser.add_argument("--full-attention-mass-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-value-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-min-processed-blocks", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _build_config(
    *,
    block_size: int,
    check_interval: int,
    mass_eps: float,
    value_eps: float,
    min_processed_blocks: int,
) -> PersistentServingConfig:
    return PersistentServingConfig(
        block_size=int(block_size),
        enable_priority=True,
        enable_early_exit=True,
        full_attention_check_interval=max(int(check_interval), 1),
        full_attention_mass_eps=float(mass_eps),
        full_attention_value_eps=float(value_eps),
        full_attention_min_processed_blocks=max(int(min_processed_blocks), 1),
        full_attention_optional_use_upper_bounds_first=True,
    )


def _summarize_variant_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "snapshot_count": 0,
            "avg_streaming_first_true_stop_fraction": 0.0,
            "avg_streaming_truth_min_nonterminal_stop_ratio": 0.0,
            "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.0,
        }
    true_stop_records = [
        record for record in records if record.get("streaming_first_true_certified_stop_block_count") is not None
    ]
    return {
        "snapshot_count": int(len(records)),
        "avg_streaming_first_true_stop_fraction": float(
            sum(
                float(record["streaming_first_true_certified_stop_token_count"])
                / max(float(record["full_token_count"]), 1.0)
                for record in true_stop_records
            )
            / max(len(true_stop_records), 1)
        ),
        "avg_streaming_truth_min_nonterminal_stop_ratio": float(
            sum(float(record.get("streaming_truth_min_nonterminal_stop_ratio", 0.0) or 0.0) for record in records)
            / len(records)
        ),
        "avg_streaming_truth_nonterminal_frontier_block_fraction": float(
            sum(
                float(record.get("streaming_truth_nonterminal_frontier_block_count", 0.0) or 0.0)
                / max(float(record["full_block_count"]), 1.0)
                for record in records
            )
            / len(records)
        ),
    }


def _variant_score(summary: dict[str, Any]) -> tuple[float, float]:
    return (
        float(summary.get("avg_streaming_truth_min_nonterminal_stop_ratio", float("inf"))),
        float(summary.get("avg_streaming_truth_nonterminal_frontier_block_fraction", float("inf"))),
    )


def main() -> None:
    args = parse_args()
    snapshot_records = _resolve_snapshot_records(
        snapshot_paths=[str(path) for path in args.snapshot_paths],
        snapshot_glob=args.snapshot_glob,
        manifest_path=args.manifest_path,
    )
    if not snapshot_records:
        raise SystemExit("no snapshot records resolved; provide --manifest-path, --snapshot-glob, or --snapshot-paths")

    config = _build_config(
        block_size=int(args.block_size),
        check_interval=int(args.full_attention_check_interval),
        mass_eps=float(args.full_attention_mass_eps),
        value_eps=float(args.full_attention_value_eps),
        min_processed_blocks=int(args.full_attention_min_processed_blocks),
    )
    variant_records: dict[str, list[dict[str, Any]]] = {str(variant): [] for variant in args.variants}
    for snapshot_record in snapshot_records:
        compare = run_qwen35_persistent_full_attention_snapshot_oracle_ordering_compare(
            snapshot_record["snapshot_path"],
            persistent_serving_config=config,
        )
        for variant in [str(item) for item in args.variants]:
            result = compare["variants"][str(variant)]
            variant_records[str(variant)].append(
                {
                    "snapshot_path": str(snapshot_record["snapshot_path"]),
                    "case_tag": str(snapshot_record.get("case_tag", "")),
                    "layer_id": int(snapshot_record.get("layer_id", 0)),
                    "kv_head_id": int(snapshot_record.get("kv_head_id", 0)),
                    "step_index": int(snapshot_record.get("step_index", 0)),
                    "full_block_count": int(result["full_block_count"]),
                    "full_token_count": int(result["full_token_count"]),
                    "streaming_checkpoint_count": int(result["streaming_checkpoint_count"]),
                    "streaming_first_true_certified_stop_block_count": result.get(
                        "streaming_first_true_certified_stop_block_count"
                    ),
                    "streaming_first_true_certified_stop_token_count": result.get(
                        "streaming_first_true_certified_stop_token_count"
                    ),
                    "streaming_first_true_certified_stop_beta_upper": result.get(
                        "streaming_first_true_certified_stop_beta_upper"
                    ),
                    "streaming_first_true_certified_stop_delta_upper": result.get(
                        "streaming_first_true_certified_stop_delta_upper"
                    ),
                    "streaming_truth_min_nonterminal_stop_ratio": result.get(
                        "streaming_truth_min_nonterminal_stop_ratio"
                    ),
                    "streaming_truth_nonterminal_frontier_block_count": result.get(
                        "streaming_truth_nonterminal_frontier_block_count"
                    ),
                    "streaming_truth_nonterminal_frontier_token_count": result.get(
                        "streaming_truth_nonterminal_frontier_token_count"
                    ),
                    "streaming_truth_nonterminal_frontier_beta_upper": result.get(
                        "streaming_truth_nonterminal_frontier_beta_upper"
                    ),
                    "streaming_truth_nonterminal_frontier_delta_upper": result.get(
                        "streaming_truth_nonterminal_frontier_delta_upper"
                    ),
                }
            )

    variants_payload: list[dict[str, Any]] = []
    for variant, records in variant_records.items():
        summary = _summarize_variant_records(records)
        variants_payload.append(
            {
                "variant": str(variant),
                "config": {
                    "block_size": int(config.block_size),
                    "full_attention_check_interval": int(config.full_attention_check_interval),
                    "full_attention_mass_eps": float(config.full_attention_mass_eps),
                    "full_attention_value_eps": float(config.full_attention_value_eps),
                    "full_attention_min_processed_blocks": int(config.full_attention_min_processed_blocks),
                },
                "summary": summary,
                "records": records,
            }
        )

    ranked_variants = [
        {"variant": item["variant"], "oracle_frontier_score": _variant_score(item["summary"])}
        for item in sorted(variants_payload, key=lambda item: _variant_score(item["summary"]))
    ]
    payload = {
        "variant_count": int(len(variants_payload)),
        "ranked_variants": ranked_variants,
        "variants": variants_payload,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({item["variant"]: item["summary"] for item in variants_payload}, sort_keys=True))


if __name__ == "__main__":
    main()
