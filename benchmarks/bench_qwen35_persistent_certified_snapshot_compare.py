from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import _resolve_snapshot_records
from dotcache.integrations.qwen35 import PersistentServingConfig, run_qwen35_persistent_full_attention_snapshot_comparison


_VARIANT_CHOICES = ("default", "region_caps", "cluster8", "region_caps_cluster8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare conservative certified-streaming bound variants on a snapshot corpus."
    )
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--variants", nargs="*", default=["default"], choices=list(_VARIANT_CHOICES))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--full-attention-check-interval", type=int, default=1)
    parser.add_argument("--full-attention-mass-eps", type=float, default=1e-3)
    parser.add_argument("--full-attention-value-eps", type=float, default=1e-3)
    parser.add_argument("--full-attention-min-processed-blocks", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _build_variant_config(
    *,
    variant: str,
    block_size: int,
    check_interval: int,
    mass_eps: float,
    value_eps: float,
    min_processed_blocks: int,
) -> PersistentServingConfig:
    config = PersistentServingConfig(
        block_size=int(block_size),
        enable_early_exit=True,
        full_attention_check_interval=max(int(check_interval), 1),
        full_attention_mass_eps=float(mass_eps),
        full_attention_value_eps=float(value_eps),
        full_attention_min_processed_blocks=max(int(min_processed_blocks), 1),
    )
    if str(variant) == "region_caps":
        config.full_attention_region_residual_caps = True
    elif str(variant) == "cluster8":
        config.full_attention_residual_cluster_count = 8
    elif str(variant) == "region_caps_cluster8":
        config.full_attention_region_residual_caps = True
        config.full_attention_residual_cluster_count = 8
    return config


def _summarize_variant_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "snapshot_count": 0,
            "streaming_certified_stop_rate": 0.0,
            "streaming_true_certified_stop_rate": 0.0,
            "avg_streaming_processed_block_count": 0.0,
            "avg_streaming_first_stop_block_count": 0.0,
            "avg_streaming_first_true_stop_block_count": 0.0,
            "avg_streaming_checkpoint_count": 0.0,
            "avg_streaming_first_stop_fraction": 0.0,
            "avg_streaming_first_true_stop_fraction": 0.0,
            "avg_streaming_processed_fraction": 0.0,
            "avg_streaming_truth_max_beta_over_true_beta_ratio": 0.0,
            "avg_streaming_truth_max_delta_over_true_delta_ratio": 0.0,
            "max_streaming_truth_max_beta_over_true_beta_ratio": 0.0,
            "max_streaming_truth_max_delta_over_true_delta_ratio": 0.0,
            "avg_streaming_truth_min_stop_ratio": 0.0,
            "avg_streaming_truth_bound_stop_ratio_at_true_frontier": 0.0,
            "avg_streaming_truth_frontier_block_fraction": 0.0,
            "avg_streaming_truth_min_nonterminal_stop_ratio": 0.0,
            "avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": 0.0,
            "avg_streaming_truth_nonterminal_frontier_block_fraction": 0.0,
            "avg_streaming_truth_nonterminal_frontier_true_beta_ratio": 0.0,
            "avg_streaming_truth_nonterminal_frontier_true_delta_ratio": 0.0,
            "avg_streaming_truth_nonterminal_frontier_bound_beta_ratio": 0.0,
            "avg_streaming_truth_nonterminal_frontier_bound_delta_ratio": 0.0,
            "streaming_truth_nonterminal_frontier_true_beta_dominant_rate": 0.0,
            "streaming_truth_nonterminal_frontier_true_delta_dominant_rate": 0.0,
            "streaming_truth_nonterminal_frontier_bound_beta_dominant_rate": 0.0,
            "streaming_truth_nonterminal_frontier_bound_delta_dominant_rate": 0.0,
            "max_streaming_max_abs_error": 0.0,
        }
    stop_records = [record for record in records if record.get("streaming_first_certified_stop_block_count") is not None]
    true_stop_records = [
        record for record in records if record.get("streaming_first_true_certified_stop_block_count") is not None
    ]
    nonterminal_frontier_records = [
        record
        for record in records
        if record.get("streaming_truth_nonterminal_frontier_true_beta_ratio") is not None
        and record.get("streaming_truth_nonterminal_frontier_true_delta_ratio") is not None
    ]
    return {
        "snapshot_count": int(len(records)),
        "streaming_certified_stop_rate": float(len(stop_records) / len(records)),
        "streaming_true_certified_stop_rate": float(len(true_stop_records) / len(records)),
        "avg_streaming_processed_block_count": float(
            sum(float(record.get("streaming_processed_block_count", 0.0)) for record in records) / len(records)
        ),
        "avg_streaming_first_stop_block_count": float(
            sum(float(record.get("streaming_first_certified_stop_block_count", 0.0)) for record in stop_records)
            / max(len(stop_records), 1)
        ),
        "avg_streaming_first_true_stop_block_count": float(
            sum(float(record.get("streaming_first_true_certified_stop_block_count", 0.0)) for record in true_stop_records)
            / max(len(true_stop_records), 1)
        ),
        "avg_streaming_checkpoint_count": float(
            sum(float(record.get("streaming_checkpoint_count", 0.0)) for record in records) / len(records)
        ),
        "avg_streaming_first_stop_fraction": float(
            sum(
                float(record["streaming_first_certified_stop_token_count"]) / max(float(record["full_token_count"]), 1.0)
                for record in stop_records
            )
            / max(len(stop_records), 1)
        ),
        "avg_streaming_first_true_stop_fraction": float(
            sum(
                float(record["streaming_first_true_certified_stop_token_count"])
                / max(float(record["full_token_count"]), 1.0)
                for record in true_stop_records
            )
            / max(len(true_stop_records), 1)
        ),
        "avg_streaming_processed_fraction": float(
            sum(
                float(record.get("streaming_processed_token_count", 0.0)) / max(float(record["full_token_count"]), 1.0)
                for record in records
            )
            / len(records)
        ),
        "avg_streaming_truth_max_beta_over_true_beta_ratio": float(
            sum(float(record.get("streaming_truth_max_beta_over_true_beta_ratio", 0.0)) for record in records)
            / len(records)
        ),
        "avg_streaming_truth_max_delta_over_true_delta_ratio": float(
            sum(float(record.get("streaming_truth_max_delta_over_true_delta_ratio", 0.0)) for record in records)
            / len(records)
        ),
        "max_streaming_truth_max_beta_over_true_beta_ratio": float(
            max(float(record.get("streaming_truth_max_beta_over_true_beta_ratio", 0.0)) for record in records)
        ),
        "max_streaming_truth_max_delta_over_true_delta_ratio": float(
            max(float(record.get("streaming_truth_max_delta_over_true_delta_ratio", 0.0)) for record in records)
        ),
        "avg_streaming_truth_min_stop_ratio": float(
            sum(float(record.get("streaming_truth_min_stop_ratio", 0.0) or 0.0) for record in records)
            / len(records)
        ),
        "avg_streaming_truth_bound_stop_ratio_at_true_frontier": float(
            sum(float(record.get("streaming_truth_bound_stop_ratio_at_true_frontier", 0.0) or 0.0) for record in records)
            / len(records)
        ),
        "avg_streaming_truth_frontier_block_fraction": float(
            sum(
                float(record.get("streaming_truth_frontier_block_count", 0.0) or 0.0)
                / max(float(record["full_block_count"]), 1.0)
                for record in records
            )
            / len(records)
        ),
        "avg_streaming_truth_min_nonterminal_stop_ratio": float(
            sum(float(record.get("streaming_truth_min_nonterminal_stop_ratio", 0.0) or 0.0) for record in records)
            / len(records)
        ),
        "avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": float(
            sum(
                float(record.get("streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier", 0.0) or 0.0)
                for record in records
            )
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
        "avg_streaming_truth_nonterminal_frontier_true_beta_ratio": float(
            sum(
                float(record.get("streaming_truth_nonterminal_frontier_true_beta_ratio", 0.0) or 0.0)
                for record in nonterminal_frontier_records
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "avg_streaming_truth_nonterminal_frontier_true_delta_ratio": float(
            sum(
                float(record.get("streaming_truth_nonterminal_frontier_true_delta_ratio", 0.0) or 0.0)
                for record in nonterminal_frontier_records
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "avg_streaming_truth_nonterminal_frontier_bound_beta_ratio": float(
            sum(
                float(record.get("streaming_truth_nonterminal_frontier_bound_beta_ratio", 0.0) or 0.0)
                for record in nonterminal_frontier_records
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "avg_streaming_truth_nonterminal_frontier_bound_delta_ratio": float(
            sum(
                float(record.get("streaming_truth_nonterminal_frontier_bound_delta_ratio", 0.0) or 0.0)
                for record in nonterminal_frontier_records
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "streaming_truth_nonterminal_frontier_true_beta_dominant_rate": float(
            sum(
                1
                for record in nonterminal_frontier_records
                if float(record.get("streaming_truth_nonterminal_frontier_true_beta_ratio", 0.0) or 0.0)
                >= float(record.get("streaming_truth_nonterminal_frontier_true_delta_ratio", 0.0) or 0.0)
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "streaming_truth_nonterminal_frontier_true_delta_dominant_rate": float(
            sum(
                1
                for record in nonterminal_frontier_records
                if float(record.get("streaming_truth_nonterminal_frontier_true_delta_ratio", 0.0) or 0.0)
                > float(record.get("streaming_truth_nonterminal_frontier_true_beta_ratio", 0.0) or 0.0)
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "streaming_truth_nonterminal_frontier_bound_beta_dominant_rate": float(
            sum(
                1
                for record in nonterminal_frontier_records
                if float(record.get("streaming_truth_nonterminal_frontier_bound_beta_ratio", 0.0) or 0.0)
                >= float(record.get("streaming_truth_nonterminal_frontier_bound_delta_ratio", 0.0) or 0.0)
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "streaming_truth_nonterminal_frontier_bound_delta_dominant_rate": float(
            sum(
                1
                for record in nonterminal_frontier_records
                if float(record.get("streaming_truth_nonterminal_frontier_bound_delta_ratio", 0.0) or 0.0)
                > float(record.get("streaming_truth_nonterminal_frontier_bound_beta_ratio", 0.0) or 0.0)
            )
            / max(len(nonterminal_frontier_records), 1)
        ),
        "max_streaming_max_abs_error": float(
            max(float(record.get("streaming_max_abs_error", 0.0)) for record in records)
        ),
    }


def main() -> None:
    args = parse_args()
    snapshot_records = _resolve_snapshot_records(
        snapshot_paths=[str(path) for path in args.snapshot_paths],
        snapshot_glob=args.snapshot_glob,
        manifest_path=args.manifest_path,
    )
    if not snapshot_records:
        raise SystemExit("no snapshot records resolved; provide --manifest-path, --snapshot-glob, or --snapshot-paths")

    variants_payload: list[dict[str, Any]] = []
    for variant in [str(item) for item in args.variants]:
        config = _build_variant_config(
            variant=variant,
            block_size=int(args.block_size),
            check_interval=int(args.full_attention_check_interval),
            mass_eps=float(args.full_attention_mass_eps),
            value_eps=float(args.full_attention_value_eps),
            min_processed_blocks=int(args.full_attention_min_processed_blocks),
        )
        records: list[dict[str, Any]] = []
        for snapshot_record in snapshot_records:
            result = run_qwen35_persistent_full_attention_snapshot_comparison(
                snapshot_record["snapshot_path"],
                persistent_serving_config=config,
            )
            records.append(
                {
                    "snapshot_path": str(snapshot_record["snapshot_path"]),
                    "case_tag": str(snapshot_record.get("case_tag", "")),
                    "layer_id": int(snapshot_record.get("layer_id", 0)),
                    "kv_head_id": int(snapshot_record.get("kv_head_id", 0)),
                    "step_index": int(snapshot_record.get("step_index", 0)),
                    "streaming_processed_block_count": int(result.get("streaming_processed_block_count", 0)),
                    "streaming_processed_token_count": int(result.get("streaming_processed_token_count", 0)),
                    "streaming_checkpoint_count": int(result.get("streaming_checkpoint_count", 0)),
                    "streaming_first_certified_stop_block_count": result.get("streaming_first_certified_stop_block_count"),
                    "streaming_first_certified_stop_token_count": result.get("streaming_first_certified_stop_token_count"),
                    "streaming_first_certified_stop_beta_upper": result.get("streaming_first_certified_stop_beta_upper"),
                    "streaming_first_certified_stop_delta_upper": result.get("streaming_first_certified_stop_delta_upper"),
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
                    "streaming_truth_max_beta_over_true_beta_ratio": float(
                        result.get("streaming_truth_max_beta_over_true_beta_ratio", 0.0)
                    ),
                    "streaming_truth_max_delta_over_true_delta_ratio": float(
                        result.get("streaming_truth_max_delta_over_true_delta_ratio", 0.0)
                    ),
                    "streaming_truth_min_stop_ratio": result.get("streaming_truth_min_stop_ratio"),
                    "streaming_truth_bound_stop_ratio_at_true_frontier": result.get(
                        "streaming_truth_bound_stop_ratio_at_true_frontier"
                    ),
                    "streaming_truth_frontier_block_count": result.get("streaming_truth_frontier_block_count"),
                    "streaming_truth_min_nonterminal_stop_ratio": result.get(
                        "streaming_truth_min_nonterminal_stop_ratio"
                    ),
                    "streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier": result.get(
                        "streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier"
                    ),
                    "streaming_truth_nonterminal_frontier_block_count": result.get(
                        "streaming_truth_nonterminal_frontier_block_count"
                    ),
                    "streaming_truth_nonterminal_frontier_true_beta_ratio": result.get(
                        "streaming_truth_nonterminal_frontier_true_beta_ratio"
                    ),
                    "streaming_truth_nonterminal_frontier_true_delta_ratio": result.get(
                        "streaming_truth_nonterminal_frontier_true_delta_ratio"
                    ),
                    "streaming_truth_nonterminal_frontier_bound_beta_ratio": result.get(
                        "streaming_truth_nonterminal_frontier_bound_beta_ratio"
                    ),
                    "streaming_truth_nonterminal_frontier_bound_delta_ratio": result.get(
                        "streaming_truth_nonterminal_frontier_bound_delta_ratio"
                    ),
                    "streaming_max_abs_error": float(result.get("streaming_max_abs_error", 0.0)),
                    "full_block_count": int(result.get("full_block_count", 0)),
                    "full_token_count": int(result.get("full_token_count", 0)),
                }
            )
        variants_payload.append(
            {
                "variant": variant,
                "config": {
                    "block_size": int(config.block_size),
                    "full_attention_check_interval": int(config.full_attention_check_interval),
                    "full_attention_mass_eps": float(config.full_attention_mass_eps),
                    "full_attention_value_eps": float(config.full_attention_value_eps),
                    "full_attention_min_processed_blocks": int(config.full_attention_min_processed_blocks),
                    "full_attention_region_residual_caps": bool(config.full_attention_region_residual_caps),
                    "full_attention_residual_cluster_count": int(config.full_attention_residual_cluster_count),
                },
                "summary": _summarize_variant_records(records),
                "records": records,
            }
        )

    payload = {
        "variant_count": int(len(variants_payload)),
        "variants": variants_payload,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({item["variant"]: item["summary"] for item in variants_payload}, sort_keys=True))


if __name__ == "__main__":
    main()
