from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_certified_snapshot_compare import _summarize_variant_records
from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import _resolve_snapshot_records
from dotcache.integrations.qwen35 import PersistentServingConfig, run_qwen35_persistent_full_attention_snapshot_comparison


_ORDERING_CHOICES = (
    "upper_ci16",
    "upper_ci8",
    "upper_ci4",
    "priority_ci16",
    "priority_ci8",
    "hybrid_ci16",
    "hybrid_ci8",
    "hybrid_ci4",
    "proxy_ci16",
    "proxy_ci8",
    "proxy_refineall_ci16",
    "proxy_refineall_ci8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare conservative certified-streaming ordering and tranche variants on a snapshot corpus."
    )
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--variants", nargs="*", default=["upper_ci16"], choices=list(_ORDERING_CHOICES))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--full-attention-mass-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-value-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-min-processed-blocks", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _build_ordering_config(
    *,
    variant: str,
    block_size: int,
    mass_eps: float,
    value_eps: float,
    min_processed_blocks: int,
) -> PersistentServingConfig:
    name = str(variant).strip().lower()
    if name not in _ORDERING_CHOICES:
        raise ValueError(f"unsupported ordering variant: {variant}")
    use_upper_bounds_first = name.startswith("upper_")
    if name.startswith("proxy_"):
        streaming_order_mode = "residual_proxy"
    elif name.startswith("hybrid_"):
        streaming_order_mode = "priority_value_hybrid"
    else:
        streaming_order_mode = "shortlist"
    refine_top_k = 100000 if "refineall" in name else 0
    if name.endswith("ci16"):
        check_interval = 16
    elif name.endswith("ci8"):
        check_interval = 8
    elif name.endswith("ci4"):
        check_interval = 4
    else:  # pragma: no cover - guarded by choices above
        raise ValueError(f"unsupported check interval variant: {variant}")
    return PersistentServingConfig(
        block_size=int(block_size),
        enable_priority=True,
        enable_early_exit=True,
        full_attention_check_interval=int(check_interval),
        full_attention_mass_eps=float(mass_eps),
        full_attention_value_eps=float(value_eps),
        full_attention_min_processed_blocks=max(int(min_processed_blocks), 1),
        full_attention_optional_use_upper_bounds_first=bool(use_upper_bounds_first),
        full_attention_streaming_order_mode=str(streaming_order_mode),
        full_attention_streaming_priority_value_upper_weight=0.25,
        full_attention_refine_top_k=int(refine_top_k),
    )


def _variant_score(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary.get("avg_streaming_truth_min_nonterminal_stop_ratio", float("inf"))),
        float(summary.get("avg_streaming_truth_nonterminal_frontier_block_fraction", float("inf"))),
        float(summary.get("avg_streaming_truth_bound_stop_ratio_at_nonterminal_true_frontier", float("inf"))),
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

    variants_payload: list[dict[str, Any]] = []
    for variant in [str(item) for item in args.variants]:
        config = _build_ordering_config(
            variant=variant,
            block_size=int(args.block_size),
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
                    "streaming_first_true_certified_stop_block_count": result.get(
                        "streaming_first_true_certified_stop_block_count"
                    ),
                    "streaming_first_true_certified_stop_token_count": result.get(
                        "streaming_first_true_certified_stop_token_count"
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
                    "streaming_max_abs_error": float(result.get("streaming_max_abs_error", 0.0)),
                    "full_block_count": int(result.get("full_block_count", 0)),
                    "full_token_count": int(result.get("full_token_count", 0)),
                }
            )
        summary = _summarize_variant_records(records)
        variants_payload.append(
            {
                "variant": variant,
                "config": {
                    "block_size": int(config.block_size),
                    "full_attention_check_interval": int(config.full_attention_check_interval),
                    "full_attention_mass_eps": float(config.full_attention_mass_eps),
                    "full_attention_value_eps": float(config.full_attention_value_eps),
                    "full_attention_min_processed_blocks": int(config.full_attention_min_processed_blocks),
                    "enable_priority": bool(config.enable_priority),
                    "full_attention_optional_use_upper_bounds_first": bool(
                        config.full_attention_optional_use_upper_bounds_first
                    ),
                    "full_attention_streaming_order_mode": str(config.full_attention_streaming_order_mode),
                    "full_attention_streaming_priority_value_upper_weight": float(
                        config.full_attention_streaming_priority_value_upper_weight
                    ),
                    "full_attention_refine_top_k": int(config.full_attention_refine_top_k),
                },
                "summary": summary,
                "records": records,
            }
        )

    ranked_variants = [
        {
            "variant": item["variant"],
            "oracle_frontier_score": _variant_score(item["summary"]),
        }
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
