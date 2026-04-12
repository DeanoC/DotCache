from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import _resolve_snapshot_records
from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen35 import (
    PersistentServingConfig,
    _build_persistent_full_attention_snapshot_runtime,
    _compute_exact_snapshot_block_statistics,
    _resolve_history_aware_persistent_serving_config,
)
from dotcache.backends.metal.persistent_runtime import (
    _load_torch,
    _resolve_streaming_proxy_scores,
    _resolve_streaming_value_upper_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export checkpoint-local exact block value signal labels and cheap feature columns for snapshot corpora."
    )
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--full-attention-check-interval", type=int, default=4)
    parser.add_argument("--full-attention-mass-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-value-eps", type=float, default=1e-2)
    parser.add_argument("--full-attention-min-processed-blocks", type=int, default=8)
    parser.add_argument("--full-attention-streaming-order-mode", default="residual_proxy")
    parser.add_argument("--full-attention-key-centroid-count-by-layer", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def _build_config(
    *,
    block_size: int,
    check_interval: int,
    mass_eps: float,
    value_eps: float,
    min_processed_blocks: int,
    order_mode: str,
    key_centroid_count_by_layer: dict[int, int] | None,
) -> PersistentServingConfig:
    return PersistentServingConfig(
        block_size=int(block_size),
        enable_priority=True,
        enable_early_exit=True,
        full_attention_check_interval=max(int(check_interval), 1),
        full_attention_mass_eps=float(mass_eps),
        full_attention_value_eps=float(value_eps),
        full_attention_min_processed_blocks=max(int(min_processed_blocks), 1),
        full_attention_streaming_order_mode=str(order_mode),
        full_attention_optional_use_upper_bounds_first=True,
        full_attention_key_centroid_count_by_layer=key_centroid_count_by_layer,
    )


def _resolve_kv_max_feature(state: Any, tensor: Any) -> Any:
    torch = _load_torch()
    q_to_kv = np.asarray(state.q_head_to_kv_head if hasattr(state, "q_head_to_kv_head") else [], dtype=np.int64)
    device = tensor.device
    dtype = torch.float32
    if int(len(q_to_kv)) <= 0:
        return torch.as_tensor(tensor, device=device, dtype=dtype)
    per_kv = [tensor[:, int(kv_head_idx)].to(device=device, dtype=dtype) for kv_head_idx in q_to_kv.tolist()]
    return torch.stack(per_kv, dim=0).max(dim=0).values


def _topk_recall(reference_ids: list[int], candidate_ids: list[int], top_k: int) -> float:
    if top_k <= 0:
        return 0.0
    ref = set(int(block_id) for block_id in reference_ids[:top_k])
    cand = set(int(block_id) for block_id in candidate_ids[:top_k])
    if not ref:
        return 1.0
    return float(len(ref & cand) / len(ref))


def _summarize_rows(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "checkpoint_count": 0,
            "snapshot_count": 0,
            "avg_rows_per_checkpoint": 0.0,
            "avg_topk_recall_upper_bound": 0.0,
            "avg_topk_recall_priority_score": 0.0,
            "avg_topk_recall_proxy_score": 0.0,
            "avg_topk_recall_value_upper_score": 0.0,
        }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    snapshot_keys: set[str] = set()
    for row in rows:
        key = (str(row["snapshot_path"]), int(row["checkpoint_index"]))
        grouped.setdefault(key, []).append(row)
        snapshot_keys.add(str(row["snapshot_path"]))
    upper_recalls: list[float] = []
    priority_recalls: list[float] = []
    proxy_recalls: list[float] = []
    value_upper_recalls: list[float] = []
    for checkpoint_rows in grouped.values():
        oracle = sorted(
            checkpoint_rows,
            key=lambda row: (-float(row["exact_value_score"]), int(row["block_id"])),
        )
        oracle_ids = [int(row["block_id"]) for row in oracle]
        upper_ids = [
            int(row["block_id"])
            for row in sorted(checkpoint_rows, key=lambda row: (-float(row["upper_bound"]), int(row["block_id"])))
        ]
        priority_ids = [
            int(row["block_id"])
            for row in sorted(checkpoint_rows, key=lambda row: (-float(row["priority_score"]), int(row["block_id"])))
        ]
        proxy_ids = [
            int(row["block_id"])
            for row in sorted(checkpoint_rows, key=lambda row: (-float(row["proxy_score"]), int(row["block_id"])))
        ]
        value_upper_ids = [
            int(row["block_id"])
            for row in sorted(
                checkpoint_rows,
                key=lambda row: (-float(row["value_upper_score"]), int(row["block_id"])),
            )
        ]
        upper_recalls.append(_topk_recall(oracle_ids, upper_ids, top_k))
        priority_recalls.append(_topk_recall(oracle_ids, priority_ids, top_k))
        proxy_recalls.append(_topk_recall(oracle_ids, proxy_ids, top_k))
        value_upper_recalls.append(_topk_recall(oracle_ids, value_upper_ids, top_k))
    checkpoint_count = int(len(grouped))
    return {
        "row_count": int(len(rows)),
        "checkpoint_count": checkpoint_count,
        "snapshot_count": int(len(snapshot_keys)),
        "avg_rows_per_checkpoint": float(len(rows) / max(checkpoint_count, 1)),
        "avg_topk_recall_upper_bound": float(sum(upper_recalls) / max(len(upper_recalls), 1)),
        "avg_topk_recall_priority_score": float(sum(priority_recalls) / max(len(priority_recalls), 1)),
        "avg_topk_recall_proxy_score": float(sum(proxy_recalls) / max(len(proxy_recalls), 1)),
        "avg_topk_recall_value_upper_score": float(sum(value_upper_recalls) / max(len(value_upper_recalls), 1)),
    }


def _export_rows_for_snapshot(
    snapshot_path: str,
    *,
    config: PersistentServingConfig,
    top_k: int,
) -> list[dict[str, Any]]:
    from dotcache.backends.mps_persistent_experimental import load_paged_attention_snapshot

    torch = _load_torch()
    snapshot = load_paged_attention_snapshot(snapshot_path)
    effective_config = _resolve_history_aware_persistent_serving_config(config, history_snapshot_count=0)
    runtime, query_tensor, key_history, _value_history, resolved_query_scale, _ = _build_persistent_full_attention_snapshot_runtime(
        snapshot,
        persistent_serving_config=effective_config,
        dotcache_config=DotCacheConfig(
            head_dim=int(snapshot.head_dim),
            group_size=32,
            bits_k=4,
            bits_v=4,
            tokens_per_page=int(snapshot.tokens_per_page),
        ),
        query_scale=None,
        prev_attention_values=None,
        prev_attention_transform="sqrt",
        prev_attention_neighbor_blend=0.2,
        prev_attention_smoothing_passes=1,
        prev_attention_floor=1e-6,
    )
    selection = runtime.select_blocks(0, query_tensor, query_scale=resolved_query_scale)
    streaming = runtime.stream_decode_layer(
        0,
        query_tensor,
        query_scale=resolved_query_scale,
        check_interval=int(effective_config.full_attention_check_interval),
        stop_on_certificate=False,
    )
    stats = _compute_exact_snapshot_block_statistics(
        runtime=runtime,
        layer_id=0,
        query_tensor=query_tensor,
        query_scale=float(resolved_query_scale),
    )
    per_head_block_logits = stats["per_head_block_logits"]
    per_head_block_values = stats["per_head_block_values"]
    state = runtime.layers[0]
    processing_order = [int(block_id) for block_id in streaming.get("processing_order_block_ids", [])]
    processing_order_index = {int(block_id): idx for idx, block_id in enumerate(processing_order)}
    mandatory_ids = {int(block_id) for block_id in selection.get("mandatory_block_ids", [])}
    upper_bounds = selection["upper_bounds"].to(dtype=torch.float32, device=query_tensor.device)
    priority_scores = selection["priority_scores"].to(dtype=torch.float32, device=query_tensor.device)
    proxy_scores = _resolve_streaming_proxy_scores(
        state=state,
        config=effective_config,
        q_head_to_kv_head=np.asarray(runtime.q_head_to_kv_head, dtype=np.int32),
        upper_bounds=upper_bounds,
        layer_id=int(getattr(snapshot, "layer_id", 0)),
        mode=str(effective_config.full_attention_streaming_order_mode),
    ).to(dtype=torch.float32, device=query_tensor.device)
    value_upper_scores = _resolve_streaming_value_upper_scores(
        state=state,
        q_head_to_kv_head=np.asarray(runtime.q_head_to_kv_head, dtype=np.int32),
        upper_bounds=upper_bounds,
    ).to(dtype=torch.float32, device=query_tensor.device)
    q_to_kv = np.asarray(runtime.q_head_to_kv_head, dtype=np.int64)
    value_norm_max = torch.stack(
        [state.block_v_norm_max[:, int(kv_head_idx)].to(dtype=torch.float32, device=query_tensor.device) for kv_head_idx in q_to_kv.tolist()],
        dim=0,
    ).max(dim=0).values
    value_center_norm = torch.stack(
        [
            torch.linalg.vector_norm(
                state.block_v_center[:, int(kv_head_idx), :].to(dtype=torch.float32, device=query_tensor.device),
                dim=-1,
            )
            for kv_head_idx in q_to_kv.tolist()
        ],
        dim=0,
    ).max(dim=0).values
    value_radius = torch.stack(
        [state.block_v_radius[:, int(kv_head_idx)].to(dtype=torch.float32, device=query_tensor.device) for kv_head_idx in q_to_kv.tolist()],
        dim=0,
    ).max(dim=0).values
    value_box_norm = torch.stack(
        [
            torch.linalg.vector_norm(
                torch.maximum(
                    state.block_v_pos_sum[:, int(kv_head_idx), :].to(dtype=torch.float32, device=query_tensor.device).abs(),
                    state.block_v_neg_sum[:, int(kv_head_idx), :].to(dtype=torch.float32, device=query_tensor.device).abs(),
                ),
                dim=-1,
            )
            for kv_head_idx in q_to_kv.tolist()
        ],
        dim=0,
    ).max(dim=0).values
    rows: list[dict[str, Any]] = []
    checkpoint_records = streaming.get("checkpoint_records", [])
    for checkpoint_index, checkpoint in enumerate(checkpoint_records):
        processed_block_count = int(checkpoint["processed_block_count"])
        unresolved_ids = [int(block_id) for block_id in processing_order[processed_block_count:]]
        if not unresolved_ids:
            continue
        per_head = checkpoint["per_head"]
        for block_id in unresolved_ids:
            exact_mass_score = 0.0
            exact_value_score = 0.0
            exact_stop_score = 0.0
            for q_head_idx, head_state in enumerate(per_head):
                logits = per_head_block_logits[q_head_idx][int(block_id)]
                if int(logits.numel()) <= 0:
                    continue
                scaled = torch.exp(logits - float(head_state["m"]))
                block_mass = float(scaled.sum().item())
                denom = float(head_state["l"] + block_mass)
                if denom <= 0.0:
                    continue
                contribution = torch.sum(scaled[:, None] * per_head_block_values[q_head_idx][int(block_id)], dim=0)
                mass_score = float(block_mass / denom)
                value_score = float(torch.linalg.vector_norm(contribution).item() / denom)
                stop_score = max(
                    mass_score / max(float(effective_config.full_attention_mass_eps), 1e-12),
                    value_score / max(float(effective_config.full_attention_value_eps), 1e-12),
                )
                exact_mass_score = max(exact_mass_score, mass_score)
                exact_value_score = max(exact_value_score, value_score)
                exact_stop_score = max(exact_stop_score, stop_score)
            rows.append(
                {
                    "snapshot_path": str(Path(snapshot_path).resolve()),
                    "case_tag": str(Path(snapshot_path).stem),
                    "layer_id": int(getattr(snapshot, "layer_id", 0)),
                    "kv_head_id": int(getattr(snapshot, "kv_head_id", 0)),
                    "step_index": int(getattr(snapshot, "step_index", 0)),
                    "full_block_count": int(len(state.block_token_starts)),
                    "full_token_count": int(key_history.shape[0]),
                    "checkpoint_index": int(checkpoint_index),
                    "processed_block_count": processed_block_count,
                    "remaining_block_count": int(len(unresolved_ids)),
                    "block_id": int(block_id),
                    "runtime_order_index": int(processing_order_index[int(block_id)]),
                    "is_mandatory": bool(int(block_id) in mandatory_ids),
                    "token_count": int(state.block_token_counts[int(block_id)]),
                    "upper_bound": float(upper_bounds[int(block_id)].item()),
                    "priority_score": float(priority_scores[int(block_id)].item()),
                    "proxy_score": float(proxy_scores[int(block_id)].item()),
                    "value_upper_score": float(value_upper_scores[int(block_id)].item()),
                    "value_norm_max": float(value_norm_max[int(block_id)].item()),
                    "value_center_radius": float((value_center_norm[int(block_id)] + value_radius[int(block_id)]).item()),
                    "value_box_norm": float(value_box_norm[int(block_id)].item()),
                    "prev_attention_ema": float(state.block_prev_attention_ema[int(block_id)].item()),
                    "exact_mass_score": float(exact_mass_score),
                    "exact_value_score": float(exact_value_score),
                    "exact_stop_score": float(exact_stop_score),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    snapshot_records = _resolve_snapshot_records(
        snapshot_paths=[str(path) for path in args.snapshot_paths],
        snapshot_glob=args.snapshot_glob,
        manifest_path=args.manifest_path,
    )
    if not snapshot_records:
        raise SystemExit("no snapshot records resolved; provide --manifest-path, --snapshot-glob, or --snapshot-paths")
    key_centroid_count_by_layer = (
        {
            int(layer_id): max(int(count), 1)
            for layer_id, count in json.loads(str(args.full_attention_key_centroid_count_by_layer)).items()
        }
        if args.full_attention_key_centroid_count_by_layer
        else None
    )
    config = _build_config(
        block_size=int(args.block_size),
        check_interval=int(args.full_attention_check_interval),
        mass_eps=float(args.full_attention_mass_eps),
        value_eps=float(args.full_attention_value_eps),
        min_processed_blocks=int(args.full_attention_min_processed_blocks),
        order_mode=str(args.full_attention_streaming_order_mode),
        key_centroid_count_by_layer=key_centroid_count_by_layer,
    )
    all_rows: list[dict[str, Any]] = []
    for snapshot_record in snapshot_records:
        all_rows.extend(
            _export_rows_for_snapshot(
                str(snapshot_record["snapshot_path"]),
                config=config,
                top_k=int(args.top_k),
            )
        )
    summary = _summarize_rows(all_rows, top_k=int(args.top_k))
    payload = {
        "config": {
            "block_size": int(config.block_size),
            "full_attention_check_interval": int(config.full_attention_check_interval),
            "full_attention_mass_eps": float(config.full_attention_mass_eps),
            "full_attention_value_eps": float(config.full_attention_value_eps),
            "full_attention_min_processed_blocks": int(config.full_attention_min_processed_blocks),
            "full_attention_streaming_order_mode": str(config.full_attention_streaming_order_mode),
            "full_attention_key_centroid_count_by_layer": key_centroid_count_by_layer,
            "top_k": int(args.top_k),
        },
        "summary": summary,
        "rows": all_rows if args.output_json else None,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
