from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.backends.mps_persistent_experimental import (
    PagedAttentionControllerConfig,
    build_synthetic_snapshot,
    load_paged_attention_snapshot,
    prepare_resident_layer_pages,
    result_error_stats,
    run_paged_attention_step,
    run_reference_step,
    save_paged_attention_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Apple-first paged-attention benchmark.")
    parser.add_argument("--input-mode", choices=["synthetic", "snapshot"], default="synthetic")
    parser.add_argument("--input-snapshot", type=str, default=None)
    parser.add_argument("--save-snapshot", type=str, default=None)
    parser.add_argument("--engine", choices=["cpu_ref", "torch_mps_baseline", "mps_experimental"], default="mps_experimental")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=32)
    parser.add_argument("--partial-last-page-tokens", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--sink-window", type=int, default=256)
    parser.add_argument("--page-chunk-size", type=int, default=4)
    parser.add_argument("--early-exit", action="store_true")
    parser.add_argument("--early-exit-eps", type=float, default=1e-4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    return parser.parse_args()


def _load_snapshot(args: argparse.Namespace):
    if args.input_mode == "snapshot":
        if args.input_snapshot is None:
            raise ValueError("--input-snapshot is required when --input-mode=snapshot")
        return load_paged_attention_snapshot(args.input_snapshot)
    partial_last_page_tokens = None if args.partial_last_page_tokens <= 0 else args.partial_last_page_tokens
    snapshot = build_synthetic_snapshot(
        num_pages=args.num_pages,
        tokens_per_page=args.tokens_per_page,
        head_dim=args.head_dim,
        seed=args.seed,
        partial_last_page_tokens=partial_last_page_tokens,
    )
    if args.save_snapshot is not None:
        save_paged_attention_snapshot(args.save_snapshot, snapshot)
    return snapshot


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    snapshot = _load_snapshot(args)
    config = PagedAttentionControllerConfig(
        sink_window_tokens=args.sink_window,
        recent_window_tokens=args.recent_window,
        top_k=args.top_k,
        page_chunk_size=args.page_chunk_size,
        early_exit=args.early_exit,
        early_exit_eps=args.early_exit_eps,
    )

    reference = run_reference_step(snapshot, config=config)

    if args.engine == "cpu_ref":
        measured = [run_reference_step(snapshot, config=config) for _ in range(max(args.measured_runs, 1))]
        host_to_device_bytes_after_warmup = 0
        resident_host_to_device_bytes = 0
        device = "cpu"
    else:
        resident = prepare_resident_layer_pages(
            page_k_mean=snapshot.page_k_mean,
            prev_attn=snapshot.prev_attn,
            distance=snapshot.distance,
            k_pages=snapshot.k_pages,
            v_pages=snapshot.v_pages,
            page_token_counts=snapshot.page_token_counts,
            page_token_starts=snapshot.page_token_starts,
            device=args.device,
            dtype=args.dtype,
        )
        for _ in range(max(args.warmup_runs, 0)):
            run_paged_attention_step(snapshot.query, resident, config=config, engine=args.engine)
        measured = [
            run_paged_attention_step(snapshot.query, resident, config=config, engine=args.engine)
            for _ in range(max(args.measured_runs, 1))
        ]
        host_to_device_bytes_after_warmup = 0
        resident_host_to_device_bytes = resident.host_to_device_bytes
        device = resident.device.type

    errors = [result_error_stats(result.output, reference.output) for result in measured]
    record = {
        "engine": args.engine,
        "input_mode": args.input_mode,
        "input_snapshot": args.input_snapshot,
        "saved_snapshot": str(Path(args.save_snapshot).resolve()) if args.save_snapshot is not None else None,
        "device": device,
        "dtype": args.dtype,
        "head_dim": snapshot.head_dim,
        "num_pages": snapshot.num_pages,
        "tokens_per_page": snapshot.tokens_per_page,
        "total_tokens": int(snapshot.page_token_starts[-1] + snapshot.page_token_counts[-1]),
        "top_k": args.top_k,
        "recent_window_tokens": args.recent_window,
        "sink_window_tokens": args.sink_window,
        "page_chunk_size": args.page_chunk_size,
        "early_exit": bool(args.early_exit),
        "early_exit_eps": float(args.early_exit_eps),
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
        "score_time_ms": _average([result.score_time_ms for result in measured]),
        "selection_time_ms": _average([result.selection_time_ms for result in measured]),
        "attention_time_ms": _average([result.attention_time_ms for result in measured]),
        "total_step_time_ms": _average([result.total_step_time_ms for result in measured]),
        "selected_page_count": _average([float(result.selected_page_count) for result in measured]),
        "processed_page_count": _average([float(result.processed_page_count) for result in measured]),
        "tokens_processed": _average([float(result.tokens_processed) for result in measured]),
        "early_exit_rate": _average([1.0 if result.early_exit_triggered else 0.0 for result in measured]),
        "host_to_device_bytes_after_warmup": int(host_to_device_bytes_after_warmup),
        "resident_host_to_device_bytes": int(resident_host_to_device_bytes),
        "max_abs_error": max(error["max_abs_error"] for error in errors),
        "max_rel_error": max(error["max_rel_error"] for error in errors),
        "reference_selected_page_count": int(reference.selected_page_count),
        "reference_processed_page_count": int(reference.processed_page_count),
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
