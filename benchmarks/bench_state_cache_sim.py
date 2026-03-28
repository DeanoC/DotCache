from __future__ import annotations

import argparse
import json

import numpy as np

from dotcache.state_cache_sim import StateTileSpec, load_captured_state_sample, simulate_state_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic StateCache simulator for recurrent-state tile compression.")
    parser.add_argument("--state-rows", type=int, default=128)
    parser.add_argument("--state-cols", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["M0", "M3"])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3])
    parser.add_argument("--renorm-intervals", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--escape-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--captured-sample", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update-scale", type=float, default=0.05)
    parser.add_argument("--readout-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sample = None
    if args.captured_sample:
        sample = load_captured_state_sample(args.captured_sample)
        initial_state = sample.initial_state
        update_deltas = sample.update_deltas
        state_rows = sample.state_rows
        state_cols = sample.state_cols
        steps = sample.steps
        input_source = "captured"
    else:
        state_rows = args.state_rows
        state_cols = args.state_cols
        steps = args.steps
        initial_state = rng.normal(size=(state_rows, state_cols)).astype(np.float32)
        update_deltas = (rng.normal(size=(steps, state_rows, state_cols)).astype(np.float32) * float(args.update_scale))
        input_source = "synthetic"
    readout_projections = rng.normal(size=(steps, state_cols, args.readout_dim)).astype(np.float32)

    escape_bits = np.dtype(np.float16 if args.escape_dtype == "float16" else np.float32).itemsize * 8
    summary_records: dict[tuple[str, int], list[dict[str, object]]] = {}
    for mode_name in args.modes:
        bit_candidates = args.bits if mode_name == "M0" else [int(escape_bits)]
        for bits in bit_candidates:
            for renorm_interval in args.renorm_intervals:
                spec = StateTileSpec(
                    state_rows=state_rows,
                    state_cols=state_cols,
                    group_size=args.group_size,
                    bits=int(bits),
                    mode=mode_name,
                    escape_dtype=args.escape_dtype,
                )
                result = simulate_state_sequence(
                    initial_state,
                    update_deltas,
                    readout_projections,
                    spec=spec,
                    renorm_interval=int(renorm_interval),
                )
                record = {
                    "benchmark": "state_cache_sim",
                    "input_source": input_source,
                    "seed": args.seed,
                    "state_rows": state_rows,
                    "state_cols": state_cols,
                    "steps": steps,
                    "update_scale": float(args.update_scale),
                    "readout_dim": args.readout_dim,
                    "final_update_error": float(result.update_error_curve[-1] if result.update_error_curve else 0.0),
                    "final_readout_error": float(result.readout_error_curve[-1] if result.readout_error_curve else 0.0),
                }
                if sample is not None:
                    record.update(
                        {
                            "captured_sample_path": args.captured_sample,
                            "captured_sample_kind": sample.state_kind,
                            "captured_layer_id": sample.layer_id,
                            "captured_prompt_length": sample.prompt_length,
                        }
                    )
                record.update(result.to_dict())
                summary_records.setdefault((mode_name, int(bits)), []).append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    for (mode_name, bits), records in sorted(summary_records.items()):
        best_update = min(records, key=lambda item: float(item["final_update_error"]))
        best_readout = min(records, key=lambda item: float(item["final_readout_error"]))
        summary = {
            "benchmark": "state_cache_sim_summary",
            "input_source": input_source,
            "mode": mode_name,
            "bits": bits,
            "best_update_renorm_interval": int(best_update["renorm_interval"]),
            "best_update_error": float(best_update["final_update_error"]),
            "best_readout_renorm_interval": int(best_readout["renorm_interval"]),
            "best_readout_error": float(best_readout["final_readout_error"]),
            "state_rows": state_rows,
            "state_cols": state_cols,
            "steps": steps,
        }
        if sample is not None:
            summary.update(
                {
                    "captured_sample_path": args.captured_sample,
                    "captured_sample_kind": sample.state_kind,
                    "captured_layer_id": sample.layer_id,
                    "captured_prompt_length": sample.prompt_length,
                }
            )
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
