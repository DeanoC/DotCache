from __future__ import annotations

import argparse
import json

import numpy as np

from dotcache.state_cache_sim import StateTileSpec, simulate_state_sequence


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update-scale", type=float, default=0.05)
    parser.add_argument("--readout-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    initial_state = rng.normal(size=(args.state_rows, args.state_cols)).astype(np.float32)
    update_deltas = (rng.normal(size=(args.steps, args.state_rows, args.state_cols)).astype(np.float32) * float(args.update_scale))
    readout_projections = rng.normal(size=(args.steps, args.state_cols, args.readout_dim)).astype(np.float32)

    escape_bits = np.dtype(np.float16 if args.escape_dtype == "float16" else np.float32).itemsize * 8
    for mode_name in args.modes:
        bit_candidates = args.bits if mode_name == "M0" else [int(escape_bits)]
        for bits in bit_candidates:
            for renorm_interval in args.renorm_intervals:
                spec = StateTileSpec(
                    state_rows=args.state_rows,
                    state_cols=args.state_cols,
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
                    "seed": args.seed,
                    "state_rows": args.state_rows,
                    "state_cols": args.state_cols,
                    "steps": args.steps,
                    "update_scale": float(args.update_scale),
                    "readout_dim": args.readout_dim,
                    "final_update_error": float(result.update_error_curve[-1] if result.update_error_curve else 0.0),
                    "final_readout_error": float(result.readout_error_curve[-1] if result.readout_error_curve else 0.0),
                }
                record.update(result.to_dict())
                print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
