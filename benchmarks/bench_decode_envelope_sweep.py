from __future__ import annotations

import argparse

from bench_common import build_config, emit, warm_backend
from bench_decode_session import run_session_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep envelope-gated session decode settings and emit Pareto-tagged results.")
    parser.add_argument("--backend", choices=["torch_mps"], default="torch_mps")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=[4096])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-sink-windows", nargs="*", type=int, default=[0, 256, 512])
    parser.add_argument("--execution-recent-windows", nargs="*", type=int, default=[512, 1024, 1536, 2048, 3072])
    parser.add_argument("--execution-relevance-top-ks", nargs="*", type=int, default=[0, 2, 4, 6, 8])
    return parser.parse_args()


def _dominates(left: dict[str, float | int | str], right: dict[str, float | int | str]) -> bool:
    left_runtime = float(left["session_runtime_ms_per_step"])
    left_error = float(left["step_max_abs_error"])
    right_runtime = float(right["session_runtime_ms_per_step"])
    right_error = float(right["step_max_abs_error"])
    return (
        left_runtime <= right_runtime
        and left_error <= right_error
        and (left_runtime < right_runtime or left_error < right_error)
    )


def _pareto_frontier(records: list[dict[str, float | int | str]]) -> list[int]:
    frontier: list[int] = []
    for index, record in enumerate(records):
        if any(_dominates(other, record) for other_index, other in enumerate(records) if other_index != index):
            continue
        frontier.append(index)
    return frontier


def main() -> None:
    args = parse_args()
    config = build_config(args)
    warm_backend(args.backend)

    for context_length in args.contexts:
        results: list[dict[str, float | int | str]] = []
        for sink_window in args.execution_sink_windows:
            for recent_window in args.execution_recent_windows:
                for relevance_top_k in args.execution_relevance_top_ks:
                    result = run_session_benchmark(
                        backend=args.backend,
                        config=config,
                        context_length=context_length,
                        decode_steps=args.decode_steps,
                        seed=args.seed,
                        execution_recent_window=recent_window,
                        execution_sink_window=sink_window,
                        execution_relevance_top_k=relevance_top_k,
                        execution_relevance_sketch_size=1,
                        execution_relevance_mode="envelope",
                        execution_exact_refine_top_k=0,
                        execution_approximate_old_pages=False,
                    )
                    result["record_type"] = "result"
                    results.append(result)

        frontier_indices = set(_pareto_frontier(results))
        for index, result in enumerate(results):
            enriched = dict(result)
            enriched["is_pareto_frontier"] = int(index in frontier_indices)
            emit(enriched)

        best_runtime = min(results, key=lambda item: float(item["session_runtime_ms_per_step"]))
        best_error = min(results, key=lambda item: float(item["step_max_abs_error"]))
        best_frontier = sorted(
            [results[index] for index in frontier_indices],
            key=lambda item: (float(item["session_runtime_ms_per_step"]), float(item["step_max_abs_error"])),
        )
        emit(
            {
                "record_type": "summary",
                "context_length": context_length,
                "backend": args.backend,
                "frontier_point_count": len(best_frontier),
                "best_runtime_sink_window": int(best_runtime["execution_sink_window"]),
                "best_runtime_recent_window": int(best_runtime["execution_recent_window"]),
                "best_runtime_relevance_top_k": int(best_runtime["execution_relevance_top_k"]),
                "best_runtime_session_ms_per_step": float(best_runtime["session_runtime_ms_per_step"]),
                "best_runtime_step_max_abs_error": float(best_runtime["step_max_abs_error"]),
                "best_error_sink_window": int(best_error["execution_sink_window"]),
                "best_error_recent_window": int(best_error["execution_recent_window"]),
                "best_error_relevance_top_k": int(best_error["execution_relevance_top_k"]),
                "best_error_session_ms_per_step": float(best_error["session_runtime_ms_per_step"]),
                "best_error_step_max_abs_error": float(best_error["step_max_abs_error"]),
            }
        )


if __name__ == "__main__":
    main()
