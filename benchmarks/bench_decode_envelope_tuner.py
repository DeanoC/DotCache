from __future__ import annotations

import argparse

from bench_common import build_config, emit, warm_backend
from bench_decode_session import run_session_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune envelope-gated session profiles under runtime budgets.")
    parser.add_argument("--backend", choices=["torch_mps"], default="torch_mps")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=[8192, 16384])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-sink-windows", nargs="*", type=int, default=[256])
    parser.add_argument("--execution-recent-windows", nargs="*", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--execution-relevance-top-ks", nargs="*", type=int, default=[2, 4, 8])
    parser.add_argument("--balanced-runtime-slack-ms", type=float, default=1.5)
    parser.add_argument("--balanced-runtime-slack-ratio", type=float, default=1.25)
    return parser.parse_args()


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
                    result["record_type"] = "candidate"
                    results.append(result)
                    emit(result)

        fastest = min(results, key=lambda item: float(item["session_runtime_ms_per_step"]))
        fastest_ms = float(fastest["session_runtime_ms_per_step"])
        balanced_budget_ms = max(
            fastest_ms + args.balanced_runtime_slack_ms,
            fastest_ms * args.balanced_runtime_slack_ratio,
        )
        balanced_candidates = [
            record for record in results if float(record["session_runtime_ms_per_step"]) <= balanced_budget_ms
        ]
        balanced = min(
            balanced_candidates,
            key=lambda item: (float(item["step_max_abs_error"]), float(item["session_runtime_ms_per_step"])),
        )
        emit(
            {
                "record_type": "summary",
                "context_length": context_length,
                "backend": args.backend,
                "balanced_runtime_budget_ms": balanced_budget_ms,
                "fast_sink_window": int(fastest["execution_sink_window"]),
                "fast_recent_window": int(fastest["execution_recent_window"]),
                "fast_relevance_top_k": int(fastest["execution_relevance_top_k"]),
                "fast_session_ms_per_step": float(fastest["session_runtime_ms_per_step"]),
                "fast_step_max_abs_error": float(fastest["step_max_abs_error"]),
                "balanced_sink_window": int(balanced["execution_sink_window"]),
                "balanced_recent_window": int(balanced["execution_recent_window"]),
                "balanced_relevance_top_k": int(balanced["execution_relevance_top_k"]),
                "balanced_session_ms_per_step": float(balanced["session_runtime_ms_per_step"]),
                "balanced_step_max_abs_error": float(balanced["step_max_abs_error"]),
            }
        )


if __name__ == "__main__":
    main()
