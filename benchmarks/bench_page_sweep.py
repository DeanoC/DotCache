from __future__ import annotations

import argparse

from dotcache.attention_runtime import decode_step

from bench_common import build_config, build_fixture, emit, prepare_context_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep tokens_per_page to find MPS/CPU crossover points.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=[4096])
    parser.add_argument("--head-dims", nargs="*", type=int, default=[128, 256])
    parser.add_argument("--page-sizes", nargs="*", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def average_ms(fn, repeats: int) -> float:
    fn()
    import time

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return ((time.perf_counter() - start) * 1000.0) / repeats


def main() -> None:
    args = parse_args()

    for context_length in args.contexts:
        for head_dim in args.head_dims:
            for tokens_per_page in args.page_sizes:
                config = build_config(
                    argparse.Namespace(
                        backend="auto",
                        config=args.config,
                        contexts=[context_length],
                        repeats=args.repeats,
                        head_dim=head_dim,
                        group_size=args.group_size,
                        tokens_per_page=tokens_per_page,
                        seed=args.seed,
                    )
                )
                fixture = build_fixture(context_length, config, seed=args.seed)
                key_pages = fixture["key_pages"]
                value_pages = fixture["value_pages"]
                query = fixture["query"]

                prepared_key_pages, key_prepare_ms, key_prep_trace = prepare_context_pages(key_pages, "torch_mps")
                prepared_value_pages, value_prepare_ms, value_prep_trace = prepare_context_pages(value_pages, "torch_mps")

                cpu_ms = average_ms(lambda: decode_step(query, key_pages, value_pages, backend="cpu_ref"), args.repeats)
                mps_ms = average_ms(
                    lambda: decode_step(query, prepared_key_pages, prepared_value_pages, backend="torch_mps"),
                    args.repeats,
                )

                emit(
                    {
                        "context_length": context_length,
                        "cpu_decode_ms": cpu_ms,
                        "head_dim": head_dim,
                        "mps_decode_ms": mps_ms,
                        "mps_host_to_device_bytes": key_prep_trace.host_to_device_bytes + value_prep_trace.host_to_device_bytes,
                        "mps_prepare_ms": key_prepare_ms + value_prepare_ms,
                        "page_count": len(key_pages),
                        "speedup_vs_cpu": cpu_ms / mps_ms,
                        "tokens_per_page": tokens_per_page,
                    }
                )


if __name__ == "__main__":
    main()
