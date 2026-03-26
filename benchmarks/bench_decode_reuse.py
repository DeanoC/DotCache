from __future__ import annotations

from dotcache.attention_runtime import decode_step
from dotcache.page_cache import PreparedPageCache
from dotcache.tracing import ExecutionTrace

from bench_common import (
    average_ms,
    build_config,
    build_fixture,
    build_queries,
    emit,
    error_stats,
    parse_args,
)


def _aggregate_step_trace(
    queries,
    key_pages,
    value_pages,
    *,
    backend: str,
    cache: PreparedPageCache | None = None,
) -> tuple[list, list, ExecutionTrace]:
    outputs = []
    logits = []
    total_trace = ExecutionTrace()
    for query in queries:
        step_trace = ExecutionTrace()
        step_logits, _, step_output = decode_step(
            query,
            key_pages,
            value_pages,
            backend=backend,
            cache=cache,
            trace=step_trace,
        )
        total_trace.merge(step_trace)
        logits.append(step_logits)
        outputs.append(step_output)
    return logits, outputs, total_trace


def main() -> None:
    args = parse_args(
        "Benchmark repeated decode steps with and without persistent prepared-page reuse.",
        default_repeats=3,
    )
    config = build_config(args)

    for context_length in args.contexts:
        fixture = build_fixture(context_length, config, seed=args.seed)
        queries = build_queries(context_length, config.head_dim, steps=args.decode_steps, seed=args.seed + 10_000)
        key_pages = fixture["key_pages"]
        value_pages = fixture["value_pages"]

        reference_logits, reference_outputs, _ = _aggregate_step_trace(
            queries,
            key_pages,
            value_pages,
            backend="cpu_ref",
        )
        cpu_ms = average_ms(
            lambda: _aggregate_step_trace(queries, key_pages, value_pages, backend="cpu_ref"),
            args.repeats,
        )

        no_cache_ms = average_ms(
            lambda: _aggregate_step_trace(queries, key_pages, value_pages, backend=args.backend),
            args.repeats,
        )
        no_cache_logits, no_cache_outputs, no_cache_trace = _aggregate_step_trace(
            queries,
            key_pages,
            value_pages,
            backend=args.backend,
        )

        cold_cache_ms = average_ms(
            lambda: _aggregate_step_trace(
                queries,
                key_pages,
                value_pages,
                backend=args.backend,
                cache=PreparedPageCache(),
            ),
            args.repeats,
        )
        cold_cache_logits, cold_cache_outputs, cold_cache_trace = _aggregate_step_trace(
            queries,
            key_pages,
            value_pages,
            backend=args.backend,
            cache=PreparedPageCache(),
        )

        warm_cache = PreparedPageCache()
        _aggregate_step_trace(queries[:1], key_pages, value_pages, backend=args.backend, cache=warm_cache)
        warm_cache_ms = average_ms(
            lambda: _aggregate_step_trace(queries, key_pages, value_pages, backend=args.backend, cache=warm_cache),
            args.repeats,
        )
        warm_cache_logits, warm_cache_outputs, warm_cache_trace = _aggregate_step_trace(
            queries,
            key_pages,
            value_pages,
            backend=args.backend,
            cache=warm_cache,
        )

        all_reference_logits = [item for step in reference_logits for item in step]
        all_no_cache_logits = [item for step in no_cache_logits for item in step]
        all_cold_cache_logits = [item for step in cold_cache_logits for item in step]
        all_warm_cache_logits = [item for step in warm_cache_logits for item in step]

        record = {
            "backend": args.backend,
            "cache_cold_decode_ms_per_step": cold_cache_ms / args.decode_steps,
            "cache_cold_hit_rate": cold_cache_trace.prepared_page_cache_hits
            / max(cold_cache_trace.prepared_page_cache_hits + cold_cache_trace.prepared_page_cache_misses, 1),
            "cache_cold_host_to_device_bytes_per_step": cold_cache_trace.host_to_device_bytes / args.decode_steps,
            "cache_resident_bytes": warm_cache_trace.cache_resident_bytes,
            "cache_warm_decode_ms_per_step": warm_cache_ms / args.decode_steps,
            "cache_warm_hit_rate": warm_cache_trace.prepared_page_cache_hits
            / max(warm_cache_trace.prepared_page_cache_hits + warm_cache_trace.prepared_page_cache_misses, 1),
            "cache_warm_host_to_device_bytes_per_step": warm_cache_trace.host_to_device_bytes / args.decode_steps,
            "context_length": context_length,
            "cpu_decode_ms_per_step": cpu_ms / args.decode_steps,
            "decode_steps": args.decode_steps,
            "no_cache_decode_ms_per_step": no_cache_ms / args.decode_steps,
            "no_cache_host_to_device_bytes_per_step": no_cache_trace.host_to_device_bytes / args.decode_steps,
            "page_count": len(key_pages),
            "speedup_cache_cold_vs_cpu": cpu_ms / max(cold_cache_ms, 1e-8),
            "speedup_cache_cold_vs_no_cache": no_cache_ms / max(cold_cache_ms, 1e-8),
            "speedup_cache_warm_vs_cpu": cpu_ms / max(warm_cache_ms, 1e-8),
            "speedup_cache_warm_vs_no_cache": no_cache_ms / max(warm_cache_ms, 1e-8),
            "tokens_per_page": config.tokens_per_page,
            **{f"no_cache_{k}": v for k, v in error_stats(all_no_cache_logits, all_reference_logits).items()},
            **{f"cache_cold_{k}": v for k, v in error_stats(all_cold_cache_logits, all_reference_logits).items()},
            **{f"cache_warm_{k}": v for k, v in error_stats(all_warm_cache_logits, all_reference_logits).items()},
            **{
                f"no_cache_output_{k}": v
                for k, v in error_stats(
                    [item for step in no_cache_outputs for item in step],
                    [item for step in reference_outputs for item in step],
                ).items()
            },
            **{
                f"cache_cold_output_{k}": v
                for k, v in error_stats(
                    [item for step in cold_cache_outputs for item in step],
                    [item for step in reference_outputs for item in step],
                ).items()
            },
            **{
                f"cache_warm_output_{k}": v
                for k, v in error_stats(
                    [item for step in warm_cache_outputs for item in step],
                    [item for step in reference_outputs for item in step],
                ).items()
            },
        }
        emit(record)


if __name__ == "__main__":
    main()
