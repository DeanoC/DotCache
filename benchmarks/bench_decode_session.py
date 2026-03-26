from __future__ import annotations

import time

import numpy as np

from dotcache.attention_runtime import decode_step
from dotcache.encode import encode_page
from dotcache.page_cache import PreparedPageCache
from dotcache.session_runtime import PagedDecodeSession
from dotcache.tracing import ExecutionTrace

from bench_common import build_config, build_queries, emit, parse_args, warm_backend


def _time_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _build_growth_fixture(
    *,
    initial_context: int,
    decode_steps: int,
    append_tokens: int,
    head_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_tokens = initial_context + decode_steps * append_tokens
    rng = np.random.default_rng(seed + total_tokens + head_dim)
    keys = rng.normal(size=(total_tokens, head_dim)).astype(np.float32)
    values = rng.normal(size=(total_tokens, head_dim)).astype(np.float32)
    return keys, values


def _encode_slice(values: np.ndarray, config, *, kind: str, token_start: int) -> list:
    pages = []
    for start in range(token_start, token_start + values.shape[0], config.tokens_per_page):
        offset = start - token_start
        end_offset = min(offset + config.tokens_per_page, values.shape[0])
        pages.append(
            encode_page(
                values[offset:end_offset],
                config,
                kind=kind,
                token_start=start,
            )
        )
    return pages


def main() -> None:
    args = parse_args(
        "Benchmark a session-shaped runtime with separate preload, append, and decode phases.",
        default_repeats=1,
    )
    config = build_config(args)
    append_tokens = config.tokens_per_page
    warm_backend(args.backend)

    for context_length in args.contexts:
        initial_context = max(config.tokens_per_page, context_length)
        keys, values = _build_growth_fixture(
            initial_context=initial_context,
            decode_steps=args.decode_steps,
            append_tokens=append_tokens,
            head_dim=config.head_dim,
            seed=args.seed,
        )
        queries = build_queries(initial_context, config.head_dim, steps=args.decode_steps, seed=args.seed + 30_000)

        raw_key_pages = _encode_slice(keys[:initial_context], config, kind="K", token_start=0)
        raw_value_pages = _encode_slice(values[:initial_context], config, kind="V", token_start=0)

        session = PagedDecodeSession(
            backend=args.backend,
            cache=PreparedPageCache(),
            recent_window_tokens=args.execution_recent_window,
            sink_window_tokens=args.execution_sink_window,
            relevance_top_k=args.execution_relevance_top_k,
            relevance_sketch_size=args.execution_relevance_sketch_size,
            relevance_mode=args.execution_relevance_mode,
            exact_refine_top_k=args.execution_exact_refine_top_k,
            approximate_old_pages=args.execution_approximate_old_pages,
        )
        preload_trace = ExecutionTrace()
        preload_ms = _time_ms(lambda: session.preload(raw_key_pages, raw_value_pages, trace=preload_trace))

        cpu_total_ms = 0.0
        decode_total_ms = 0.0
        append_total_ms = 0.0
        decode_trace_total = ExecutionTrace()
        append_trace_total = ExecutionTrace()
        cpu_outputs: list[np.ndarray] = []
        session_outputs: list[np.ndarray] = []
        active_page_counts: list[int] = []
        active_token_counts: list[int] = []
        current_context = initial_context

        for query in queries:
            cpu_total_ms += _time_ms(
                lambda q=query, kp=list(raw_key_pages), vp=list(raw_value_pages): cpu_outputs.append(
                    decode_step(q, kp, vp, backend="cpu_ref")[2]
                )
            )

            step_trace = ExecutionTrace()
            decode_total_ms += _time_ms(
                lambda q=query, st=step_trace: session_outputs.append(session.decode(q, trace=st)[2])
            )
            decode_trace_total.merge(step_trace)
            active_indices = session.last_selected_indices
            active_page_counts.append(len(active_indices))
            active_token_counts.append(sum(session.key_pages[index].header.token_count for index in active_indices))

            next_key_tokens = keys[current_context : current_context + append_tokens]
            next_value_tokens = values[current_context : current_context + append_tokens]
            if next_key_tokens.shape[0] == 0:
                continue
            new_key_pages = _encode_slice(next_key_tokens, config, kind="K", token_start=current_context)
            new_value_pages = _encode_slice(next_value_tokens, config, kind="V", token_start=current_context)
            append_trace = ExecutionTrace()
            append_total_ms += _time_ms(
                lambda nk=new_key_pages, nv=new_value_pages, at=append_trace: session.append(nk, nv, trace=at)
            )
            append_trace_total.merge(append_trace)
            raw_key_pages.extend(new_key_pages)
            raw_value_pages.extend(new_value_pages)
            current_context += append_tokens

        output_delta = np.abs(np.stack(session_outputs) - np.stack(cpu_outputs))
        output_ref = np.maximum(np.abs(np.stack(cpu_outputs)), 1e-8)

        emit(
            {
                "active_page_count": float(np.mean(active_page_counts)),
                "active_page_count_last": active_page_counts[-1],
                "active_token_count": float(np.mean(active_token_counts)),
                "active_token_count_last": active_token_counts[-1],
                "backend": args.backend,
                "append_host_to_device_bytes_per_step": append_trace_total.host_to_device_bytes / args.decode_steps,
                "append_ms_per_step": append_total_ms / args.decode_steps,
                "append_tokens": append_tokens,
                "cache_resident_bytes": session.cache.resident_bytes if session.cache is not None else 0,
                "context_length": context_length,
                "cpu_decode_ms_per_step": cpu_total_ms / args.decode_steps,
                "decode_host_to_device_bytes_per_step": decode_trace_total.host_to_device_bytes / args.decode_steps,
                "decode_ms_per_step": decode_total_ms / args.decode_steps,
                "decode_steps": args.decode_steps,
                "final_page_count": session.page_count,
                "initial_context": initial_context,
                "preload_host_to_device_bytes": preload_trace.host_to_device_bytes,
                "preload_ms": preload_ms,
                "execution_recent_window": -1 if args.execution_recent_window is None else args.execution_recent_window,
                "execution_approximate_old_pages": int(args.execution_approximate_old_pages),
                "execution_exact_refine_top_k": args.execution_exact_refine_top_k,
                "execution_relevance_mode": args.execution_relevance_mode,
                "execution_relevance_top_k": args.execution_relevance_top_k,
                "execution_relevance_sketch_size": args.execution_relevance_sketch_size,
                "execution_sink_window": args.execution_sink_window,
                "session_runtime_ms_per_step": (decode_total_ms + append_total_ms) / args.decode_steps,
                "speedup_decode_vs_cpu": cpu_total_ms / max(decode_total_ms, 1e-8),
                "speedup_session_runtime_vs_cpu": cpu_total_ms / max(decode_total_ms + append_total_ms, 1e-8),
                "step_max_abs_error": float(np.max(output_delta)),
                "step_max_rel_error": float(np.max(output_delta / output_ref)),
                "tokens_per_page": config.tokens_per_page,
            }
        )


if __name__ == "__main__":
    main()
