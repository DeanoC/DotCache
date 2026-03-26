from __future__ import annotations

import argparse

import numpy as np

from dotcache.attention_runtime import decode_step
from dotcache.backends import prepare_page_mps
from dotcache.encode import encode_page
from dotcache.page_cache import PreparedPageCache
from dotcache.tracing import ExecutionTrace

from bench_common import build_config, build_queries, emit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep prepared-page cache capacity under growing-context decode.")
    parser.add_argument("--backend", choices=["torch_mps"], default="torch_mps")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=[4096])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cache-policies",
        nargs="*",
        choices=["fifo", "lru", "pinned_recent_fifo"],
        default=["fifo", "lru", "pinned_recent_fifo"],
    )
    parser.add_argument("--capacity-page-pairs", nargs="*", default=["1", "2", "4", "8", "initial", "final", "unbounded"])
    parser.add_argument("--working-set-page-pairs", nargs="*", default=["all", "4"])
    parser.add_argument("--pinned-recent-page-pairs", nargs="*", default=["4"])
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _time_ms(fn) -> float:
    import time

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


def _parse_capacity_multiplier(raw: str, *, initial_page_pairs: int, final_page_pairs: int) -> int | None:
    if raw.lower() in {"unbounded", "none", "inf"}:
        return None
    if raw.lower() == "initial":
        return initial_page_pairs
    if raw.lower() == "final":
        return final_page_pairs
    return int(raw)


def _slice_working_set(key_pages: list, value_pages: list, working_set_page_pairs: int | None) -> tuple[list, list]:
    if working_set_page_pairs is None:
        return key_pages, value_pages
    return key_pages[-working_set_page_pairs:], value_pages[-working_set_page_pairs:]


def _run_growth_once(
    *,
    backend: str,
    config,
    context_length: int,
    decode_steps: int,
    seed: int,
    max_resident_bytes: int | None,
    cache_policy: str,
    pinned_recent_pages: int,
    working_set_page_pairs: int | None,
) -> dict[str, float | int]:
    append_tokens = config.tokens_per_page
    initial_context = max(config.tokens_per_page, context_length)
    keys, values = _build_growth_fixture(
        initial_context=initial_context,
        decode_steps=decode_steps,
        append_tokens=append_tokens,
        head_dim=config.head_dim,
        seed=seed,
    )
    queries = build_queries(initial_context, config.head_dim, steps=decode_steps, seed=seed + 20_000)

    base_keys = keys[:initial_context]
    base_values = values[:initial_context]
    key_pages = _encode_slice(base_keys, config, kind="K", token_start=0)
    value_pages = _encode_slice(base_values, config, kind="V", token_start=0)

    cpu_total_ms = 0.0
    mps_total_ms = 0.0
    cpu_outputs: list[np.ndarray] = []
    mps_outputs: list[np.ndarray] = []

    growth_trace = ExecutionTrace()
    cache = PreparedPageCache(
        max_resident_bytes=max_resident_bytes,
        policy=cache_policy,
        pinned_recent_pages=pinned_recent_pages,
    )

    initial_prep_trace = ExecutionTrace()
    cache.append_pages(key_pages + value_pages, trace=initial_prep_trace)

    current_context = initial_context
    for query in queries:
        cpu_total_ms += _time_ms(
            lambda q=query, kp=list(key_pages), vp=list(value_pages): cpu_outputs.append(
                decode_step(
                    q,
                    *_slice_working_set(kp, vp, working_set_page_pairs),
                    backend="cpu_ref",
                )[2]
            )
        )

        step_trace = ExecutionTrace()
        mps_total_ms += _time_ms(
            lambda q=query, kp=list(key_pages), vp=list(value_pages), st=step_trace: mps_outputs.append(
                decode_step(
                    q,
                    *_slice_working_set(kp, vp, working_set_page_pairs),
                    backend=backend,
                    cache=cache,
                    trace=st,
                )[2]
            )
        )
        growth_trace.merge(step_trace)

        next_key_tokens = keys[current_context : current_context + append_tokens]
        next_value_tokens = values[current_context : current_context + append_tokens]
        if next_key_tokens.shape[0] == 0:
            continue
        new_key_pages = _encode_slice(next_key_tokens, config, kind="K", token_start=current_context)
        new_value_pages = _encode_slice(next_value_tokens, config, kind="V", token_start=current_context)
        append_trace = ExecutionTrace()
        cache.append_pages(new_key_pages + new_value_pages, trace=append_trace)
        growth_trace.merge(append_trace)
        key_pages.extend(new_key_pages)
        value_pages.extend(new_value_pages)
        current_context += append_tokens

    output_delta = np.abs(np.stack(mps_outputs) - np.stack(cpu_outputs))
    output_ref = np.maximum(np.abs(np.stack(cpu_outputs)), 1e-8)
    return {
        "cache_hit_rate": growth_trace.prepared_page_cache_hits
        / max(growth_trace.prepared_page_cache_hits + growth_trace.prepared_page_cache_misses, 1),
        "cache_resident_bytes": growth_trace.cache_resident_bytes,
        "cpu_decode_ms_per_step": cpu_total_ms / decode_steps,
        "initial_host_to_device_bytes": initial_prep_trace.host_to_device_bytes,
        "mps_decode_ms_per_step": mps_total_ms / decode_steps,
        "page_count_final": len(key_pages),
        "prepared_page_cache_evictions": growth_trace.prepared_page_cache_evictions,
        "prepared_page_cache_hits": growth_trace.prepared_page_cache_hits,
        "prepared_page_cache_misses": growth_trace.prepared_page_cache_misses,
        "step_host_to_device_bytes": growth_trace.host_to_device_bytes / decode_steps,
        "step_max_abs_error": float(np.max(output_delta)),
        "step_max_rel_error": float(np.max(output_delta / output_ref)),
        "speedup_vs_cpu": cpu_total_ms / max(mps_total_ms, 1e-8),
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # Derive a single appended K+V page-pair size from the chosen config.
    sample_keys, sample_values = _build_growth_fixture(
        initial_context=config.tokens_per_page,
        decode_steps=1,
        append_tokens=config.tokens_per_page,
        head_dim=config.head_dim,
        seed=args.seed,
    )
    sample_key_page = _encode_slice(sample_keys[: config.tokens_per_page], config, kind="K", token_start=0)[0]
    sample_value_page = _encode_slice(sample_values[: config.tokens_per_page], config, kind="V", token_start=0)[0]
    prepared_key_page = prepare_page_mps(sample_key_page)
    prepared_value_page = prepare_page_mps(sample_value_page)
    append_page_pair_bytes = prepared_key_page.host_to_device_nbytes + prepared_value_page.host_to_device_nbytes

    for context_length in args.contexts:
        initial_page_pairs = max(config.tokens_per_page, context_length) // config.tokens_per_page
        final_page_pairs = initial_page_pairs + args.decode_steps
        for raw_capacity in args.capacity_page_pairs:
            multiplier = _parse_capacity_multiplier(
                raw_capacity,
                initial_page_pairs=initial_page_pairs,
                final_page_pairs=final_page_pairs,
            )
            max_resident_bytes = None if multiplier is None else multiplier * append_page_pair_bytes
            for cache_policy in args.cache_policies:
                pinned_page_pair_options = [0]
                if cache_policy == "pinned_recent_fifo":
                    pinned_page_pair_options = [int(raw_value) for raw_value in args.pinned_recent_page_pairs]
                for raw_working_set in args.working_set_page_pairs:
                    working_set_page_pairs = None if raw_working_set == "all" else int(raw_working_set)
                    for pinned_recent_page_pairs in pinned_page_pair_options:
                        result = _run_growth_once(
                            backend=args.backend,
                            config=config,
                            context_length=context_length,
                            decode_steps=args.decode_steps,
                            seed=args.seed,
                            max_resident_bytes=max_resident_bytes,
                            cache_policy=cache_policy,
                            pinned_recent_pages=2 * pinned_recent_page_pairs,
                            working_set_page_pairs=working_set_page_pairs,
                        )
                        emit(
                            {
                                "append_page_pair_bytes": append_page_pair_bytes,
                                "backend": args.backend,
                                "cache_policy": cache_policy,
                                "capacity_label": raw_capacity,
                                "capacity_page_pairs": -1 if multiplier is None else multiplier,
                                "context_length": context_length,
                                "decode_steps": args.decode_steps,
                                "max_resident_bytes": -1 if max_resident_bytes is None else max_resident_bytes,
                                "pinned_recent_page_pairs": pinned_recent_page_pairs,
                                "max_pinned_recent_pages": 2 * pinned_recent_page_pairs,
                                "tokens_per_page": config.tokens_per_page,
                                "working_set_label": raw_working_set,
                                "working_set_page_pairs": -1 if working_set_page_pairs is None else working_set_page_pairs,
                                **result,
                            }
                        )


if __name__ == "__main__":
    main()
