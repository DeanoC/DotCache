from __future__ import annotations

import numpy as np

from dotcache.attention_runtime import score_page
from dotcache.tracing import ExecutionTrace

from bench_common import (
    average_ms,
    build_config,
    build_fixture,
    emit,
    error_stats,
    page_byte_totals,
    parse_args,
    prepare_context_pages,
)


def main() -> None:
    args = parse_args("Benchmark page scoring across a short context ladder.", default_repeats=20)
    config = build_config(args)

    for context_length in args.contexts:
        fixture = build_fixture(context_length, config, seed=args.seed)
        key_pages = fixture["key_pages"]
        query = fixture["query"]
        prepared_key_pages, prepare_ms, prep_trace = prepare_context_pages(key_pages, args.backend)
        payload_bytes, metadata_bytes = page_byte_totals(key_pages)

        reference_logits = [
            score_page(query, page, backend="cpu_ref")
            for page in key_pages
        ]

        def run() -> None:
            for page in prepared_key_pages:
                score_page(query, page, backend=args.backend)

        score_ms = average_ms(run, args.repeats)
        exec_trace = ExecutionTrace()
        backend_logits = [
            score_page(query, page, backend=args.backend, trace=exec_trace)
            for page in prepared_key_pages
        ]

        record = {
            "backend": args.backend,
            "context_length": context_length,
            "execution_host_to_device_bytes": exec_trace.host_to_device_bytes,
            "execution_max_temporary_bytes": exec_trace.max_temporary_bytes,
            "execution_metadata_bytes_read": exec_trace.metadata_bytes_read,
            "execution_m0_full_page_materializations": exec_trace.m0_full_page_materializations,
            "execution_payload_bytes_read": exec_trace.payload_bytes_read,
            "host_to_device_bytes": prep_trace.host_to_device_bytes,
            "metadata_bytes": metadata_bytes,
            "page_count": len(key_pages),
            "payload_bytes": payload_bytes,
            "prepare_ms": prepare_ms,
            "score_ms": score_ms,
            "score_ms_per_page": score_ms / max(len(key_pages), 1),
            **error_stats(
                np.concatenate(backend_logits),
                np.concatenate(reference_logits),
            ),
        }
        emit(record)


if __name__ == "__main__":
    main()
