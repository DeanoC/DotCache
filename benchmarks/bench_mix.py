from __future__ import annotations

import numpy as np

from dotcache.attention_runtime import mix_page
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
    split_weights,
)


def main() -> None:
    args = parse_args("Benchmark page mixing across a short context ladder.", default_repeats=20)
    config = build_config(args)

    for context_length in args.contexts:
        fixture = build_fixture(context_length, config, seed=args.seed)
        value_pages = fixture["value_pages"]
        weight_chunks = split_weights(fixture["attn"], value_pages)
        prepared_value_pages, prepare_ms, prep_trace = prepare_context_pages(value_pages, args.backend)
        payload_bytes, metadata_bytes = page_byte_totals(value_pages)

        reference_output = np.zeros(config.head_dim, dtype=np.float32)
        for weights, page in zip(weight_chunks, value_pages, strict=True):
            reference_output = mix_page(weights, page, out_acc=reference_output, backend="cpu_ref")

        def run() -> None:
            output = np.zeros(config.head_dim, dtype=np.float32)
            for weights, page in zip(weight_chunks, prepared_value_pages, strict=True):
                output = mix_page(weights, page, out_acc=output, backend=args.backend)

        mix_ms = average_ms(run, args.repeats)
        exec_trace = ExecutionTrace()
        backend_output = np.zeros(config.head_dim, dtype=np.float32)
        for weights, page in zip(weight_chunks, prepared_value_pages, strict=True):
            backend_output = mix_page(weights, page, out_acc=backend_output, backend=args.backend, trace=exec_trace)

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
            "mix_ms": mix_ms,
            "mix_ms_per_page": mix_ms / max(len(value_pages), 1),
            "page_count": len(value_pages),
            "payload_bytes": payload_bytes,
            "prepare_ms": prepare_ms,
            **error_stats(backend_output, reference_output),
        }
        emit(record)


if __name__ == "__main__":
    main()
