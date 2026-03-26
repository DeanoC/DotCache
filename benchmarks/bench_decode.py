from __future__ import annotations

from dotcache.attention_runtime import decode_step
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
    args = parse_args("Benchmark full decode-step execution across a short context ladder.", default_repeats=10)
    config = build_config(args)

    for context_length in args.contexts:
        fixture = build_fixture(context_length, config, seed=args.seed)
        key_pages = fixture["key_pages"]
        value_pages = fixture["value_pages"]
        query = fixture["query"]
        prepared_key_pages, key_prepare_ms, key_prep_trace = prepare_context_pages(key_pages, args.backend)
        prepared_value_pages, value_prepare_ms, value_prep_trace = prepare_context_pages(value_pages, args.backend)
        payload_bytes_k, metadata_bytes_k = page_byte_totals(key_pages)
        payload_bytes_v, metadata_bytes_v = page_byte_totals(value_pages)

        reference_logits, _, reference_output = decode_step(query, key_pages, value_pages, backend="cpu_ref")

        def run() -> None:
            decode_step(query, prepared_key_pages, prepared_value_pages, backend=args.backend)

        decode_ms = average_ms(run, args.repeats)
        exec_trace = ExecutionTrace()
        backend_logits, _, backend_output = decode_step(
            query,
            prepared_key_pages,
            prepared_value_pages,
            backend=args.backend,
            trace=exec_trace,
        )

        record = {
            "backend": args.backend,
            "context_length": context_length,
            "decode_ms": decode_ms,
            "execution_host_to_device_bytes": exec_trace.host_to_device_bytes,
            "execution_max_temporary_bytes": exec_trace.max_temporary_bytes,
            "execution_metadata_bytes_read": exec_trace.metadata_bytes_read,
            "execution_m0_full_page_materializations": exec_trace.m0_full_page_materializations,
            "execution_payload_bytes_read": exec_trace.payload_bytes_read,
            "host_to_device_bytes": key_prep_trace.host_to_device_bytes + value_prep_trace.host_to_device_bytes,
            "key_metadata_bytes": metadata_bytes_k,
            "key_payload_bytes": payload_bytes_k,
            "page_count": len(key_pages),
            "prepare_ms": key_prepare_ms + value_prepare_ms,
            "value_metadata_bytes": metadata_bytes_v,
            "value_payload_bytes": payload_bytes_v,
            **error_stats(backend_logits, reference_logits),
            **{f"output_{k}": v for k, v in error_stats(backend_output, reference_output).items()},
        }
        emit(record)


if __name__ == "__main__":
    main()
