from __future__ import annotations

from dotcache.attention_runtime import prepare_pages
from dotcache.encode import encode_page
from dotcache.tracing import ExecutionTrace

from bench_common import (
    average_ms,
    build_config,
    build_fixture,
    emit,
    page_byte_totals,
    parse_args,
)


def main() -> None:
    args = parse_args("Benchmark page encoding across a short context ladder.", default_repeats=10)
    config = build_config(args)

    for context_length in args.contexts:
        fixture = build_fixture(context_length, config, seed=args.seed)
        keys = fixture["keys"]
        values = fixture["values"]

        def run_encode():
            key_pages = []
            value_pages = []
            for token_start in range(0, context_length, config.tokens_per_page):
                token_end = min(token_start + config.tokens_per_page, context_length)
                key_pages.append(
                    encode_page(
                        keys[token_start:token_end],
                        config,
                        kind="K",
                        token_start=token_start,
                    )
                )
                value_pages.append(
                    encode_page(
                        values[token_start:token_end],
                        config,
                        kind="V",
                        token_start=token_start,
                    )
                )
            return key_pages, value_pages

        encode_ms = average_ms(lambda: run_encode(), args.repeats)
        key_pages, value_pages = run_encode()
        payload_bytes_k, metadata_bytes_k = page_byte_totals(key_pages)
        payload_bytes_v, metadata_bytes_v = page_byte_totals(value_pages)

        prep_trace = ExecutionTrace()
        prepare_pages(key_pages + value_pages, backend=args.backend, trace=prep_trace)

        record = {
            "backend": args.backend,
            "context_length": context_length,
            "encode_ms": encode_ms,
            "encode_ms_per_page": encode_ms / max(len(key_pages) + len(value_pages), 1),
            "key_metadata_bytes": metadata_bytes_k,
            "key_payload_bytes": payload_bytes_k,
            "page_count": len(key_pages) + len(value_pages),
            "value_metadata_bytes": metadata_bytes_v,
            "value_payload_bytes": payload_bytes_v,
            **prep_trace.to_dict(),
        }
        emit(record)


if __name__ == "__main__":
    main()
