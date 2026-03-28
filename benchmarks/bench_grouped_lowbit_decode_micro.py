from __future__ import annotations

import argparse
import json
import time
from typing import Iterable

import numpy as np

from dotcache.attention_runtime import prepare_pages
from dotcache.backends import (
    clear_prepared_chunk_cache,
    configure_prepared_chunk_cache,
    decode_grouped_multiquery_step_prepared_torch_tensor_output_only,
    mps_available,
    prepared_chunk_cache_resident_bytes,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic grouped low-bit decode microbenchmark.")
    parser.add_argument("--device", choices=["mps"], default="mps")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--bits-ladder", type=int, nargs="+", choices=[2, 3, 4], default=None)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--tokens-per-page-ladder", type=int, nargs="+", default=None)
    parser.add_argument("--page-count", type=int, default=4)
    parser.add_argument("--page-count-ladder", type=int, nargs="+", default=None)
    parser.add_argument("--kv-group-count", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-count-ladder", type=int, nargs="+", default=None)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-format", choices=["pretty", "json"], default="pretty")
    return parser.parse_args()


def _synchronize(device: str) -> None:
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _encode_group(values: np.ndarray, config: DotCacheConfig, *, kind: str, kv_head_id: int) -> list:
    pages = []
    for page_index in range(values.shape[0]):
        token_start = page_index * config.tokens_per_page
        pages.append(
            encode_page(
                values[page_index],
                config,
                kind=kind,
                kv_head_id=kv_head_id,
                token_start=token_start,
            )
        )
    return pages


def _as_ladder(values: Iterable[int] | None, *, fallback: int) -> list[int]:
    if values is None:
        return [int(fallback)]
    ladder = [int(value) for value in values]
    if not ladder:
        return [int(fallback)]
    return ladder


def _run_case(
    *,
    device: str,
    bits: int,
    head_dim: int,
    group_size: int,
    tokens_per_page: int,
    page_count: int,
    kv_group_count: int,
    query_count: int,
    warmup_iters: int,
    bench_iters: int,
    seed: int,
):
    import torch

    rng = np.random.default_rng(seed)
    config = DotCacheConfig(
        head_dim=head_dim,
        group_size=group_size,
        bits_k=bits,
        bits_v=bits,
        tokens_per_page=tokens_per_page,
    )

    key_pages_by_group = []
    value_pages_by_group = []
    query_groups = []
    for kv_group_id in range(kv_group_count):
        keys = rng.normal(size=(page_count, tokens_per_page, head_dim)).astype(np.float32)
        values = rng.normal(size=(page_count, tokens_per_page, head_dim)).astype(np.float32)
        key_pages = prepare_pages(_encode_group(keys, config, kind="K", kv_head_id=kv_group_id), backend="torch_mps")
        value_pages = prepare_pages(_encode_group(values, config, kind="V", kv_head_id=kv_group_id), backend="torch_mps")
        key_pages_by_group.append(key_pages)
        value_pages_by_group.append(value_pages)
        query_groups.append(
            torch.from_numpy(rng.normal(size=(query_count, head_dim)).astype(np.float32)).to(device=device)
        )

    clear_prepared_chunk_cache()
    configure_prepared_chunk_cache(cached_kinds=("K", "V"), min_page_count=1)
    try:
        for _ in range(max(warmup_iters, 0)):
            decode_grouped_multiquery_step_prepared_torch_tensor_output_only(
                query_groups,
                key_pages_by_group,
                value_pages_by_group,
            )
        _synchronize(device)
        resident_after_warmup = prepared_chunk_cache_resident_bytes()

        start = time.perf_counter()
        for _ in range(max(bench_iters, 1)):
            output = decode_grouped_multiquery_step_prepared_torch_tensor_output_only(
                query_groups,
                key_pages_by_group,
                value_pages_by_group,
            )
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(bench_iters, 1)
    finally:
        clear_prepared_chunk_cache()
        configure_prepared_chunk_cache(cached_kinds=("K", "V"), min_page_count=4)

    return {
        "benchmark": "grouped_lowbit_decode_micro",
        "device": device,
        "bits": bits,
        "head_dim": head_dim,
        "group_size": group_size,
        "tokens_per_page": tokens_per_page,
        "page_count": page_count,
        "kv_group_count": kv_group_count,
        "query_count": query_count,
        "decode_ms": elapsed_ms,
        "resident_chunk_cache_bytes": resident_after_warmup,
        "output_shape": tuple(int(dim) for dim in output.shape),
    }


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not mps_available():
        raise SystemExit("MPS is unavailable")

    import torch

    if args.head_dim % args.group_size != 0:
        raise SystemExit("head_dim must be divisible by group_size")

    records = []
    case_index = 0
    for bits in _as_ladder(args.bits_ladder, fallback=args.bits):
        for tokens_per_page in _as_ladder(args.tokens_per_page_ladder, fallback=args.tokens_per_page):
            for page_count in _as_ladder(args.page_count_ladder, fallback=args.page_count):
                for query_count in _as_ladder(args.query_count_ladder, fallback=args.query_count):
                    records.append(
                        _run_case(
                            device=args.device,
                            bits=bits,
                            head_dim=args.head_dim,
                            group_size=args.group_size,
                            tokens_per_page=tokens_per_page,
                            page_count=page_count,
                            kv_group_count=args.kv_group_count,
                            query_count=query_count,
                            warmup_iters=args.warmup_iters,
                            bench_iters=args.bench_iters,
                            seed=args.seed + case_index,
                        )
                    )
                    case_index += 1

    if args.output_format == "json":
        payload = records[0] if len(records) == 1 else records
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if len(records) == 1:
        print(json.dumps(records[0], indent=2, sort_keys=True))
        return

    for record in records:
        print(
            json.dumps(
                {
                    "bits": record["bits"],
                    "tokens_per_page": record["tokens_per_page"],
                    "page_count": record["page_count"],
                    "query_count": record["query_count"],
                    "decode_ms": round(float(record["decode_ms"]), 3),
                    "resident_chunk_cache_bytes": int(record["resident_chunk_cache_bytes"]),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
