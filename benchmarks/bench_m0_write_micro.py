from __future__ import annotations

import argparse
import json
import time

import numpy as np

from dotcache.modes.m0_affine import quantize_tensor
from dotcache.page_format import build_payload
from dotcache.packing import words_per_group


def _legacy_pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint32)
    symbol_count = values.shape[-1]
    word_count = words_per_group(symbol_count, bits)
    flat = values.reshape(-1, symbol_count)
    packed = np.zeros((flat.shape[0], word_count), dtype=np.uint32)
    mask = (1 << bits) - 1
    for row_index, row in enumerate(flat):
        bit_offset = 0
        for raw_value in row:
            value = int(raw_value) & mask
            word_index = bit_offset // 32
            bit_index = bit_offset % 32
            current = int(packed[row_index, word_index])
            current |= (value << bit_index) & 0xFFFFFFFF
            packed[row_index, word_index] = np.uint32(current)
            spill = bit_index + bits - 32
            if spill > 0:
                next_word = int(packed[row_index, word_index + 1])
                next_word |= (value >> (bits - spill)) & 0xFFFFFFFF
                packed[row_index, word_index + 1] = np.uint32(next_word)
            bit_offset += bits
    return packed.reshape(*values.shape[:-1], word_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microbenchmark M0 affine write path components.")
    parser.add_argument("--token-count", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-format", choices=["pretty", "json"], default="pretty")
    return parser.parse_args()


def _record(*, bits: int, token_count: int, head_dim: int, group_size: int, iters: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((token_count, head_dim), dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(iters):
        codes, scales, bias, padded_head_dim = quantize_tensor(values, group_size=group_size, bits=bits, scheme="affine")
    quantize_s = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        payload = build_payload(codes, bits, "group_major")
    payload_s = (time.perf_counter() - t0) / iters

    record = {
        "benchmark": "m0_write_micro",
        "bits": bits,
        "token_count": token_count,
        "head_dim": head_dim,
        "group_size": group_size,
        "iters": iters,
        "quantize_ms": quantize_s * 1000.0,
        "payload_ms": payload_s * 1000.0,
        "total_ms": (quantize_s + payload_s) * 1000.0,
        "payload_shape": list(payload.shape),
        "padded_head_dim": int(padded_head_dim),
    }

    if bits == 3:
        legacy_payload = np.stack([_legacy_pack_bits(codes[:, group_index, :], 3) for group_index in range(codes.shape[1])], axis=0)
        if not np.array_equal(payload, legacy_payload):
            raise RuntimeError("current and legacy 3-bit payload builders disagree")
        t0 = time.perf_counter()
        for _ in range(iters):
            np.stack([_legacy_pack_bits(codes[:, group_index, :], 3) for group_index in range(codes.shape[1])], axis=0)
        legacy_payload_s = (time.perf_counter() - t0) / iters
        record["legacy_payload_ms"] = legacy_payload_s * 1000.0
        record["payload_speedup_vs_legacy"] = legacy_payload_s / max(payload_s, 1e-12)

    return record


def main() -> None:
    args = parse_args()
    records = [
        _record(
            bits=bits,
            token_count=args.token_count,
            head_dim=args.head_dim,
            group_size=args.group_size,
            iters=args.iters,
            seed=args.seed,
        )
        for bits in args.bits
    ]
    if args.output_format == "json":
        print(json.dumps(records, sort_keys=True))
        return
    for record in records:
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
