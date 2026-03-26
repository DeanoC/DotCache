from __future__ import annotations

import time

import numpy as np

from dotcache.attention_reference import explicit_dequantized_attention, run_attention_reference
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def main() -> None:
    rng = np.random.default_rng(1)
    config = DotCacheConfig(head_dim=128, group_size=32, tokens_per_page=64)
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)

    key_page = encode_page(keys, config, kind="K")
    value_page = encode_page(values, config, kind="V")

    start = time.perf_counter()
    for _ in range(200):
        run_attention_reference(query, key_page, value_page)
    streaming_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for _ in range(200):
        explicit_dequantized_attention(query, key_page, value_page)
    dense_ms = (time.perf_counter() - start) * 1000

    print(f"streaming_decode_ms={streaming_ms:.3f}")
    print(f"explicit_dequant_decode_ms={dense_ms:.3f}")


if __name__ == "__main__":
    main()

