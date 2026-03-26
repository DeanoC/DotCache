from __future__ import annotations

import time

import numpy as np

from dotcache.attention_reference import explicit_dequantized_score, score_page_ref
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page


def main() -> None:
    rng = np.random.default_rng(0)
    config = DotCacheConfig(head_dim=128, group_size=32, tokens_per_page=64)
    keys = rng.normal(size=(config.tokens_per_page, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    page = encode_page(keys, config, kind="K")

    start = time.perf_counter()
    for _ in range(200):
        score_page_ref(query, page)
    streaming_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for _ in range(200):
        explicit_dequantized_score(query, page)
    dense_ms = (time.perf_counter() - start) * 1000

    print(f"streaming_score_ms={streaming_ms:.3f}")
    print(f"explicit_dequant_ms={dense_ms:.3f}")
    print(f"payload_bytes={page.payload_nbytes}")
    print(f"metadata_bytes={page.metadata_nbytes}")


if __name__ == "__main__":
    main()

