from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from dotcache.attention_reference import softmax
from dotcache.attention_runtime import BackendName, prepare_pages
from dotcache.config import DotCacheConfig
from dotcache.config_io import load_dotcache_config
from dotcache.encode import encode_page
from dotcache.tracing import ExecutionTrace

DEFAULT_CONTEXTS = [64, 256, 1024]
CONFIG_OVERRIDES = ("head_dim", "group_size", "tokens_per_page")
_MPS_WARMED = False


def parse_args(description: str, *, default_repeats: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", choices=["cpu_ref", "torch_mps", "auto"], default="auto")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--repeats", type=int, default=default_repeats)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--execution-profile",
        choices=["none", "m4_envelope_fast", "m4_envelope_balanced", "m4_envelope_auto"],
        default="none",
    )
    parser.add_argument("--execution-recent-window", type=int, default=None)
    parser.add_argument("--execution-sink-window", type=int, default=0)
    parser.add_argument("--execution-relevance-top-k", type=int, default=0)
    parser.add_argument("--execution-relevance-sketch-size", type=int, default=1)
    parser.add_argument("--execution-relevance-mode", choices=["sketch", "envelope"], default="sketch")
    parser.add_argument("--execution-exact-refine-top-k", type=int, default=0)
    parser.add_argument("--execution-approximate-old-pages", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DotCacheConfig:
    base_config = load_dotcache_config(args.config) if args.config is not None else DotCacheConfig(head_dim=128)
    config_values = asdict(base_config)
    for field_name in CONFIG_OVERRIDES:
        value = getattr(args, field_name)
        if value is not None:
            config_values[field_name] = value
    return DotCacheConfig(**config_values)


def encode_context(values: np.ndarray, config: DotCacheConfig, *, kind: str, mode: str = "M0") -> list:
    pages = []
    for token_start in range(0, values.shape[0], config.tokens_per_page):
        token_end = min(token_start + config.tokens_per_page, values.shape[0])
        pages.append(
            encode_page(
                values[token_start:token_end],
                config,
                kind=kind,
                mode=mode,
                token_start=token_start,
            )
        )
    return pages


def build_fixture(context_length: int, config: DotCacheConfig, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + context_length)
    keys = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    values = rng.normal(size=(context_length, config.head_dim)).astype(np.float32)
    query = rng.normal(size=(config.head_dim,)).astype(np.float32)
    attn = softmax(rng.normal(size=(context_length,)).astype(np.float32))
    return {
        "keys": keys,
        "values": values,
        "query": query,
        "attn": attn,
        "key_pages": encode_context(keys, config, kind="K"),
        "value_pages": encode_context(values, config, kind="V"),
    }


def build_queries(context_length: int, head_dim: int, *, steps: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + context_length + head_dim)
    return [rng.normal(size=(head_dim,)).astype(np.float32) for _ in range(steps)]


def split_weights(weights: np.ndarray, pages: list) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    offset = 0
    for page in pages:
        token_count = page.header.token_count
        chunks.append(weights[offset : offset + token_count])
        offset += token_count
    return chunks


def page_byte_totals(pages: list) -> tuple[int, int]:
    payload_bytes = sum(page.payload_nbytes for page in pages)
    metadata_bytes = sum(page.metadata_nbytes for page in pages)
    return payload_bytes, metadata_bytes


def prepare_context_pages(pages: list, backend: BackendName) -> tuple[list, float, ExecutionTrace]:
    warm_backend(backend)
    trace = ExecutionTrace()
    start = time.perf_counter()
    prepared_pages = prepare_pages(pages, backend=backend, trace=trace)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return prepared_pages, elapsed_ms, trace


def average_ms(fn, repeats: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return ((time.perf_counter() - start) * 1000.0) / repeats


def error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=np.float32)
    reference_arr = np.asarray(reference, dtype=np.float32)
    delta = np.abs(actual_arr - reference_arr)
    denom = np.maximum(np.abs(reference_arr), 1e-8)
    return {
        "max_abs_error": float(np.max(delta)),
        "max_rel_error": float(np.max(delta / denom)),
    }


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True))


def warm_backend(backend: BackendName) -> None:
    global _MPS_WARMED

    if backend != "torch_mps" or _MPS_WARMED:
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.backends.mps.is_available():
        return
    sample = torch.ones(16, device="mps", dtype=torch.float32)
    _ = (sample + 1).sum().item()
    _MPS_WARMED = True
