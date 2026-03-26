from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .attention_reference import softmax
from .backends import (
    PreparedPageMPS,
    decode_step_mps,
    mix_page_cpu_ref,
    mix_page_mps,
    mps_available,
    page_supported_mps,
    prepare_page_mps,
    prepare_pages_mps,
    score_page_cpu_ref,
    score_page_mps,
)
from .page_cache import PreparedPageCache
from .tracing import ExecutionTrace
from .types import EncodedPage

BackendName = Literal["cpu_ref", "torch_mps", "auto"]
PageLike = EncodedPage | PreparedPageMPS


def _resolve_backend(backend: BackendName, page: PageLike) -> Literal["cpu_ref", "torch_mps"]:
    if backend == "cpu_ref":
        return "cpu_ref"
    if backend == "torch_mps":
        if not mps_available():
            raise RuntimeError("torch_mps is unavailable on this machine")
        if not page_supported_mps(page):
            raise ValueError("page is unsupported by torch_mps in this phase")
        return "torch_mps"
    if isinstance(page, PreparedPageMPS):
        return "torch_mps"
    if mps_available() and page_supported_mps(page):
        return "torch_mps"
    return "cpu_ref"


def prepare_page(
    page: PageLike,
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> PageLike:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        if cache is not None:
            return cache.prepare_page(page, trace=trace)
        return prepare_page_mps(page, trace=trace)
    return page.source_page if isinstance(page, PreparedPageMPS) else page


def prepare_pages(
    pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> list[PageLike]:
    if pages:
        resolved_backend = _resolve_backend(backend, pages[0])
        if resolved_backend == "torch_mps":
            if cache is not None:
                return cache.prepare_pages(list(pages), trace=trace)
            return prepare_pages_mps(pages, trace=trace)
    return [prepare_page(page, backend=backend, cache=cache, trace=trace) for page in pages]


def score_page(
    query_slice: np.ndarray,
    page: PageLike,
    *,
    backend: BackendName = "auto",
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        return score_page_mps(query_slice, page, trace=trace)
    return score_page_cpu_ref(query_slice, page, trace=trace)


def mix_page(
    attn_weights: np.ndarray,
    page: PageLike,
    *,
    out_acc: np.ndarray | None = None,
    backend: BackendName = "auto",
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        return mix_page_mps(attn_weights, page, out_acc=out_acc, trace=trace)
    return mix_page_cpu_ref(attn_weights, page, out_acc=out_acc, trace=trace)


def attention_step(
    query_slice: np.ndarray,
    key_page: PageLike,
    value_page: PageLike,
    *,
    backend: BackendName = "cpu_ref",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared_key_page = prepare_page(key_page, backend=backend, cache=cache, trace=trace)
    prepared_value_page = prepare_page(value_page, backend=backend, cache=cache, trace=trace)
    logits = score_page(query_slice, prepared_key_page, backend=backend, trace=trace)
    weights = softmax(logits)
    output = mix_page(weights, prepared_value_page, backend=backend, trace=trace)
    return logits, weights, output


def decode_step(
    query_slice: np.ndarray,
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        raise ValueError("decode_step requires at least one page")

    prepared_key_pages = prepare_pages(key_pages, backend=backend, cache=cache, trace=trace)
    prepared_value_pages = prepare_pages(value_pages, backend=backend, cache=cache, trace=trace)

    if (
        backend != "cpu_ref"
        and all(isinstance(page, PreparedPageMPS) for page in prepared_key_pages)
        and all(isinstance(page, PreparedPageMPS) for page in prepared_value_pages)
    ):
        return decode_step_mps(query_slice, prepared_key_pages, prepared_value_pages, trace=trace)

    page_logits = [score_page(query_slice, page, backend=backend, trace=trace) for page in prepared_key_pages]
    logits = np.concatenate(page_logits).astype(np.float32, copy=False)
    weights = softmax(logits)

    output = np.zeros(prepared_key_pages[0].header.head_dim, dtype=np.float32)
    offset = 0
    for value_page in prepared_value_pages:
        token_count = value_page.header.token_count
        page_weights = weights[offset : offset + token_count]
        output = mix_page(page_weights, value_page, out_acc=output, backend=backend, trace=trace)
        offset += token_count

    return logits, weights, output
