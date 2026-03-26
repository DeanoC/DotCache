from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .attention_runtime import BackendName, decode_step, prepare_pages
from .modes.m0_affine import dequantize_group
from .page_cache import PreparedPageCache
from .page_format import load_group_words
from .packing import unpack_bits
from .tracing import ExecutionTrace
from .types import EncodedPage
from .backends import PreparedPageMPS

PageLike = EncodedPage | PreparedPageMPS


def summarize_key_page(page: PageLike) -> np.ndarray:
    source_page = page.source_page if isinstance(page, PreparedPageMPS) else page
    header = source_page.header

    if header.mode_default == "M3":
        if source_page.escape_payload is None:
            raise ValueError("escape payload is missing")
        return np.asarray(source_page.escape_payload[:, : header.head_dim], dtype=np.float32).mean(axis=0)

    if source_page.payload is None or source_page.scales is None:
        raise ValueError("M0 page is missing payload or scales")

    summary = np.zeros(header.padded_head_dim, dtype=np.float32)
    for group_index in range(header.num_groups):
        words = load_group_words(source_page, group_index)
        codes = unpack_bits(words, header.bits, header.group_size)
        scales = source_page.scales[:, group_index].astype(np.float32)[:, None]
        bias = None
        if source_page.bias is not None:
            bias = source_page.bias[:, group_index].astype(np.float32)[:, None]
        group_values = dequantize_group(
            codes,
            scales=scales,
            bias=bias,
            bits=header.bits,
            scheme=header.quant_scheme,
        )
        start = group_index * header.group_size
        end = start + header.group_size
        summary[start:end] = group_values.mean(axis=0)

    return summary[: header.head_dim]


def select_execution_page_indices(
    key_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
    query_slice: np.ndarray | None = None,
    key_page_summaries: Sequence[np.ndarray] | None = None,
    relevance_top_k: int = 0,
) -> list[int]:
    if not key_pages:
        return []

    context_end = max(page.header.token_start + page.header.token_count for page in key_pages)
    sink_end = max(0, sink_window_tokens)
    recent_start = context_end
    if recent_window_tokens is not None and recent_window_tokens > 0:
        recent_start = max(0, context_end - recent_window_tokens)

    selected_indices: set[int] = set()
    for index, page in enumerate(key_pages):
        page_start = page.header.token_start
        page_end = page_start + page.header.token_count
        in_sink = sink_end > 0 and page_start < sink_end and page_end > 0
        in_recent = recent_window_tokens is not None and recent_window_tokens > 0 and page_end > recent_start
        if in_sink or in_recent:
            selected_indices.add(index)

    if relevance_top_k > 0:
        if query_slice is None or key_page_summaries is None:
            raise ValueError("relevance gating requires query_slice and key_page_summaries")
        if len(key_page_summaries) != len(key_pages):
            raise ValueError("key_page_summaries must align with key_pages")
        candidate_indices = [index for index in range(len(key_pages)) if index not in selected_indices]
        if candidate_indices:
            scores = [
                float(np.dot(np.asarray(key_page_summaries[index], dtype=np.float32), np.asarray(query_slice, dtype=np.float32)))
                for index in candidate_indices
            ]
            ranked_candidates = [
                index
                for _, index in sorted(
                    zip(scores, candidate_indices, strict=True),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
            selected_indices.update(ranked_candidates[:relevance_top_k])

    if not selected_indices:
        return list(range(len(key_pages)))
    return sorted(selected_indices)


def select_execution_page_pairs(
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
    query_slice: np.ndarray | None = None,
    key_page_summaries: Sequence[np.ndarray] | None = None,
    relevance_top_k: int = 0,
) -> tuple[list[PageLike], list[PageLike]]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        return [], []
    if (
        (recent_window_tokens is None or recent_window_tokens <= 0)
        and sink_window_tokens <= 0
        and relevance_top_k <= 0
    ):
        return list(key_pages), list(value_pages)
    selected_indices = select_execution_page_indices(
        key_pages,
        recent_window_tokens=recent_window_tokens,
        sink_window_tokens=sink_window_tokens,
        query_slice=query_slice,
        key_page_summaries=key_page_summaries,
        relevance_top_k=relevance_top_k,
    )
    return (
        [key_pages[index] for index in selected_indices],
        [value_pages[index] for index in selected_indices],
    )


@dataclass(slots=True)
class PagedDecodeSession:
    backend: BackendName = "auto"
    cache: PreparedPageCache | None = None
    recent_window_tokens: int | None = None
    sink_window_tokens: int = 0
    relevance_top_k: int = 0
    key_pages: list[PageLike] = field(default_factory=list)
    value_pages: list[PageLike] = field(default_factory=list)
    key_page_summaries: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.key_pages.clear()
        self.value_pages.clear()
        self.key_page_summaries.clear()
        if self.cache is not None:
            self.cache.clear()

    @property
    def page_count(self) -> int:
        return len(self.key_pages)

    @property
    def active_page_count(self) -> int:
        return len(self.execution_pages()[0])

    @property
    def active_token_count(self) -> int:
        return sum(page.header.token_count for page in self.execution_pages()[0])

    def preload(
        self,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        self.clear()
        self.append(key_pages, value_pages, trace=trace)

    def append(
        self,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if len(key_pages) != len(value_pages):
            raise ValueError("key_pages and value_pages must contain the same number of pages")
        prepared_key_pages = prepare_pages(key_pages, backend=self.backend, cache=self.cache, trace=trace)
        prepared_value_pages = prepare_pages(value_pages, backend=self.backend, cache=self.cache, trace=trace)
        self.key_pages.extend(prepared_key_pages)
        self.value_pages.extend(prepared_value_pages)
        self.key_page_summaries.extend(summarize_key_page(page) for page in prepared_key_pages)

    def execution_pages(self, query_slice: np.ndarray | None = None) -> tuple[list[PageLike], list[PageLike]]:
        return select_execution_page_pairs(
            self.key_pages,
            self.value_pages,
            recent_window_tokens=self.recent_window_tokens,
            sink_window_tokens=self.sink_window_tokens,
            query_slice=query_slice,
            key_page_summaries=self.key_page_summaries,
            relevance_top_k=self.relevance_top_k,
        )

    def decode(
        self,
        query_slice: np.ndarray,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.key_pages or not self.value_pages:
            raise ValueError("PagedDecodeSession requires preloaded pages before decode")
        key_pages, value_pages = self.execution_pages(query_slice)
        return decode_step(
            query_slice,
            key_pages,
            value_pages,
            backend=self.backend,
            trace=trace,
        )
