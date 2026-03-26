from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .attention_runtime import BackendName, decode_step, prepare_pages
from .page_cache import PreparedPageCache
from .tracing import ExecutionTrace
from .types import EncodedPage
from .backends import PreparedPageMPS

PageLike = EncodedPage | PreparedPageMPS


def select_execution_page_pairs(
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
) -> tuple[list[PageLike], list[PageLike]]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        return [], []
    if (recent_window_tokens is None or recent_window_tokens <= 0) and sink_window_tokens <= 0:
        return list(key_pages), list(value_pages)

    context_end = max(page.header.token_start + page.header.token_count for page in key_pages)
    sink_end = max(0, sink_window_tokens)
    recent_start = context_end
    if recent_window_tokens is not None and recent_window_tokens > 0:
        recent_start = max(0, context_end - recent_window_tokens)

    selected_indices: list[int] = []
    for index, page in enumerate(key_pages):
        page_start = page.header.token_start
        page_end = page_start + page.header.token_count
        in_sink = sink_end > 0 and page_start < sink_end and page_end > 0
        in_recent = recent_window_tokens is not None and recent_window_tokens > 0 and page_end > recent_start
        if in_sink or in_recent:
            selected_indices.append(index)

    if not selected_indices:
        return list(key_pages), list(value_pages)
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
    key_pages: list[PageLike] = field(default_factory=list)
    value_pages: list[PageLike] = field(default_factory=list)

    def clear(self) -> None:
        self.key_pages.clear()
        self.value_pages.clear()
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

    def execution_pages(self) -> tuple[list[PageLike], list[PageLike]]:
        return select_execution_page_pairs(
            self.key_pages,
            self.value_pages,
            recent_window_tokens=self.recent_window_tokens,
            sink_window_tokens=self.sink_window_tokens,
        )

    def decode(
        self,
        query_slice: np.ndarray,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.key_pages or not self.value_pages:
            raise ValueError("PagedDecodeSession requires preloaded pages before decode")
        key_pages, value_pages = self.execution_pages()
        return decode_step(
            query_slice,
            key_pages,
            value_pages,
            backend=self.backend,
            trace=trace,
        )
