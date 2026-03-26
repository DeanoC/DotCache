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


@dataclass(slots=True)
class PagedDecodeSession:
    backend: BackendName = "auto"
    cache: PreparedPageCache | None = None
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

    def decode(
        self,
        query_slice: np.ndarray,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.key_pages or not self.value_pages:
            raise ValueError("PagedDecodeSession requires preloaded pages before decode")
        return decode_step(
            query_slice,
            self.key_pages,
            self.value_pages,
            backend=self.backend,
            trace=trace,
        )
