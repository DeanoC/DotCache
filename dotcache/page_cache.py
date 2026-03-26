from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Literal

from .backends import PreparedPageMPS, prepare_page_mps
from .tracing import ExecutionTrace
from .types import EncodedPage

CachePolicy = Literal["fifo", "lru"]


@dataclass(slots=True)
class PreparedPageCache:
    max_resident_bytes: int | None = None
    policy: CachePolicy = "fifo"
    _mps_pages: dict[int, PreparedPageMPS] = field(default_factory=dict)
    _resident_bytes: int = 0
    _order: OrderedDict[int, None] = field(default_factory=OrderedDict)

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def size(self) -> int:
        return len(self._mps_pages)

    def clear(self) -> None:
        self._mps_pages.clear()
        self._resident_bytes = 0
        self._order.clear()

    def _page_nbytes(self, page: PreparedPageMPS) -> int:
        return int(page.host_to_device_nbytes)

    def _evict_one(self, *, trace: ExecutionTrace | None = None) -> bool:
        while self._order:
            cache_key, _ = self._order.popitem(last=False)
            cached_page = self._mps_pages.pop(cache_key, None)
            if cached_page is None:
                continue
            evicted_bytes = self._page_nbytes(cached_page)
            self._resident_bytes = max(0, self._resident_bytes - evicted_bytes)
            if trace is not None:
                trace.record_cache_eviction(evicted_bytes)
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return True
        return False

    def _ensure_capacity(self, incoming_nbytes: int, *, trace: ExecutionTrace | None = None) -> None:
        if self.max_resident_bytes is None:
            return
        while self._resident_bytes + incoming_nbytes > self.max_resident_bytes and self._mps_pages:
            if not self._evict_one(trace=trace):
                break

    def append_page(
        self,
        page: EncodedPage | PreparedPageMPS,
        *,
        trace: ExecutionTrace | None = None,
    ) -> EncodedPage | PreparedPageMPS:
        return self.prepare_page(page, trace=trace)

    def append_pages(
        self,
        pages: list[EncodedPage | PreparedPageMPS],
        *,
        trace: ExecutionTrace | None = None,
    ) -> list[EncodedPage | PreparedPageMPS]:
        return [self.append_page(page, trace=trace) for page in pages]

    def prepare_page(
        self,
        page: EncodedPage | PreparedPageMPS,
        *,
        trace: ExecutionTrace | None = None,
    ) -> EncodedPage | PreparedPageMPS:
        if isinstance(page, PreparedPageMPS):
            if trace is not None:
                trace.record_cache_hit()
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return page

        cache_key = id(page)
        cached_page = self._mps_pages.get(cache_key)
        if cached_page is not None:
            if self.policy == "lru":
                self._order.move_to_end(cache_key)
            if trace is not None:
                trace.record_cache_hit()
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return cached_page

        prepared_page = prepare_page_mps(page, trace=trace)
        self._ensure_capacity(self._page_nbytes(prepared_page), trace=trace)
        self._mps_pages[cache_key] = prepared_page
        self._order[cache_key] = None
        self._resident_bytes += self._page_nbytes(prepared_page)
        if trace is not None:
            trace.record_cache_miss()
            trace.observe_cache_resident_bytes(self._resident_bytes)
        return prepared_page

    def prepare_pages(
        self,
        pages: list[EncodedPage | PreparedPageMPS],
        *,
        trace: ExecutionTrace | None = None,
    ) -> list[EncodedPage | PreparedPageMPS]:
        return [self.prepare_page(page, trace=trace) for page in pages]
