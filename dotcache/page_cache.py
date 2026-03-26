from __future__ import annotations

from dataclasses import dataclass, field

from .backends import PreparedPageMPS, prepare_page_mps
from .tracing import ExecutionTrace
from .types import EncodedPage


@dataclass(slots=True)
class PreparedPageCache:
    _mps_pages: dict[int, PreparedPageMPS] = field(default_factory=dict)
    _resident_bytes: int = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def size(self) -> int:
        return len(self._mps_pages)

    def clear(self) -> None:
        self._mps_pages.clear()
        self._resident_bytes = 0

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
            if trace is not None:
                trace.record_cache_hit()
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return cached_page

        prepared_page = prepare_page_mps(page, trace=trace)
        self._mps_pages[cache_key] = prepared_page
        self._resident_bytes += prepared_page.host_to_device_nbytes
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
