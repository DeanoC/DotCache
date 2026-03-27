from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionTrace:
    capture_timings: bool = False
    m0_full_page_materializations: int = 0
    payload_bytes_read: int = 0
    metadata_bytes_read: int = 0
    host_to_device_bytes: int = 0
    max_temporary_bytes: int = 0
    prepared_page_cache_hits: int = 0
    prepared_page_cache_misses: int = 0
    cache_resident_bytes: int = 0
    prepared_page_cache_evictions: int = 0
    cache_evicted_bytes: int = 0
    prepare_ms_total: float = 0.0
    prepare_calls: int = 0
    score_ms_total: float = 0.0
    score_calls: int = 0
    mix_ms_total: float = 0.0
    mix_calls: int = 0
    softmax_ms_total: float = 0.0
    softmax_calls: int = 0
    unpack_ms_total: float = 0.0
    unpack_calls: int = 0
    fwht_ms_total: float = 0.0
    fwht_calls: int = 0

    def record_page_read(self, payload_bytes: int, metadata_bytes: int) -> None:
        self.payload_bytes_read += int(payload_bytes)
        self.metadata_bytes_read += int(metadata_bytes)

    def record_host_to_device(self, nbytes: int) -> None:
        self.host_to_device_bytes += int(nbytes)

    def record_temporary(self, nbytes: int) -> None:
        self.max_temporary_bytes = max(self.max_temporary_bytes, int(nbytes))

    def record_m0_full_page_materialization(self, count: int = 1) -> None:
        self.m0_full_page_materializations += int(count)

    def record_cache_hit(self, count: int = 1) -> None:
        self.prepared_page_cache_hits += int(count)

    def record_cache_miss(self, count: int = 1) -> None:
        self.prepared_page_cache_misses += int(count)

    def observe_cache_resident_bytes(self, nbytes: int) -> None:
        self.cache_resident_bytes = max(self.cache_resident_bytes, int(nbytes))

    def record_cache_eviction(self, nbytes: int, count: int = 1) -> None:
        self.prepared_page_cache_evictions += int(count)
        self.cache_evicted_bytes += int(nbytes)

    def record_timing(self, section: str, ms: float, count: int = 1) -> None:
        if section == "prepare":
            self.prepare_ms_total += float(ms)
            self.prepare_calls += int(count)
            return
        if section == "score":
            self.score_ms_total += float(ms)
            self.score_calls += int(count)
            return
        if section == "mix":
            self.mix_ms_total += float(ms)
            self.mix_calls += int(count)
            return
        if section == "softmax":
            self.softmax_ms_total += float(ms)
            self.softmax_calls += int(count)
            return
        if section == "unpack":
            self.unpack_ms_total += float(ms)
            self.unpack_calls += int(count)
            return
        if section == "fwht":
            self.fwht_ms_total += float(ms)
            self.fwht_calls += int(count)
            return
        raise ValueError(f"unknown timing section: {section}")

    def merge(self, other: "ExecutionTrace") -> None:
        self.m0_full_page_materializations += other.m0_full_page_materializations
        self.payload_bytes_read += other.payload_bytes_read
        self.metadata_bytes_read += other.metadata_bytes_read
        self.host_to_device_bytes += other.host_to_device_bytes
        self.max_temporary_bytes = max(self.max_temporary_bytes, other.max_temporary_bytes)
        self.prepared_page_cache_hits += other.prepared_page_cache_hits
        self.prepared_page_cache_misses += other.prepared_page_cache_misses
        self.cache_resident_bytes = max(self.cache_resident_bytes, other.cache_resident_bytes)
        self.prepared_page_cache_evictions += other.prepared_page_cache_evictions
        self.cache_evicted_bytes += other.cache_evicted_bytes
        self.prepare_ms_total += other.prepare_ms_total
        self.prepare_calls += other.prepare_calls
        self.score_ms_total += other.score_ms_total
        self.score_calls += other.score_calls
        self.mix_ms_total += other.mix_ms_total
        self.mix_calls += other.mix_calls
        self.softmax_ms_total += other.softmax_ms_total
        self.softmax_calls += other.softmax_calls
        self.unpack_ms_total += other.unpack_ms_total
        self.unpack_calls += other.unpack_calls
        self.fwht_ms_total += other.fwht_ms_total
        self.fwht_calls += other.fwht_calls

    def to_dict(self) -> dict[str, int]:
        return {
            "m0_full_page_materializations": self.m0_full_page_materializations,
            "payload_bytes_read": self.payload_bytes_read,
            "metadata_bytes_read": self.metadata_bytes_read,
            "host_to_device_bytes": self.host_to_device_bytes,
            "max_temporary_bytes": self.max_temporary_bytes,
            "prepared_page_cache_hits": self.prepared_page_cache_hits,
            "prepared_page_cache_misses": self.prepared_page_cache_misses,
            "cache_resident_bytes": self.cache_resident_bytes,
            "prepared_page_cache_evictions": self.prepared_page_cache_evictions,
            "cache_evicted_bytes": self.cache_evicted_bytes,
            "prepare_ms_total": self.prepare_ms_total,
            "prepare_calls": self.prepare_calls,
            "score_ms_total": self.score_ms_total,
            "score_calls": self.score_calls,
            "mix_ms_total": self.mix_ms_total,
            "mix_calls": self.mix_calls,
            "softmax_ms_total": self.softmax_ms_total,
            "softmax_calls": self.softmax_calls,
            "unpack_ms_total": self.unpack_ms_total,
            "unpack_calls": self.unpack_calls,
            "fwht_ms_total": self.fwht_ms_total,
            "fwht_calls": self.fwht_calls,
        }
