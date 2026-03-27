from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionTrace:
    collect_runtime_breakdown: bool = False
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
    score_runtime_ms_total: float = 0.0
    mix_runtime_ms_total: float = 0.0
    unpack_runtime_ms_total: float = 0.0
    softmax_runtime_ms_total: float = 0.0
    chunk_assembly_runtime_ms_total: float = 0.0

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

    def record_score_runtime_ms(self, ms: float) -> None:
        self.score_runtime_ms_total += float(ms)

    def record_mix_runtime_ms(self, ms: float) -> None:
        self.mix_runtime_ms_total += float(ms)

    def record_unpack_runtime_ms(self, ms: float) -> None:
        self.unpack_runtime_ms_total += float(ms)

    def record_softmax_runtime_ms(self, ms: float) -> None:
        self.softmax_runtime_ms_total += float(ms)

    def record_chunk_assembly_runtime_ms(self, ms: float) -> None:
        self.chunk_assembly_runtime_ms_total += float(ms)

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
        self.score_runtime_ms_total += other.score_runtime_ms_total
        self.mix_runtime_ms_total += other.mix_runtime_ms_total
        self.unpack_runtime_ms_total += other.unpack_runtime_ms_total
        self.softmax_runtime_ms_total += other.softmax_runtime_ms_total
        self.chunk_assembly_runtime_ms_total += other.chunk_assembly_runtime_ms_total

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
            "score_runtime_ms_total": float(self.score_runtime_ms_total),
            "mix_runtime_ms_total": float(self.mix_runtime_ms_total),
            "unpack_runtime_ms_total": float(self.unpack_runtime_ms_total),
            "softmax_runtime_ms_total": float(self.softmax_runtime_ms_total),
            "chunk_assembly_runtime_ms_total": float(self.chunk_assembly_runtime_ms_total),
        }
