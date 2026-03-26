from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionTrace:
    m0_full_page_materializations: int = 0
    payload_bytes_read: int = 0
    metadata_bytes_read: int = 0
    host_to_device_bytes: int = 0
    max_temporary_bytes: int = 0

    def record_page_read(self, payload_bytes: int, metadata_bytes: int) -> None:
        self.payload_bytes_read += int(payload_bytes)
        self.metadata_bytes_read += int(metadata_bytes)

    def record_host_to_device(self, nbytes: int) -> None:
        self.host_to_device_bytes += int(nbytes)

    def record_temporary(self, nbytes: int) -> None:
        self.max_temporary_bytes = max(self.max_temporary_bytes, int(nbytes))

    def record_m0_full_page_materialization(self, count: int = 1) -> None:
        self.m0_full_page_materializations += int(count)

    def to_dict(self) -> dict[str, int]:
        return {
            "m0_full_page_materializations": self.m0_full_page_materializations,
            "payload_bytes_read": self.payload_bytes_read,
            "metadata_bytes_read": self.metadata_bytes_read,
            "host_to_device_bytes": self.host_to_device_bytes,
            "max_temporary_bytes": self.max_temporary_bytes,
        }
