from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .attention_runtime import BackendName, decode_multi_query_step, prepare_pages
from .backends import (
    PreparedPageTorch,
    clear_prepared_chunk_cache,
    cuda_available,
    decode_grouped_multiquery_step_prepared_torch_tensor,
    decode_multi_query_step_torch_tensor,
    mps_available,
    prepared_chunk_cache_resident_bytes,
)
from .config import DotCacheConfig
from .encode import encode_page
from .page_cache import PreparedPageCache
from .packing import words_per_group
from .session_runtime import PagedDecodeSession
from .tracing import ExecutionTrace
from .types import EncodedPage, PageHeader

PageLike = EncodedPage | PreparedPageTorch


def default_q_head_to_kv_head(num_attention_heads: int, num_key_value_heads: int) -> np.ndarray:
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads for the Llama path")
    return (np.arange(num_attention_heads, dtype=np.int64) // (num_attention_heads // num_key_value_heads)).astype(
        np.int64,
        copy=False,
    )


def _group_query_heads(mapping: np.ndarray, *, num_key_value_heads: int) -> tuple[tuple[int, ...], ...]:
    grouped: list[list[int]] = [[] for _ in range(num_key_value_heads)]
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        if kv_head_id < 0 or kv_head_id >= num_key_value_heads:
            raise ValueError("q_head_to_kv_head contains an invalid KV head id")
        grouped[kv_head_id].append(q_head_id)
    return tuple(tuple(group) for group in grouped)


def _grouped_pages_can_batch(
    key_pages_by_group: Sequence[Sequence[PageLike]],
    value_pages_by_group: Sequence[Sequence[PageLike]],
    query_groups: Sequence[Any],
) -> bool:
    if not key_pages_by_group or len(key_pages_by_group) != len(value_pages_by_group):
        return False
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        return False
    try:
        query_count = int(query_groups[0].shape[0])
    except Exception:
        return False
    page_count = len(key_pages_by_group[0])
    if page_count == 0:
        return False
    for group_index in range(group_count):
        if len(key_pages_by_group[group_index]) != page_count or len(value_pages_by_group[group_index]) != page_count:
            return False
        if int(query_groups[group_index].shape[0]) != query_count:
            return False
        if not all(isinstance(page, PreparedPageTorch) for page in key_pages_by_group[group_index]):
            return False
        if not all(isinstance(page, PreparedPageTorch) for page in value_pages_by_group[group_index]):
            return False
        if any(page.device_type != key_pages_by_group[0][0].device_type for page in key_pages_by_group[group_index]):
            return False
        if any(page.device_type != value_pages_by_group[0][0].device_type for page in value_pages_by_group[group_index]):
            return False
    for page_index in range(page_count):
        key_signature = (
            key_pages_by_group[0][page_index].header.mode_default,
            key_pages_by_group[0][page_index].header.token_count,
            key_pages_by_group[0][page_index].header.head_dim,
            key_pages_by_group[0][page_index].header.padded_head_dim,
            key_pages_by_group[0][page_index].header.group_size,
            key_pages_by_group[0][page_index].header.num_groups,
            key_pages_by_group[0][page_index].header.bits,
            key_pages_by_group[0][page_index].header.words_per_group,
            key_pages_by_group[0][page_index].header.layout,
            key_pages_by_group[0][page_index].header.quant_scheme,
        )
        value_signature = (
            value_pages_by_group[0][page_index].header.mode_default,
            value_pages_by_group[0][page_index].header.token_count,
            value_pages_by_group[0][page_index].header.head_dim,
            value_pages_by_group[0][page_index].header.padded_head_dim,
            value_pages_by_group[0][page_index].header.group_size,
            value_pages_by_group[0][page_index].header.num_groups,
            value_pages_by_group[0][page_index].header.bits,
            value_pages_by_group[0][page_index].header.words_per_group,
            value_pages_by_group[0][page_index].header.layout,
            value_pages_by_group[0][page_index].header.quant_scheme,
        )
        for group_index in range(1, group_count):
            key_page = key_pages_by_group[group_index][page_index]
            value_page = value_pages_by_group[group_index][page_index]
            if (
                key_page.header.mode_default,
                key_page.header.token_count,
                key_page.header.head_dim,
                key_page.header.padded_head_dim,
                key_page.header.group_size,
                key_page.header.num_groups,
                key_page.header.bits,
                key_page.header.words_per_group,
                key_page.header.layout,
                key_page.header.quant_scheme,
            ) != key_signature:
                return False
            if (
                value_page.header.mode_default,
                value_page.header.token_count,
                value_page.header.head_dim,
                value_page.header.padded_head_dim,
                value_page.header.group_size,
                value_page.header.num_groups,
                value_page.header.bits,
                value_page.header.words_per_group,
                value_page.header.layout,
                value_page.header.quant_scheme,
            ) != value_signature:
                return False
    return True


def _normalize_prefill_tensor(
    values: np.ndarray,
    *,
    num_key_value_heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, seq_len, head_dim] or [1, kv_heads, seq_len, head_dim]")
    if array.shape[0] != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if array.shape[2] != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


def _normalize_step_tensor(
    values: np.ndarray,
    *,
    num_key_value_heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, token_count, head_dim]")
    if array.shape[0] != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if array.shape[2] != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


def _normalize_query_step(query_step: np.ndarray, *, num_attention_heads: int, head_dim: int) -> np.ndarray:
    queries = np.asarray(query_step, dtype=np.float32)
    if queries.ndim == 4:
        if queries.shape[0] != 1 or queries.shape[2] != 1:
            raise ValueError("query_step must have shape [q_heads, head_dim] or [1, q_heads, 1, head_dim]")
        queries = queries[0, :, 0, :]
    if queries.ndim != 2:
        raise ValueError("query_step must have shape [q_heads, head_dim]")
    if queries.shape[0] != num_attention_heads:
        raise ValueError(f"query_step must contain {num_attention_heads} query heads")
    if queries.shape[1] != head_dim:
        raise ValueError(f"query_step head_dim must equal {head_dim}")
    return queries


def _normalize_prefill_tensor_torch(values, *, num_key_value_heads: int, head_dim: int, name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for torch-native prefill ingest") from exc
    if not torch.is_tensor(values):
        raise TypeError(f"{name} must be a torch.Tensor")
    array = values.detach().to(dtype=torch.float32)
    if array.ndim == 4:
        if int(array.shape[0]) != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, seq_len, head_dim] or [1, kv_heads, seq_len, head_dim]")
    if int(array.shape[0]) != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if int(array.shape[2]) != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


@dataclass(slots=True)
class _TailPageBuilder:
    config: DotCacheConfig
    layer_id: int
    kv_head_id: int
    token_start: int | None = None
    key_rows: list[np.ndarray] = field(default_factory=list)
    value_rows: list[np.ndarray] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.key_rows)

    def clear(self) -> None:
        self.token_start = None
        self.key_rows.clear()
        self.value_rows.clear()

    def load_prefill_remainder(
        self,
        key_rows: np.ndarray,
        value_rows: np.ndarray,
        *,
        token_start: int,
    ) -> None:
        self.clear()
        if key_rows.shape[0] != value_rows.shape[0]:
            raise ValueError("prefill remainder key/value rows must align")
        if key_rows.shape[0] == 0:
            return
        self.token_start = int(token_start)
        self.key_rows.extend(np.asarray(key_rows, dtype=np.float32))
        self.value_rows.extend(np.asarray(value_rows, dtype=np.float32))

    def append_step_rows(
        self,
        key_rows: np.ndarray,
        value_rows: np.ndarray,
        *,
        token_start: int,
    ) -> tuple[list[EncodedPage], list[EncodedPage]]:
        if key_rows.shape != value_rows.shape:
            raise ValueError("step key/value rows must align")
        if key_rows.ndim != 2:
            raise ValueError("step rows must have shape [token_count, head_dim]")
        if key_rows.shape[0] == 0:
            return [], []

        finalized_key_pages: list[EncodedPage] = []
        finalized_value_pages: list[EncodedPage] = []
        expected_token = self.next_token_index
        if expected_token is not None and token_start != expected_token:
            raise ValueError(f"tail-page append expected token_index {expected_token}, received {token_start}")
        if self.token_start is None:
            self.token_start = int(token_start)

        for offset in range(key_rows.shape[0]):
            self.key_rows.append(np.asarray(key_rows[offset], dtype=np.float32))
            self.value_rows.append(np.asarray(value_rows[offset], dtype=np.float32))
            if len(self.key_rows) < self.config.tokens_per_page:
                continue

            if self.token_start is None:
                raise RuntimeError("tail-page token_start is missing while finalizing a page")
            dense_keys = np.stack(self.key_rows, axis=0).astype(np.float32, copy=False)
            dense_values = np.stack(self.value_rows, axis=0).astype(np.float32, copy=False)
            finalized_key_pages.append(
                encode_page(
                    dense_keys,
                    self.config,
                    kind="K",
                    layer_id=self.layer_id,
                    kv_head_id=self.kv_head_id,
                    token_start=self.token_start,
                )
            )
            finalized_value_pages.append(
                encode_page(
                    dense_values,
                    self.config,
                    kind="V",
                    layer_id=self.layer_id,
                    kv_head_id=self.kv_head_id,
                    token_start=self.token_start,
                )
            )
            self.key_rows.clear()
            self.value_rows.clear()
            self.token_start += self.config.tokens_per_page
        if self.token_count == 0:
            self.token_start = None
        return finalized_key_pages, finalized_value_pages

    @property
    def next_token_index(self) -> int | None:
        if self.token_start is None:
            return None
        return self.token_start + self.token_count

    def build_temp_pages(self) -> tuple[EncodedPage, EncodedPage] | None:
        if self.token_count == 0:
            return None
        if self.token_start is None:
            raise RuntimeError("tail-page token_start is missing")
        dense_keys = np.stack(self.key_rows, axis=0).astype(np.float32, copy=False)
        dense_values = np.stack(self.value_rows, axis=0).astype(np.float32, copy=False)
        return (
            encode_page(
                dense_keys,
                self.config,
                kind="K",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
            ),
            encode_page(
                dense_values,
                self.config,
                kind="V",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
            ),
        )


def _tail_escape_dtype_numpy(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"unsupported tail escape dtype: {dtype_name}")


@dataclass(slots=True)
class _PersistentTailPage:
    config: DotCacheConfig
    layer_id: int
    kv_head_id: int
    kind: str
    device_type: str
    source_page: EncodedPage | None = None
    prepared_page: PreparedPageTorch | None = None
    host_buffer: np.ndarray | None = None
    token_count: int = 0
    resident_nbytes: int = 0

    def clear(self) -> None:
        self.token_count = 0
        if self.source_page is not None:
            self.source_page.header.token_count = 0
            self.source_page.escape_payload = None if self.host_buffer is None else self.host_buffer[:0]

    def _ensure_allocated(self, *, token_start: int) -> None:
        if self.source_page is not None and self.prepared_page is not None and self.host_buffer is not None:
            self.source_page.header.token_start = int(token_start)
            self.prepared_page.header.token_start = int(token_start)
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is required only for the MPS tail path
            raise RuntimeError("torch is required for the persistent torch tail path") from exc

        dtype_name = self.config.escape_dtype
        np_dtype = _tail_escape_dtype_numpy(dtype_name)
        torch_dtype = getattr(torch, dtype_name)
        host_buffer = np.zeros((self.config.tokens_per_page, self.config.head_dim), dtype=np_dtype)
        header = PageHeader(
            layer_id=self.layer_id,
            kv_head_id=self.kv_head_id,
            kind=self.kind,
            token_start=int(token_start),
            token_count=0,
            head_dim=self.config.head_dim,
            padded_head_dim=self.config.padded_head_dim,
            group_size=self.config.group_size,
            num_groups=self.config.num_groups,
            bits=self.config.bits_k if self.kind == "K" else self.config.bits_v,
            words_per_group=words_per_group(self.config.group_size, self.config.bits_k if self.kind == "K" else self.config.bits_v),
            mode_default="M3",
            layout=self.config.payload_layout_k if self.kind == "K" else self.config.payload_layout_v,
            quant_scheme=self.config.quant_scheme_k if self.kind == "K" else self.config.quant_scheme_v,
            escape_dtype=dtype_name,
        )
        source_page = EncodedPage(header=header, escape_payload=host_buffer[:0])
        device_payload = torch.zeros(
            (self.config.tokens_per_page, self.config.head_dim),
            dtype=torch_dtype,
            device=self.device_type,
        )
        prepared_page = PreparedPageTorch(
            device_type=self.device_type,
            source_page=source_page,
            header=header,
            escape_payload=device_payload,
            host_to_device_nbytes=int(device_payload.numel() * device_payload.element_size()),
        )
        self.source_page = source_page
        self.prepared_page = prepared_page
        self.host_buffer = host_buffer
        self.resident_nbytes = int(device_payload.numel() * device_payload.element_size())

    def load_rows(
        self,
        rows: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        self.clear()
        if values.shape[0] == 0:
            return
        self._ensure_allocated(token_start=token_start)
        self.append_rows(values, token_start=token_start, trace=trace)

    def prepare_append_span(self, *, token_start: int, row_count: int) -> tuple[int, int]:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.source_page is None or self.prepared_page is None or self.host_buffer is None:
            raise RuntimeError("persistent tail page is not initialized")
        expected_token = self.source_page.header.token_start + self.token_count
        if token_start != expected_token:
            raise ValueError(f"persistent tail expected token_start {expected_token}, received {token_start}")
        end = self.token_count + row_count
        if end > self.config.tokens_per_page:
            raise ValueError("persistent tail cannot exceed tokens_per_page")
        start = self.token_count
        self.source_page.header.token_count = end
        self.prepared_page.header.token_count = end
        self.source_page.escape_payload = self.host_buffer[:end]
        self.token_count = end
        return start, end

    def append_rows_from_device(
        self,
        *,
        rows: np.ndarray,
        device_rows: Any,
        token_start: int,
    ) -> None:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if values.shape[0] == 0:
            return
        if self.host_buffer is None or self.prepared_page is None:
            raise RuntimeError("persistent tail page is not initialized")
        start, end = self.prepare_append_span(token_start=token_start, row_count=values.shape[0])
        converted = values.astype(self.host_buffer.dtype, copy=False)
        self.host_buffer[start:end] = converted
        self.prepared_page.escape_payload[start:end, : self.config.head_dim] = device_rows

    def append_device_rows(
        self,
        device_rows,
        *,
        token_start: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        if not torch.is_tensor(device_rows):
            raise TypeError("append_device_rows requires a torch.Tensor")
        if device_rows.ndim != 2 or int(device_rows.shape[1]) != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if int(device_rows.shape[0]) == 0:
            return
        if self.prepared_page is None:
            self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.prepared_page is None:
            raise RuntimeError("persistent tail page is not initialized")
        start, end = self.prepare_append_span(token_start=token_start, row_count=int(device_rows.shape[0]))
        self.prepared_page.escape_payload[start:end, : self.config.head_dim] = device_rows.to(
            dtype=self.prepared_page.escape_payload.dtype
        )

    def materialize_rows(self) -> np.ndarray:
        if self.prepared_page is None or self.token_count <= 0:
            return np.zeros((0, self.config.head_dim), dtype=np.float32)
        return (
            self.prepared_page.escape_payload[: self.token_count, : self.config.head_dim]
            .detach()
            .to(dtype=self.prepared_page.escape_payload.dtype)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def append_rows(
        self,
        rows: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if values.shape[0] == 0:
            return
        self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.source_page is None or self.prepared_page is None or self.host_buffer is None:
            raise RuntimeError("persistent tail page is not initialized")
        converted = values.astype(self.host_buffer.dtype, copy=False)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        row_tensor = torch.from_numpy(np.ascontiguousarray(converted)).to(device=self.device_type)
        start, end = self.prepare_append_span(token_start=token_start, row_count=values.shape[0])
        self.host_buffer[start:end] = converted
        self.prepared_page.escape_payload[start:end, : self.config.head_dim] = row_tensor
        if trace is not None:
            trace.record_host_to_device(int(row_tensor.numel() * row_tensor.element_size()))

    @property
    def active_page(self) -> PreparedPageTorch | None:
        if self.token_count <= 0:
            return None
        return self.prepared_page


@dataclass(slots=True)
class _HeadSessionState:
    session: PagedDecodeSession
    tail: _TailPageBuilder
    persistent_key_tail: _PersistentTailPage | None = None
    persistent_value_tail: _PersistentTailPage | None = None
    decode_key_pages_with_tail: list[PageLike] | None = None
    decode_value_pages_with_tail: list[PageLike] | None = None
    sequence_length: int = 0

    def invalidate_decode_views(self) -> None:
        self.decode_key_pages_with_tail = None
        self.decode_value_pages_with_tail = None

    def clear(self, *, clear_prepared_cache: bool) -> None:
        self.session.key_pages.clear()
        self.session.value_pages.clear()
        self.session.key_page_sketches.clear()
        self.session.key_page_minima.clear()
        self.session.key_page_maxima.clear()
        self.session.value_page_summaries.clear()
        self.session.last_selected_indices.clear()
        if clear_prepared_cache and self.session.cache is not None:
            self.session.cache.clear()
        self.tail.clear()
        if self.persistent_key_tail is not None:
            self.persistent_key_tail.clear()
        if self.persistent_value_tail is not None:
            self.persistent_value_tail.clear()
        self.invalidate_decode_views()
        self.sequence_length = 0


class ModelPagedKVCache:
    def __init__(
        self,
        *,
        config: DotCacheConfig,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        backend: BackendName = "auto",
        cache: PreparedPageCache | None = None,
    ) -> None:
        self.config = config
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.backend = backend
        self.cache = cache if cache is not None else PreparedPageCache()
        self.default_q_head_to_kv_head = default_q_head_to_kv_head(self.num_attention_heads, self.num_key_value_heads)
        self.default_grouped_query_heads = _group_query_heads(
            self.default_q_head_to_kv_head,
            num_key_value_heads=self.num_key_value_heads,
        )
        self._states: dict[tuple[int, int], _HeadSessionState] = {}

    @property
    def resident_bytes(self) -> int:
        tail_resident_bytes = 0
        for state in self._states.values():
            if state.persistent_key_tail is not None:
                tail_resident_bytes += state.persistent_key_tail.resident_nbytes
            if state.persistent_value_tail is not None:
                tail_resident_bytes += state.persistent_value_tail.resident_nbytes
        chunk_resident_bytes = prepared_chunk_cache_resident_bytes() if self._torch_device_type == "mps" else 0
        return self.cache.resident_bytes + tail_resident_bytes + chunk_resident_bytes

    def clear(self) -> None:
        for state in self._states.values():
            state.clear(clear_prepared_cache=False)
        self.cache.clear()
        clear_prepared_chunk_cache()

    def _grouped_query_heads_for_mapping(self, q_head_to_kv_head: Sequence[int] | np.ndarray) -> tuple[tuple[int, ...], ...]:
        mapping = np.asarray(q_head_to_kv_head, dtype=np.int64)
        if mapping.shape != (self.num_attention_heads,):
            raise ValueError("q_head_to_kv_head must have shape [num_attention_heads]")
        if np.array_equal(mapping, self.default_q_head_to_kv_head):
            return self.default_grouped_query_heads
        return _group_query_heads(mapping, num_key_value_heads=self.num_key_value_heads)

    def _encode_full_prefill_pages(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        full_tokens: int,
    ) -> tuple[list[list[EncodedPage]], list[list[EncodedPage]]]:
        key_pages_by_head: list[list[EncodedPage]] = [[] for _ in range(self.num_key_value_heads)]
        value_pages_by_head: list[list[EncodedPage]] = [[] for _ in range(self.num_key_value_heads)]
        if full_tokens <= 0:
            return key_pages_by_head, value_pages_by_head

        page_size = self.config.tokens_per_page
        full_keys = np.ascontiguousarray(keys[:, :full_tokens], dtype=np.float32)
        full_values = np.ascontiguousarray(values[:, :full_tokens], dtype=np.float32)

        for page_start in range(0, full_tokens, page_size):
            page_end = page_start + page_size
            for kv_head_id in range(self.num_key_value_heads):
                key_pages_by_head[kv_head_id].append(
                    encode_page(
                        full_keys[kv_head_id, page_start:page_end],
                        self.config,
                        kind="K",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=page_start,
                        build_runtime_metadata=False,
                    )
                )
                value_pages_by_head[kv_head_id].append(
                    encode_page(
                        full_values[kv_head_id, page_start:page_end],
                        self.config,
                        kind="V",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=page_start,
                        build_runtime_metadata=False,
                    )
                )
        return key_pages_by_head, value_pages_by_head

    def prepare_static_pages(self, *, trace: ExecutionTrace | None = None) -> None:
        if self._torch_device_type is None:
            return
        key_refs: list[tuple[_HeadSessionState, int]] = []
        key_pages: list[PageLike] = []
        value_refs: list[tuple[_HeadSessionState, int]] = []
        value_pages: list[PageLike] = []

        for state in self._states.values():
            for index, page in enumerate(state.session.key_pages):
                if isinstance(page, PreparedPageTorch):
                    continue
                key_refs.append((state, index))
                key_pages.append(page)
            for index, page in enumerate(state.session.value_pages):
                if isinstance(page, PreparedPageTorch):
                    continue
                value_refs.append((state, index))
                value_pages.append(page)

        if key_pages:
            prepared_keys = prepare_pages(key_pages, backend=self.backend, cache=self.cache, trace=trace)
            for (state, index), prepared in zip(key_refs, prepared_keys, strict=True):
                state.session.key_pages[index] = prepared
                state.invalidate_decode_views()
        if value_pages:
            prepared_values = prepare_pages(value_pages, backend=self.backend, cache=self.cache, trace=trace)
            for (state, index), prepared in zip(value_refs, prepared_values, strict=True):
                state.session.value_pages[index] = prepared
                state.invalidate_decode_views()

    def _ensure_prepared_static_pages(
        self,
        state: _HeadSessionState,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if self._torch_device_type is None:
            return
        if state.session.key_pages and not all(isinstance(page, PreparedPageTorch) for page in state.session.key_pages):
            state.session.key_pages = prepare_pages(
                state.session.key_pages,
                backend=self.backend,
                cache=self.cache,
                trace=trace,
            )
            state.invalidate_decode_views()
        if state.session.value_pages and not all(isinstance(page, PreparedPageTorch) for page in state.session.value_pages):
            state.session.value_pages = prepare_pages(
                state.session.value_pages,
                backend=self.backend,
                cache=self.cache,
                trace=trace,
            )
            state.invalidate_decode_views()

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.num_hidden_layers:
            raise ValueError(f"layer_id must be in [0, {self.num_hidden_layers})")

    def _state(self, layer_id: int, kv_head_id: int) -> _HeadSessionState:
        self._validate_layer_id(layer_id)
        if kv_head_id < 0 or kv_head_id >= self.num_key_value_heads:
            raise ValueError(f"kv_head_id must be in [0, {self.num_key_value_heads})")
        key = (layer_id, kv_head_id)
        state = self._states.get(key)
        if state is None:
            torch_device_type = self._torch_device_type
            state = _HeadSessionState(
                session=PagedDecodeSession(backend=self.backend, cache=self.cache),
                tail=_TailPageBuilder(self.config, layer_id=layer_id, kv_head_id=kv_head_id),
                persistent_key_tail=_PersistentTailPage(
                    self.config,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    kind="K",
                    device_type=torch_device_type,
                )
                if torch_device_type is not None
                else None,
                persistent_value_tail=_PersistentTailPage(
                    self.config,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    kind="V",
                    device_type=torch_device_type,
                )
                if torch_device_type is not None
                else None,
            )
            self._states[key] = state
        return state

    @property
    def _torch_device_type(self) -> str | None:
        if self.backend == "torch_mps":
            return "mps" if mps_available() else None
        if self.backend == "torch_cuda":
            return "cuda" if cuda_available() else None
        if self.backend == "auto":
            if cuda_available():
                return "cuda"
            if mps_available():
                return "mps"
        return None

    @property
    def _use_persistent_torch_tail(self) -> bool:
        return self._torch_device_type is not None

    def layer_sequence_length(self, layer_id: int) -> int:
        self._validate_layer_id(layer_id)
        lengths = {self._state(layer_id, kv_head_id).sequence_length for kv_head_id in range(self.num_key_value_heads)}
        if len(lengths) > 1:
            raise RuntimeError(f"layer {layer_id} KV heads disagree on sequence length")
        return next(iter(lengths), 0)

    def _batch_upload_persistent_tail_rows(
        self,
        tails: Sequence[_PersistentTailPage | None],
        rows_by_head: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        active_pairs = [(tail, rows_by_head[index]) for index, tail in enumerate(tails) if tail is not None]
        if not active_pairs:
            return
        non_empty_pairs = [(tail, rows) for tail, rows in active_pairs if rows.shape[0] > 0]
        if not non_empty_pairs:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        contiguous_rows = np.ascontiguousarray(np.stack([rows.astype(np.float16, copy=False) for _, rows in non_empty_pairs], axis=0))
        device_rows = torch.from_numpy(contiguous_rows).to(device=non_empty_pairs[0][0].device_type)
        if trace is not None:
            trace.record_host_to_device(int(device_rows.numel() * device_rows.element_size()))
        for batch_index, (tail, rows) in enumerate(non_empty_pairs):
            if tail is None:
                continue
            if tail.host_buffer is None or tail.prepared_page is None:
                tail._ensure_allocated(token_start=token_start if tail.token_count == 0 else tail.source_page.header.token_start)
            tail.append_rows_from_device(
                rows=rows,
                device_rows=device_rows[batch_index],
                token_start=token_start,
            )

    def _batch_append_persistent_tail_tensors(
        self,
        tails: Sequence[_PersistentTailPage | None],
        rows_by_head,
        *,
        token_start: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        if not torch.is_tensor(rows_by_head):
            raise TypeError("rows_by_head must be a torch.Tensor")
        if rows_by_head.ndim != 3:
            raise ValueError("rows_by_head must have shape [kv_heads, token_count, head_dim]")
        for index, tail in enumerate(tails):
            if tail is None or int(rows_by_head[index].shape[0]) == 0:
                continue
            if tail.prepared_page is None:
                tail._ensure_allocated(token_start=token_start if tail.token_count == 0 else tail.source_page.header.token_start)
            tail.append_device_rows(rows_by_head[index], token_start=token_start)

    def ingest_prefill_cache(
        self,
        layer_id: int,
        layer_k: np.ndarray,
        layer_v: np.ndarray,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        keys = _normalize_prefill_tensor(
            layer_k,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_k",
        )
        values = _normalize_prefill_tensor(
            layer_v,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_v",
        )
        if keys.shape[1] != values.shape[1]:
            raise ValueError("layer_k and layer_v sequence lengths must match")

        seq_len = int(keys.shape[1])
        full_page_count = seq_len // self.config.tokens_per_page
        full_tokens = full_page_count * self.config.tokens_per_page
        preload_key_pages_by_head, preload_value_pages_by_head = self._encode_full_prefill_pages(
            layer_id,
            keys,
            values,
            full_tokens=full_tokens,
        )
        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            state.clear(clear_prepared_cache=False)
            preload_key_pages = preload_key_pages_by_head[kv_head_id]
            preload_value_pages = preload_value_pages_by_head[kv_head_id]
            if preload_key_pages:
                state.session.append(preload_key_pages, preload_value_pages, prepare=False, trace=trace)
                state.invalidate_decode_views()
            remainder_keys = keys[kv_head_id, full_tokens:]
            remainder_values = values[kv_head_id, full_tokens:]
            state.tail.load_prefill_remainder(remainder_keys, remainder_values, token_start=full_tokens)
            state.sequence_length = seq_len
        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                tail = key_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if keys[kv_head_id, full_tokens:].shape[0] > 0:
                        tail._ensure_allocated(token_start=full_tokens)
                tail = value_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if values[kv_head_id, full_tokens:].shape[0] > 0:
                        tail._ensure_allocated(token_start=full_tokens)
            self._batch_upload_persistent_tail_rows(
                key_tails,
                keys[:, full_tokens:],
                token_start=full_tokens,
                trace=trace,
            )
            self._batch_upload_persistent_tail_rows(
                value_tails,
                values[:, full_tokens:],
                token_start=full_tokens,
                trace=trace,
            )

    def ingest_prefill_cache_torch(
        self,
        layer_id: int,
        layer_k,
        layer_v,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for ingest_prefill_cache_torch") from exc
        if self._torch_device_type is None:
            raise RuntimeError("ingest_prefill_cache_torch is only available for a torch accelerator backend")
        keys = _normalize_prefill_tensor_torch(
            layer_k,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_k",
        )
        values = _normalize_prefill_tensor_torch(
            layer_v,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_v",
        )
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("layer_k and layer_v sequence lengths must match")

        seq_len = int(keys.shape[1])
        full_page_count = seq_len // self.config.tokens_per_page
        full_tokens = full_page_count * self.config.tokens_per_page
        if full_tokens > 0:
            full_keys_cpu = keys[:, :full_tokens].detach().cpu().numpy()
            full_values_cpu = values[:, :full_tokens].detach().cpu().numpy()
            preload_key_pages_by_head, preload_value_pages_by_head = self._encode_full_prefill_pages(
                layer_id,
                full_keys_cpu,
                full_values_cpu,
                full_tokens=full_tokens,
            )
        else:
            preload_key_pages_by_head = [[] for _ in range(self.num_key_value_heads)]
            preload_value_pages_by_head = [[] for _ in range(self.num_key_value_heads)]
        if not self._use_persistent_torch_tail:
            remainder_keys_cpu = keys[:, full_tokens:].detach().cpu().numpy()
            remainder_values_cpu = values[:, full_tokens:].detach().cpu().numpy()

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            state.clear(clear_prepared_cache=False)
            preload_key_pages = preload_key_pages_by_head[kv_head_id]
            preload_value_pages = preload_value_pages_by_head[kv_head_id]
            if preload_key_pages:
                state.session.append(preload_key_pages, preload_value_pages, prepare=False, trace=trace)
                state.invalidate_decode_views()
            if self._use_persistent_torch_tail:
                state.tail.clear()
            else:
                remainder_keys = remainder_keys_cpu[kv_head_id]
                remainder_values = remainder_values_cpu[kv_head_id]
                state.tail.load_prefill_remainder(remainder_keys, remainder_values, token_start=full_tokens)
            state.sequence_length = seq_len

        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                tail = key_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if int(keys[kv_head_id, full_tokens:].shape[0]) > 0:
                        tail._ensure_allocated(token_start=full_tokens)
                tail = value_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if int(values[kv_head_id, full_tokens:].shape[0]) > 0:
                        tail._ensure_allocated(token_start=full_tokens)
            self._batch_append_persistent_tail_tensors(
                key_tails,
                keys[:, full_tokens:],
                token_start=full_tokens,
            )
            self._batch_append_persistent_tail_tensors(
                value_tails,
                values[:, full_tokens:],
                token_start=full_tokens,
            )

    def append_step(
        self,
        layer_id: int,
        key_step: np.ndarray,
        value_step: np.ndarray,
        token_index: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        keys = _normalize_step_tensor(
            key_step,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="key_step",
        )
        values = _normalize_step_tensor(
            value_step,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="value_step",
        )
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key_step and value_step token counts must match")
        token_count = int(keys.shape[1])

        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                key_tail = key_tails[kv_head_id]
                value_tail = value_tails[kv_head_id]
                if key_tail is not None:
                    key_tail._ensure_allocated(
                        token_start=token_index if key_tail.token_count == 0 else key_tail.source_page.header.token_start
                    )
                if value_tail is not None:
                    value_tail._ensure_allocated(
                        token_start=token_index if value_tail.token_count == 0 else value_tail.source_page.header.token_start
                    )
            self._batch_upload_persistent_tail_rows(key_tails, keys, token_start=token_index, trace=trace)
            self._batch_upload_persistent_tail_rows(value_tails, values, token_start=token_index, trace=trace)

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.sequence_length != token_index:
                raise ValueError(
                    f"layer {layer_id} kv_head {kv_head_id} expected token_index {state.sequence_length}, received {token_index}"
                )
            finalized_key_pages, finalized_value_pages = state.tail.append_step_rows(
                keys[kv_head_id],
                values[kv_head_id],
                token_start=token_index,
            )
            if finalized_key_pages:
                state.session.append(finalized_key_pages, finalized_value_pages, trace=trace)
                state.invalidate_decode_views()
                if state.persistent_key_tail is not None and state.persistent_value_tail is not None:
                    state.persistent_key_tail.clear()
                    state.persistent_value_tail.clear()
            state.sequence_length += token_count

    def append_step_torch(
        self,
        layer_id: int,
        key_step,
        value_step,
        token_index: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for append_step_torch") from exc
        if not torch.is_tensor(key_step) or not torch.is_tensor(value_step):
            raise TypeError("append_step_torch requires torch.Tensor inputs")
        if self._torch_device_type is None:
            raise RuntimeError("append_step_torch is only available for a torch accelerator backend")

        keys = key_step.detach().to(dtype=torch.float32)
        values = value_step.detach().to(dtype=torch.float32)
        if keys.ndim == 4:
            if int(keys.shape[0]) != 1:
                raise ValueError("key_step batch dimension must be 1 for the Phase 5 Llama path")
            keys = keys[0]
        if values.ndim == 4:
            if int(values.shape[0]) != 1:
                raise ValueError("value_step batch dimension must be 1 for the Phase 5 Llama path")
            values = values[0]
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("key_step and value_step must have shape [kv_heads, token_count, head_dim]")
        if int(keys.shape[0]) != self.num_key_value_heads or int(values.shape[0]) != self.num_key_value_heads:
            raise ValueError(f"append steps must contain {self.num_key_value_heads} KV heads")
        if int(keys.shape[2]) != self.config.head_dim or int(values.shape[2]) != self.config.head_dim:
            raise ValueError(f"append steps head_dim must equal {self.config.head_dim}")
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("key_step and value_step token counts must match")
        token_count = int(keys.shape[1])

        if not self._use_persistent_torch_tail:
            self.append_step(
                layer_id,
                keys.cpu().numpy(),
                values.cpu().numpy(),
                token_index,
                trace=trace,
            )
            return

        key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
        value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.sequence_length != token_index:
                raise ValueError(
                    f"layer {layer_id} kv_head {kv_head_id} expected token_index {state.sequence_length}, received {token_index}"
                )
            if key_tails[kv_head_id] is not None:
                key_tails[kv_head_id]._ensure_allocated(
                    token_start=token_index
                    if key_tails[kv_head_id].token_count == 0
                    else key_tails[kv_head_id].source_page.header.token_start
                )
            if value_tails[kv_head_id] is not None:
                value_tails[kv_head_id]._ensure_allocated(
                    token_start=token_index
                    if value_tails[kv_head_id].token_count == 0
                    else value_tails[kv_head_id].source_page.header.token_start
                )

        self._batch_append_persistent_tail_tensors(key_tails, keys, token_start=token_index)
        self._batch_append_persistent_tail_tensors(value_tails, values, token_start=token_index)

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.persistent_key_tail is None or state.persistent_value_tail is None:
                raise RuntimeError("persistent torch tail path requires allocated key/value tails")
            if state.tail.token_count > 0:
                state.tail.clear()
            if state.persistent_key_tail.token_count >= self.config.tokens_per_page:
                token_start_full = state.persistent_key_tail.source_page.header.token_start
                dense_keys = state.persistent_key_tail.materialize_rows()
                dense_values = state.persistent_value_tail.materialize_rows()
                finalized_key_page = encode_page(
                    dense_keys,
                    self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start_full,
                    build_runtime_metadata=False,
                )
                finalized_value_page = encode_page(
                    dense_values,
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start_full,
                    build_runtime_metadata=False,
                )
                state.session.append([finalized_key_page], [finalized_value_page], trace=trace)
                state.invalidate_decode_views()
                state.persistent_key_tail.clear()
                state.persistent_value_tail.clear()
            state.sequence_length += token_count

    def _prepared_pages_with_tail(
        self,
        layer_id: int,
        kv_head_id: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[PageLike], list[PageLike]]:
        state = self._state(layer_id, kv_head_id)
        self._ensure_prepared_static_pages(state, trace=trace)
        if state.persistent_key_tail is not None and state.persistent_value_tail is not None:
            prepared_key_tail = state.persistent_key_tail.active_page
            prepared_value_tail = state.persistent_value_tail.active_page
            if prepared_key_tail is not None and prepared_value_tail is not None:
                cached_key_pages = state.decode_key_pages_with_tail
                cached_value_pages = state.decode_value_pages_with_tail
                if (
                    cached_key_pages is not None
                    and cached_value_pages is not None
                    and cached_key_pages
                    and cached_value_pages
                    and cached_key_pages[-1] is prepared_key_tail
                    and cached_value_pages[-1] is prepared_value_tail
                    and len(cached_key_pages) == len(state.session.key_pages) + 1
                    and len(cached_value_pages) == len(state.session.value_pages) + 1
                ):
                    return cached_key_pages, cached_value_pages
                key_pages = list(state.session.key_pages)
                value_pages = list(state.session.value_pages)
                key_pages.append(prepared_key_tail)
                value_pages.append(prepared_value_tail)
                state.decode_key_pages_with_tail = key_pages
                state.decode_value_pages_with_tail = value_pages
                return key_pages, value_pages
            state.invalidate_decode_views()
            return state.session.key_pages, state.session.value_pages
        temp_pages = state.tail.build_temp_pages()
        if temp_pages is None:
            return state.session.key_pages, state.session.value_pages
        temp_key_page, temp_value_page = temp_pages
        prepared_temp_key_page = prepare_pages([temp_key_page], backend=self.backend, cache=self.cache, trace=trace)[0]
        prepared_temp_value_page = prepare_pages([temp_value_page], backend=self.backend, cache=self.cache, trace=trace)[0]
        key_pages = list(state.session.key_pages)
        value_pages = list(state.session.value_pages)
        key_pages.append(prepared_temp_key_page)
        value_pages.append(prepared_temp_value_page)
        return key_pages, value_pages

    def decode_layer(
        self,
        layer_id: int,
        query_step: np.ndarray,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace: ExecutionTrace | None = None,
    ) -> np.ndarray:
        queries = _normalize_query_step(
            query_step,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.config.head_dim,
        )
        scaled_queries = queries * np.float32(query_scale)
        grouped_query_heads = self._grouped_query_heads_for_mapping(q_head_to_kv_head)

        outputs = np.zeros((self.num_attention_heads, self.config.head_dim), dtype=np.float32)
        for kv_head_id, q_head_ids in enumerate(grouped_query_heads):
            if not q_head_ids:
                continue
            key_pages, value_pages = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            kv_queries = scaled_queries[list(q_head_ids)]
            _, _, kv_outputs = decode_multi_query_step(
                kv_queries,
                key_pages,
                value_pages,
                backend=self.backend,
                trace=trace,
            )
            outputs[list(q_head_ids)] = kv_outputs
        return outputs

    def decode_layer_torch(
        self,
        layer_id: int,
        query_step,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace: ExecutionTrace | None = None,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for decode_layer_torch") from exc
        if not torch.is_tensor(query_step):
            raise TypeError("decode_layer_torch requires a torch.Tensor query_step")
        if self._torch_device_type is None:
            raise RuntimeError("decode_layer_torch is only available for a torch accelerator backend")
        if query_step.ndim == 4:
            if tuple(query_step.shape[:1] + query_step.shape[2:3]) != (1, 1):
                raise ValueError("query_step must have shape [q_heads, head_dim] or [1, q_heads, 1, head_dim]")
            queries = query_step[0, :, 0, :]
        elif query_step.ndim == 2:
            queries = query_step
        else:
            raise ValueError("query_step must have shape [q_heads, head_dim]")
        if int(queries.shape[0]) != self.num_attention_heads:
            raise ValueError(f"query_step must contain {self.num_attention_heads} query heads")
        if int(queries.shape[1]) != self.config.head_dim:
            raise ValueError(f"query_step head_dim must equal {self.config.head_dim}")

        scaled_queries = queries.to(dtype=torch.float32) * float(query_scale)
        grouped_query_heads = self._grouped_query_heads_for_mapping(q_head_to_kv_head)

        outputs = torch.zeros(
            (self.num_attention_heads, self.config.head_dim),
            dtype=torch.float32,
            device=scaled_queries.device,
        )
        active_q_head_ids: list[tuple[int, ...]] = []
        active_queries: list[Any] = []
        active_key_pages: list[Sequence[PageLike]] = []
        active_value_pages: list[Sequence[PageLike]] = []
        for kv_head_id, q_head_ids in enumerate(grouped_query_heads):
            if not q_head_ids:
                continue
            key_pages, value_pages = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            active_q_head_ids.append(q_head_ids)
            active_queries.append(scaled_queries[list(q_head_ids)])
            active_key_pages.append(key_pages)
            active_value_pages.append(value_pages)

        if _grouped_pages_can_batch(active_key_pages, active_value_pages, active_queries):
            _, _, grouped_outputs = decode_grouped_multiquery_step_prepared_torch_tensor(
                active_queries,
                active_key_pages,
                active_value_pages,
                trace=trace,
            )
            for q_head_ids, kv_outputs in zip(active_q_head_ids, grouped_outputs, strict=True):
                outputs[list(q_head_ids)] = kv_outputs
            return outputs

        for q_head_ids, kv_queries, key_pages, value_pages in zip(
            active_q_head_ids,
            active_queries,
            active_key_pages,
            active_value_pages,
            strict=True,
        ):
            _, _, kv_outputs = decode_multi_query_step_torch_tensor(
                kv_queries,
                key_pages,
                value_pages,
                device_type=self._torch_device_type,
                trace=trace,
            )
            outputs[list(q_head_ids)] = kv_outputs
        return outputs
