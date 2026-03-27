from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ...config import DotCacheConfig
from ...encode import encode_page
from ...model_kv_cache import ModelPagedKVCache, default_q_head_to_kv_head
from ...page_cache import PreparedPageCache
from ...tracing import ExecutionTrace
from ...types import EncodedPage


@dataclass(frozen=True, slots=True)
class VllmBlockKey:
    layer_id: int
    kv_head_id: int
    block_id: int
    kind: str


@dataclass(slots=True)
class VllmBlockEntry:
    key: VllmBlockKey
    page: EncodedPage
    finalized: bool
    token_count: int
    token_start: int


@dataclass(slots=True)
class _LiveBlockState:
    block_id: int | None = None
    token_start: int | None = None
    key_rows: list[np.ndarray] = field(default_factory=list)
    value_rows: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.block_id = None
        self.token_start = None
        self.key_rows.clear()
        self.value_rows.clear()


def _normalize_block_tensor(
    values: np.ndarray,
    *,
    num_key_value_heads: int,
    block_size: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1")
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape [kv_heads, block_count, block_size, head_dim]")
    if int(array.shape[0]) != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if int(array.shape[2]) != block_size:
        raise ValueError(f"{name} block size must equal {block_size}")
    if int(array.shape[3]) != head_dim:
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
            raise ValueError(f"{name} batch dimension must be 1")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, token_count, head_dim]")
    if int(array.shape[0]) != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if int(array.shape[2]) != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


class VllmPagedKVCache:
    def __init__(
        self,
        *,
        config: DotCacheConfig,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        block_size: int,
        backend: str = "torch_cuda",
        cache: PreparedPageCache | None = None,
    ) -> None:
        if config.tokens_per_page != block_size:
            raise ValueError("DotCache tokens_per_page must equal the vLLM block_size for this phase")
        self.config = config
        self.block_size = int(block_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.backend = backend
        self.cache = cache if cache is not None else PreparedPageCache()
        self.model_kv_cache = ModelPagedKVCache(
            config=config,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            backend=backend,
            cache=self.cache,
        )
        self.default_q_head_to_kv_head = default_q_head_to_kv_head(num_attention_heads, num_key_value_heads)
        self._blocks: dict[VllmBlockKey, VllmBlockEntry] = {}
        self._live_states: dict[tuple[int, int], _LiveBlockState] = {}

    @property
    def resident_bytes(self) -> int:
        return self.model_kv_cache.resident_bytes

    def clear(self) -> None:
        self._blocks.clear()
        self._live_states.clear()
        self.model_kv_cache.clear()

    def block_entry(self, layer_id: int, kv_head_id: int, block_id: int, kind: str) -> VllmBlockEntry:
        key = VllmBlockKey(int(layer_id), int(kv_head_id), int(block_id), kind)
        return self._blocks[key]

    def block_entries_for_layer(self, layer_id: int, *, kind: str) -> list[VllmBlockEntry]:
        return sorted(
            [entry for key, entry in self._blocks.items() if key.layer_id == layer_id and key.kind == kind],
            key=lambda entry: (entry.key.kv_head_id, entry.key.block_id),
        )

    def _remove_layer_blocks(self, layer_id: int) -> None:
        stale_keys = [key for key in self._blocks if key.layer_id == layer_id]
        for key in stale_keys:
            del self._blocks[key]
        stale_live = [key for key in self._live_states if key[0] == layer_id]
        for key in stale_live:
            del self._live_states[key]

    def sync_layer_blocks(
        self,
        layer_id: int,
        key_blocks: np.ndarray,
        value_blocks: np.ndarray,
        *,
        block_ids: Sequence[int] | None = None,
        live_block_token_count: int = 0,
        trace: ExecutionTrace | None = None,
    ) -> None:
        keys = _normalize_block_tensor(
            key_blocks,
            num_key_value_heads=self.num_key_value_heads,
            block_size=self.block_size,
            head_dim=self.config.head_dim,
            name="key_blocks",
        )
        values = _normalize_block_tensor(
            value_blocks,
            num_key_value_heads=self.num_key_value_heads,
            block_size=self.block_size,
            head_dim=self.config.head_dim,
            name="value_blocks",
        )
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key_blocks and value_blocks must contain the same number of blocks")

        block_count = int(keys.shape[1])
        resolved_block_ids = tuple(range(block_count)) if block_ids is None else tuple(int(block_id) for block_id in block_ids)
        if len(resolved_block_ids) != block_count:
            raise ValueError("block_ids must align with the number of blocks")
        if len(set(resolved_block_ids)) != len(resolved_block_ids):
            raise ValueError("block_ids must be unique")
        if live_block_token_count < 0 or live_block_token_count > self.block_size:
            raise ValueError("live_block_token_count must be in [0, block_size]")

        self._remove_layer_blocks(layer_id)
        self.model_kv_cache.clear_layer(layer_id)

        if block_count == 0:
            return

        finalized_block_count = block_count if live_block_token_count in (0, self.block_size) else block_count - 1
        full_tokens = finalized_block_count * self.block_size

        dense_keys = keys[:, :finalized_block_count].reshape(
            self.num_key_value_heads,
            full_tokens,
            self.config.head_dim,
        ) if finalized_block_count > 0 else np.zeros((self.num_key_value_heads, 0, self.config.head_dim), dtype=np.float32)
        dense_values = values[:, :finalized_block_count].reshape(
            self.num_key_value_heads,
            full_tokens,
            self.config.head_dim,
        ) if finalized_block_count > 0 else np.zeros((self.num_key_value_heads, 0, self.config.head_dim), dtype=np.float32)

        if finalized_block_count < block_count:
            live_keys = keys[:, finalized_block_count, :live_block_token_count]
            live_values = values[:, finalized_block_count, :live_block_token_count]
            dense_keys = np.concatenate([dense_keys, live_keys], axis=1)
            dense_values = np.concatenate([dense_values, live_values], axis=1)

        self.model_kv_cache.ingest_prefill_cache(layer_id, dense_keys, dense_values, trace=trace)
        self.model_kv_cache.prepare_static_pages(trace=trace)

        for block_index, block_id in enumerate(resolved_block_ids[:finalized_block_count]):
            token_start = block_index * self.block_size
            for kv_head_id in range(self.num_key_value_heads):
                key_page = encode_page(
                    keys[kv_head_id, block_index],
                    self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start,
                    build_runtime_metadata=False,
                )
                value_page = encode_page(
                    values[kv_head_id, block_index],
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start,
                    build_runtime_metadata=False,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, block_id, "K")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, block_id, "K"),
                    page=key_page,
                    finalized=True,
                    token_count=self.block_size,
                    token_start=token_start,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, block_id, "V")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, block_id, "V"),
                    page=value_page,
                    finalized=True,
                    token_count=self.block_size,
                    token_start=token_start,
                )

        if finalized_block_count < block_count:
            live_block_id = resolved_block_ids[finalized_block_count]
            live_token_start = finalized_block_count * self.block_size
            for kv_head_id in range(self.num_key_value_heads):
                live_key_rows = keys[kv_head_id, finalized_block_count, :live_block_token_count]
                live_value_rows = values[kv_head_id, finalized_block_count, :live_block_token_count]
                key_page = encode_page(
                    live_key_rows,
                    self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=live_token_start,
                    mode="M3",
                    build_runtime_metadata=False,
                )
                value_page = encode_page(
                    live_value_rows,
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=live_token_start,
                    mode="M3",
                    build_runtime_metadata=False,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, live_block_id, "K")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, live_block_id, "K"),
                    page=key_page,
                    finalized=False,
                    token_count=live_block_token_count,
                    token_start=live_token_start,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, live_block_id, "V")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, live_block_id, "V"),
                    page=value_page,
                    finalized=False,
                    token_count=live_block_token_count,
                    token_start=live_token_start,
                )
                state = _LiveBlockState(
                    block_id=live_block_id,
                    token_start=live_token_start,
                    key_rows=[np.asarray(row, dtype=np.float32) for row in live_key_rows],
                    value_rows=[np.asarray(row, dtype=np.float32) for row in live_value_rows],
                )
                self._live_states[(layer_id, kv_head_id)] = state

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
        self.model_kv_cache.append_step(layer_id, keys, values, token_index, trace=trace)
        self._update_block_entries_from_steps(layer_id, keys, values, token_index)

    def append_step_torch(
        self,
        layer_id: int,
        key_step,
        value_step,
        token_index: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        self.model_kv_cache.append_step_torch(layer_id, key_step, value_step, token_index, trace=trace)
        keys = key_step.detach().to(dtype=key_step.dtype).cpu().numpy().astype(np.float32, copy=False)
        values = value_step.detach().to(dtype=value_step.dtype).cpu().numpy().astype(np.float32, copy=False)
        self._update_block_entries_from_steps(layer_id, keys, values, token_index)

    def append_tokens_torch(
        self,
        layer_id: int,
        key_rows,
        value_rows,
        token_positions,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        positions = token_positions.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.size == 0:
            return
        expected = np.arange(int(positions[0]), int(positions[0]) + positions.size, dtype=np.int64)
        if not np.array_equal(positions, expected):
            raise ValueError("Phase 6 vLLM adapter requires contiguous batch=1 token positions")
        self.append_step_torch(layer_id, key_rows.transpose(0, 1), value_rows.transpose(0, 1), int(positions[0]), trace=trace)

    def _update_block_entries_from_steps(
        self,
        layer_id: int,
        key_rows_by_head: np.ndarray,
        value_rows_by_head: np.ndarray,
        token_index: int,
    ) -> None:
        token_count = int(key_rows_by_head.shape[1])
        for offset in range(token_count):
            absolute_token = int(token_index) + offset
            block_id = absolute_token // self.block_size
            block_offset = absolute_token % self.block_size
            token_start = block_id * self.block_size
            for kv_head_id in range(self.num_key_value_heads):
                state = self._live_states.setdefault((layer_id, kv_head_id), _LiveBlockState())
                if block_offset == 0 or state.block_id != block_id:
                    state.clear()
                    state.block_id = block_id
                    state.token_start = token_start
                state.key_rows.append(np.asarray(key_rows_by_head[kv_head_id, offset], dtype=np.float32))
                state.value_rows.append(np.asarray(value_rows_by_head[kv_head_id, offset], dtype=np.float32))
                key_rows = np.stack(state.key_rows, axis=0).astype(np.float32, copy=False)
                value_rows = np.stack(state.value_rows, axis=0).astype(np.float32, copy=False)
                finalized = key_rows.shape[0] == self.block_size
                key_mode = None if finalized else "M3"
                value_mode = None if finalized else "M3"
                key_page = encode_page(
                    key_rows,
                    self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start,
                    mode=key_mode,
                    build_runtime_metadata=False,
                )
                value_page = encode_page(
                    value_rows,
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start,
                    mode=value_mode,
                    build_runtime_metadata=False,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, block_id, "K")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, block_id, "K"),
                    page=key_page,
                    finalized=finalized,
                    token_count=int(key_rows.shape[0]),
                    token_start=token_start,
                )
                self._blocks[VllmBlockKey(layer_id, kv_head_id, block_id, "V")] = VllmBlockEntry(
                    key=VllmBlockKey(layer_id, kv_head_id, block_id, "V"),
                    page=value_page,
                    finalized=finalized,
                    token_count=int(value_rows.shape[0]),
                    token_start=token_start,
                )
                if finalized:
                    state.clear()

    def decode_layer(
        self,
        layer_id: int,
        query_step: np.ndarray,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace: ExecutionTrace | None = None,
    ) -> np.ndarray:
        return self.model_kv_cache.decode_layer(
            layer_id,
            query_step,
            q_head_to_kv_head,
            query_scale=query_scale,
            trace=trace,
        )

    def decode_layer_torch(
        self,
        layer_id: int,
        query_step,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace: ExecutionTrace | None = None,
    ):
        return self.model_kv_cache.decode_layer_torch(
            layer_id,
            query_step,
            q_head_to_kv_head,
            query_scale=query_scale,
            trace=trace,
        )
