from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .attention_runtime import BackendName, decode_step, prepare_pages
from .backends import PreparedPageMPS
from .config import DotCacheConfig
from .encode import encode_page
from .page_cache import PreparedPageCache
from .session_runtime import PagedDecodeSession
from .tracing import ExecutionTrace
from .types import EncodedPage

PageLike = EncodedPage | PreparedPageMPS


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


@dataclass(slots=True)
class _HeadSessionState:
    session: PagedDecodeSession
    tail: _TailPageBuilder
    sequence_length: int = 0

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
        self._states: dict[tuple[int, int], _HeadSessionState] = {}

    @property
    def resident_bytes(self) -> int:
        return self.cache.resident_bytes

    def clear(self) -> None:
        for state in self._states.values():
            state.clear(clear_prepared_cache=False)
        self.cache.clear()

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
            state = _HeadSessionState(
                session=PagedDecodeSession(backend=self.backend, cache=self.cache),
                tail=_TailPageBuilder(self.config, layer_id=layer_id, kv_head_id=kv_head_id),
            )
            self._states[key] = state
        return state

    def layer_sequence_length(self, layer_id: int) -> int:
        self._validate_layer_id(layer_id)
        lengths = {self._state(layer_id, kv_head_id).sequence_length for kv_head_id in range(self.num_key_value_heads)}
        if len(lengths) > 1:
            raise RuntimeError(f"layer {layer_id} KV heads disagree on sequence length")
        return next(iter(lengths), 0)

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
        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            state.clear(clear_prepared_cache=False)
            dense_keys = keys[kv_head_id]
            dense_values = values[kv_head_id]
            full_page_count = seq_len // self.config.tokens_per_page
            full_tokens = full_page_count * self.config.tokens_per_page

            preload_key_pages: list[EncodedPage] = []
            preload_value_pages: list[EncodedPage] = []
            for page_start in range(0, full_tokens, self.config.tokens_per_page):
                page_end = page_start + self.config.tokens_per_page
                preload_key_pages.append(
                    encode_page(
                        dense_keys[page_start:page_end],
                        self.config,
                        kind="K",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=page_start,
                    )
                )
                preload_value_pages.append(
                    encode_page(
                        dense_values[page_start:page_end],
                        self.config,
                        kind="V",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=page_start,
                    )
                )
            if preload_key_pages:
                state.session.append(preload_key_pages, preload_value_pages, trace=trace)
            remainder_keys = dense_keys[full_tokens:]
            remainder_values = dense_values[full_tokens:]
            state.tail.load_prefill_remainder(remainder_keys, remainder_values, token_start=full_tokens)
            state.sequence_length = seq_len

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
            state.sequence_length += token_count

    def _prepared_pages_with_tail(
        self,
        layer_id: int,
        kv_head_id: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[PageLike], list[PageLike]]:
        state = self._state(layer_id, kv_head_id)
        key_pages = list(state.session.key_pages)
        value_pages = list(state.session.value_pages)
        temp_pages = state.tail.build_temp_pages()
        if temp_pages is None:
            return key_pages, value_pages
        temp_key_page, temp_value_page = temp_pages
        prepared_temp_key_page = prepare_pages([temp_key_page], backend=self.backend, cache=self.cache, trace=trace)[0]
        prepared_temp_value_page = prepare_pages([temp_value_page], backend=self.backend, cache=self.cache, trace=trace)[0]
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
        mapping = np.asarray(q_head_to_kv_head, dtype=np.int64)
        if mapping.shape != (self.num_attention_heads,):
            raise ValueError("q_head_to_kv_head must have shape [num_attention_heads]")

        grouped_query_heads: dict[int, list[int]] = defaultdict(list)
        for q_head_id, kv_head_id in enumerate(mapping.tolist()):
            if kv_head_id < 0 or kv_head_id >= self.num_key_value_heads:
                raise ValueError("q_head_to_kv_head contains an invalid KV head id")
            grouped_query_heads[kv_head_id].append(q_head_id)

        outputs = np.zeros((self.num_attention_heads, self.config.head_dim), dtype=np.float32)
        prepared_page_pairs: dict[int, tuple[list[PageLike], list[PageLike]]] = {}
        for kv_head_id, q_head_ids in grouped_query_heads.items():
            prepared_page_pairs[kv_head_id] = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            key_pages, value_pages = prepared_page_pairs[kv_head_id]
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            for q_head_id in q_head_ids:
                _, _, output = decode_step(
                    scaled_queries[q_head_id],
                    key_pages,
                    value_pages,
                    backend=self.backend,
                    trace=trace,
                )
                outputs[q_head_id] = output
        return outputs
