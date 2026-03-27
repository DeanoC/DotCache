from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ..config import DotCacheConfig
from ..model_kv_cache import ModelPagedKVCache
from ..page_cache import PreparedPageCache
from ..tracing import ExecutionTrace


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


if transformers_available():
    import torch
    import torch.nn as nn
    import transformers.models.llama.modeling_llama as llama_mod
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    nn = object  # type: ignore[assignment]
    llama_mod = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


AttentionMode = Literal["dense", "dotcache"]


@dataclass(slots=True)
class LlamaReplayRecord:
    step_index: int
    layer_id: int
    token_index: int
    query_states: np.ndarray
    key_states: np.ndarray
    value_states: np.ndarray
    context_states: np.ndarray
    output_states: np.ndarray


@dataclass(slots=True)
class LlamaLayerRuntimeProfile:
    layer_id: int
    call_count: int = 0
    qkv_projection_ms_total: float = 0.0
    append_ms_total: float = 0.0
    decode_ms_total: float = 0.0
    output_projection_ms_total: float = 0.0

    def reset(self) -> None:
        self.call_count = 0
        self.qkv_projection_ms_total = 0.0
        self.append_ms_total = 0.0
        self.decode_ms_total = 0.0
        self.output_projection_ms_total = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "layer_id": self.layer_id,
            "call_count": self.call_count,
            "qkv_projection_ms_total": float(self.qkv_projection_ms_total),
            "append_ms_total": float(self.append_ms_total),
            "decode_ms_total": float(self.decode_ms_total),
            "output_projection_ms_total": float(self.output_projection_ms_total),
            "qkv_projection_ms_per_call": float(self.qkv_projection_ms_total / max(self.call_count, 1)),
            "append_ms_per_call": float(self.append_ms_total / max(self.call_count, 1)),
            "decode_ms_per_call": float(self.decode_ms_total / max(self.call_count, 1)),
            "output_projection_ms_per_call": float(self.output_projection_ms_total / max(self.call_count, 1)),
        }


def _require_transformers() -> None:
    if not transformers_available():
        raise RuntimeError("transformers and torch are required for the Llama integration path")


def _torch_backend_matches_device(backend: str, device_type: str) -> bool:
    if device_type == "mps":
        return backend in {"torch_mps", "auto"}
    if device_type == "cuda":
        return backend in {"torch_cuda", "auto"}
    return False


def _default_model_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_type(device: Any) -> str:
    if hasattr(device, "type"):
        return str(device.type)
    return str(device)


def _synchronize_device(device: Any) -> None:
    device_type = _device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed_call(fn, *, device: Any) -> tuple[Any, float]:
    _synchronize_device(device)
    start = time.perf_counter()
    result = fn()
    _synchronize_device(device)
    return result, (time.perf_counter() - start) * 1000.0


def _prewarm_torch_decode_layers(adapter: "LlamaDotCacheModelAdapter", *, device: Any) -> None:
    device_type = _device_type(device)
    if device_type != "cuda" or not _torch_backend_matches_device(adapter.backend, device_type):
        return
    if adapter.model_kv_cache._torch_device_type is None:
        return

    zero_query = torch.zeros(
        (adapter.model.config.num_attention_heads, adapter.dotcache_config.head_dim),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        for layer_id in range(adapter.model.config.num_hidden_layers):
            if adapter.model_kv_cache.layer_sequence_length(layer_id) <= 0:
                continue
            adapter.model_kv_cache.decode_layer_torch(
                layer_id,
                zero_query,
                adapter.q_head_to_kv_head,
                query_scale=1.0,
                trace=None,
            )
    _synchronize_device(device)


def _begin_cuda_memory_region(device: Any) -> dict[str, int] | None:
    if _device_type(device) != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device=device)
    torch.cuda.reset_peak_memory_stats(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "allocation_count": int(stats.get("allocation.all.allocated", 0)),
        "segment_count": int(stats.get("segment.all.allocated", 0)),
    }


def _end_cuda_memory_region(device: Any, baseline: dict[str, int] | None) -> dict[str, int]:
    if baseline is None or _device_type(device) != "cuda" or not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize(device=device)
    stats = torch.cuda.memory_stats(device)
    return {
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_allocation_count": int(stats.get("allocation.all.allocated", 0)) - baseline["allocation_count"],
        "cuda_segment_allocation_count": int(stats.get("segment.all.allocated", 0)) - baseline["segment_count"],
    }


def _default_attention_mask(input_ids) -> Any:
    return torch.ones_like(input_ids, dtype=torch.long)


def _clone_attention_mask(attention_mask) -> Any:
    if attention_mask is None:
        return None
    return attention_mask.clone()


def _normalize_input_ids(input_ids, *, device) -> Any:
    tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError("Phase 5 Llama harness requires input_ids with shape [1, seq_len]")
    return tensor


def _ensure_attention_mask(input_ids, attention_mask, *, device) -> Any:
    if attention_mask is None:
        return _default_attention_mask(input_ids).to(device=device)
    mask = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    if mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    return mask


def _can_skip_decode_attention_mask(attention_mask) -> bool:
    if attention_mask is None:
        return True
    return bool(torch.all(attention_mask != 0).item())


def extract_past_key_values_arrays(past_key_values) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        raise ValueError("past_key_values must expose a .layers cache structure")
    extracted: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        keys = layer.keys.detach().to(dtype=torch.float32).cpu().numpy()
        values = layer.values.detach().to(dtype=torch.float32).cpu().numpy()
        if keys.shape[0] != 1 or values.shape[0] != 1:
            raise ValueError("Phase 5 Llama harness requires batch=1 past_key_values")
        extracted.append((keys, values))
    return extracted


def extract_past_key_values_tensors(past_key_values) -> list[tuple[Any, Any]]:
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        raise ValueError("past_key_values must expose a .layers cache structure")
    extracted: list[tuple[Any, Any]] = []
    for layer in layers:
        keys = layer.keys.detach().to(dtype=torch.float32)
        values = layer.values.detach().to(dtype=torch.float32)
        if keys.shape[0] != 1 or values.shape[0] != 1:
            raise ValueError("Phase 5 Llama harness requires batch=1 past_key_values")
        extracted.append((keys, values))
    return extracted


def _prefill_layer_nbytes(prefill_layers: Sequence[tuple[Any, Any]]) -> int:
    total = 0
    for layer_keys, layer_values in prefill_layers:
        if torch.is_tensor(layer_keys):
            total += int(layer_keys.numel() * layer_keys.element_size())
        else:
            keys = np.asarray(layer_keys)
            total += int(keys.nbytes)
        if torch.is_tensor(layer_values):
            total += int(layer_values.numel() * layer_values.element_size())
        else:
            values = np.asarray(layer_values)
            total += int(values.nbytes)
    return total


def _dense_kv_bytes_after_decode(
    prefill_layers: Sequence[tuple[Any, Any]],
    *,
    generated_token_count: int,
) -> int:
    if not prefill_layers:
        return 0
    layer_keys, _ = prefill_layers[0]
    if torch.is_tensor(layer_keys):
        seq_len = int(layer_keys.shape[2])
        kv_heads = int(layer_keys.shape[1])
        head_dim = int(layer_keys.shape[3])
        dtype_bytes = int(layer_keys.element_size())
    else:
        keys = np.asarray(layer_keys)
        seq_len = int(keys.shape[2])
        kv_heads = int(keys.shape[1])
        head_dim = int(keys.shape[3])
        dtype_bytes = int(keys.dtype.itemsize)
    total_tokens = seq_len + max(generated_token_count - 1, 0)
    layer_count = len(prefill_layers)
    return int(layer_count * 2 * kv_heads * total_tokens * head_dim * dtype_bytes)


class DotCacheLlamaAttention(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "LlamaDotCacheModelAdapter") -> None:
        super().__init__()
        self.base_attention = base_attention
        self.adapter = adapter
        self.layer_idx = int(base_attention.layer_idx)
        self.config = base_attention.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.adapter.mode == "dense" and not self.adapter.capture_enabled:
            return self.base_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if self.adapter.mode == "dense":
            return self._forward_dense_with_capture(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return self._forward_dotcache(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Llama attention path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        return llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin), value_states

    def _forward_dense_with_capture(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        (query_states, key_states), value_states = self._project_qkv(hidden_states, position_embeddings)
        fresh_key_states = key_states
        fresh_value_states = value_states

        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = llama_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.base_attention.config._attn_implementation,
            llama_mod.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base_attention.attention_dropout,
            scaling=self.base_attention.scaling,
            **kwargs,
        )
        reshaped_output = attn_output.reshape(*input_shape, -1).contiguous()
        projected_output = self.base_attention.o_proj(reshaped_output)

        if self.adapter.capture_enabled and tuple(hidden_states.shape[:2]) == (1, 1):
            token_index = self.adapter.current_token_index(cache_position)
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    key_states=fresh_key_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    value_states=fresh_value_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    context_states=attn_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, attn_weights

    def _forward_dotcache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, kwargs
        if past_key_values is not None:
            raise ValueError("DotCache decode mode manages its own KV cache and requires past_key_values=None")
        if tuple(hidden_states.shape[:2]) != (1, 1):
            raise ValueError("DotCache decode mode only supports batch=1 and query_len=1")
        token_index = self.adapter.current_token_index(cache_position)
        ((query_states, key_states), value_states), qkv_ms = _timed_call(
            lambda: self._project_qkv(hidden_states, position_embeddings),
            device=hidden_states.device,
        )
        query_step = query_states[0, :, 0, :].detach().to(dtype=torch.float32)
        key_step = key_states[0].detach().to(dtype=torch.float32)
        value_step = value_states[0].detach().to(dtype=torch.float32)
        self.adapter.record_layer_runtime(self.layer_idx, qkv_projection_ms=qkv_ms)

        _, append_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.append_step_torch(
                self.layer_idx,
                key_step,
                value_step,
                token_index,
                trace=self.adapter.active_trace,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.append_step(
                self.layer_idx,
                key_step.cpu().numpy(),
                value_step.cpu().numpy(),
                token_index,
                trace=self.adapter.active_trace,
            ),
            device=hidden_states.device,
        )
        self.adapter.append_runtime_ms_total += append_ms

        context_states, decode_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.decode_layer_torch(
                self.layer_idx,
                query_step,
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=self.adapter.active_trace,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.decode_layer(
                self.layer_idx,
                query_step.detach().cpu().numpy(),
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=self.adapter.active_trace,
            ),
            device=hidden_states.device,
        )
        self.adapter.decode_runtime_ms_total += decode_ms

        def _project_output():
            local_context_states = context_states
            if not torch.is_tensor(local_context_states):
                local_context_states = torch.as_tensor(local_context_states, dtype=torch.float32, device=hidden_states.device)
            context_tensor = local_context_states.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
            return self.base_attention.o_proj(context_tensor.reshape(1, 1, -1).contiguous())

        projected_output, output_projection_ms = _timed_call(_project_output, device=hidden_states.device)
        self.adapter.record_layer_runtime(
            self.layer_idx,
            append_ms=append_ms,
            decode_ms=decode_ms,
            output_projection_ms=output_projection_ms,
        )

        if self.adapter.capture_enabled:
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_step.detach().cpu().numpy(),
                    key_states=key_step[:, 0, :].detach().cpu().numpy(),
                    value_states=value_step[:, 0, :].detach().cpu().numpy(),
                    context_states=context_states.detach().cpu().numpy().astype(np.float32, copy=False),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, None


class LlamaDotCacheModelAdapter:
    def __init__(
        self,
        model,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
        cache: PreparedPageCache | None = None,
    ) -> None:
        _require_transformers()
        self.model = model
        self.dotcache_config = dotcache_config
        self.backend = backend
        self.cache = cache if cache is not None else PreparedPageCache()
        self.model_kv_cache = ModelPagedKVCache(
            config=dotcache_config,
            num_hidden_layers=model.config.num_hidden_layers,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            backend=backend,
            cache=self.cache,
        )
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()
        self.mode: AttentionMode = "dense"
        self.capture_enabled = False
        self.capture_step_index = -1
        self.active_trace: ExecutionTrace | None = None
        self._pending_records: list[LlamaReplayRecord] = []
        self._wrappers: list[DotCacheLlamaAttention] = []
        self.append_runtime_ms_total = 0.0
        self.decode_runtime_ms_total = 0.0
        self.qkv_projection_ms_total = 0.0
        self.output_projection_ms_total = 0.0
        self.layer_runtime_profiles = [LlamaLayerRuntimeProfile(layer_id=layer_id) for layer_id in range(model.config.num_hidden_layers)]
        self._current_token_index_override: int | None = None
        self._install_wrappers()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _install_wrappers(self) -> None:
        for layer in self.model.model.layers[: self.model.config.num_hidden_layers]:
            wrapper = DotCacheLlamaAttention(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def set_mode(self, mode: AttentionMode) -> None:
        self.mode = mode

    def set_capture(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)

    def begin_capture_step(self, step_index: int) -> None:
        self.capture_step_index = int(step_index)
        self._pending_records = []
        self.capture_enabled = True

    def end_capture_step(self) -> list[LlamaReplayRecord]:
        records = list(self._pending_records)
        self._pending_records = []
        self.capture_enabled = False
        self.capture_step_index = -1
        return records

    def record_replay(self, record: LlamaReplayRecord) -> None:
        if self.capture_step_index < 0:
            return
        self._pending_records.append(record)

    def current_token_index(self, cache_position) -> int:
        if self._current_token_index_override is not None:
            return self._current_token_index_override
        if cache_position is None:
            raise ValueError("cache_position is required for the Phase 5 Llama path")
        token_positions = cache_position.reshape(-1)
        if token_positions.numel() != 1:
            raise ValueError("Phase 5 Llama path requires a single cache_position per decode step")
        return int(token_positions.item())

    def set_current_token_index(self, token_index: int | None) -> None:
        self._current_token_index_override = None if token_index is None else int(token_index)

    def clear(self) -> None:
        self.model_kv_cache.clear()
        self._pending_records = []
        self.capture_enabled = False
        self.capture_step_index = -1
        self.active_trace = None
        self._current_token_index_override = None
        self.reset_runtime_metrics()

    def reconfigure(self, dotcache_config: DotCacheConfig, *, backend: str | None = None) -> None:
        self.dotcache_config = dotcache_config
        if backend is not None:
            self.backend = backend
        self.model_kv_cache = ModelPagedKVCache(
            config=dotcache_config,
            num_hidden_layers=self.model.config.num_hidden_layers,
            num_attention_heads=self.model.config.num_attention_heads,
            num_key_value_heads=self.model.config.num_key_value_heads,
            backend=self.backend,
            cache=self.cache,
        )
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()
        self.clear()

    def reset_runtime_metrics(self) -> None:
        self.append_runtime_ms_total = 0.0
        self.decode_runtime_ms_total = 0.0
        self.qkv_projection_ms_total = 0.0
        self.output_projection_ms_total = 0.0
        for profile in self.layer_runtime_profiles:
            profile.reset()

    def record_layer_runtime(
        self,
        layer_id: int,
        *,
        qkv_projection_ms: float = 0.0,
        append_ms: float = 0.0,
        decode_ms: float = 0.0,
        output_projection_ms: float = 0.0,
    ) -> None:
        profile = self.layer_runtime_profiles[layer_id]
        if qkv_projection_ms > 0.0:
            profile.call_count += 1
            profile.qkv_projection_ms_total += qkv_projection_ms
            self.qkv_projection_ms_total += qkv_projection_ms
        if append_ms > 0.0:
            profile.append_ms_total += append_ms
        if decode_ms > 0.0:
            profile.decode_ms_total += decode_ms
        if output_projection_ms > 0.0:
            profile.output_projection_ms_total += output_projection_ms
            self.output_projection_ms_total += output_projection_ms

    def runtime_profile_summary(self, *, model_forward_ms_total: float) -> dict[str, Any]:
        per_layer = [profile.to_dict() for profile in self.layer_runtime_profiles if profile.call_count > 0]
        accounted_ms_total = (
            self.qkv_projection_ms_total
            + self.append_runtime_ms_total
            + self.decode_runtime_ms_total
            + self.output_projection_ms_total
        )
        return {
            "model_forward_ms_total": float(model_forward_ms_total),
            "qkv_projection_ms_total": float(self.qkv_projection_ms_total),
            "append_runtime_ms_total": float(self.append_runtime_ms_total),
            "decode_runtime_ms_total": float(self.decode_runtime_ms_total),
            "output_projection_ms_total": float(self.output_projection_ms_total),
            "other_overhead_ms_total": float(max(model_forward_ms_total - accounted_ms_total, 0.0)),
            "per_layer": per_layer,
        }

    def load_prefill_cache(self, past_key_values, *, trace: ExecutionTrace | None = None) -> None:
        if _torch_backend_matches_device(self.backend, self.device.type):
            self.load_prefill_cache_tensors(extract_past_key_values_tensors(past_key_values), trace=trace)
        else:
            self.load_prefill_cache_arrays(extract_past_key_values_arrays(past_key_values), trace=trace)

    def load_prefill_cache_arrays(
        self,
        prefill_layers: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if len(prefill_layers) != self.model.config.num_hidden_layers:
            raise ValueError("prefill_layers must align with model.config.num_hidden_layers")
        self.model_kv_cache.clear()
        for layer_idx, (layer_keys, layer_values) in enumerate(prefill_layers):
            self.model_kv_cache.ingest_prefill_cache(layer_idx, layer_keys, layer_values, trace=trace)
        self.model_kv_cache.prepare_static_pages(trace=trace)

    def load_prefill_cache_tensors(
        self,
        prefill_layers: Sequence[tuple[Any, Any]],
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if len(prefill_layers) != self.model.config.num_hidden_layers:
            raise ValueError("prefill_layers must align with model.config.num_hidden_layers")
        self.model_kv_cache.clear()
        for layer_idx, (layer_keys, layer_values) in enumerate(prefill_layers):
            self.model_kv_cache.ingest_prefill_cache_torch(layer_idx, layer_keys, layer_values, trace=trace)
        self.model_kv_cache.prepare_static_pages(trace=trace)


@dataclass(slots=True)
class LlamaDotCacheHarness:
    model: Any
    tokenizer: Any | None
    adapter: LlamaDotCacheModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "LlamaDotCacheHarness":
        _require_transformers()
        dtype = getattr(torch, torch_dtype)
        resolved_device = _default_model_device() if device is None else device
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(resolved_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        adapter = LlamaDotCacheModelAdapter(model, dotcache_config, backend=backend)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(self, prompt: str) -> tuple[Any, Any]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is unavailable for text prompt input")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.adapter.device)
        attention_mask = encoded["attention_mask"].to(self.adapter.device)
        return input_ids, attention_mask

    def run_replay(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
    ) -> dict[str, float | int]:
        return run_llama_replay_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=decode_steps,
            tokenizer=self.tokenizer,
        )

    def generate_greedy(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 8,
        profile: bool = False,
    ) -> dict[str, Any]:
        return run_llama_generation_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer,
            profile=profile,
        )

    def evaluate_loss(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
    ) -> dict[str, Any]:
        return run_llama_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            tokenizer=self.tokenizer,
        )


def _prefill_prompt(
    model,
    adapter: LlamaDotCacheModelAdapter,
    input_ids,
    attention_mask,
):
    adapter.set_mode("dense")
    adapter.set_capture(False)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    if _torch_backend_matches_device(adapter.backend, input_ids.device.type):
        prefill_layers = extract_past_key_values_tensors(outputs.past_key_values)
    else:
        prefill_layers = extract_past_key_values_arrays(outputs.past_key_values)
    first_generated_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return outputs, prefill_layers, first_generated_token


def _run_dense_greedy_decode(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    capture: bool,
) -> dict[str, Any]:
    (prefill_outputs, prefill_layers, first_generated_token), prefill_ms = _timed_call(
        lambda: _prefill_prompt(model, adapter, input_ids, attention_mask),
        device=input_ids.device,
    )
    if max_new_tokens <= 0:
        return {
            "prefill_layers": prefill_layers,
            "generated_ids": [],
            "decode_inputs": [],
            "step_logits": [],
            "capture_records": [],
            "prefill_outputs": prefill_outputs,
            "prefill_ms": prefill_ms,
        }

    generated_ids = [int(first_generated_token.item())]
    if max_new_tokens == 1:
        return {
            "prefill_layers": prefill_layers,
            "generated_ids": generated_ids,
            "decode_inputs": [],
            "step_logits": [],
            "capture_records": [],
            "prefill_outputs": prefill_outputs,
            "prefill_ms": prefill_ms,
        }

    adapter.set_mode("dense")
    adapter.set_capture(False)
    past_key_values = prefill_outputs.past_key_values
    current_input_ids = first_generated_token
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    decode_inputs: list[Any] = []
    step_logits: list[np.ndarray] = []
    capture_records: list[list[LlamaReplayRecord]] = []
    dense_decode_ms_total = 0.0

    for step_index in range(max_new_tokens - 1):
        decode_inputs.append(current_input_ids.detach().clone())
        if capture:
            adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
        try:
            outputs, step_ms = _timed_call(
                lambda: model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                ),
                device=input_ids.device,
            )
            dense_decode_ms_total += step_ms
        finally:
            adapter.set_current_token_index(None)
        if capture:
            capture_records.append(adapter.end_capture_step())
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()
        step_logits.append(logits)
        current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(current_input_ids.item()))
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    return {
        "prefill_layers": prefill_layers,
        "generated_ids": generated_ids,
        "decode_inputs": decode_inputs,
        "step_logits": step_logits,
        "capture_records": capture_records,
        "prefill_outputs": prefill_outputs,
        "prefill_ms": prefill_ms,
        "dense_decode_ms_total": dense_decode_ms_total,
    }


def _run_dotcache_decode_inputs(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    input_ids,
    attention_mask,
    prefill_layers: Sequence[tuple[Any, Any]],
    decode_inputs: Sequence[Any],
    profile_backend: bool = False,
) -> dict[str, Any]:
    if prefill_layers and torch.is_tensor(prefill_layers[0][0]):
        adapter.load_prefill_cache_tensors(prefill_layers)
    else:
        adapter.load_prefill_cache_arrays(prefill_layers)
    _prewarm_torch_decode_layers(adapter, device=input_ids.device)
    adapter.set_mode("dotcache")
    adapter.reset_runtime_metrics()
    use_attention_mask = not _can_skip_decode_attention_mask(attention_mask)
    current_attention_mask = attention_mask if use_attention_mask else None
    step_logits: list[np.ndarray] = []
    decode_ms_total = 0.0
    trace_total = ExecutionTrace(capture_timings=profile_backend)

    for offset, decode_input in enumerate(decode_inputs):
        cache_position = torch.tensor([input_ids.shape[1] + offset], dtype=torch.long, device=input_ids.device)
        step_trace = ExecutionTrace(capture_timings=profile_backend)
        adapter.active_trace = step_trace
        adapter.set_current_token_index(int(input_ids.shape[1] + offset))
        try:
            outputs, step_ms = _timed_call(
                lambda: model(
                    input_ids=decode_input,
                    attention_mask=current_attention_mask,
                    use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                ),
                device=input_ids.device,
            )
        finally:
            adapter.active_trace = None
            adapter.set_current_token_index(None)
        decode_ms_total += step_ms
        trace_total.merge(step_trace)
        step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())

    return {
        "decode_ms_total": decode_ms_total,
        "append_runtime_ms_total": adapter.append_runtime_ms_total,
        "decode_runtime_ms_total": adapter.decode_runtime_ms_total,
        "step_logits": step_logits,
        "trace": trace_total,
    }


def _run_dense_decode_inputs(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    input_ids,
    attention_mask,
    prefill_outputs,
    decode_inputs: Sequence[Any],
) -> dict[str, Any]:
    adapter.set_mode("dense")
    adapter.set_capture(False)
    past_key_values = prefill_outputs.past_key_values
    current_attention_mask = attention_mask
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    step_logits: list[np.ndarray] = []
    decode_ms_total = 0.0

    for decode_input in decode_inputs:
        start = time.perf_counter()
        outputs = model(
            input_ids=decode_input,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
        )
        decode_ms_total += (time.perf_counter() - start) * 1000.0
        step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        past_key_values = outputs.past_key_values
        if current_attention_mask is not None:
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
        cache_position = cache_position + 1

    return {
        "decode_ms_total": decode_ms_total,
        "step_logits": step_logits,
    }


def _run_dotcache_greedy_decode(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    input_ids,
    attention_mask,
    prefill_layers: Sequence[tuple[Any, Any]],
    first_generated_token,
    max_new_tokens: int,
    profile_backend: bool = False,
) -> dict[str, Any]:
    if prefill_layers and torch.is_tensor(prefill_layers[0][0]):
        adapter.load_prefill_cache_tensors(prefill_layers)
    else:
        adapter.load_prefill_cache_arrays(prefill_layers)
    _prewarm_torch_decode_layers(adapter, device=input_ids.device)
    adapter.set_mode("dotcache")
    adapter.reset_runtime_metrics()
    generated_ids = [int(first_generated_token.item())]
    if max_new_tokens <= 1:
        return {
            "generated_ids": generated_ids,
            "decode_ms_total": 0.0,
            "append_runtime_ms_total": 0.0,
            "decode_runtime_ms_total": 0.0,
            "step_count": 0,
            "trace": ExecutionTrace(capture_timings=profile_backend),
        }

    current_input_ids = first_generated_token
    use_attention_mask = not _can_skip_decode_attention_mask(attention_mask)
    current_attention_mask = (
        torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        if use_attention_mask
        else None
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    current_token_index = int(input_ids.shape[1])
    step_count = 0
    decode_ms_total = 0.0
    trace_total = ExecutionTrace(capture_timings=profile_backend)

    for _ in range(max_new_tokens - 1):
        step_trace = ExecutionTrace(capture_timings=profile_backend)
        adapter.active_trace = step_trace
        adapter.set_current_token_index(current_token_index)
        try:
            outputs, step_ms = _timed_call(
                lambda: model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                ),
                device=input_ids.device,
            )
        finally:
            adapter.active_trace = None
            adapter.set_current_token_index(None)
        decode_ms_total += step_ms
        trace_total.merge(step_trace)
        current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(current_input_ids.item()))
        step_count += 1
        if current_attention_mask is not None:
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
        cache_position = cache_position + 1
        current_token_index += 1

    return {
        "generated_ids": generated_ids,
        "decode_ms_total": decode_ms_total,
        "append_runtime_ms_total": adapter.append_runtime_ms_total,
        "decode_runtime_ms_total": adapter.decode_runtime_ms_total,
        "step_count": step_count,
        "trace": trace_total,
    }


def run_llama_replay_harness(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    decode_steps: int = 4,
    tokenizer=None,
) -> dict[str, float | int]:
    _require_transformers()
    if prompt is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when prompt text is provided")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)
    dense_result = _run_dense_greedy_decode(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=decode_steps + 1,
        capture=True,
    )

    replay_cache = ModelPagedKVCache(
        config=adapter.dotcache_config,
        num_hidden_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        backend=adapter.backend,
        cache=PreparedPageCache(),
    )
    for layer_idx, (layer_keys, layer_values) in enumerate(dense_result["prefill_layers"]):
        if torch.is_tensor(layer_keys):
            replay_cache.ingest_prefill_cache_torch(layer_idx, layer_keys, layer_values)
        else:
            replay_cache.ingest_prefill_cache(layer_idx, layer_keys, layer_values)

    replay_context_max_abs = 0.0
    replay_context_max_rel = 0.0
    for step_records in dense_result["capture_records"]:
        for record in step_records:
            replay_cache.append_step(
                record.layer_id,
                record.key_states[:, None, :],
                record.value_states[:, None, :],
                record.token_index,
            )
            replay_context = replay_cache.decode_layer(record.layer_id, record.query_states, adapter.q_head_to_kv_head)
            delta = np.abs(replay_context - record.context_states)
            denom = np.maximum(np.abs(record.context_states), 1e-8)
            replay_context_max_abs = max(replay_context_max_abs, float(np.max(delta)))
            replay_context_max_rel = max(replay_context_max_rel, float(np.max(delta / denom)))

    dotcache_teacher_forced = _run_dotcache_decode_inputs(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefill_layers=dense_result["prefill_layers"],
        decode_inputs=dense_result["decode_inputs"],
    )
    dense_logits = np.stack(dense_result["step_logits"], axis=0) if dense_result["step_logits"] else np.zeros((0, 1))
    dotcache_logits = (
        np.stack(dotcache_teacher_forced["step_logits"], axis=0) if dotcache_teacher_forced["step_logits"] else np.zeros((0, 1))
    )
    if dense_logits.size == 0:
        max_abs_logit_drift = 0.0
        max_rel_logit_drift = 0.0
    else:
        logit_delta = np.abs(dotcache_logits - dense_logits)
        logit_denom = np.maximum(np.abs(dense_logits), 1e-8)
        max_abs_logit_drift = float(np.max(logit_delta))
        max_rel_logit_drift = float(np.max(logit_delta / logit_denom))

    return {
        "decode_steps": max(0, decode_steps),
        "replay_context_max_abs_error": replay_context_max_abs,
        "replay_context_max_rel_error": replay_context_max_rel,
        "teacher_forced_logit_max_abs_error": max_abs_logit_drift,
        "teacher_forced_logit_max_rel_error": max_rel_logit_drift,
    }


def run_llama_generation_harness(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    max_new_tokens: int = 8,
    tokenizer=None,
    profile: bool = False,
) -> dict[str, Any]:
    _require_transformers()
    if prompt is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when prompt text is provided")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)

    dense_result = _run_dense_greedy_decode(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        capture=False,
    )

    prefill_trace = ExecutionTrace(capture_timings=profile)
    prefill_cuda_baseline = _begin_cuda_memory_region(input_ids.device) if profile else None
    _, prefill_ingest_ms = _timed_call(
        lambda: adapter.load_prefill_cache_tensors(dense_result["prefill_layers"], trace=prefill_trace)
        if dense_result["prefill_layers"] and torch.is_tensor(dense_result["prefill_layers"][0][0])
        else adapter.load_prefill_cache_arrays(dense_result["prefill_layers"], trace=prefill_trace),
        device=input_ids.device,
    )
    prefill_cuda_stats = _end_cuda_memory_region(input_ids.device, prefill_cuda_baseline) if profile else {}

    if max_new_tokens <= 1:
        generated_ids = dense_result["generated_ids"]
        decode_ms_per_step = 0.0
        append_ms_per_step = 0.0
        decode_trace = ExecutionTrace(capture_timings=profile)
        step_count = 0
        append_runtime_ms_per_step = 0.0
        decode_runtime_ms_per_step = 0.0
        dotcache_profile = adapter.runtime_profile_summary(model_forward_ms_total=0.0) if profile else None
        dotcache_cuda_stats: dict[str, int] = {}
    else:
        dotcache_cuda_baseline = _begin_cuda_memory_region(input_ids.device) if profile else None
        dotcache_result = _run_dotcache_greedy_decode(
            model,
            adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefill_layers=dense_result["prefill_layers"],
            first_generated_token=torch.as_tensor([[dense_result["generated_ids"][0]]], dtype=torch.long, device=input_ids.device),
            max_new_tokens=max_new_tokens,
            profile_backend=profile,
        )
        step_count = int(dotcache_result["step_count"])
        decode_trace = dotcache_result["trace"]
        generated_ids = dotcache_result["generated_ids"]
        decode_ms_per_step = dotcache_result["decode_ms_total"] / max(step_count, 1)
        append_runtime_ms_per_step = dotcache_result["append_runtime_ms_total"] / max(step_count, 1)
        decode_runtime_ms_per_step = dotcache_result["decode_runtime_ms_total"] / max(step_count, 1)
        append_ms_per_step = append_runtime_ms_per_step
        dotcache_profile = adapter.runtime_profile_summary(model_forward_ms_total=float(dotcache_result["decode_ms_total"])) if profile else None
        dotcache_cuda_stats = _end_cuda_memory_region(input_ids.device, dotcache_cuda_baseline) if profile else {}

    dense_generated_ids = dense_result["generated_ids"]
    dense_step_count = max(max_new_tokens - 1, 0)
    dense_decode_ms_per_step = float(dense_result["dense_decode_ms_total"] / max(dense_step_count, 1)) if dense_step_count > 0 else 0.0
    dense_prefill_kv_cache_bytes = _prefill_layer_nbytes(dense_result["prefill_layers"])
    dense_final_kv_cache_bytes = _dense_kv_bytes_after_decode(
        dense_result["prefill_layers"],
        generated_token_count=len(dense_generated_ids),
    )
    agreement_prefix = sum(
        int(lhs == rhs)
        for lhs, rhs in zip(generated_ids, dense_generated_ids, strict=False)
    )
    agreement_rate = agreement_prefix / max(min(len(generated_ids), len(dense_generated_ids)), 1)

    teacher_forced = _run_dotcache_decode_inputs(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefill_layers=dense_result["prefill_layers"],
        decode_inputs=dense_result["decode_inputs"],
    )
    if dense_result["step_logits"]:
        dense_logits = np.stack(dense_result["step_logits"], axis=0)
        forced_logits = np.stack(teacher_forced["step_logits"], axis=0)
        logit_delta = np.abs(forced_logits - dense_logits)
        logit_denom = np.maximum(np.abs(dense_logits), 1e-8)
        max_abs_logit_drift = float(np.max(logit_delta))
        max_rel_logit_drift = float(np.max(logit_delta / logit_denom))
    else:
        max_abs_logit_drift = 0.0
        max_rel_logit_drift = 0.0

    resident_byte_summary = adapter.model_kv_cache.resident_byte_summary()
    result: dict[str, Any] = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": max(max_new_tokens - 1, 0),
        "prefill_ms": float(dense_result["prefill_ms"]),
        "dense_decode_ms_per_step": dense_decode_ms_per_step,
        "dense_prefill_kv_cache_bytes": dense_prefill_kv_cache_bytes,
        "dense_final_kv_cache_bytes": dense_final_kv_cache_bytes,
        "dense_generated_ids": dense_generated_ids,
        "dotcache_generated_ids": generated_ids,
        "greedy_token_agreement_rate": float(agreement_rate),
        "prefill_cache_ingest_host_to_device_bytes": prefill_trace.host_to_device_bytes,
        "prefill_cache_ingest_ms": float(prefill_ingest_ms),
        "decode_ms_per_step": float(decode_ms_per_step),
        "append_ms_per_step": float(append_ms_per_step),
        "append_runtime_ms_per_step": float(append_runtime_ms_per_step),
        "decode_runtime_ms_per_step": float(decode_runtime_ms_per_step),
        "resident_bytes": int(resident_byte_summary["resident_bytes"]),
        "kv_resident_bytes": int(resident_byte_summary["kv_resident_bytes"]),
        "prepared_page_cache_resident_bytes": int(resident_byte_summary["prepared_page_cache_resident_bytes"]),
        "direct_page_resident_bytes": int(resident_byte_summary["direct_page_resident_bytes"]),
        "tail_resident_bytes": int(resident_byte_summary["tail_resident_bytes"]),
        "prepared_chunk_cache_budget_bytes": int(resident_byte_summary["prepared_chunk_cache_budget_bytes"]),
        "prepared_chunk_resident_bytes": int(resident_byte_summary["prepared_chunk_resident_bytes"]),
        "dotcache_vs_dense_kv_bytes_ratio": float(
            resident_byte_summary["kv_resident_bytes"] / max(dense_final_kv_cache_bytes, 1)
        ),
        "dotcache_vs_dense_total_resident_bytes_ratio": float(
            resident_byte_summary["resident_bytes"] / max(dense_final_kv_cache_bytes, 1)
        ),
        "dotcache_vs_dense_decode_speedup": float(dense_decode_ms_per_step / max(decode_ms_per_step, 1e-8))
        if decode_ms_per_step > 0.0
        else 0.0,
        "decode_host_to_device_bytes_per_step": decode_trace.host_to_device_bytes / max(step_count, 1),
        "prefill_prepare_ms": float(prefill_trace.prepare_ms_total),
        "decode_prepare_ms_per_step": float(decode_trace.prepare_ms_total / max(step_count, 1)),
        "decode_score_ms_per_step": float(decode_trace.score_ms_total / max(step_count, 1)),
        "decode_softmax_ms_per_step": float(decode_trace.softmax_ms_total / max(step_count, 1)),
        "decode_mix_ms_per_step": float(decode_trace.mix_ms_total / max(step_count, 1)),
        "decode_unpack_ms_per_step": float(decode_trace.unpack_ms_total / max(step_count, 1)),
        "decode_fwht_ms_per_step": float(decode_trace.fwht_ms_total / max(step_count, 1)),
        "teacher_forced_logit_max_abs_error": max_abs_logit_drift,
        "teacher_forced_logit_max_rel_error": max_rel_logit_drift,
    }
    result.update(adapter.model_kv_cache.page_mode_summary())
    if tokenizer is not None:
        result["dense_text"] = tokenizer.decode(dense_generated_ids, skip_special_tokens=True)
        result["dotcache_text"] = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if profile:
        result["profile"] = {
            "device_type": input_ids.device.type,
            "prefill_cache_ingest": {
                "ms_total": float(prefill_ingest_ms),
                "host_to_device_bytes": int(prefill_trace.host_to_device_bytes),
                "trace": prefill_trace.to_dict(),
                **prefill_cuda_stats,
            },
            "dotcache_decode": {
                **({} if dotcache_profile is None else dotcache_profile),
                "step_count": int(step_count),
                "host_to_device_bytes_total": int(decode_trace.host_to_device_bytes),
                "host_to_device_bytes_per_step": float(decode_trace.host_to_device_bytes / max(step_count, 1)),
                "trace": decode_trace.to_dict(),
                **dotcache_cuda_stats,
            },
        }
    return result


def run_llama_loss_harness(
    model,
    adapter: LlamaDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    tokenizer=None,
) -> dict[str, Any]:
    _require_transformers()
    if prompt is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when prompt text is provided")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)
    if prefix_length <= 0 or prefix_length >= int(input_ids.shape[1]):
        raise ValueError("prefix_length must be in [1, sequence_length)")
    available_eval_steps = int(input_ids.shape[1]) - prefix_length
    if eval_steps <= 0 or eval_steps > available_eval_steps:
        raise ValueError("eval_steps must be positive and fit inside the provided sequence after prefix_length")

    prefix_input_ids = input_ids[:, :prefix_length]
    prefix_attention_mask = attention_mask[:, :prefix_length]
    continuation_ids = input_ids[:, prefix_length : prefix_length + eval_steps]
    decode_inputs = [continuation_ids[:, index : index + 1] for index in range(max(eval_steps - 1, 0))]

    prefill_start = time.perf_counter()
    prefill_outputs, prefill_layers, _ = _prefill_prompt(model, adapter, prefix_input_ids, prefix_attention_mask)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    dense_prefill_logits = prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()
    dense_decode = _run_dense_decode_inputs(
        model,
        adapter,
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        prefill_outputs=prefill_outputs,
        decode_inputs=decode_inputs,
    )
    dotcache_decode = _run_dotcache_decode_inputs(
        model,
        adapter,
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        prefill_layers=prefill_layers,
        decode_inputs=decode_inputs,
    )

    dense_logits_list = [dense_prefill_logits, *dense_decode["step_logits"]]
    dotcache_logits_list = [dense_prefill_logits, *dotcache_decode["step_logits"]]
    dense_logits = np.concatenate(dense_logits_list, axis=0).astype(np.float32, copy=False)
    dotcache_logits = np.concatenate(dotcache_logits_list, axis=0).astype(np.float32, copy=False)
    target_tokens = continuation_ids[0, : dense_logits.shape[0]].detach().cpu().numpy().astype(np.int64, copy=False)

    def _loss_metrics(logits: np.ndarray) -> tuple[float, float, np.ndarray]:
        max_logits = np.max(logits, axis=-1, keepdims=True)
        stabilized = logits - max_logits
        log_probs = stabilized - np.log(np.sum(np.exp(stabilized), axis=-1, keepdims=True))
        token_losses = -log_probs[np.arange(target_tokens.shape[0]), target_tokens]
        mean_loss = float(np.mean(token_losses))
        perplexity = float(np.exp(min(mean_loss, 50.0)))
        predictions = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
        return mean_loss, perplexity, predictions

    dense_loss, dense_perplexity, dense_predictions = _loss_metrics(dense_logits)
    dotcache_loss, dotcache_perplexity, dotcache_predictions = _loss_metrics(dotcache_logits)
    token_agreement = float(np.mean((dense_predictions == dotcache_predictions).astype(np.float32)))
    target_agreement = float(np.mean((dotcache_predictions == target_tokens).astype(np.float32)))
    logit_delta = np.abs(dotcache_logits - dense_logits)
    logit_denom = np.maximum(np.abs(dense_logits), 1e-8)

    result: dict[str, Any] = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode["decode_ms_total"] / max(len(decode_inputs), 1)),
        "dotcache_decode_ms_per_step": float(dotcache_decode["decode_ms_total"] / max(len(decode_inputs), 1)),
        "dotcache_append_runtime_ms_per_step": float(dotcache_decode["append_runtime_ms_total"] / max(len(decode_inputs), 1)),
        "dotcache_decode_runtime_ms_per_step": float(dotcache_decode["decode_runtime_ms_total"] / max(len(decode_inputs), 1)),
        "dense_teacher_forced_loss": dense_loss,
        "dense_teacher_forced_perplexity": dense_perplexity,
        "dotcache_teacher_forced_loss": dotcache_loss,
        "dotcache_teacher_forced_perplexity": dotcache_perplexity,
        "teacher_forced_loss_delta": float(dotcache_loss - dense_loss),
        "teacher_forced_perplexity_ratio": float(dotcache_perplexity / max(dense_perplexity, 1e-8)),
        "teacher_forced_token_agreement_rate": token_agreement,
        "teacher_forced_target_match_rate": target_agreement,
        "teacher_forced_logit_max_abs_error": float(np.max(logit_delta)),
        "teacher_forced_logit_max_rel_error": float(np.max(logit_delta / logit_denom)),
        "prefill_cache_ingest_host_to_device_bytes": dotcache_decode["trace"].host_to_device_bytes,
    }
    result.update(adapter.model_kv_cache.page_mode_summary())
    return result
