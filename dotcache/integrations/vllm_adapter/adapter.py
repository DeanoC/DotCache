from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...page_cache import PreparedPageCache
from ...tracing import ExecutionTrace
from .block_cache import VllmPagedKVCache
from .compat import require_supported_vllm_version
from .config import VllmAdapterConfig, VllmAdapterMode

try:  # pragma: no cover - torch is optional for the base repo
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for the vLLM adapter path")


def _torch_backend_matches_device(backend: str, device_type: str) -> bool:
    if device_type == "cuda":
        return backend in {"torch_cuda", "auto"}
    return False


def _looks_like_vllm_llama_attention(module: Any) -> bool:
    required = ("qkv_proj", "o_proj", "rotary_emb", "q_size", "kv_size", "num_heads", "num_kv_heads", "head_dim", "scaling")
    return all(hasattr(module, name) for name in required)


def _looks_like_vllm_llama_model(model: Any) -> bool:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return False
    layers = getattr(model.model, "layers")
    if not layers:
        return False
    first_layer = layers[0]
    return hasattr(first_layer, "self_attn") and _looks_like_vllm_llama_attention(first_layer.self_attn)


def _extract_qkv(base_attention: Any, hidden_states) -> Any:
    projected = base_attention.qkv_proj(hidden_states)
    if isinstance(projected, tuple):
        projected = projected[0]
    return projected


def _split_qkv(base_attention: Any, qkv) -> tuple[Any, Any, Any]:
    return qkv.split([int(base_attention.q_size), int(base_attention.kv_size), int(base_attention.kv_size)], dim=-1)


def _apply_rope(base_attention: Any, positions, q, k) -> tuple[Any, Any]:
    rotated = base_attention.rotary_emb(positions, q, k)
    if isinstance(rotated, tuple) and len(rotated) == 2:
        return rotated
    raise ValueError("vLLM rotary_emb must return (query, key)")


def _project_dotcache_output(base_attention: Any, context_tensor) -> Any:
    projected = base_attention.o_proj(context_tensor)
    if isinstance(projected, tuple):
        return projected[0]
    return projected


def _resolve_token_positions(positions) -> np.ndarray:
    if torch is not None and torch.is_tensor(positions):
        return positions.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(positions, dtype=np.int64).reshape(-1)


class DotCacheVllmLlamaAttention(nn.Module):  # type: ignore[misc]
    def __init__(self, base_attention: Any, adapter: "VllmDotCacheModelAdapter", *, layer_id: int) -> None:
        _require_torch()
        super().__init__()
        self.base_attention = base_attention
        self.adapter = adapter
        self.layer_id = int(layer_id)

    def forward(self, positions, hidden_states):
        token_count = int(hidden_states.shape[0])
        active_trace = self.adapter.active_trace if self.adapter.active_trace is not None else self.adapter.runtime_trace
        if self.adapter.mode == "dense":
            return self.base_attention(positions, hidden_states)

        qkv = _extract_qkv(self.base_attention, hidden_states)
        query_states, key_states, value_states = _split_qkv(self.base_attention, qkv)
        query_states, key_states = _apply_rope(self.base_attention, positions, query_states, key_states)
        query_rows = query_states.view(token_count, int(self.base_attention.num_heads), int(self.base_attention.head_dim))
        key_rows = key_states.view(token_count, int(self.base_attention.num_kv_heads), int(self.base_attention.head_dim))
        value_rows = value_states.view(token_count, int(self.base_attention.num_kv_heads), int(self.base_attention.head_dim))
        token_positions = _resolve_token_positions(positions)

        if token_count != 1:
            dense_output = self.base_attention(positions, hidden_states)
            encode_start = time.perf_counter()
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type):
                self.adapter.block_cache.append_tokens_torch(
                    self.layer_id,
                    key_rows.detach().to(dtype=torch.float32),
                    value_rows.detach().to(dtype=torch.float32),
                    positions,
                    trace=active_trace,
                )
            else:
                self.adapter.block_cache.append_step(
                    self.layer_id,
                    key_rows.detach().permute(1, 0, 2).to(dtype=torch.float32).cpu().numpy(),
                    value_rows.detach().permute(1, 0, 2).to(dtype=torch.float32).cpu().numpy(),
                    int(token_positions[0]),
                    trace=active_trace,
                )
            self.adapter.prefill_block_encode_ms_total += (time.perf_counter() - encode_start) * 1000.0
            return dense_output

        dense_output = self.base_attention(positions, hidden_states)
        append_start = time.perf_counter()
        if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type):
            self.adapter.block_cache.append_step_torch(
                self.layer_id,
                key_rows.detach().permute(1, 0, 2).to(dtype=torch.float32),
                value_rows.detach().permute(1, 0, 2).to(dtype=torch.float32),
                int(token_positions[0]),
                trace=active_trace,
            )
        else:
            self.adapter.block_cache.append_step(
                self.layer_id,
                key_rows.detach().permute(1, 0, 2).to(dtype=torch.float32).cpu().numpy(),
                value_rows.detach().permute(1, 0, 2).to(dtype=torch.float32).cpu().numpy(),
                int(token_positions[0]),
                trace=active_trace,
            )
        self.adapter.append_runtime_ms_total += (time.perf_counter() - append_start) * 1000.0

        decode_start = time.perf_counter()
        if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type):
            context_states = self.adapter.block_cache.decode_layer_torch(
                self.layer_id,
                query_rows[0].detach().to(dtype=torch.float32),
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=active_trace,
            )
        else:
            context_states = self.adapter.block_cache.decode_layer(
                self.layer_id,
                query_rows[0].detach().to(dtype=torch.float32).cpu().numpy(),
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=active_trace,
            )
        self.adapter.decode_runtime_ms_total += (time.perf_counter() - decode_start) * 1000.0

        if not torch.is_tensor(context_states):
            context_states = torch.as_tensor(context_states, dtype=torch.float32, device=hidden_states.device)
        dotcache_output = _project_dotcache_output(
            self.base_attention,
            context_states.to(dtype=hidden_states.dtype, device=hidden_states.device).reshape(1, -1),
        )

        self.adapter.record_last_dotcache_output(self.layer_id, dotcache_output)
        if self.adapter.mode == "dotcache_shadow":
            self.adapter.record_shadow_output(dense_output, dotcache_output)
            return dense_output
        return dotcache_output


@dataclass
class VllmDotCacheModelAdapter:
    model: Any
    adapter_config: VllmAdapterConfig
    backend: str = "torch_cuda"
    cache: PreparedPageCache | None = None

    def __post_init__(self) -> None:
        self.cache = self.cache if self.cache is not None else PreparedPageCache()
        self.block_cache = VllmPagedKVCache(
            config=self.adapter_config.dotcache_config,
            num_hidden_layers=self.model.config.num_hidden_layers,
            num_attention_heads=self.model.config.num_attention_heads,
            num_key_value_heads=getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads),
            block_size=self.adapter_config.block_size,
            backend=self.backend,
            cache=self.cache,
        )
        self.q_head_to_kv_head = self.block_cache.default_q_head_to_kv_head.copy()
        self.mode: VllmAdapterMode = self.adapter_config.mode
        self.active_trace: ExecutionTrace | None = None
        self.runtime_trace = ExecutionTrace()
        self.prefill_block_encode_ms_total = 0.0
        self.append_runtime_ms_total = 0.0
        self.decode_runtime_ms_total = 0.0
        self.shadow_output_max_abs_error = 0.0
        self.shadow_output_max_rel_error = 0.0
        self._last_dotcache_outputs: dict[int, Any] = {}
        self._wrappers: list[DotCacheVllmLlamaAttention] = []
        self._install_wrappers()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def resident_bytes(self) -> int:
        return self.block_cache.resident_bytes

    def _install_wrappers(self) -> None:
        for layer_id, layer in enumerate(self.model.model.layers[: self.model.config.num_hidden_layers]):
            wrapper = DotCacheVllmLlamaAttention(layer.self_attn, self, layer_id=layer_id)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def set_mode(self, mode: VllmAdapterMode) -> None:
        self.mode = mode

    def clear(self) -> None:
        self.block_cache.clear()
        self.active_trace = None
        self._last_dotcache_outputs.clear()
        self.reset_runtime_metrics()

    def reset_runtime_metrics(self) -> None:
        self.runtime_trace = ExecutionTrace()
        self.prefill_block_encode_ms_total = 0.0
        self.append_runtime_ms_total = 0.0
        self.decode_runtime_ms_total = 0.0
        self.shadow_output_max_abs_error = 0.0
        self.shadow_output_max_rel_error = 0.0

    def record_shadow_output(self, dense_output, dotcache_output) -> None:
        dense = dense_output.detach().to(dtype=torch.float32).cpu().numpy()
        dotcache = dotcache_output.detach().to(dtype=torch.float32).cpu().numpy()
        delta = np.abs(dotcache - dense)
        denom = np.maximum(np.abs(dense), 1e-8)
        self.shadow_output_max_abs_error = max(self.shadow_output_max_abs_error, float(np.max(delta)))
        self.shadow_output_max_rel_error = max(self.shadow_output_max_rel_error, float(np.max(delta / denom)))

    def record_last_dotcache_output(self, layer_id: int, output) -> None:
        self._last_dotcache_outputs[int(layer_id)] = output.detach().clone()

    def last_dotcache_output(self, layer_id: int):
        return self._last_dotcache_outputs[int(layer_id)]


def install_dotcache_on_vllm_model(
    model: Any,
    dotcache_config,
    *,
    block_size: int,
    backend: str = "torch_cuda",
    mode: VllmAdapterMode = "dense",
    cache: PreparedPageCache | None = None,
) -> VllmDotCacheModelAdapter:
    if not _looks_like_vllm_llama_model(model):
        raise ValueError("target model is not a supported vLLM Llama-family executor model")
    adapter_config = VllmAdapterConfig(
        dotcache_config=dotcache_config,
        block_size=block_size,
        mode=mode,
        model_family="llama",
    )
    return VllmDotCacheModelAdapter(model=model, adapter_config=adapter_config, backend=backend, cache=cache)


def _infer_block_size_from_target(target: Any) -> int | None:
    for attr_name in ("cache_config", "vllm_config"):
        attr = getattr(target, attr_name, None)
        if attr is not None and hasattr(attr, "block_size"):
            return int(attr.block_size)
        cache_config = getattr(attr, "cache_config", None)
        if cache_config is not None and hasattr(cache_config, "block_size"):
            return int(cache_config.block_size)
    return None


def _search_for_model(target: Any, *, max_depth: int = 6, visited: set[int] | None = None) -> Any | None:
    if visited is None:
        visited = set()
    if target is None:
        return None
    target_id = id(target)
    if target_id in visited or max_depth < 0:
        return None
    visited.add(target_id)

    if _looks_like_vllm_llama_model(target):
        return target

    for attr_name in (
        "model",
        "runner",
        "model_runner",
        "driver_worker",
        "worker",
        "model_executor",
        "llm_engine",
        "engine",
        "engine_core",
        "executor",
    ):
        child = getattr(target, attr_name, None)
        found = _search_for_model(child, max_depth=max_depth - 1, visited=visited)
        if found is not None:
            return found
    return None


def install_dotcache_on_vllm_runtime(
    target: Any,
    dotcache_config,
    *,
    block_size: int | None = None,
    backend: str = "torch_cuda",
    mode: VllmAdapterMode = "dense",
    cache: PreparedPageCache | None = None,
) -> VllmDotCacheModelAdapter:
    require_supported_vllm_version()
    model = _search_for_model(target)
    if model is None:
        raise RuntimeError("could not locate a supported vLLM Llama-family executor model inside the target runtime")
    resolved_block_size = int(block_size) if block_size is not None else _infer_block_size_from_target(target)
    if resolved_block_size is None:
        raise RuntimeError("could not infer vLLM block_size from the target runtime; pass block_size explicitly")
    return install_dotcache_on_vllm_model(
        model,
        dotcache_config,
        block_size=resolved_block_size,
        backend=backend,
        mode=mode,
        cache=cache,
    )
