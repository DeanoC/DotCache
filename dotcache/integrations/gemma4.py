from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from ..config import DotCacheConfig
from ..model_kv_cache import ModelPagedKVCache
from ..page_cache import PreparedPageCache
from .llama import (
    LlamaDotCacheHarness,
    LlamaDotCacheModelAdapter,
    LlamaReplayRecord,
    _default_model_device,
    _ensure_attention_mask,
    _normalize_input_ids,
    _require_transformers,
    _run_dense_greedy_decode,
    _run_dotcache_decode_inputs,
    _timed_call,
    _torch_backend_matches_device,
    resolve_hf_auth_kwargs,
    run_llama_generation_harness,
    run_llama_loss_harness,
    transformers_available,
)

if transformers_available():
    import torch
    import torch.nn as nn
    import transformers.models.gemma4.modeling_gemma4 as gemma4_mod
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    nn = object  # type: ignore[assignment]
    gemma4_mod = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _gemma4_text_config(model_or_config: Any) -> Any:
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def _gemma4_text_model(model_or_config: Any) -> Any:
    model = getattr(model_or_config, "model", model_or_config)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return language_model
    return model


def _gemma4_layer_types(model_or_config: Any) -> tuple[str, ...]:
    config = _gemma4_text_config(model_or_config)
    raw_types = tuple(str(layer_type) for layer_type in getattr(config, "layer_types", ()))
    if raw_types:
        return raw_types
    return tuple("sliding_attention" for _ in range(int(config.num_hidden_layers)))


def _gemma4_layer_head_dim(model_or_config: Any, layer_idx: int) -> int:
    config = _gemma4_text_config(model_or_config)
    layer_types = _gemma4_layer_types(config)
    layer_type = layer_types[int(layer_idx)] if layer_types else "sliding_attention"
    global_head_dim = int(getattr(config, "global_head_dim", int(config.head_dim)))
    return global_head_dim if layer_type == "full_attention" else int(config.head_dim)


def gemma4_full_attention_source_layers(model_or_config: Any) -> tuple[int, ...]:
    layer_types = _gemma4_layer_types(model_or_config)
    unique_layer_count = _gemma4_unique_kv_layer_count(model_or_config)
    return tuple(
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types[:unique_layer_count])
        if layer_type == "full_attention"
    )


def gemma4_sliding_attention_source_layers(model_or_config: Any) -> tuple[int, ...]:
    layer_types = _gemma4_layer_types(model_or_config)
    unique_layer_count = _gemma4_unique_kv_layer_count(model_or_config)
    return tuple(
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types[:unique_layer_count])
        if layer_type == "sliding_attention"
    )


def _gemma4_first_shared_layer_idx(model_or_config: Any) -> int:
    config = _gemma4_text_config(model_or_config)
    shared_count = int(getattr(config, "num_kv_shared_layers", 0) or 0)
    return int(config.num_hidden_layers) - shared_count


def _gemma4_unique_kv_layer_count(model_or_config: Any) -> int:
    return _gemma4_first_shared_layer_idx(model_or_config)


def _gemma4_shared_source_layer_index(model_or_config: Any, layer_idx: int) -> int | None:
    config = _gemma4_text_config(model_or_config)
    first_shared_layer_idx = _gemma4_first_shared_layer_idx(config)
    if int(layer_idx) < first_shared_layer_idx:
        return None
    layer_types = _gemma4_layer_types(config)
    prev_layers = layer_types[:first_shared_layer_idx]
    if not prev_layers:
        return None
    layer_type = layer_types[int(layer_idx)]
    return len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)


def gemma4_text_dotcache_supported(model_or_config: Any) -> bool:
    config = _gemma4_text_config(model_or_config)
    if int(config.num_hidden_layers) <= 0:
        return False
    if int(config.num_attention_heads) <= 0 or int(config.num_key_value_heads) <= 0:
        return False
    layer_types = _gemma4_layer_types(config)
    if layer_types and len(layer_types) != int(config.num_hidden_layers):
        return False
    return all(_gemma4_layer_head_dim(config, layer_idx) > 0 for layer_idx in range(int(config.num_hidden_layers)))


def gemma4_text_tuned_profile_for_workload(*, prompt_length: int, decode_budget: int) -> str:
    prompt_tokens = int(prompt_length)
    decode_tokens = int(decode_budget)
    if prompt_tokens <= 0:
        raise ValueError("prompt_length must be positive")
    if decode_tokens <= 0:
        raise ValueError("decode_budget must be positive")
    if prompt_tokens >= 4096:
        if decode_tokens >= 32:
            return "balanced_layer0"
        if decode_tokens >= 24:
            return "balanced_layer0_8"
        return "balanced"
    if prompt_tokens >= 2048:
        return "balanced_layer0_8"
    if prompt_tokens >= 1024:
        return "balanced" if decode_tokens >= 24 else "balanced_layer0_8"
    return "balanced_layer0"


def gemma4_text_tuned_knobs_for_workload(*, prompt_length: int, decode_budget: int) -> tuple[int, int, int]:
    prompt_tokens = int(prompt_length)
    decode_tokens = int(decode_budget)
    if prompt_tokens <= 0:
        raise ValueError("prompt_length must be positive")
    if decode_tokens <= 0:
        raise ValueError("decode_budget must be positive")
    if prompt_tokens >= 4096 and decode_tokens >= 24:
        return 4, 16, 8
    if prompt_tokens >= 2048 and decode_tokens >= 24:
        return 4, 16, 4
    return 4, 32, 4


def gemma4_text_tuned_value_layers_for_workload(*, prompt_length: int, decode_budget: int) -> tuple[int, ...] | None:
    prompt_tokens = int(prompt_length)
    decode_tokens = int(decode_budget)
    if prompt_tokens <= 0:
        raise ValueError("prompt_length must be positive")
    if decode_tokens <= 0:
        raise ValueError("decode_budget must be positive")
    if prompt_tokens >= 4096 and decode_tokens >= 24:
        return (0, 4, 8, 9, 14)
    return None


def gemma4_text_recommended_dotcache_config(
    model_or_config: Any,
    *,
    bits_k: int = 4,
    bits_v: int = 4,
    tokens_per_page: int = 4,
    group_size: int = 32,
    profile: str = "balanced",
    prompt_length: int | None = None,
    decode_budget: int | None = None,
    adaptive_knobs: bool = False,
    adaptive_values: bool = False,
    extra_exact_key_layers: Sequence[int] = (),
    extra_exact_value_layers: Sequence[int] = (),
) -> DotCacheConfig:
    normalized_profile = str(profile).strip().lower()
    if normalized_profile == "adaptive":
        if prompt_length is None or decode_budget is None:
            raise ValueError("adaptive Gemma 4 tuning requires prompt_length and decode_budget")
        if adaptive_knobs:
            bits_k, group_size, tokens_per_page = gemma4_text_tuned_knobs_for_workload(
                prompt_length=int(prompt_length),
                decode_budget=int(decode_budget),
            )
        if adaptive_values:
            tuned_value_layers = gemma4_text_tuned_value_layers_for_workload(
                prompt_length=int(prompt_length),
                decode_budget=int(decode_budget),
            )
            if tuned_value_layers is not None:
                extra_exact_value_layers = tuple(extra_exact_value_layers) + tuple(tuned_value_layers)
        normalized_profile = gemma4_text_tuned_profile_for_workload(
            prompt_length=int(prompt_length),
            decode_budget=int(decode_budget),
        )
    if normalized_profile == "balanced_layer0":
        normalized_profile = "balanced"
        extra_exact_key_layers = tuple(extra_exact_key_layers) + (0,)
    elif normalized_profile == "balanced_layer0_8":
        normalized_profile = "balanced"
        extra_exact_key_layers = tuple(extra_exact_key_layers) + (0, 8)
    base = DotCacheConfig(
        head_dim=int(_gemma4_text_config(model_or_config).head_dim),
        group_size=int(group_size),
        bits_k=int(bits_k),
        bits_v=int(bits_v),
        tokens_per_page=int(tokens_per_page),
    )
    full_attention_layers = gemma4_full_attention_source_layers(model_or_config)
    extra_key_layers = tuple(sorted({int(layer_idx) for layer_idx in extra_exact_key_layers}))
    extra_value_layers = tuple(sorted({int(layer_idx) for layer_idx in extra_exact_value_layers}))
    if normalized_profile == "aggressive":
        return replace(
            base,
            key_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in sorted(set(full_attention_layers) | set(extra_key_layers))),
            value_mode_overrides=tuple(
                f"layer:{layer_idx}=M3" for layer_idx in sorted(set(full_attention_layers) | set(extra_value_layers))
            ),
        )
    if normalized_profile == "value_exact":
        return replace(
            base,
            default_mode_v="M3",
            key_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in extra_key_layers),
            value_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in extra_value_layers),
        )
    if normalized_profile == "balanced":
        balanced = replace(
            base,
            default_mode_v="M3" if not adaptive_values else "M0",
            key_mode_overrides=tuple(
                f"layer:{layer_idx}=M3" for layer_idx in sorted(set(full_attention_layers) | set(extra_key_layers))
            ),
            value_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in extra_value_layers),
        )
        if not adaptive_values:
            return balanced
        if extra_value_layers:
            return balanced
        return replace(balanced, default_mode_v="M3")
    if normalized_profile == "exact":
        return replace(
            base,
            default_mode_k="M3",
            default_mode_v="M3",
            key_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in extra_key_layers),
            value_mode_overrides=tuple(f"layer:{layer_idx}=M3" for layer_idx in extra_value_layers),
        )
    raise ValueError(
        "profile must be one of aggressive, value_exact, balanced, balanced_layer0, balanced_layer0_8, adaptive, exact"
    )


def _merge_summary_dicts(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(lhs)
    for key, value in rhs.items():
        if key not in merged:
            merged[key] = value
            continue
        current = merged[key]
        if isinstance(current, dict) and isinstance(value, dict):
            nested: dict[str, Any] = dict(current)
            for nested_key, nested_value in value.items():
                if nested_key not in nested:
                    nested[nested_key] = nested_value
                    continue
                existing_value = nested[nested_key]
                if isinstance(existing_value, bool) and isinstance(nested_value, bool):
                    nested[nested_key] = bool(existing_value or nested_value)
                elif isinstance(existing_value, int) and isinstance(nested_value, int):
                    nested[nested_key] = int(existing_value) + int(nested_value)
                elif isinstance(existing_value, float) and isinstance(nested_value, float):
                    nested[nested_key] = float(existing_value) + float(nested_value)
            merged[key] = nested
            continue
        if isinstance(current, bool) and isinstance(value, bool):
            merged[key] = bool(current or value)
        elif isinstance(current, int) and isinstance(value, int):
            merged[key] = int(current) + int(value)
        elif isinstance(current, float) and isinstance(value, float):
            if key.endswith("_max") or key.endswith("_p95"):
                merged[key] = max(float(current), float(value))
            else:
                merged[key] = float(current) + float(value)
    return merged


class _Gemma4AggregateModelPagedKVCache:
    def __init__(
        self,
        *,
        model,
        config: DotCacheConfig,
        backend: str = "auto",
    ) -> None:
        self.model = model
        self.config = config
        self.backend = backend
        self.num_hidden_layers = int(model.config.num_hidden_layers)
        self.num_attention_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self._source_layer_ids = tuple(range(_gemma4_unique_kv_layer_count(model)))
        self._layer_source_map = {
            layer_idx: (
                layer_idx
                if layer_idx < _gemma4_first_shared_layer_idx(model)
                else int(_gemma4_shared_source_layer_index(model, layer_idx))
            )
            for layer_idx in range(self.num_hidden_layers)
        }
        self._layer_head_dims = {
            layer_idx: _gemma4_layer_head_dim(model, layer_idx)
            for layer_idx in range(self.num_hidden_layers)
        }
        self._source_caches = {
            source_layer_id: ModelPagedKVCache(
                config=replace(config, head_dim=self._layer_head_dims[source_layer_id]),
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                backend=backend,
                cache=PreparedPageCache(),
            )
            for source_layer_id in self._source_layer_ids
        }
        first_cache = self._source_caches[self._source_layer_ids[0]]
        self.default_q_head_to_kv_head = first_cache.default_q_head_to_kv_head.copy()
        self._torch_device_type = first_cache._torch_device_type

    @property
    def resident_bytes(self) -> int:
        return self.resident_byte_summary()["resident_bytes"]

    def layer_head_dim(self, layer_id: int) -> int:
        return int(self._layer_head_dims[int(layer_id)])

    def _resolve_cache(self, layer_id: int) -> tuple[ModelPagedKVCache, int]:
        public_layer_id = int(layer_id)
        if public_layer_id < 0 or public_layer_id >= self.num_hidden_layers:
            raise ValueError(f"layer_id {layer_id} is out of range")
        source_layer_id = int(self._layer_source_map[public_layer_id])
        return self._source_caches[source_layer_id], source_layer_id

    def clear(self) -> None:
        for cache in self._source_caches.values():
            cache.clear()

    def ingest_prefill_cache(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        context_length: int | None = None,
        trace=None,
    ) -> None:
        cache, source_layer_id = self._resolve_cache(layer_id)
        cache.ingest_prefill_cache(source_layer_id, keys, values, context_length=context_length, trace=trace)

    def ingest_prefill_cache_torch(
        self,
        layer_id: int,
        keys,
        values,
        *,
        context_length: int | None = None,
        trace=None,
    ) -> None:
        cache, source_layer_id = self._resolve_cache(layer_id)
        cache.ingest_prefill_cache_torch(source_layer_id, keys, values, context_length=context_length, trace=trace)

    def prepare_static_pages(self, *, trace=None) -> None:
        for cache in self._source_caches.values():
            cache.prepare_static_pages(trace=trace)

    def append_step(self, layer_id: int, key_step: np.ndarray, value_step: np.ndarray, token_index: int, *, trace=None) -> None:
        cache, source_layer_id = self._resolve_cache(layer_id)
        cache.append_step(source_layer_id, key_step, value_step, token_index, trace=trace)

    def append_step_torch(self, layer_id: int, key_step, value_step, token_index: int, *, trace=None) -> None:
        cache, source_layer_id = self._resolve_cache(layer_id)
        cache.append_step_torch(source_layer_id, key_step, value_step, token_index, trace=trace)

    def decode_layer(
        self,
        layer_id: int,
        query_step: np.ndarray,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace=None,
    ) -> np.ndarray:
        cache, source_layer_id = self._resolve_cache(layer_id)
        return cache.decode_layer(
            source_layer_id,
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
        prefer_grouped_batching: bool = True,
        trace=None,
    ):
        cache, source_layer_id = self._resolve_cache(layer_id)
        return cache.decode_layer_torch(
            source_layer_id,
            query_step,
            q_head_to_kv_head,
            query_scale=query_scale,
            prefer_grouped_batching=prefer_grouped_batching,
            trace=trace,
        )

    def layer_sequence_length(self, layer_id: int) -> int:
        cache, source_layer_id = self._resolve_cache(layer_id)
        return cache.layer_sequence_length(source_layer_id)

    def resident_byte_summary(self) -> dict[str, int]:
        kv_resident_bytes = 0
        prepared_page_cache_resident_bytes = 0
        direct_page_resident_bytes = 0
        tail_resident_bytes = 0
        prepared_chunk_cache_budget_bytes = 0
        prepared_chunk_resident_bytes = 0
        for cache in self._source_caches.values():
            summary = cache.resident_byte_summary()
            kv_resident_bytes += int(summary["kv_resident_bytes"])
            prepared_page_cache_resident_bytes += int(summary["prepared_page_cache_resident_bytes"])
            direct_page_resident_bytes += int(summary["direct_page_resident_bytes"])
            tail_resident_bytes += int(summary["tail_resident_bytes"])
            prepared_chunk_cache_budget_bytes = max(
                prepared_chunk_cache_budget_bytes,
                int(summary["prepared_chunk_cache_budget_bytes"]),
            )
            prepared_chunk_resident_bytes = max(
                prepared_chunk_resident_bytes,
                int(summary["prepared_chunk_resident_bytes"]),
            )
        return {
            "prepared_page_cache_resident_bytes": int(prepared_page_cache_resident_bytes),
            "direct_page_resident_bytes": int(direct_page_resident_bytes),
            "tail_resident_bytes": int(tail_resident_bytes),
            "kv_resident_bytes": int(kv_resident_bytes),
            "prepared_chunk_cache_budget_bytes": int(prepared_chunk_cache_budget_bytes),
            "prepared_chunk_resident_bytes": int(prepared_chunk_resident_bytes),
            "resident_bytes": int(kv_resident_bytes + prepared_chunk_resident_bytes),
        }

    def page_mode_summary(self) -> dict[str, object]:
        summaries = [cache.page_mode_summary() for cache in self._source_caches.values()]
        if not summaries:
            return {}
        merged = dict(summaries[0])
        for summary in summaries[1:]:
            merged = _merge_summary_dicts(merged, summary)
        if "m2_prefilter_top_k" in summaries[0]:
            merged["m2_prefilter_top_k"] = summaries[0]["m2_prefilter_top_k"]
        if "m2_prefilter_min_pages" in summaries[0]:
            merged["m2_prefilter_min_pages"] = summaries[0]["m2_prefilter_min_pages"]
        return merged


class Gemma4TextModelWrapper(nn.Module):
    def __init__(self, root_model: nn.Module) -> None:
        super().__init__()
        self.root_model = root_model
        self.model = _gemma4_text_model(root_model)
        self.config = _gemma4_text_config(root_model)
        self.lm_head = getattr(root_model, "lm_head", None)

    def forward(self, *args: Any, **kwargs: Any):
        return self.root_model(*args, **kwargs)


def load_gemma4_text_only_from_pretrained(
    model_id: str,
    *,
    device: str | None = None,
    torch_dtype: str = "bfloat16",
) -> tuple[Gemma4TextModelWrapper, Any]:
    _require_transformers()
    dtype = getattr(torch, torch_dtype)
    resolved_device = _default_model_device() if device is None else device
    auth_kwargs = resolve_hf_auth_kwargs()
    root_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=False,
        **auth_kwargs,
    )
    root_model.to(resolved_device)
    root_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **auth_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return Gemma4TextModelWrapper(root_model), tokenizer


class DotCacheGemma4TextAttention(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "Gemma4TextModelAdapter") -> None:
        super().__init__()
        self.base_attention = base_attention
        self.adapter = adapter
        self.layer_idx = int(base_attention.layer_idx)
        self.config = base_attention.config
        self.cache_source_layer_id = _gemma4_shared_source_layer_index(self.config, self.layer_idx)

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

    @property
    def is_kv_shared_layer(self) -> bool:
        return self.cache_source_layer_id is not None

    def _project_query(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Gemma4 text attention path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        cos, sin = position_embeddings
        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape)
        query_states = self.base_attention.q_norm(query_states)
        query_states = gemma4_mod.apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        return query_states.transpose(1, 2)

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Gemma4 text attention path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        cos, sin = position_embeddings
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape)
        value_states = (
            self.base_attention.v_proj(hidden_states).view(hidden_shape)
            if self.base_attention.v_proj is not None
            else key_states
        )
        key_states = self.base_attention.k_norm(key_states)
        key_states = gemma4_mod.apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value_states = self.base_attention.v_norm(value_states).transpose(1, 2)
        return key_states, value_states

    def _attention_interface(self):
        if self.base_attention.config._attn_implementation == "eager":
            return gemma4_mod.eager_attention_forward
        return gemma4_mod.ALL_ATTENTION_FUNCTIONS[self.base_attention.config._attn_implementation]

    def _record_cache_source_layer_id(self) -> int | None:
        return self.cache_source_layer_id

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
        query_states = self._project_query(hidden_states, position_embeddings)
        if self.is_kv_shared_layer and past_key_values is not None:
            assert self.cache_source_layer_id is not None
            key_states, value_states = past_key_values.shared_layers[self.cache_source_layer_id]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
            fresh_key_states = key_states[:, :, -1:, :]
            fresh_value_states = value_states[:, :, -1:, :]
        else:
            key_states, value_states = self._project_kv(hidden_states, position_embeddings)
            fresh_key_states = key_states
            fresh_value_states = value_states
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                if getattr(self.base_attention, "store_full_length_kv", False):
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    past_key_values.shared_layers[self.layer_idx] = (key_states, value_states)

        attention_interface = self._attention_interface()
        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.base_attention.attention_dropout if self.training else 0.0,
            scaling=self.base_attention.scaling,
            sliding_window=self.base_attention.sliding_window,
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
                    key_states=fresh_key_states[0, :, -1, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    value_states=fresh_value_states[0, :, -1, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    context_states=attn_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    cache_source_layer_id=self._record_cache_source_layer_id(),
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
        query_states, q_ms = _timed_call(
            lambda: self._project_query(hidden_states, position_embeddings),
            device=hidden_states.device,
        )
        query_step = query_states[0, :, 0, :].detach().to(dtype=torch.float32)
        append_ms = 0.0
        cache_layer_id = self.layer_idx if self.cache_source_layer_id is None else self.cache_source_layer_id

        if self.cache_source_layer_id is None:
            (key_states, value_states), kv_ms = _timed_call(
                lambda: self._project_kv(hidden_states, position_embeddings),
                device=hidden_states.device,
            )
            self.adapter.record_layer_runtime(self.layer_idx, qkv_projection_ms=q_ms + kv_ms)
            key_step = key_states[0].detach().to(dtype=torch.float32)
            value_step = value_states[0].detach().to(dtype=torch.float32)
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
        else:
            self.adapter.record_layer_runtime(self.layer_idx, qkv_projection_ms=q_ms)
            key_step = None
            value_step = None

        context_states, decode_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.decode_layer_torch(
                cache_layer_id,
                query_step,
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=self.adapter.active_trace,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.decode_layer(
                cache_layer_id,
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
            if key_step is None or value_step is None:
                capture_key_states = np.zeros((query_step.shape[0], query_step.shape[1]), dtype=np.float32)
                capture_value_states = np.zeros((query_step.shape[0], query_step.shape[1]), dtype=np.float32)
            else:
                capture_key_states = key_step[:, 0, :].detach().cpu().numpy()
                capture_value_states = value_step[:, 0, :].detach().cpu().numpy()
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_step.detach().cpu().numpy(),
                    key_states=capture_key_states,
                    value_states=capture_value_states,
                    context_states=context_states.detach().cpu().numpy().astype(np.float32, copy=False),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    cache_source_layer_id=self._record_cache_source_layer_id(),
                )
            )
        return projected_output, None


class Gemma4TextModelAdapter(LlamaDotCacheModelAdapter):
    def __init__(
        self,
        model,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
    ) -> None:
        self.supports_uniform_head_dim_dotcache = gemma4_text_dotcache_supported(model)
        super().__init__(model, dotcache_config, backend=backend)
        self.model_kv_cache = self._build_model_kv_cache(dotcache_config, backend=self.backend)
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()
        self.clear()

    def _build_model_kv_cache(self, dotcache_config: DotCacheConfig, *, backend: str) -> _Gemma4AggregateModelPagedKVCache:
        return _Gemma4AggregateModelPagedKVCache(
            model=self.model,
            config=dotcache_config,
            backend=backend,
        )

    def _install_wrappers(self) -> None:
        for layer in self.model.model.layers[: self.model.config.num_hidden_layers]:
            wrapper = DotCacheGemma4TextAttention(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def reconfigure(self, dotcache_config: DotCacheConfig, *, backend: str | None = None) -> None:
        self.dotcache_config = dotcache_config
        if backend is not None:
            self.backend = backend
        self.model_kv_cache = self._build_model_kv_cache(dotcache_config, backend=self.backend)
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()
        self.clear()

    def load_prefill_cache_arrays(
        self,
        prefill_layers: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        context_length: int | None = None,
        trace=None,
    ) -> None:
        unique_layer_count = _gemma4_unique_kv_layer_count(self.model)
        if len(prefill_layers) not in {unique_layer_count, self.model.config.num_hidden_layers}:
            raise ValueError("prefill_layers must align with Gemma4 unique KV cache layers")
        self.model_kv_cache.clear()
        for layer_idx, (layer_keys, layer_values) in enumerate(prefill_layers[:unique_layer_count]):
            self.model_kv_cache.ingest_prefill_cache(
                layer_idx,
                layer_keys,
                layer_values,
                context_length=context_length,
                trace=trace,
            )
        self.model_kv_cache.prepare_static_pages(trace=trace)

    def load_prefill_cache_tensors(
        self,
        prefill_layers: Sequence[tuple[Any, Any]],
        *,
        context_length: int | None = None,
        trace=None,
    ) -> None:
        unique_layer_count = _gemma4_unique_kv_layer_count(self.model)
        if len(prefill_layers) not in {unique_layer_count, self.model.config.num_hidden_layers}:
            raise ValueError("prefill_layers must align with Gemma4 unique KV cache layers")
        self.model_kv_cache.clear()
        for layer_idx, (layer_keys, layer_values) in enumerate(prefill_layers[:unique_layer_count]):
            self.model_kv_cache.ingest_prefill_cache_torch(
                layer_idx,
                layer_keys,
                layer_values,
                context_length=context_length,
                trace=trace,
            )
        self.model_kv_cache.prepare_static_pages(trace=trace)


@dataclass(slots=True)
class Gemma4TextHarness(LlamaDotCacheHarness):
    adapter: Gemma4TextModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
    ) -> "Gemma4TextHarness":
        model, tokenizer = load_gemma4_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Gemma4TextModelAdapter(model, dotcache_config, backend=backend)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def run_replay(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
    ) -> dict[str, float | int]:
        return run_gemma4_text_replay_harness(
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
        return run_gemma4_text_generation_harness(
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
        return run_gemma4_text_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            tokenizer=self.tokenizer,
        )


def run_gemma4_text_replay_harness(
    model,
    adapter: Gemma4TextModelAdapter,
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

    replay_cache = _Gemma4AggregateModelPagedKVCache(
        model=adapter.model,
        config=adapter.dotcache_config,
        backend=adapter.backend,
    )
    for layer_idx, (layer_keys, layer_values) in enumerate(dense_result["prefill_layers"]):
        if torch.is_tensor(layer_keys):
            replay_cache.ingest_prefill_cache_torch(layer_idx, layer_keys, layer_values)
        else:
            replay_cache.ingest_prefill_cache(layer_idx, layer_keys, layer_values)
    replay_cache.prepare_static_pages()

    replay_context_max_abs = 0.0
    replay_context_max_rel = 0.0
    for step_records in dense_result["capture_records"]:
        for record in step_records:
            cache_layer_id = record.layer_id if record.cache_source_layer_id is None else int(record.cache_source_layer_id)
            if cache_layer_id == record.layer_id:
                replay_cache.append_step(
                    cache_layer_id,
                    record.key_states[:, None, :],
                    record.value_states[:, None, :],
                    record.token_index,
                )
            replay_context = replay_cache.decode_layer(cache_layer_id, record.query_states, adapter.q_head_to_kv_head)
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


def run_gemma4_text_generation_harness(
    model,
    adapter: Gemma4TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    max_new_tokens: int = 8,
    tokenizer=None,
    profile: bool = False,
) -> dict[str, Any]:
    return run_llama_generation_harness(
        model,
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        profile=profile,
    )


def run_gemma4_text_loss_harness(
    model,
    adapter: Gemma4TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    tokenizer=None,
) -> dict[str, Any]:
    return run_llama_loss_harness(
        model,
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=prefix_length,
        eval_steps=eval_steps,
        tokenizer=tokenizer,
    )
