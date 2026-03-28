from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..config import DotCacheConfig
from ..decode_reference import decode_page
from ..encode import encode_page
from ..model_kv_cache import ModelPagedKVCache, PreparedPageCache
from .llama import (
    _default_model_device,
    _ensure_attention_mask,
    _normalize_input_ids,
    _require_transformers,
    _run_inference,
    LlamaReplayRecord,
    _torch_backend_matches_device,
    _timed_call,
    resolve_hf_auth_kwargs,
    transformers_available,
)

if transformers_available():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    AutoTokenizer = None
    Qwen3_5ForConditionalGeneration = None
    nn = object  # type: ignore[assignment]
    qwen35_mod = None


Qwen35Mode = Literal["dense", "dotcache_attention_subset"]


def _require_qwen35_model_class() -> None:
    _require_transformers()
    if Qwen3_5ForConditionalGeneration is None:
        raise RuntimeError("transformers installation does not expose Qwen3_5ForConditionalGeneration")


def _text_only_error() -> ValueError:
    return ValueError("Qwen3.5 v1 is text-only; image/video or multimodal inputs are not supported")


def _hybrid_layer_types(model_or_config: Any) -> tuple[str, ...]:
    config = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return ()
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        return ()
    return tuple(str(layer_type) for layer_type in layer_types)


def _hybrid_block_summary(model_or_config: Any) -> dict[str, Any]:
    layer_types = _hybrid_layer_types(model_or_config)
    summary = {
        "hybrid_family": "qwen3_5",
        "hybrid_layer_count": len(layer_types),
        "hybrid_layer_types": list(layer_types),
        "hybrid_linear_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "linear_attention"),
        "hybrid_full_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "full_attention"),
        "hybrid_other_layer_type_count": sum(
            1 for layer_type in layer_types if layer_type not in {"linear_attention", "full_attention"}
        ),
    }
    config = getattr(model_or_config, "config", model_or_config)
    summary["vision_config_present"] = bool(getattr(config, "vision_config", None) is not None)
    return summary


def _qwen35_text_model(model_or_config: Any) -> Any | None:
    model = getattr(model_or_config, "model", model_or_config)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return language_model
    root_model = getattr(model, "model", None)
    if root_model is not None:
        return getattr(root_model, "language_model", None)
    return None


def _qwen35_text_config(model_or_config: Any) -> Any:
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def _qwen35_attention_head_dim(model_or_config: Any) -> int:
    text_model = _qwen35_text_model(model_or_config)
    layers = getattr(text_model, "layers", None)
    layer_types = _hybrid_layer_types(model_or_config)
    if layers is not None:
        for layer_id, layer_type in enumerate(layer_types):
            if layer_type == "full_attention" and layer_id < len(layers) and hasattr(layers[layer_id], "self_attn"):
                attention_module = layers[layer_id].self_attn
                if hasattr(attention_module, "base_attention"):
                    attention_module = attention_module.base_attention
                return int(attention_module.head_dim)
    text_config = _qwen35_text_config(model_or_config)
    return int(getattr(text_config, "head_dim", int(text_config.hidden_size) // int(text_config.num_attention_heads)))


def _configure_qwen35_linear_attention_runtime(model_or_config: Any) -> None:
    if qwen35_mod is None or torch is None:
        return
    text_model = _qwen35_text_model(model_or_config)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return
    layer_types = _hybrid_layer_types(model_or_config)
    model = getattr(model_or_config, "model", model_or_config)
    try:
        use_cuda_fast_path = next(model.parameters()).device.type == "cuda"
    except StopIteration:  # pragma: no cover - defensive only
        use_cuda_fast_path = False
    for layer_id, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention" or layer_id >= len(layers):
            continue
        linear_attn = getattr(layers[layer_id], "linear_attn", None)
        if linear_attn is None:
            continue
        if use_cuda_fast_path:
            continue
        linear_attn.causal_conv1d_fn = None
        linear_attn.causal_conv1d_update = qwen35_mod.torch_causal_conv1d_update
        linear_attn.chunk_gated_delta_rule = qwen35_mod.torch_chunk_gated_delta_rule
        linear_attn.recurrent_gated_delta_rule = qwen35_mod.torch_recurrent_gated_delta_rule
        if type(linear_attn.norm).__name__ != "Qwen3_5RMSNormGated":
            fallback_norm = qwen35_mod.Qwen3_5RMSNormGated(
                linear_attn.head_v_dim,
                eps=linear_attn.layer_norm_epsilon,
            )
            if hasattr(linear_attn.norm, "weight") and hasattr(fallback_norm, "weight"):
                fallback_norm.weight.data.copy_(linear_attn.norm.weight.detach().to(dtype=fallback_norm.weight.dtype))
            fallback_norm = fallback_norm.to(device=linear_attn.out_proj.weight.device)
            linear_attn.norm = fallback_norm


def _hybrid_layer_records(model_or_config: Any) -> list[dict[str, Any]]:
    layer_types = _hybrid_layer_types(model_or_config)
    text_model = _qwen35_text_model(model_or_config)
    layers = getattr(text_model, "layers", None)
    records: list[dict[str, Any]] = []
    for layer_id, layer_type in enumerate(layer_types):
        record = {
            "layer_id": int(layer_id),
            "layer_type": layer_type,
            "state_family": "attention_kv" if layer_type == "full_attention" else "linear_recurrent",
            "dotcache_candidate": bool(layer_type == "full_attention"),
            "requires_hybrid_state": bool(layer_type != "full_attention"),
            "token_mixer_module": None,
            "layer_module": None,
        }
        if layers is not None and layer_id < len(layers):
            layer = layers[layer_id]
            record["layer_module"] = type(layer).__name__
            if hasattr(layer, "self_attn"):
                record["token_mixer_module"] = type(layer.self_attn).__name__
            elif hasattr(layer, "linear_attn"):
                record["token_mixer_module"] = type(layer.linear_attn).__name__
        records.append(record)
    return records


def _extract_attention_subset_prefill_tensors(cache: Any, layer_ids: list[int]) -> dict[int, tuple[Any, Any]]:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is None or value_cache is None:
        raise ValueError("Qwen3.5 attention-subset DotCache path requires key_cache/value_cache on past_key_values")
    extracted: dict[int, tuple[Any, Any]] = {}
    for layer_id in layer_ids:
        layer_keys = key_cache[layer_id]
        layer_values = value_cache[layer_id]
        if layer_keys is None or layer_values is None:
            raise ValueError(f"Qwen3.5 attention layer {layer_id} is missing prefill KV tensors")
        extracted[int(layer_id)] = (layer_keys, layer_values)
    return extracted


def _replace_attention_subset_cache_with_placeholders(cache: Any, layer_ids: list[int]) -> None:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is None or value_cache is None:
        raise ValueError("Qwen3.5 attention-subset DotCache path requires key_cache/value_cache on past_key_values")
    for layer_id in layer_ids:
        layer_keys = key_cache[layer_id]
        layer_values = value_cache[layer_id]
        if layer_keys is None or layer_values is None:
            continue
        key_cache[layer_id] = layer_keys[..., :0].contiguous()
        value_cache[layer_id] = layer_values[..., :0].contiguous()


def _advance_attention_subset_cache_placeholder(cache: Any, layer_id: int) -> None:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is None or value_cache is None:
        return
    layer_keys = key_cache[layer_id]
    layer_values = value_cache[layer_id]
    if layer_keys is None or layer_values is None:
        return
    key_cache[layer_id] = torch.cat([layer_keys, layer_keys[:, :, :1, :]], dim=2)
    value_cache[layer_id] = torch.cat([layer_values, layer_values[:, :, :1, :]], dim=2)


def _default_q_head_to_kv_head(num_attention_heads: int, num_key_value_heads: int) -> np.ndarray:
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    heads_per_kv = num_attention_heads // num_key_value_heads
    return (np.arange(num_attention_heads, dtype=np.int32) // heads_per_kv).astype(np.int32, copy=False)


def _reconstruct_prefill_history(
    tensor_4d,
    *,
    config: DotCacheConfig,
    kind: str,
    layer_id: int,
) -> np.ndarray:
    values = np.asarray(tensor_4d.detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("attention-subset prefill tensors must have shape [1, kv_heads, seq_len, head_dim]")
    kv_heads = values.shape[1]
    seq_len = values.shape[2]
    reconstructed = np.zeros((kv_heads, seq_len, values.shape[3]), dtype=np.float32)
    page_size = int(config.tokens_per_page)
    for kv_head_id in range(kv_heads):
        token_start = 0
        while token_start < seq_len:
            token_end = min(token_start + page_size, seq_len)
            encoded = encode_page(
                values[0, kv_head_id, token_start:token_end, :],
                config,
                kind=kind,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=token_start,
            )
            reconstructed[kv_head_id, token_start:token_end, :] = decode_page(encoded)
            token_start = token_end
    return reconstructed


def _append_dense_decode_history(
    prefill_history: np.ndarray,
    decode_records: list[LlamaReplayRecord],
    *,
    kind: str,
    step_index: int,
) -> np.ndarray:
    dense_suffix = []
    for record in decode_records[: step_index + 1]:
        dense_suffix.append(record.key_states if kind == "K" else record.value_states)
    if not dense_suffix:
        return prefill_history
    suffix = np.stack(dense_suffix, axis=1).astype(np.float32, copy=False)
    return np.concatenate([prefill_history, suffix], axis=1)


def _replay_attention_subset_context(
    *,
    query_states: np.ndarray,
    key_history: np.ndarray,
    value_history: np.ndarray,
    q_head_to_kv_head: np.ndarray,
    scaling: float,
) -> np.ndarray:
    num_attention_heads, head_dim = query_states.shape
    context = np.zeros((num_attention_heads, head_dim), dtype=np.float32)
    for q_head_id in range(num_attention_heads):
        kv_head_id = int(q_head_to_kv_head[q_head_id])
        logits = key_history[kv_head_id] @ query_states[q_head_id]
        logits = logits.astype(np.float32, copy=False) * np.float32(scaling)
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        weights = weights / np.sum(weights)
        context[q_head_id] = weights.astype(np.float32, copy=False) @ value_history[kv_head_id]
    return context.reshape(-1)


def _iter_cache_tensors(cache: Any, *, _seen: set[int] | None = None):
    if _seen is None:
        _seen = set()
    if cache is None:
        return
    cache_id = id(cache)
    if cache_id in _seen:
        return
    _seen.add(cache_id)

    if torch is not None and torch.is_tensor(cache):
        yield cache
        return
    if isinstance(cache, (list, tuple)):
        for item in cache:
            yield from _iter_cache_tensors(item, _seen=_seen)
        return

    for attr_name in (
        "key_cache",
        "value_cache",
        "kv_states",
        "conv_states",
        "ssm_states",
        "recurrent_states",
        "attention_mask_cache",
        "position_cache",
    ):
        if hasattr(cache, attr_name):
            yield from _iter_cache_tensors(getattr(cache, attr_name), _seen=_seen)

    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        try:
            legacy = to_legacy()
        except Exception:  # pragma: no cover - defensive only
            legacy = None
        if legacy is not None and legacy is not cache:
            yield from _iter_cache_tensors(legacy, _seen=_seen)


def _hybrid_cache_nbytes(cache: Any) -> int:
    total = 0
    for tensor in _iter_cache_tensors(cache):
        total += int(tensor.nelement() * tensor.element_size())
    return total


def _cache_component_nbytes(cache: Any, attr_name: str, layer_id: int) -> int:
    values = getattr(cache, attr_name, None)
    if values is None:
        return 0
    if not isinstance(values, list | tuple):
        return 0
    if layer_id >= len(values):
        return 0
    value = values[layer_id]
    if value is None:
        return 0
    return _hybrid_cache_nbytes(value)


def _cache_component_value(cache: Any, attr_name: str, layer_id: int) -> Any | None:
    values = getattr(cache, attr_name, None)
    if values is None:
        return None
    if not isinstance(values, list | tuple):
        return None
    if layer_id >= len(values):
        return None
    return values[layer_id]


@dataclass(slots=True)
class Qwen35HybridLayerStateSlice:
    layer_id: int
    layer_type: str
    state_growth_family: Literal["fixed_resident", "token_growing"]
    key_cache: Any | None = None
    value_cache: Any | None = None
    conv_state: Any | None = None
    recurrent_state: Any | None = None

    @property
    def key_cache_bytes(self) -> int:
        return _hybrid_cache_nbytes(self.key_cache)

    @property
    def value_cache_bytes(self) -> int:
        return _hybrid_cache_nbytes(self.value_cache)

    @property
    def conv_state_bytes(self) -> int:
        return _hybrid_cache_nbytes(self.conv_state)

    @property
    def recurrent_state_bytes(self) -> int:
        return _hybrid_cache_nbytes(self.recurrent_state)

    @property
    def layer_state_bytes(self) -> int:
        return self.key_cache_bytes + self.value_cache_bytes + self.conv_state_bytes + self.recurrent_state_bytes

    @property
    def fixed_resident_state_bytes(self) -> int:
        return self.conv_state_bytes + self.recurrent_state_bytes if self.state_growth_family == "fixed_resident" else 0

    @property
    def token_growing_state_bytes(self) -> int:
        return self.key_cache_bytes + self.value_cache_bytes if self.state_growth_family == "token_growing" else 0

    def summary_record(self, base_record: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {} if base_record is None else dict(base_record)
        record.update(
            {
                "layer_id": int(self.layer_id),
                "layer_type": self.layer_type,
                "state_growth_family": self.state_growth_family,
                "key_cache_bytes": int(self.key_cache_bytes),
                "value_cache_bytes": int(self.value_cache_bytes),
                "conv_state_bytes": int(self.conv_state_bytes),
                "recurrent_state_bytes": int(self.recurrent_state_bytes),
                "layer_state_bytes": int(self.layer_state_bytes),
                "fixed_resident_state_bytes": int(self.fixed_resident_state_bytes),
                "token_growing_state_bytes": int(self.token_growing_state_bytes),
            }
        )
        return record


@dataclass(slots=True)
class Qwen35HybridStatePartition:
    fixed_resident_layers: tuple[Qwen35HybridLayerStateSlice, ...]
    token_growing_layers: tuple[Qwen35HybridLayerStateSlice, ...]

    @property
    def all_layers(self) -> tuple[Qwen35HybridLayerStateSlice, ...]:
        return tuple(sorted(self.fixed_resident_layers + self.token_growing_layers, key=lambda record: record.layer_id))

    @property
    def fixed_resident_layer_ids(self) -> list[int]:
        return [int(record.layer_id) for record in self.fixed_resident_layers]

    @property
    def token_growing_layer_ids(self) -> list[int]:
        return [int(record.layer_id) for record in self.token_growing_layers]

    def to_summary(self, *, model_or_config: Any | None = None) -> dict[str, Any]:
        layer_records = _hybrid_layer_records(model_or_config) if model_or_config is not None else []
        layer_records_by_id = {int(record["layer_id"]): record for record in layer_records}
        all_layers = self.all_layers
        attention_kv_bytes = sum(record.key_cache_bytes + record.value_cache_bytes for record in all_layers)
        linear_conv_bytes = sum(record.conv_state_bytes for record in all_layers)
        linear_recurrent_bytes = sum(record.recurrent_state_bytes for record in all_layers)
        fixed_resident_bytes = sum(record.fixed_resident_state_bytes for record in self.fixed_resident_layers)
        token_growing_bytes = sum(record.token_growing_state_bytes for record in self.token_growing_layers)
        return {
            "hybrid_state_total_bytes": int(attention_kv_bytes + linear_conv_bytes + linear_recurrent_bytes),
            "hybrid_attention_kv_bytes": int(attention_kv_bytes),
            "hybrid_linear_conv_state_bytes": int(linear_conv_bytes),
            "hybrid_linear_recurrent_state_bytes": int(linear_recurrent_bytes),
            "hybrid_fixed_resident_bytes": int(fixed_resident_bytes),
            "hybrid_token_growing_bytes": int(token_growing_bytes),
            "hybrid_fixed_resident_layer_count": len(self.fixed_resident_layers),
            "hybrid_token_growing_layer_count": len(self.token_growing_layers),
            "hybrid_fixed_resident_layer_ids": self.fixed_resident_layer_ids,
            "hybrid_token_growing_layer_ids": self.token_growing_layer_ids,
            "hybrid_state_layers": [
                record.summary_record(layer_records_by_id.get(int(record.layer_id)))
                for record in all_layers
            ],
        }


@dataclass(slots=True)
class Qwen35NativeHybridRuntimeState:
    model_or_config: Any
    past_key_values: Any
    prefill_partition: Qwen35HybridStatePartition
    current_partition: Qwen35HybridStatePartition

    @classmethod
    def from_post_handoff_cache(cls, past_key_values: Any, model_or_config: Any) -> "Qwen35NativeHybridRuntimeState":
        partition = partition_qwen35_hybrid_state(past_key_values, model_or_config)
        return cls(
            model_or_config=model_or_config,
            past_key_values=past_key_values,
            prefill_partition=partition,
            current_partition=partition,
        )

    @property
    def fixed_resident_layer_ids(self) -> list[int]:
        return self.current_partition.fixed_resident_layer_ids

    @property
    def token_growing_layer_ids(self) -> list[int]:
        return self.current_partition.token_growing_layer_ids

    def refresh(self, past_key_values: Any) -> None:
        self.past_key_values = past_key_values
        self.current_partition = partition_qwen35_hybrid_state(past_key_values, self.model_or_config)

    def prefill_summary(self) -> dict[str, Any]:
        return self.prefill_partition.to_summary(model_or_config=self.model_or_config)

    def current_summary(self) -> dict[str, Any]:
        return self.current_partition.to_summary(model_or_config=self.model_or_config)

    def summary(self) -> dict[str, Any]:
        prefill_summary = self.prefill_summary()
        final_summary = self.current_summary()
        result = {
            "hybrid_state_partition_ready": True,
            "native_hybrid_fixed_resident_layer_ids": self.fixed_resident_layer_ids,
            "native_hybrid_token_growing_layer_ids": self.token_growing_layer_ids,
            "native_hybrid_prefill_fixed_resident_bytes": int(prefill_summary["hybrid_fixed_resident_bytes"]),
            "native_hybrid_prefill_token_growing_bytes": int(prefill_summary["hybrid_token_growing_bytes"]),
            "native_hybrid_final_fixed_resident_bytes": int(final_summary["hybrid_fixed_resident_bytes"]),
            "native_hybrid_final_token_growing_bytes": int(final_summary["hybrid_token_growing_bytes"]),
            "native_hybrid_fixed_resident_growth_bytes": int(
                final_summary["hybrid_fixed_resident_bytes"] - prefill_summary["hybrid_fixed_resident_bytes"]
            ),
            "native_hybrid_token_growing_growth_bytes": int(
                final_summary["hybrid_token_growing_bytes"] - prefill_summary["hybrid_token_growing_bytes"]
            ),
            "native_hybrid_prefill_state_layers": prefill_summary["hybrid_state_layers"],
            "native_hybrid_final_state_layers": final_summary["hybrid_state_layers"],
        }
        result["native_hybrid_fixed_resident_preserved"] = (
            result["native_hybrid_fixed_resident_growth_bytes"] == 0
        )
        return result


@dataclass(slots=True)
class Qwen35HybridDotCacheRuntimeState:
    native_state: Qwen35NativeHybridRuntimeState
    model_kv_cache: ModelPagedKVCache

    @property
    def model_past_key_values(self) -> Any:
        return self.native_state.past_key_values

    def refresh_native(self, past_key_values: Any) -> None:
        self.native_state.refresh(past_key_values)

    def advance(self, past_key_values: Any) -> None:
        self.refresh_native(past_key_values)

    def summary(self) -> dict[str, Any]:
        result = self.native_state.summary()
        result.update(
            {
                "hybrid_dotcache_runtime_ready": True,
                "hybrid_runtime_state_kind": "qwen35_attention_subset",
                "hybrid_runtime_token_growing_layer_ids": self.native_state.token_growing_layer_ids,
                "hybrid_runtime_fixed_resident_layer_ids": self.native_state.fixed_resident_layer_ids,
            }
        )
        result.update(self.model_kv_cache.resident_byte_summary())
        result.update(self.model_kv_cache.page_mode_summary())
        return result


def partition_qwen35_hybrid_state(cache: Any, model_or_config: Any) -> Qwen35HybridStatePartition:
    fixed_resident_layers: list[Qwen35HybridLayerStateSlice] = []
    token_growing_layers: list[Qwen35HybridLayerStateSlice] = []
    for layer_record in _hybrid_layer_records(model_or_config):
        layer_id = int(layer_record["layer_id"])
        layer_type = str(layer_record["layer_type"])
        growth_family: Literal["fixed_resident", "token_growing"] = (
            "token_growing" if layer_type == "full_attention" else "fixed_resident"
        )
        state_slice = Qwen35HybridLayerStateSlice(
            layer_id=layer_id,
            layer_type=layer_type,
            state_growth_family=growth_family,
            key_cache=_cache_component_value(cache, "key_cache", layer_id),
            value_cache=_cache_component_value(cache, "value_cache", layer_id),
            conv_state=_cache_component_value(cache, "conv_states", layer_id),
            recurrent_state=_cache_component_value(cache, "recurrent_states", layer_id),
        )
        if growth_family == "fixed_resident":
            fixed_resident_layers.append(state_slice)
        else:
            token_growing_layers.append(state_slice)
    return Qwen35HybridStatePartition(
        fixed_resident_layers=tuple(fixed_resident_layers),
        token_growing_layers=tuple(token_growing_layers),
    )


def summarize_qwen35_hybrid_state(cache: Any, model_or_config: Any) -> dict[str, Any]:
    return partition_qwen35_hybrid_state(cache, model_or_config).to_summary(model_or_config=model_or_config)


def summarize_qwen35_hybrid_state_growth(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_layers = {int(record["layer_id"]): record for record in before.get("hybrid_state_layers", [])}
    after_layers = {int(record["layer_id"]): record for record in after.get("hybrid_state_layers", [])}
    layer_growth: list[dict[str, Any]] = []
    for layer_id in sorted(set(before_layers) | set(after_layers)):
        before_record = before_layers.get(layer_id, {})
        after_record = after_layers.get(layer_id, {})
        layer_growth.append(
            {
                "layer_id": int(layer_id),
                "layer_type": after_record.get("layer_type", before_record.get("layer_type")),
                "state_growth_family": after_record.get("state_growth_family", before_record.get("state_growth_family")),
                "key_cache_growth_bytes": int(after_record.get("key_cache_bytes", 0) - before_record.get("key_cache_bytes", 0)),
                "value_cache_growth_bytes": int(
                    after_record.get("value_cache_bytes", 0) - before_record.get("value_cache_bytes", 0)
                ),
                "conv_state_growth_bytes": int(
                    after_record.get("conv_state_bytes", 0) - before_record.get("conv_state_bytes", 0)
                ),
                "recurrent_state_growth_bytes": int(
                    after_record.get("recurrent_state_bytes", 0) - before_record.get("recurrent_state_bytes", 0)
                ),
                "layer_state_growth_bytes": int(
                    after_record.get("layer_state_bytes", 0) - before_record.get("layer_state_bytes", 0)
                ),
            }
        )
    return {
        "hybrid_state_growth_bytes": int(after.get("hybrid_state_total_bytes", 0) - before.get("hybrid_state_total_bytes", 0)),
        "hybrid_attention_kv_growth_bytes": int(
            after.get("hybrid_attention_kv_bytes", 0) - before.get("hybrid_attention_kv_bytes", 0)
        ),
        "hybrid_linear_conv_state_growth_bytes": int(
            after.get("hybrid_linear_conv_state_bytes", 0) - before.get("hybrid_linear_conv_state_bytes", 0)
        ),
        "hybrid_linear_recurrent_state_growth_bytes": int(
            after.get("hybrid_linear_recurrent_state_bytes", 0) - before.get("hybrid_linear_recurrent_state_bytes", 0)
        ),
        "hybrid_fixed_resident_growth_bytes": int(
            after.get("hybrid_fixed_resident_bytes", 0) - before.get("hybrid_fixed_resident_bytes", 0)
        ),
        "hybrid_token_growing_growth_bytes": int(
            after.get("hybrid_token_growing_bytes", 0) - before.get("hybrid_token_growing_bytes", 0)
        ),
        "hybrid_state_growth_layers": layer_growth,
    }


def summarize_qwen35_dotcache_fit(model_or_config: Any) -> dict[str, Any]:
    layer_records = _hybrid_layer_records(model_or_config)
    attention_candidate_layers = [record["layer_id"] for record in layer_records if record["dotcache_candidate"]]
    hybrid_only_layers = [record["layer_id"] for record in layer_records if record["requires_hybrid_state"]]
    linear_layer_count = len(hybrid_only_layers)
    full_attention_layer_count = len(attention_candidate_layers)
    suggested_next_step = (
        "attention_subset_only_then_generalize_state"
        if linear_layer_count > 0 and full_attention_layer_count > 0
        else "dotcache_attention_path_only"
        if full_attention_layer_count > 0
        else "no_attention_subset_available"
    )
    return {
        "attention_candidate_layer_ids": attention_candidate_layers,
        "attention_candidate_layer_count": len(attention_candidate_layers),
        "hybrid_only_layer_ids": hybrid_only_layers,
        "hybrid_only_layer_count": len(hybrid_only_layers),
        "requires_hybrid_state_abstraction": bool(linear_layer_count > 0),
        "suggested_next_step": suggested_next_step,
    }


def _decode_text(tokenizer: Any | None, token_ids: list[int]) -> str | None:
    if tokenizer is None:
        return None
    return str(tokenizer.decode(token_ids, skip_special_tokens=True))


def load_qwen35_text_only_from_pretrained(
    model_id: str,
    *,
    device: str | None = None,
    torch_dtype: str = "float16",
):
    _require_qwen35_model_class()
    dtype = getattr(torch, torch_dtype)
    resolved_device = _default_model_device() if device is None else device
    auth_kwargs = resolve_hf_auth_kwargs()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=False,
        **auth_kwargs,
    )
    model.to(resolved_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **auth_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@dataclass(slots=True)
class Qwen35TextModelAdapter:
    model: Any
    mode: Qwen35Mode = "dense"

    def __post_init__(self) -> None:
        _configure_qwen35_linear_attention_runtime(self.model)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_mode(self, mode: str) -> None:
        if mode != "dense":
            raise ValueError(
                "Qwen3.5 v1 only supports dense mode; DotCache interception is not implemented for the hybrid text stack yet"
            )
        self.mode = "dense"

    def hybrid_block_summary(self) -> dict[str, Any]:
        return _hybrid_block_summary(self.model)

    def hybrid_layer_summary(self) -> list[dict[str, Any]]:
        return _hybrid_layer_records(self.model)

    def hybrid_fit_summary(self) -> dict[str, Any]:
        return summarize_qwen35_dotcache_fit(self.model)

    def partition_hybrid_state(self, cache: Any) -> Qwen35HybridStatePartition:
        return partition_qwen35_hybrid_state(cache, self.model)


class DotCacheQwen35AttentionSubset(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "Qwen35AttentionSubsetModelAdapter") -> None:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Qwen3.5 attention subset capture path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        query_states, gate = torch.chunk(
            self.base_attention.q_proj(hidden_states).view(*input_shape, -1, self.base_attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.base_attention.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.base_attention.k_norm(self.base_attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen35_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

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
        query_states, key_states, value_states, gate = self._project_qkv(hidden_states, position_embeddings)
        fresh_key_states = key_states
        fresh_value_states = value_states

        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = qwen35_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.base_attention.config._attn_implementation,
            qwen35_mod.eager_attention_forward,
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
        gated_context = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        projected_output = self.base_attention.o_proj(gated_context)

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
                    context_states=gated_context[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    gate_states=gate[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
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
        if past_key_values is None:
            raise ValueError("Qwen3.5 attention-subset DotCache mode requires the native hybrid past_key_values state")
        if tuple(hidden_states.shape[:2]) != (1, 1):
            raise ValueError("Qwen3.5 attention-subset DotCache mode only supports batch=1 and query_len=1")
        token_index = self.adapter.current_token_index(cache_position)
        input_shape = hidden_states.shape[:-1]
        (query_states, key_states, value_states, gate), qkv_ms = _timed_call(
            lambda: self._project_qkv(hidden_states, position_embeddings),
            device=hidden_states.device,
        )
        query_step = query_states[0, :, 0, :].detach().to(dtype=torch.float32)
        key_step = key_states[0].detach().to(dtype=torch.float32)
        value_step = value_states[0].detach().to(dtype=torch.float32)
        self.adapter.qkv_projection_ms_total += qkv_ms

        _, append_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.append_step_torch(
                self.layer_idx,
                key_step,
                value_step,
                token_index,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.append_step(
                self.layer_idx,
                key_step.cpu().numpy(),
                value_step.cpu().numpy(),
                token_index,
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
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.decode_layer(
                self.layer_idx,
                query_step.detach().cpu().numpy(),
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
            ),
            device=hidden_states.device,
        )
        self.adapter.decode_runtime_ms_total += decode_ms

        if torch.is_tensor(context_states):
            context_tensor = context_states.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
        else:
            context_tensor = torch.as_tensor(context_states, dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
        gated_context = context_tensor.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        projected_output, output_projection_ms = _timed_call(
            lambda: self.base_attention.o_proj(gated_context),
            device=hidden_states.device,
        )
        self.adapter.output_projection_ms_total += output_projection_ms

        _advance_attention_subset_cache_placeholder(past_key_values, self.layer_idx)

        if self.adapter.capture_enabled:
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_step.detach().cpu().numpy(),
                    key_states=key_step[:, 0, :].detach().cpu().numpy(),
                    value_states=value_step[:, 0, :].detach().cpu().numpy(),
                    context_states=gated_context[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    gate_states=gate[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, None


@dataclass(slots=True)
class Qwen35AttentionSubsetModelAdapter(Qwen35TextModelAdapter):
    capture_enabled: bool = False
    capture_step_index: int = -1
    _pending_records: list[LlamaReplayRecord] = field(default_factory=list, init=False, repr=False)
    _wrappers: list[DotCacheQwen35AttentionSubset] = field(default_factory=list, init=False, repr=False)
    _current_token_index_override: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        Qwen35TextModelAdapter.__post_init__(self)
        self._install_wrappers()

    def _install_wrappers(self) -> None:
        text_model = _qwen35_text_model(self.model)
        layers = getattr(text_model, "layers", None)
        if layers is None:
            return
        layer_types = _hybrid_layer_types(self.model)
        for layer_id, layer in enumerate(layers[: len(layer_types)]):
            if layer_types[layer_id] != "full_attention" or not hasattr(layer, "self_attn"):
                continue
            wrapper = DotCacheQwen35AttentionSubset(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def begin_capture_step(self, step_index: int) -> None:
        self.capture_step_index = int(step_index)
        self.capture_enabled = True
        self._pending_records = []

    def end_capture_step(self) -> list[LlamaReplayRecord]:
        records = list(self._pending_records)
        self.capture_step_index = -1
        self.capture_enabled = False
        self._pending_records = []
        return records

    def record_replay(self, record: LlamaReplayRecord) -> None:
        if self.capture_step_index < 0:
            return
        self._pending_records.append(record)

    def current_token_index(self, cache_position) -> int:
        if self._current_token_index_override is not None:
            return self._current_token_index_override
        if cache_position is None:
            raise ValueError("cache_position is required for Qwen3.5 attention-subset capture")
        token_positions = cache_position.reshape(-1)
        if token_positions.numel() != 1:
            raise ValueError("Qwen3.5 attention-subset capture requires a single cache_position per decode step")
        return int(token_positions.item())

    def set_current_token_index(self, token_index: int | None) -> None:
        self._current_token_index_override = None if token_index is None else int(token_index)

    def attention_subset_layer_ids(self) -> list[int]:
        return self.hybrid_fit_summary()["attention_candidate_layer_ids"]


@dataclass(slots=True)
class Qwen35AttentionSubsetDotCacheModelAdapter(Qwen35AttentionSubsetModelAdapter):
    dotcache_config: DotCacheConfig = field(default_factory=lambda: DotCacheConfig(head_dim=256, group_size=32, bits_k=4, bits_v=4, tokens_per_page=16))
    backend: str = "cpu_ref"
    cache: PreparedPageCache = field(default_factory=PreparedPageCache)
    model_kv_cache: ModelPagedKVCache = field(init=False, repr=False)
    q_head_to_kv_head: np.ndarray = field(init=False, repr=False)
    append_runtime_ms_total: float = field(default=0.0, init=False, repr=False)
    decode_runtime_ms_total: float = field(default=0.0, init=False, repr=False)
    qkv_projection_ms_total: float = field(default=0.0, init=False, repr=False)
    output_projection_ms_total: float = field(default=0.0, init=False, repr=False)
    native_hybrid_runtime_state: Qwen35NativeHybridRuntimeState | None = field(default=None, init=False, repr=False)
    hybrid_dotcache_runtime_state: Qwen35HybridDotCacheRuntimeState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        Qwen35AttentionSubsetModelAdapter.__post_init__(self)
        text_config = _qwen35_text_config(self.model)
        expected_head_dim = _qwen35_attention_head_dim(self.model)
        if self.dotcache_config.head_dim != expected_head_dim:
            self.dotcache_config = replace(self.dotcache_config, head_dim=expected_head_dim)
        self.model_kv_cache = ModelPagedKVCache(
            config=self.dotcache_config,
            num_hidden_layers=int(text_config.num_hidden_layers),
            num_attention_heads=int(text_config.num_attention_heads),
            num_key_value_heads=int(text_config.num_key_value_heads),
            backend=self.backend,
            cache=self.cache,
        )
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()

    def set_mode(self, mode: str) -> None:
        if mode not in {"dense", "dotcache_attention_subset"}:
            raise ValueError("Qwen3.5 attention-subset adapter only supports dense and dotcache_attention_subset modes")
        self.mode = mode  # type: ignore[assignment]

    def clear(self) -> None:
        self.model_kv_cache.clear()
        self._pending_records = []
        self.capture_enabled = False
        self.capture_step_index = -1
        self._current_token_index_override = None
        self.append_runtime_ms_total = 0.0
        self.decode_runtime_ms_total = 0.0
        self.qkv_projection_ms_total = 0.0
        self.output_projection_ms_total = 0.0
        self.native_hybrid_runtime_state = None
        self.hybrid_dotcache_runtime_state = None

    def token_growing_layer_ids(self) -> list[int]:
        if self.native_hybrid_runtime_state is not None:
            return list(self.native_hybrid_runtime_state.token_growing_layer_ids)
        return self.attention_subset_layer_ids()

    def fixed_resident_layer_ids(self) -> list[int]:
        if self.native_hybrid_runtime_state is not None:
            return list(self.native_hybrid_runtime_state.fixed_resident_layer_ids)
        return [
            layer_id
            for layer_id, layer_type in enumerate(_hybrid_layer_types(self.model))
            if layer_type != "full_attention"
        ]

    def load_attention_subset_prefill_cache(self, past_key_values: Any) -> None:
        source_prefill_partition = self.partition_hybrid_state(past_key_values)
        attention_layer_ids = source_prefill_partition.token_growing_layer_ids
        extracted = _extract_attention_subset_prefill_tensors(past_key_values, attention_layer_ids)
        self.model_kv_cache.clear()
        use_torch_prefill = _torch_backend_matches_device(self.backend, self.device.type)
        for layer_id in attention_layer_ids:
            layer_keys, layer_values = extracted[layer_id]
            if use_torch_prefill:
                self.model_kv_cache.ingest_prefill_cache_torch(layer_id, layer_keys, layer_values)
            else:
                self.model_kv_cache.ingest_prefill_cache(
                    layer_id,
                    layer_keys.detach().cpu().numpy(),
                    layer_values.detach().cpu().numpy(),
                )
        self.model_kv_cache.prepare_static_pages()
        _replace_attention_subset_cache_with_placeholders(past_key_values, attention_layer_ids)
        self.native_hybrid_runtime_state = Qwen35NativeHybridRuntimeState.from_post_handoff_cache(
            past_key_values,
            self.model,
        )
        self.hybrid_dotcache_runtime_state = Qwen35HybridDotCacheRuntimeState(
            native_state=self.native_hybrid_runtime_state,
            model_kv_cache=self.model_kv_cache,
        )

    def refresh_native_hybrid_runtime_state(self, past_key_values: Any) -> None:
        if self.hybrid_dotcache_runtime_state is not None:
            self.hybrid_dotcache_runtime_state.refresh_native(past_key_values)
            self.native_hybrid_runtime_state = self.hybrid_dotcache_runtime_state.native_state
            return
        self.native_hybrid_runtime_state = Qwen35NativeHybridRuntimeState.from_post_handoff_cache(
            past_key_values,
            self.model,
        )
        self.hybrid_dotcache_runtime_state = Qwen35HybridDotCacheRuntimeState(
            native_state=self.native_hybrid_runtime_state,
            model_kv_cache=self.model_kv_cache,
        )

    def summarize_dotcache_native_hybrid_state(self, past_key_values: Any) -> dict[str, Any]:
        self.refresh_native_hybrid_runtime_state(past_key_values)
        if self.hybrid_dotcache_runtime_state is not None:
            return self.hybrid_dotcache_runtime_state.summary()
        if self.native_hybrid_runtime_state is None:  # pragma: no cover - defensive only
            return {"hybrid_state_partition_ready": False, "hybrid_dotcache_runtime_ready": False}
        result = self.native_hybrid_runtime_state.summary()
        result["hybrid_dotcache_runtime_ready"] = False
        return result

    def require_hybrid_dotcache_runtime_state(self) -> Qwen35HybridDotCacheRuntimeState:
        if self.hybrid_dotcache_runtime_state is None:
            raise ValueError("Qwen3.5 attention-subset DotCache runtime state is not initialized")
        return self.hybrid_dotcache_runtime_state


@dataclass(slots=True)
class Qwen35TextHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35TextModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen35TextHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Qwen35TextModelAdapter(model=model)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        if multimodal_inputs is not None:
            raise _text_only_error()
        if self.tokenizer is None:
            raise ValueError("tokenizer is unavailable for text prompt input")
        if not isinstance(prompt, str):
            raise ValueError("Qwen3.5 v1 expects a plain text prompt string")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.adapter.device)
        attention_mask = encoded["attention_mask"].to(self.adapter.device)
        return input_ids, attention_mask

    def generate_greedy(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 8,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_text_generation_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer,
            multimodal_inputs=multimodal_inputs,
        )

    def evaluate_loss(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_text_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            tokenizer=self.tokenizer,
            multimodal_inputs=multimodal_inputs,
        )

    def inspect_hybrid_state(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 0,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return inspect_qwen35_hybrid_state(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            multimodal_inputs=multimodal_inputs,
        )


@dataclass(slots=True)
class Qwen35AttentionSubsetHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35AttentionSubsetModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen35AttentionSubsetHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Qwen35AttentionSubsetModelAdapter(model=model)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        helper = Qwen35TextHarness(model=self.model, tokenizer=self.tokenizer, adapter=self.adapter)
        return helper.tokenize_prompt(prompt, multimodal_inputs=multimodal_inputs)

    def run_attention_subset_replay(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
        ) -> dict[str, Any]:
        return run_qwen35_attention_subset_replay_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            multimodal_inputs=multimodal_inputs,
        )

    def run_prefill_ablation(
        self,
        dotcache_config: DotCacheConfig,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_prefill_ablation_harness(
            self.model,
            self.adapter,
            dotcache_config=dotcache_config,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            multimodal_inputs=multimodal_inputs,
        )


@dataclass(slots=True)
class Qwen35AttentionSubsetDotCacheHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen35AttentionSubsetDotCacheHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
            model=model,
            dotcache_config=dotcache_config,
            backend=backend,
        )
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        helper = Qwen35TextHarness(model=self.model, tokenizer=self.tokenizer, adapter=self.adapter)
        return helper.tokenize_prompt(prompt, multimodal_inputs=multimodal_inputs)

    def run_attention_subset_dotcache(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            multimodal_inputs=multimodal_inputs,
        )


def _normalize_text_inputs(
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None,
    input_ids,
    attention_mask,
    tokenizer,
    multimodal_inputs: Any | None,
) -> tuple[Any, Any]:
    if multimodal_inputs is not None:
        raise _text_only_error()
    if prompt is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when prompt text is provided")
        if not isinstance(prompt, str):
            raise ValueError("Qwen3.5 v1 expects a plain text prompt string")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)
    return input_ids, attention_mask


def _run_dense_prefill(model, *, input_ids, attention_mask):
    return _run_inference(lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True))


def _run_dense_decode_step(
    model,
    *,
    decode_input_ids,
    attention_mask,
    past_key_values,
    cache_position,
):
    return _run_inference(
        lambda: model(
            input_ids=decode_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
    )


def run_qwen35_text_generation_harness(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    max_new_tokens: int = 8,
    tokenizer=None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    prefill_cache_bytes = _hybrid_cache_nbytes(prefill_outputs.past_key_values)
    generated_ids: list[int] = []
    dense_decode_ms_total = 0.0
    final_past_key_values = prefill_outputs.past_key_values

    if max_new_tokens > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(current_input_ids.item()))
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        past_key_values = prefill_outputs.past_key_values

        for _ in range(max(max_new_tokens - 1, 0)):
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
                device=input_ids.device,
            )
            dense_decode_ms_total += step_ms
            past_key_values = outputs.past_key_values
            final_past_key_values = outputs.past_key_values
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(current_input_ids.item()))
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": max(max_new_tokens - 1, 0),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(max_new_tokens - 1, 1)) if max_new_tokens > 1 else 0.0,
        "dense_generated_ids": list(generated_ids),
        "dense_prefill_cache_bytes": int(prefill_cache_bytes),
        "dense_final_cache_bytes": int(_hybrid_cache_nbytes(final_past_key_values)),
        "cache_metric_kind": "hybrid_cache_bytes",
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    decoded_text = _decode_text(tokenizer, generated_ids)
    if decoded_text is not None:
        result["dense_text"] = decoded_text
    return result


def run_qwen35_text_loss_harness(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    tokenizer=None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    if prefix_length <= 0 or prefix_length >= int(input_ids.shape[1]):
        raise ValueError("prefix_length must be in [1, sequence_length)")
    available_eval_steps = int(input_ids.shape[1]) - prefix_length
    if eval_steps <= 0 or eval_steps > available_eval_steps:
        raise ValueError("eval_steps must be positive and fit inside the provided sequence after prefix_length")

    prefix_input_ids = input_ids[:, :prefix_length]
    prefix_attention_mask = attention_mask[:, :prefix_length]
    continuation_ids = input_ids[:, prefix_length : prefix_length + eval_steps]

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=input_ids.device,
    )
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu()]
    past_key_values = prefill_outputs.past_key_values
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    dense_decode_ms_total = 0.0

    for step_index in range(max(eval_steps - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]
        outputs, step_ms = _timed_call(
            lambda: _run_dense_decode_step(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            ),
            device=input_ids.device,
        )
        dense_decode_ms_total += step_ms
        past_key_values = outputs.past_key_values
        logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu())
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    logits = torch.cat(logits_list, dim=0).numpy()
    target_tokens = continuation_ids[0, : logits.shape[0]].detach().cpu().numpy()
    max_logits = logits.max(axis=-1, keepdims=True)
    stabilized = logits - max_logits
    log_probs = stabilized - torch.from_numpy(stabilized).exp().sum(dim=-1, keepdim=True).log().numpy()
    token_losses = -log_probs[range(target_tokens.shape[0]), target_tokens]
    mean_loss = float(token_losses.mean())
    perplexity = float(math.exp(min(mean_loss, 50.0)))
    predictions = logits.argmax(axis=-1)
    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "dense_teacher_forced_loss": mean_loss,
        "dense_teacher_forced_perplexity": perplexity,
        "dense_teacher_forced_target_match_rate": float((predictions == target_tokens).mean()),
        "dense_prefill_cache_bytes": int(_hybrid_cache_nbytes(prefill_outputs.past_key_values)),
        "dense_final_cache_bytes": int(_hybrid_cache_nbytes(past_key_values)),
        "cache_metric_kind": "hybrid_cache_bytes",
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    return result


def inspect_qwen35_hybrid_state(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 0,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    prefill_cache = prefill_outputs.past_key_values
    prefill_partition = adapter.partition_hybrid_state(prefill_cache)
    prefill_state = prefill_partition.to_summary(model_or_config=model)
    dense_decode_ms_total = 0.0
    final_cache = prefill_cache
    if decode_steps > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        for _step_index in range(decode_steps):
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=final_cache,
                    cache_position=cache_position,
                ),
                device=input_ids.device,
            )
            dense_decode_ms_total += step_ms
            final_cache = outputs.past_key_values
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1
    final_partition = adapter.partition_hybrid_state(final_cache)
    final_state = final_partition.to_summary(model_or_config=model)
    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
        "hybrid_state_partition_ready": True,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(prefill_state)
    result.update(
        {
            "hybrid_prefill_state_total_bytes": int(prefill_state["hybrid_state_total_bytes"]),
            "hybrid_prefill_attention_kv_bytes": int(prefill_state["hybrid_attention_kv_bytes"]),
            "hybrid_prefill_linear_conv_state_bytes": int(prefill_state["hybrid_linear_conv_state_bytes"]),
            "hybrid_prefill_linear_recurrent_state_bytes": int(prefill_state["hybrid_linear_recurrent_state_bytes"]),
            "hybrid_prefill_fixed_resident_bytes": int(prefill_state["hybrid_fixed_resident_bytes"]),
            "hybrid_prefill_token_growing_bytes": int(prefill_state["hybrid_token_growing_bytes"]),
            "hybrid_prefill_state_layers": prefill_state["hybrid_state_layers"],
            "hybrid_final_state_total_bytes": int(final_state["hybrid_state_total_bytes"]),
            "hybrid_final_attention_kv_bytes": int(final_state["hybrid_attention_kv_bytes"]),
            "hybrid_final_linear_conv_state_bytes": int(final_state["hybrid_linear_conv_state_bytes"]),
            "hybrid_final_linear_recurrent_state_bytes": int(final_state["hybrid_linear_recurrent_state_bytes"]),
            "hybrid_final_fixed_resident_bytes": int(final_state["hybrid_fixed_resident_bytes"]),
            "hybrid_final_token_growing_bytes": int(final_state["hybrid_token_growing_bytes"]),
            "hybrid_final_state_layers": final_state["hybrid_state_layers"],
        }
    )
    result.update(summarize_qwen35_hybrid_state_growth(prefill_state, final_state))
    result.update(adapter.hybrid_fit_summary())
    return result


def _run_qwen35_attention_subset_dense_capture(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    input_ids,
    attention_mask,
    decode_steps: int,
) -> dict[str, Any]:
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    per_step_records: list[list[LlamaReplayRecord]] = []
    decode_inputs: list[Any] = []
    step_logits: list[np.ndarray] = []
    dense_decode_ms_total = 0.0

    if decode_steps > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        past_key_values = prefill_outputs.past_key_values
        for step_index in range(decode_steps):
            decode_inputs.append(current_input_ids.detach().clone())
            adapter.begin_capture_step(step_index)
            adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
            try:
                outputs, step_ms = _timed_call(
                    lambda: _run_dense_decode_step(
                        model,
                        decode_input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                    ),
                    device=input_ids.device,
                )
            finally:
                adapter.set_current_token_index(None)
            dense_decode_ms_total += step_ms
            per_step_records.append(adapter.end_capture_step())
            step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
            past_key_values = outputs.past_key_values
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1

    return {
        "prefill_outputs": prefill_outputs,
        "prefill_ms": float(prefill_ms),
        "decode_ms_total": float(dense_decode_ms_total),
        "decode_inputs": decode_inputs,
        "step_logits": step_logits,
        "capture_records": per_step_records,
    }


def _summarize_attention_subset_capture(
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    input_ids,
    decode_steps: int,
    prefill_ms: float,
    dense_decode_ms_total: float,
    per_step_records: list[list[LlamaReplayRecord]],
) -> dict[str, Any]:
    records = [record for step_records in per_step_records for record in step_records]
    per_layer_counts: dict[int, int] = {}
    per_layer_shapes: dict[int, dict[str, list[int]]] = {}
    for record in records:
        per_layer_counts[record.layer_id] = per_layer_counts.get(record.layer_id, 0) + 1
        per_layer_shapes.setdefault(
            record.layer_id,
            {
                "query_states": list(record.query_states.shape),
                "key_states": list(record.key_states.shape),
                "value_states": list(record.value_states.shape),
                "context_states": list(record.context_states.shape),
            },
        )

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "attention_subset_layer_ids": adapter.attention_subset_layer_ids(),
        "attention_subset_capture_layer_count": len(adapter.attention_subset_layer_ids()),
        "attention_subset_capture_record_count": len(records),
        "attention_subset_capture_counts_by_layer": {str(layer_id): count for layer_id, count in sorted(per_layer_counts.items())},
        "attention_subset_capture_shapes_by_layer": {str(layer_id): shapes for layer_id, shapes in sorted(per_layer_shapes.items())},
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense_attention_subset_capture",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_attention_subset_replay_harness(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    dense_capture = _run_qwen35_attention_subset_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    return _summarize_attention_subset_capture(
        adapter,
        input_ids=input_ids,
        decode_steps=decode_steps,
        prefill_ms=float(dense_capture["prefill_ms"]),
        dense_decode_ms_total=float(dense_capture["decode_ms_total"]),
        per_step_records=dense_capture["capture_records"],
    )


def run_qwen35_attention_subset_prefill_ablation_harness(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    dotcache_config: DotCacheConfig,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    dense_capture = _run_qwen35_attention_subset_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    attention_layer_ids = adapter.attention_subset_layer_ids()
    prefill_tensors = _extract_attention_subset_prefill_tensors(dense_capture["prefill_outputs"].past_key_values, attention_layer_ids)
    text_config = _qwen35_text_config(model)
    q_head_to_kv_head = _default_q_head_to_kv_head(
        int(text_config.num_attention_heads),
        int(text_config.num_key_value_heads),
    )
    per_layer_decode_records: dict[int, list[LlamaReplayRecord]] = {layer_id: [] for layer_id in attention_layer_ids}
    for step_records in dense_capture["capture_records"]:
        for record in step_records:
            per_layer_decode_records.setdefault(record.layer_id, []).append(record)

    text_model = _qwen35_text_model(model)
    if text_model is None or not hasattr(text_model, "layers"):
        raise ValueError("Qwen3.5 attention-subset ablation requires text_model.layers")

    k_only_context_by_layer: dict[str, float] = {}
    v_only_context_by_layer: dict[str, float] = {}
    kv_context_by_layer: dict[str, float] = {}
    k_only_output_by_layer: dict[str, float] = {}
    v_only_output_by_layer: dict[str, float] = {}
    kv_output_by_layer: dict[str, float] = {}
    dominant_kind_by_layer: dict[str, str] = {}

    for layer_id in attention_layer_ids:
        layer_keys, layer_values = prefill_tensors[layer_id]
        dense_prefill_keys = np.asarray(layer_keys[0].detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)
        dense_prefill_values = np.asarray(layer_values[0].detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)
        quant_prefill_keys = _reconstruct_prefill_history(layer_keys, config=dotcache_config, kind="K", layer_id=layer_id)
        quant_prefill_values = _reconstruct_prefill_history(layer_values, config=dotcache_config, kind="V", layer_id=layer_id)

        attention_module = text_model.layers[layer_id].self_attn
        if hasattr(attention_module, "base_attention"):
            attention_module = attention_module.base_attention
        scaling = float(attention_module.scaling)
        output_weight = next(attention_module.o_proj.parameters())

        layer_k_only_context = 0.0
        layer_v_only_context = 0.0
        layer_kv_context = 0.0
        layer_k_only_output = 0.0
        layer_v_only_output = 0.0
        layer_kv_output = 0.0

        layer_records = per_layer_decode_records.get(layer_id, [])
        for step_index, record in enumerate(layer_records):
            if record.gate_states is None:
                raise ValueError("Qwen3.5 prefill ablation requires gate_states in the replay record")
            dense_key_history = _append_dense_decode_history(dense_prefill_keys, layer_records, kind="K", step_index=step_index)
            dense_value_history = _append_dense_decode_history(dense_prefill_values, layer_records, kind="V", step_index=step_index)
            quant_key_history = _append_dense_decode_history(quant_prefill_keys, layer_records, kind="K", step_index=step_index)
            quant_value_history = _append_dense_decode_history(quant_prefill_values, layer_records, kind="V", step_index=step_index)

            k_only_context = _replay_attention_subset_context(
                query_states=record.query_states,
                key_history=quant_key_history,
                value_history=dense_value_history,
                q_head_to_kv_head=q_head_to_kv_head,
                scaling=scaling,
            )
            v_only_context = _replay_attention_subset_context(
                query_states=record.query_states,
                key_history=dense_key_history,
                value_history=quant_value_history,
                q_head_to_kv_head=q_head_to_kv_head,
                scaling=scaling,
            )
            kv_context = _replay_attention_subset_context(
                query_states=record.query_states,
                key_history=quant_key_history,
                value_history=quant_value_history,
                q_head_to_kv_head=q_head_to_kv_head,
                scaling=scaling,
            )
            gate = 1.0 / (1.0 + np.exp(-record.gate_states.astype(np.float32, copy=False)))
            k_only_gated = (k_only_context * gate).astype(np.float32, copy=False)
            v_only_gated = (v_only_context * gate).astype(np.float32, copy=False)
            kv_gated = (kv_context * gate).astype(np.float32, copy=False)

            layer_k_only_context = max(layer_k_only_context, float(np.max(np.abs(k_only_gated - record.context_states))))
            layer_v_only_context = max(layer_v_only_context, float(np.max(np.abs(v_only_gated - record.context_states))))
            layer_kv_context = max(layer_kv_context, float(np.max(np.abs(kv_gated - record.context_states))))

            with torch.no_grad():
                k_only_output = attention_module.o_proj(
                    torch.as_tensor(k_only_gated, dtype=output_weight.dtype, device=output_weight.device).reshape(1, 1, -1)
                )[0, 0].detach().to(dtype=torch.float32).cpu().numpy()
                v_only_output = attention_module.o_proj(
                    torch.as_tensor(v_only_gated, dtype=output_weight.dtype, device=output_weight.device).reshape(1, 1, -1)
                )[0, 0].detach().to(dtype=torch.float32).cpu().numpy()
                kv_output = attention_module.o_proj(
                    torch.as_tensor(kv_gated, dtype=output_weight.dtype, device=output_weight.device).reshape(1, 1, -1)
                )[0, 0].detach().to(dtype=torch.float32).cpu().numpy()

            layer_k_only_output = max(layer_k_only_output, float(np.max(np.abs(k_only_output - record.output_states))))
            layer_v_only_output = max(layer_v_only_output, float(np.max(np.abs(v_only_output - record.output_states))))
            layer_kv_output = max(layer_kv_output, float(np.max(np.abs(kv_output - record.output_states))))

        layer_key = str(layer_id)
        k_only_context_by_layer[layer_key] = layer_k_only_context
        v_only_context_by_layer[layer_key] = layer_v_only_context
        kv_context_by_layer[layer_key] = layer_kv_context
        k_only_output_by_layer[layer_key] = layer_k_only_output
        v_only_output_by_layer[layer_key] = layer_v_only_output
        kv_output_by_layer[layer_key] = layer_kv_output
        if layer_k_only_context > layer_v_only_context * 1.1:
            dominant_kind_by_layer[layer_key] = "K"
        elif layer_v_only_context > layer_k_only_context * 1.1:
            dominant_kind_by_layer[layer_key] = "V"
        else:
            dominant_kind_by_layer[layer_key] = "mixed"

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "attention_subset_layer_ids": attention_layer_ids,
        "attention_subset_prefill_ablation_ready": True,
        "prefill_k_only_context_max_abs_error": max(k_only_context_by_layer.values(), default=0.0),
        "prefill_v_only_context_max_abs_error": max(v_only_context_by_layer.values(), default=0.0),
        "prefill_kv_context_max_abs_error": max(kv_context_by_layer.values(), default=0.0),
        "prefill_k_only_output_max_abs_error": max(k_only_output_by_layer.values(), default=0.0),
        "prefill_v_only_output_max_abs_error": max(v_only_output_by_layer.values(), default=0.0),
        "prefill_kv_output_max_abs_error": max(kv_output_by_layer.values(), default=0.0),
        "prefill_k_only_context_max_abs_error_by_layer": dict(sorted(k_only_context_by_layer.items())),
        "prefill_v_only_context_max_abs_error_by_layer": dict(sorted(v_only_context_by_layer.items())),
        "prefill_kv_context_max_abs_error_by_layer": dict(sorted(kv_context_by_layer.items())),
        "prefill_k_only_output_max_abs_error_by_layer": dict(sorted(k_only_output_by_layer.items())),
        "prefill_v_only_output_max_abs_error_by_layer": dict(sorted(v_only_output_by_layer.items())),
        "prefill_kv_output_max_abs_error_by_layer": dict(sorted(kv_output_by_layer.items())),
        "prefill_dominant_kind_by_layer": dict(sorted(dominant_kind_by_layer.items())),
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense_attention_subset_prefill_ablation",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_attention_subset_dotcache_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.clear()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    dense_capture = _run_qwen35_attention_subset_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )

    dotcache_prefill_outputs, dotcache_prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    adapter.clear()
    adapter.load_attention_subset_prefill_cache(dotcache_prefill_outputs.past_key_values)
    adapter.set_mode("dotcache_attention_subset")

    runtime_state = adapter.require_hybrid_dotcache_runtime_state()
    dotcache_step_logits: list[np.ndarray] = []
    dotcache_records: list[list[LlamaReplayRecord]] = []
    dotcache_decode_ms_total = 0.0
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    for step_index, decode_input_ids in enumerate(dense_capture["decode_inputs"]):
        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
        try:
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                ),
                device=input_ids.device,
            )
        finally:
            adapter.set_current_token_index(None)
        dotcache_decode_ms_total += step_ms
        dotcache_records.append(adapter.end_capture_step())
        dotcache_step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        runtime_state.advance(outputs.past_key_values)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    dense_record_map = {
        (record.step_index, record.layer_id): record
        for step_records in dense_capture["capture_records"]
        for record in step_records
    }
    dotcache_record_map = {
        (record.step_index, record.layer_id): record
        for step_records in dotcache_records
        for record in step_records
    }
    replay_context_max_abs = 0.0
    replay_context_max_rel = 0.0
    replay_output_max_abs = 0.0
    replay_output_max_rel = 0.0
    per_layer_context_max_abs: dict[str, float] = {}
    per_layer_output_max_abs: dict[str, float] = {}
    for replay_key, dense_record in dense_record_map.items():
        dotcache_record = dotcache_record_map.get(replay_key)
        if dotcache_record is None:
            raise ValueError(f"missing DotCache replay record for step/layer {replay_key}")
        context_delta = np.abs(dotcache_record.context_states - dense_record.context_states)
        context_denom = np.maximum(np.abs(dense_record.context_states), 1e-8)
        replay_context_max_abs = max(replay_context_max_abs, float(np.max(context_delta)))
        replay_context_max_rel = max(replay_context_max_rel, float(np.max(context_delta / context_denom)))
        layer_key = str(dense_record.layer_id)
        per_layer_context_max_abs[layer_key] = max(
            per_layer_context_max_abs.get(layer_key, 0.0),
            float(np.max(context_delta)),
        )
        output_delta = np.abs(dotcache_record.output_states - dense_record.output_states)
        output_denom = np.maximum(np.abs(dense_record.output_states), 1e-8)
        replay_output_max_abs = max(replay_output_max_abs, float(np.max(output_delta)))
        replay_output_max_rel = max(replay_output_max_rel, float(np.max(output_delta / output_denom)))
        per_layer_output_max_abs[layer_key] = max(
            per_layer_output_max_abs.get(layer_key, 0.0),
            float(np.max(output_delta)),
        )

    dense_logits = np.stack(dense_capture["step_logits"], axis=0) if dense_capture["step_logits"] else np.zeros((0, 1))
    dotcache_logits = np.stack(dotcache_step_logits, axis=0) if dotcache_step_logits else np.zeros((0, 1))
    if dense_logits.size == 0:
        teacher_forced_max_abs = 0.0
        teacher_forced_max_rel = 0.0
    else:
        logit_delta = np.abs(dotcache_logits - dense_logits)
        logit_denom = np.maximum(np.abs(dense_logits), 1e-8)
        teacher_forced_max_abs = float(np.max(logit_delta))
        teacher_forced_max_rel = float(np.max(logit_delta / logit_denom))

    result = _summarize_attention_subset_capture(
        adapter,
        input_ids=input_ids,
        decode_steps=decode_steps,
        prefill_ms=float(dense_capture["prefill_ms"]),
        dense_decode_ms_total=float(dense_capture["decode_ms_total"]),
        per_step_records=dense_capture["capture_records"],
    )
    result.update(
        {
            "dotcache_attention_subset_ready": True,
            "dotcache_ready": False,
            "runtime_mode": "dotcache_attention_subset",
            "dotcache_prefill_ms": float(dotcache_prefill_ms),
            "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "replay_context_max_abs_error": replay_context_max_abs,
            "replay_context_max_rel_error": replay_context_max_rel,
            "replay_output_max_abs_error": replay_output_max_abs,
            "replay_output_max_rel_error": replay_output_max_rel,
            "replay_context_max_abs_error_by_layer": dict(sorted(per_layer_context_max_abs.items())),
            "replay_output_max_abs_error_by_layer": dict(sorted(per_layer_output_max_abs.items())),
            "teacher_forced_logit_max_abs_error": teacher_forced_max_abs,
            "teacher_forced_logit_max_rel_error": teacher_forced_max_rel,
            "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
            "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
            "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
            "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
        }
    )
    result.update(runtime_state.summary())
    return result


__all__ = [
    "Qwen35AttentionSubsetDotCacheHarness",
    "Qwen35AttentionSubsetDotCacheModelAdapter",
    "Qwen35AttentionSubsetHarness",
    "Qwen35AttentionSubsetModelAdapter",
    "Qwen35TextHarness",
    "Qwen35TextModelAdapter",
    "inspect_qwen35_hybrid_state",
    "load_qwen35_text_only_from_pretrained",
    "run_qwen35_attention_subset_prefill_ablation_harness",
    "run_qwen35_attention_subset_dotcache_harness",
    "run_qwen35_attention_subset_replay_harness",
    "run_qwen35_text_generation_harness",
    "run_qwen35_text_loss_harness",
    "summarize_qwen35_dotcache_fit",
    "summarize_qwen35_hybrid_state",
    "summarize_qwen35_hybrid_state_growth",
]
