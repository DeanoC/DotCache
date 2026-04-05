from __future__ import annotations

import copy
import gc
import json
import math
import numpy as np
import os
import sys
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..config import DotCacheConfig
from ..decode_reference import decode_page
from ..encode import encode_page
from ..model_kv_cache import ModelPagedKVCache, PreparedPageCache
from ..page_oracle import PageTraceRecord, save_page_trace
from ..state_cache_sim import StateAblationResult, StateLayerRecord, StateTileSpec, simulate_state_codec
from ..tracing import ExecutionTrace
from .llama import (
    _begin_cuda_memory_region,
    _end_cuda_memory_region,
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
    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # pragma: no cover - optional dependency surface
        BitsAndBytesConfig = None  # type: ignore[assignment]
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    AutoTokenizer = None
    Qwen3_5ForConditionalGeneration = None
    BitsAndBytesConfig = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    qwen35_mod = None


Qwen35Mode = Literal["dense", "dotcache_attention_subset"]
Qwen35DeltaNetStateCacheStage = Literal["readout_only_m0", "post_update_m0"]
Qwen35DeltaNetStateCacheMode = Literal["M0", "M3"]
Qwen35DeltaNetStateCacheScope = Literal["recurrent_only", "conv_only", "conv_plus_recurrent"]
Qwen35DeltaNetStateCacheReadoutPolicy = Literal["890m_context_banded_v1"]
Qwen35DeltaNetStateCacheRecurrentModePolicy = Literal["890m_m3_outlier_pair_midband_v1"]
Qwen35DeltaNetStateCacheRecurrentGroupSizePolicy = Literal["890m_long_horizon_group_escape_v1"]
Qwen35DeltaNetStateCacheReadoutModePolicy = Literal["890m_m3_outlier_pair_midband_v1"]

_VALID_QWEN35_DELTANET_STATECACHE_MODES: tuple[str, ...] = ("M0", "M3")
_VALID_QWEN35_DELTANET_STATECACHE_SCOPES: tuple[str, ...] = ("recurrent_only", "conv_only", "conv_plus_recurrent")
_VALID_QWEN35_DELTANET_STATECACHE_READOUT_POLICIES: tuple[str, ...] = ("890m_context_banded_v1",)
_VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_MODE_POLICIES: tuple[str, ...] = ("890m_m3_outlier_pair_midband_v1",)
_VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_GROUP_SIZE_POLICIES: tuple[str, ...] = ("890m_long_horizon_group_escape_v1",)
_VALID_QWEN35_DELTANET_STATECACHE_READOUT_MODE_POLICIES: tuple[str, ...] = ("890m_m3_outlier_pair_midband_v1",)

_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_LATE: dict[int, int] = {12: 2, 18: 2, 21: 2}
_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_FULL: dict[int, int] = {5: 2, 8: 2, 12: 2, 18: 2, 21: 2}
_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_EARLY: dict[int, int] = {5: 2, 8: 2}
_QWEN35_890M_M3_OUTLIER_PAIR_OVERRIDES: dict[int, Qwen35DeltaNetStateCacheMode] = {4: "M3", 20: "M3"}
_QWEN35_890M_LONG_HORIZON_GROUP_ESCAPE_OVERRIDES: dict[int, int] = {18: 8, 20: 8}


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


def resolve_qwen35_deltanet_statecache_readout_policy(
    *,
    prompt_length: int,
    policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None,
) -> tuple[dict[int, int], str | None]:
    if policy is None:
        return {}, None
    policy_name = str(policy)
    if policy_name not in _VALID_QWEN35_DELTANET_STATECACHE_READOUT_POLICIES:
        raise ValueError(
            "unknown Qwen3.5 DeltaNet StateCache readout policy "
            f"{policy_name!r}; expected one of {_VALID_QWEN35_DELTANET_STATECACHE_READOUT_POLICIES}"
        )
    if int(prompt_length) <= 0:
        raise ValueError("prompt_length must be positive when resolving a readout policy")
    if policy_name == "890m_context_banded_v1":
        if int(prompt_length) < 1024:
            return {}, "baseline"
        if int(prompt_length) <= 2048:
            return dict(_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_LATE), "late"
        if int(prompt_length) <= 4096:
            return dict(_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_FULL), "full"
        return dict(_QWEN35_890M_CONTEXT_BANDED_READOUT_RENORM_EARLY), "early"
    raise AssertionError(f"unreachable readout policy selector branch for {policy_name!r}")


def resolve_qwen35_deltanet_statecache_readout_mode_policy(
    *,
    prompt_length: int,
    policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None,
) -> tuple[dict[int, Qwen35DeltaNetStateCacheMode], str | None]:
    if policy is None:
        return {}, None
    policy_name = str(policy)
    if policy_name not in _VALID_QWEN35_DELTANET_STATECACHE_READOUT_MODE_POLICIES:
        raise ValueError(
            "unknown Qwen3.5 DeltaNet StateCache readout mode policy "
            f"{policy_name!r}; expected one of {_VALID_QWEN35_DELTANET_STATECACHE_READOUT_MODE_POLICIES}"
        )
    if int(prompt_length) <= 0:
        raise ValueError("prompt_length must be positive when resolving a readout mode policy")
    if policy_name == "890m_m3_outlier_pair_midband_v1":
        if 3072 <= int(prompt_length) <= 4096:
            return dict(_QWEN35_890M_M3_OUTLIER_PAIR_OVERRIDES), "midband_outliers"
        return {}, "baseline"
    raise AssertionError(f"unreachable readout mode policy selector branch for {policy_name!r}")


def resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
    *,
    prompt_length: int,
    policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None,
) -> tuple[dict[int, Qwen35DeltaNetStateCacheMode], str | None]:
    if policy is None:
        return {}, None
    policy_name = str(policy)
    if policy_name not in _VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_MODE_POLICIES:
        raise ValueError(
            "unknown Qwen3.5 DeltaNet StateCache recurrent mode policy "
            f"{policy_name!r}; expected one of {_VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_MODE_POLICIES}"
        )
    if int(prompt_length) <= 0:
        raise ValueError("prompt_length must be positive when resolving a recurrent mode policy")
    if policy_name == "890m_m3_outlier_pair_midband_v1":
        if 3072 <= int(prompt_length) <= 4096:
            return dict(_QWEN35_890M_M3_OUTLIER_PAIR_OVERRIDES), "midband_outliers"
        return {}, "baseline"
    raise AssertionError(f"unreachable recurrent mode policy selector branch for {policy_name!r}")


def resolve_qwen35_deltanet_statecache_recurrent_group_size_policy(
    *,
    prompt_length: int,
    decode_steps: int,
    policy: Qwen35DeltaNetStateCacheRecurrentGroupSizePolicy | str | None,
) -> tuple[dict[int, int], str | None]:
    if policy is None:
        return {}, None
    policy_name = str(policy)
    if policy_name not in _VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_GROUP_SIZE_POLICIES:
        raise ValueError(
            "unknown Qwen3.5 DeltaNet StateCache recurrent group-size policy "
            f"{policy_name!r}; expected one of {_VALID_QWEN35_DELTANET_STATECACHE_RECURRENT_GROUP_SIZE_POLICIES}"
        )
    if int(prompt_length) <= 0:
        raise ValueError("prompt_length must be positive when resolving a recurrent group-size policy")
    if int(decode_steps) <= 0:
        raise ValueError("decode_steps must be positive when resolving a recurrent group-size policy")
    if policy_name == "890m_long_horizon_group_escape_v1":
        if int(prompt_length) >= 6144 and int(decode_steps) >= 16:
            return dict(_QWEN35_890M_LONG_HORIZON_GROUP_ESCAPE_OVERRIDES), "long_horizon"
        return {}, "baseline"
    raise AssertionError(f"unreachable recurrent group-size policy selector branch for {policy_name!r}")


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
        _maybe_wrap_qwen35_rocm_float32_fast_path(linear_attn)
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


def _maybe_wrap_qwen35_rocm_float32_fast_path(linear_attn: Any) -> None:
    if torch is None or not bool(getattr(torch.version, "hip", None)):
        return
    for attr_name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        kernel = getattr(linear_attn, attr_name, None)
        if kernel is None or getattr(kernel, "_dotcache_rocm_float32_fast_path_wrapped", False):
            continue

        def _wrapped(q, k, v, *args, _kernel=kernel, **kwargs):
            input_dtype = q.dtype
            if input_dtype == torch.float32:
                out, state = _kernel(
                    q.to(torch.float16),
                    k.to(torch.float16),
                    v.to(torch.float16),
                    *args,
                    **kwargs,
                )
                return out.to(input_dtype), state
            return _kernel(q, k, v, *args, **kwargs)

        setattr(_wrapped, "_dotcache_rocm_float32_fast_path_wrapped", True)
        setattr(linear_attn, attr_name, _wrapped)


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
    extracted: dict[int, tuple[Any, Any]] = {}
    for layer_id in layer_ids:
        layer_keys = _cache_component_value(cache, "key_cache", layer_id)
        layer_values = _cache_component_value(cache, "value_cache", layer_id)
        if layer_keys is None or layer_values is None:
            raise ValueError(f"Qwen3.5 attention layer {layer_id} is missing prefill KV tensors")
        extracted[int(layer_id)] = (layer_keys, layer_values)
    return extracted


def _clone_attention_subset_prefill_tensors(prefill_tensors: dict[int, tuple[Any, Any]]) -> dict[int, tuple[Any, Any]]:
    cloned: dict[int, tuple[Any, Any]] = {}
    for layer_id, (layer_keys, layer_values) in prefill_tensors.items():
        if torch is not None and torch.is_tensor(layer_keys) and torch.is_tensor(layer_values):
            cloned[int(layer_id)] = (layer_keys.detach().clone(), layer_values.detach().clone())
        else:
            cloned[int(layer_id)] = (
                np.asarray(layer_keys, dtype=np.float32).copy(),
                np.asarray(layer_values, dtype=np.float32).copy(),
            )
    return cloned


def _tensor_to_float32_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return np.asarray(value.detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _replace_attention_subset_cache_with_placeholders(cache: Any, layer_ids: list[int]) -> None:
    for layer_id in layer_ids:
        layer_keys = _cache_component_value(cache, "key_cache", layer_id)
        layer_values = _cache_component_value(cache, "value_cache", layer_id)
        if layer_keys is None or layer_values is None:
            continue
        key_placeholder = layer_keys[:, :, :0, :].contiguous()
        value_placeholder = layer_values[:, :, :0, :].contiguous()
        if not _set_cache_component_value(cache, "key_cache", layer_id, key_placeholder):
            raise ValueError(f"Qwen3.5 attention layer {layer_id} key cache is not mutable")
        if not _set_cache_component_value(cache, "value_cache", layer_id, value_placeholder):
            raise ValueError(f"Qwen3.5 attention layer {layer_id} value cache is not mutable")


def _advance_attention_subset_cache_placeholder(cache: Any, layer_id: int) -> None:
    layer_keys = _cache_component_value(cache, "key_cache", layer_id)
    layer_values = _cache_component_value(cache, "value_cache", layer_id)
    if layer_keys is None or layer_values is None:
        return
    if layer_keys.shape[2] == 0 or layer_values.shape[2] == 0:
        return
    if not _set_cache_component_value(cache, "key_cache", layer_id, torch.cat([layer_keys, layer_keys[:, :, :1, :]], dim=2)):
        return
    _set_cache_component_value(cache, "value_cache", layer_id, torch.cat([layer_values, layer_values[:, :, :1, :]], dim=2))


def _clone_qwen35_past_key_values(cache: Any) -> Any:
    return copy.deepcopy(cache)


def _default_q_head_to_kv_head(num_attention_heads: int, num_key_value_heads: int) -> np.ndarray:
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    heads_per_kv = num_attention_heads // num_key_value_heads
    return (np.arange(num_attention_heads, dtype=np.int32) // heads_per_kv).astype(np.int32, copy=False)


def _qwen35_mps_serving_shortlist_heuristic(
    config: DotCacheConfig,
    *,
    backend: str,
    prompt_length: int,
) -> tuple[DotCacheConfig, bool]:
    if backend != "torch_mps":
        return config, False
    if int(prompt_length) < 4096:
        return config, False
    if config.execution_shortlist_enabled():
        return config, False
    updated = replace(
        config,
        execution_recent_window=1024,
        execution_sink_window=256,
        execution_relevance_top_k=4,
        execution_relevance_mode="envelope",
        execution_relevance_top_k_context_overrides=("layer:23:min_ctx:8192=8",),
    )
    return updated, True


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
        "layers",
        "key_cache",
        "value_cache",
        "kv_states",
        "keys",
        "values",
        "conv_states",
        "conv_state",
        "ssm_states",
        "recurrent_states",
        "recurrent_state",
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
    value = _cache_component_value(cache, attr_name, layer_id)
    if value is None:
        return 0
    return _hybrid_cache_nbytes(value)


def _cache_component_value(cache: Any, attr_name: str, layer_id: int) -> Any | None:
    values = getattr(cache, attr_name, None)
    if values is not None:
        try:
            return values[layer_id]
        except Exception:
            pass
    layers = getattr(cache, "layers", None)
    if layers is not None:
        try:
            layer = layers[layer_id]
        except Exception:
            layer = None
        if layer is not None:
            for layer_attr_name in _cache_component_layer_attr_names(attr_name):
                if hasattr(layer, layer_attr_name):
                    return getattr(layer, layer_attr_name)
    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        try:
            legacy = to_legacy()
        except Exception:
            legacy = None
        if legacy is not None and legacy is not cache:
            return _cache_component_value(legacy, attr_name, layer_id)
    return None


def _set_cache_component_value(cache: Any, attr_name: str, layer_id: int, value: Any) -> bool:
    values = getattr(cache, attr_name, None)
    if values is not None:
        try:
            values[layer_id] = value
            return True
        except Exception:
            pass
    layers = getattr(cache, "layers", None)
    if layers is not None:
        try:
            layer = layers[layer_id]
        except Exception:
            layer = None
        if layer is not None:
            for layer_attr_name in _cache_component_layer_attr_names(attr_name):
                if hasattr(layer, layer_attr_name):
                    setattr(layer, layer_attr_name, value)
                    return True
    return False


def _cache_component_layer_attr_names(attr_name: str) -> tuple[str, ...]:
    if attr_name == "key_cache":
        return ("keys", "key_cache")
    if attr_name == "value_cache":
        return ("values", "value_cache")
    if attr_name == "conv_states":
        return ("conv_states", "conv_state")
    if attr_name == "recurrent_states":
        return ("recurrent_states", "recurrent_state")
    return (attr_name,)


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
        result.update(self.model_kv_cache.execution_shortlist_summary())
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


@dataclass(slots=True)
class Qwen35DeltaNetStateRecord:
    step_index: int
    layer_id: int
    token_index: int
    hidden_states: torch.Tensor
    output_states: torch.Tensor
    pre_conv_state: torch.Tensor | None
    post_conv_state: torch.Tensor | None
    pre_recurrent_state: torch.Tensor | None
    post_recurrent_state: torch.Tensor | None


@dataclass(slots=True)
class _Qwen35DeltaNetCacheStub:
    layer_count: int
    target_layer_id: int
    conv_state: torch.Tensor | None
    recurrent_state: torch.Tensor | None
    has_previous_state: bool = True
    conv_states: list[torch.Tensor | None] = field(init=False)
    recurrent_states: list[torch.Tensor | None] = field(init=False)

    def __post_init__(self) -> None:
        self.conv_states = [None] * self.layer_count
        self.recurrent_states = [None] * self.layer_count
        self.conv_states[self.target_layer_id] = self.conv_state
        self.recurrent_states[self.target_layer_id] = self.recurrent_state


def _clone_state_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to(dtype=torch.float32).cpu().clone()


def _tensor_rms(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    value = tensor.detach().to(dtype=torch.float32)
    return float(torch.sqrt(torch.mean(value * value)).item())


def _tensor_max_abs(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().to(dtype=torch.float32).abs().max().item())


def _max_abs_error(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None or rhs is None:
        return 0.0
    return float((lhs.detach().to(dtype=torch.float32) - rhs.detach().to(dtype=torch.float32)).abs().max().item())


def _max_rel_error(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None or rhs is None:
        return 0.0
    lhs_f = lhs.detach().to(dtype=torch.float32)
    rhs_f = rhs.detach().to(dtype=torch.float32)
    denom = torch.maximum(rhs_f.abs(), torch.full_like(rhs_f, 1e-8))
    return float(((lhs_f - rhs_f).abs() / denom).max().item())


def _quantize_state_tensor(
    tensor: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if mode == "M3":
        return tensor.detach().clone()
    if torch is not None:
        value = tensor.detach().to(dtype=torch.float32)
        flat = value.reshape(-1, value.shape[-1])
        cols = int(flat.shape[-1])
        effective_group_size = int(max(min(group_size, cols), 1))
        group_count = int(math.ceil(cols / effective_group_size))
        padded_cols = int(group_count * effective_group_size)
        pad_cols = padded_cols - cols

        if pad_cols > 0:
            padded = torch.nn.functional.pad(flat, (0, pad_cols), mode="constant", value=0.0)
            valid_mask = torch.ones((flat.shape[0], cols), dtype=torch.bool, device=flat.device)
            valid_mask = torch.nn.functional.pad(valid_mask, (0, pad_cols), mode="constant", value=False)
        else:
            padded = flat
            valid_mask = torch.ones_like(flat, dtype=torch.bool)

        grouped = padded.reshape(flat.shape[0], group_count, effective_group_size)
        grouped_mask = valid_mask.reshape(flat.shape[0], group_count, effective_group_size)
        group_min = torch.where(grouped_mask, grouped, torch.full_like(grouped, float("inf"))).amin(dim=-1, keepdim=True)
        group_max = torch.where(grouped_mask, grouped, torch.full_like(grouped, float("-inf"))).amax(dim=-1, keepdim=True)
        levels = max((1 << int(bits)) - 1, 1)
        scale = (group_max - group_min) / float(levels)
        is_constant = (group_max - group_min).abs() <= 1e-8
        safe_scale = torch.where(is_constant, torch.ones_like(scale), scale)
        quantized = torch.round((grouped - group_min) / safe_scale).clamp_(0, levels)
        decoded = quantized * safe_scale + group_min
        decoded = torch.where(is_constant.expand_as(decoded), group_min.expand_as(decoded), decoded)
        decoded = torch.where(grouped_mask, decoded, torch.zeros_like(decoded))
        decoded = decoded.reshape(flat.shape[0], padded_cols)[..., :cols]
        return decoded.reshape(value.shape).to(dtype=tensor.dtype)
    spec = StateTileSpec(
        state_rows=int(np.prod(tensor.shape[:-1])) if tensor.ndim > 1 else 1,
        state_cols=int(tensor.shape[-1]),
        group_size=int(max(min(group_size, int(tensor.shape[-1])), 1)),
        bits=int(bits),
        mode="M0",
    )
    decoded, _, _ = simulate_state_codec(tensor.detach().to(dtype=torch.float32).cpu().numpy(), spec)
    return torch.as_tensor(decoded, dtype=tensor.dtype, device=tensor.device)


def _build_state_quantization_telemetry_record(
    tensor: torch.Tensor | None,
    decoded_tensor: torch.Tensor | None,
    *,
    layer_id: int,
    state_family: str,
    phase: str,
    step_index: int | None,
    bits: int,
    group_size: int,
    mode: str,
    renorm_applied: bool,
) -> dict[str, Any] | None:
    if tensor is None or decoded_tensor is None:
        return None
    value = tensor.detach().to(dtype=torch.float32)
    decoded_value = decoded_tensor.detach().to(dtype=torch.float32)
    flat = value.reshape(-1, value.shape[-1])
    decoded_flat = decoded_value.reshape(-1, decoded_value.shape[-1])
    cols = int(flat.shape[-1])
    effective_group_size = int(max(min(group_size, cols), 1))
    group_count = int(math.ceil(cols / effective_group_size))
    padded_cols = int(group_count * effective_group_size)
    pad_cols = padded_cols - cols

    if pad_cols > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_cols), mode="constant", value=0.0)
        decoded_padded = torch.nn.functional.pad(decoded_flat, (0, pad_cols), mode="constant", value=0.0)
        valid_mask = torch.ones((flat.shape[0], cols), dtype=torch.bool, device=flat.device)
        valid_mask = torch.nn.functional.pad(valid_mask, (0, pad_cols), mode="constant", value=False)
    else:
        padded = flat
        decoded_padded = decoded_flat
        valid_mask = torch.ones_like(flat, dtype=torch.bool)

    grouped = padded.reshape(flat.shape[0], group_count, effective_group_size)
    decoded_grouped = decoded_padded.reshape(flat.shape[0], group_count, effective_group_size)
    grouped_mask = valid_mask.reshape(flat.shape[0], group_count, effective_group_size)
    valid_values = int(grouped_mask.sum().item())
    if valid_values <= 0:
        return None

    error = torch.where(grouped_mask, (decoded_grouped - grouped).abs(), torch.zeros_like(grouped))
    valid_count_by_group = grouped_mask.sum(dim=(0, 2)).to(dtype=torch.float32)
    error_sum_by_group = error.sum(dim=(0, 2))
    group_mean_abs_error = torch.where(
        valid_count_by_group > 0,
        error_sum_by_group / torch.clamp(valid_count_by_group, min=1.0),
        torch.zeros_like(valid_count_by_group),
    )

    if mode == "M3":
        zero_group = [0.0] * group_count
        zero_value = 0.0
        return {
            "layer_id": int(layer_id),
            "state_family": str(state_family),
            "phase": str(phase),
            "step_index": None if step_index is None else int(step_index),
            "bits": int(bits),
            "group_size": int(effective_group_size),
            "group_count": int(group_count),
            "row_count": int(flat.shape[0]),
            "mode": str(mode),
            "renorm_applied": bool(renorm_applied),
            "input_rms": _tensor_rms(value),
            "input_max_abs": _tensor_max_abs(value),
            "decoded_rms": _tensor_rms(decoded_value),
            "error_mean_abs": float(error.sum().item() / max(valid_values, 1)),
            "error_max_abs": float(error.max().item()),
            "scale_min": zero_value,
            "scale_mean": zero_value,
            "scale_max": zero_value,
            "constant_group_fraction": zero_value,
            "lower_edge_code_fraction": zero_value,
            "upper_edge_code_fraction": zero_value,
            "edge_code_fraction": zero_value,
            "group_mean_abs_error_by_group": [float(v) for v in group_mean_abs_error.detach().cpu().tolist()],
            "group_edge_code_fraction_by_group": list(zero_group),
            "group_scale_mean_by_group": list(zero_group),
        }

    group_min = torch.where(grouped_mask, grouped, torch.full_like(grouped, float("inf"))).amin(dim=-1, keepdim=True)
    group_max = torch.where(grouped_mask, grouped, torch.full_like(grouped, float("-inf"))).amax(dim=-1, keepdim=True)
    levels = max((1 << int(bits)) - 1, 1)
    raw_scale = (group_max - group_min) / float(levels)
    is_constant = (group_max - group_min).abs() <= 1e-8
    safe_scale = torch.where(is_constant, torch.ones_like(raw_scale), raw_scale)
    quantized = torch.round((grouped - group_min) / safe_scale).clamp_(0, levels)
    quantized = torch.where(grouped_mask, quantized, torch.zeros_like(quantized))
    lower_edge_mask = grouped_mask & quantized.eq(0)
    upper_edge_mask = grouped_mask & quantized.eq(levels)
    edge_mask = lower_edge_mask | upper_edge_mask
    lower_edge_count = int(lower_edge_mask.sum().item())
    upper_edge_count = int(upper_edge_mask.sum().item())
    edge_count = int(edge_mask.sum().item())

    edge_count_by_group = edge_mask.sum(dim=(0, 2)).to(dtype=torch.float32)
    group_edge_fraction = torch.where(
        valid_count_by_group > 0,
        edge_count_by_group / torch.clamp(valid_count_by_group, min=1.0),
        torch.zeros_like(valid_count_by_group),
    )
    scale_values = raw_scale.squeeze(-1)
    scale_sum_by_group = torch.where(is_constant.squeeze(-1), torch.zeros_like(scale_values), scale_values).sum(dim=0)
    non_constant_count_by_group = (~is_constant.squeeze(-1)).sum(dim=0).to(dtype=torch.float32)
    group_scale_mean = torch.where(
        non_constant_count_by_group > 0,
        scale_sum_by_group / torch.clamp(non_constant_count_by_group, min=1.0),
        torch.zeros_like(scale_sum_by_group),
    )
    non_constant_scales = scale_values[~is_constant.squeeze(-1)]
    if int(non_constant_scales.numel()) > 0:
        scale_min = float(non_constant_scales.min().item())
        scale_mean = float(non_constant_scales.mean().item())
        scale_max = float(non_constant_scales.max().item())
    else:
        scale_min = 0.0
        scale_mean = 0.0
        scale_max = 0.0

    return {
        "layer_id": int(layer_id),
        "state_family": str(state_family),
        "phase": str(phase),
        "step_index": None if step_index is None else int(step_index),
        "bits": int(bits),
        "group_size": int(effective_group_size),
        "group_count": int(group_count),
        "row_count": int(flat.shape[0]),
        "mode": str(mode),
        "renorm_applied": bool(renorm_applied),
        "input_rms": _tensor_rms(value),
        "input_max_abs": _tensor_max_abs(value),
        "decoded_rms": _tensor_rms(decoded_value),
        "error_mean_abs": float(error.sum().item() / max(valid_values, 1)),
        "error_max_abs": float(error.max().item()),
        "scale_min": float(scale_min),
        "scale_mean": float(scale_mean),
        "scale_max": float(scale_max),
        "constant_group_fraction": float(is_constant.sum().item() / max(is_constant.numel(), 1)),
        "lower_edge_code_fraction": float(lower_edge_count / max(valid_values, 1)),
        "upper_edge_code_fraction": float(upper_edge_count / max(valid_values, 1)),
        "edge_code_fraction": float(edge_count / max(valid_values, 1)),
        "group_mean_abs_error_by_group": [float(v) for v in group_mean_abs_error.detach().cpu().tolist()],
        "group_edge_code_fraction_by_group": [float(v) for v in group_edge_fraction.detach().cpu().tolist()],
        "group_scale_mean_by_group": [float(v) for v in group_scale_mean.detach().cpu().tolist()],
    }


def _summarize_state_quantization_telemetry_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    per_layer_phase: dict[str, dict[str, Any]] = {}
    for record in records:
        layer_id = int(record["layer_id"])
        state_family = str(record["state_family"])
        phase = str(record["phase"])
        key = f"{state_family}:{layer_id}:{phase}"
        group_error = [float(v) for v in record.get("group_mean_abs_error_by_group", [])]
        group_edge = [float(v) for v in record.get("group_edge_code_fraction_by_group", [])]
        group_scale = [float(v) for v in record.get("group_scale_mean_by_group", [])]
        summary = per_layer_phase.setdefault(
            key,
            {
                "layer_id": layer_id,
                "state_family": state_family,
                "phase": phase,
                "event_count": 0,
                "renorm_applied_count": 0,
                "modes": set(),
                "bits": set(),
                "group_size": int(record["group_size"]),
                "group_count": int(record["group_count"]),
                "step_indices": [],
                "input_rms_sum": 0.0,
                "input_max_abs_max": 0.0,
                "decoded_rms_sum": 0.0,
                "error_mean_abs_sum": 0.0,
                "error_max_abs_max": 0.0,
                "scale_min_min": None,
                "scale_mean_sum": 0.0,
                "scale_max_max": 0.0,
                "constant_group_fraction_sum": 0.0,
                "lower_edge_code_fraction_sum": 0.0,
                "upper_edge_code_fraction_sum": 0.0,
                "edge_code_fraction_sum": 0.0,
                "group_mean_abs_error_sum": [0.0] * len(group_error),
                "group_edge_code_fraction_sum": [0.0] * len(group_edge),
                "group_scale_mean_sum": [0.0] * len(group_scale),
            },
        )
        summary["event_count"] += 1
        summary["renorm_applied_count"] += int(bool(record.get("renorm_applied", False)))
        summary["modes"].add(str(record["mode"]))
        summary["bits"].add(int(record["bits"]))
        if record.get("step_index") is not None:
            summary["step_indices"].append(int(record["step_index"]))
        summary["input_rms_sum"] += float(record["input_rms"])
        summary["input_max_abs_max"] = max(float(summary["input_max_abs_max"]), float(record["input_max_abs"]))
        summary["decoded_rms_sum"] += float(record["decoded_rms"])
        summary["error_mean_abs_sum"] += float(record["error_mean_abs"])
        summary["error_max_abs_max"] = max(float(summary["error_max_abs_max"]), float(record["error_max_abs"]))
        scale_min = float(record["scale_min"])
        summary["scale_min_min"] = scale_min if summary["scale_min_min"] is None else min(float(summary["scale_min_min"]), scale_min)
        summary["scale_mean_sum"] += float(record["scale_mean"])
        summary["scale_max_max"] = max(float(summary["scale_max_max"]), float(record["scale_max"]))
        summary["constant_group_fraction_sum"] += float(record["constant_group_fraction"])
        summary["lower_edge_code_fraction_sum"] += float(record["lower_edge_code_fraction"])
        summary["upper_edge_code_fraction_sum"] += float(record["upper_edge_code_fraction"])
        summary["edge_code_fraction_sum"] += float(record["edge_code_fraction"])
        for index, value in enumerate(group_error):
            summary["group_mean_abs_error_sum"][index] += float(value)
        for index, value in enumerate(group_edge):
            summary["group_edge_code_fraction_sum"][index] += float(value)
        for index, value in enumerate(group_scale):
            summary["group_scale_mean_sum"][index] += float(value)

    summarized: dict[str, Any] = {}
    for key, summary in sorted(per_layer_phase.items()):
        event_count = int(summary["event_count"])
        group_error_mean = [float(value / max(event_count, 1)) for value in summary["group_mean_abs_error_sum"]]
        group_edge_mean = [float(value / max(event_count, 1)) for value in summary["group_edge_code_fraction_sum"]]
        group_scale_mean = [float(value / max(event_count, 1)) for value in summary["group_scale_mean_sum"]]

        def _top_groups(values: list[float]) -> list[dict[str, Any]]:
            ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
            return [
                {"group_index": int(group_index), "value": float(value)}
                for group_index, value in ranked[: min(len(ranked), 5)]
            ]

        summarized[key] = {
            "layer_id": int(summary["layer_id"]),
            "state_family": str(summary["state_family"]),
            "phase": str(summary["phase"]),
            "event_count": event_count,
            "renorm_applied_count": int(summary["renorm_applied_count"]),
            "step_indices": sorted(set(int(step) for step in summary["step_indices"])),
            "modes": sorted(str(mode) for mode in summary["modes"]),
            "bits": sorted(int(bits) for bits in summary["bits"]),
            "group_size": int(summary["group_size"]),
            "group_count": int(summary["group_count"]),
            "input_rms_mean": float(summary["input_rms_sum"] / max(event_count, 1)),
            "input_max_abs_max": float(summary["input_max_abs_max"]),
            "decoded_rms_mean": float(summary["decoded_rms_sum"] / max(event_count, 1)),
            "error_mean_abs_mean": float(summary["error_mean_abs_sum"] / max(event_count, 1)),
            "error_max_abs_max": float(summary["error_max_abs_max"]),
            "scale_min_min": float(summary["scale_min_min"] or 0.0),
            "scale_mean_mean": float(summary["scale_mean_sum"] / max(event_count, 1)),
            "scale_max_max": float(summary["scale_max_max"]),
            "constant_group_fraction_mean": float(summary["constant_group_fraction_sum"] / max(event_count, 1)),
            "lower_edge_code_fraction_mean": float(summary["lower_edge_code_fraction_sum"] / max(event_count, 1)),
            "upper_edge_code_fraction_mean": float(summary["upper_edge_code_fraction_sum"] / max(event_count, 1)),
            "edge_code_fraction_mean": float(summary["edge_code_fraction_sum"] / max(event_count, 1)),
            "top_groups_by_mean_abs_error": _top_groups(group_error_mean),
            "top_groups_by_edge_code_fraction": _top_groups(group_edge_mean),
            "top_groups_by_scale_mean": _top_groups(group_scale_mean),
        }
    return summarized


def parse_qwen35_deltanet_statecache_mode_overrides(
    overrides: list[str] | tuple[str, ...] | None,
) -> dict[int, Qwen35DeltaNetStateCacheMode]:
    parsed: dict[int, Qwen35DeltaNetStateCacheMode] = {}
    if overrides is None:
        return parsed
    for spec in overrides:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError("state_mode_overrides entries must use layer:<id>=<mode>")
        target, mode = raw.split("=", 1)
        parts = target.strip().split(":")
        if len(parts) != 2 or parts[0] != "layer":
            raise ValueError("state_mode_overrides entries must use layer:<id>=<mode>")
        resolved_mode = mode.strip().upper()
        if resolved_mode not in _VALID_QWEN35_DELTANET_STATECACHE_MODES:
            allowed = ", ".join(_VALID_QWEN35_DELTANET_STATECACHE_MODES)
            raise ValueError(f"state_mode_overrides mode must be one of {allowed}")
        parsed[int(parts[1])] = resolved_mode  # type: ignore[assignment]
    return parsed


def parse_qwen35_deltanet_statecache_renorm_overrides(
    overrides: list[str] | tuple[str, ...] | None,
) -> dict[int, int]:
    parsed: dict[int, int] = {}
    if overrides is None:
        return parsed
    for spec in overrides:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError("state_renorm_overrides entries must use layer:<id>=<interval>")
        target, interval_text = raw.split("=", 1)
        parts = target.strip().split(":")
        if len(parts) != 2 or parts[0] != "layer":
            raise ValueError("state_renorm_overrides entries must use layer:<id>=<interval>")
        interval = int(interval_text.strip())
        if interval < 0:
            raise ValueError("state_renorm_overrides interval must be >= 0")
        parsed[int(parts[1])] = interval
    return parsed


def parse_qwen35_deltanet_statecache_int_overrides(
    overrides: list[str] | tuple[str, ...] | None,
    *,
    value_name: str,
    minimum: int = 0,
) -> dict[int, int]:
    parsed: dict[int, int] = {}
    if overrides is None:
        return parsed
    for spec in overrides:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"{value_name}_overrides entries must use layer:<id>=<{value_name}>")
        target, value_text = raw.split("=", 1)
        parts = target.strip().split(":")
        if len(parts) != 2 or parts[0] != "layer":
            raise ValueError(f"{value_name}_overrides entries must use layer:<id>=<{value_name}>")
        value = int(value_text.strip())
        if value < int(minimum):
            raise ValueError(f"{value_name}_overrides {value_name} must be >= {minimum}")
        parsed[int(parts[1])] = value
    return parsed


def _resolve_qwen35_deltanet_statecache_scope(
    scope: Qwen35DeltaNetStateCacheScope | str,
) -> Qwen35DeltaNetStateCacheScope:
    resolved = str(scope)
    if resolved not in _VALID_QWEN35_DELTANET_STATECACHE_SCOPES:
        allowed = ", ".join(_VALID_QWEN35_DELTANET_STATECACHE_SCOPES)
        raise ValueError(f"Qwen3.5 DeltaNet StateCache scope must be one of {allowed}")
    return resolved  # type: ignore[return-value]


def _statecache_scope_includes_recurrent(scope: Qwen35DeltaNetStateCacheScope | str) -> bool:
    resolved = _resolve_qwen35_deltanet_statecache_scope(scope)
    return resolved in {"recurrent_only", "conv_plus_recurrent"}


def _statecache_scope_includes_conv(scope: Qwen35DeltaNetStateCacheScope | str) -> bool:
    resolved = _resolve_qwen35_deltanet_statecache_scope(scope)
    return resolved in {"conv_only", "conv_plus_recurrent"}


def _resolve_qwen35_deltanet_statecache_mode(
    layer_id: int,
    *,
    default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
) -> Qwen35DeltaNetStateCacheMode:
    if mode_overrides is None:
        return default_mode
    resolved = mode_overrides.get(int(layer_id), default_mode)
    if resolved not in _VALID_QWEN35_DELTANET_STATECACHE_MODES:
        allowed = ", ".join(_VALID_QWEN35_DELTANET_STATECACHE_MODES)
        raise ValueError(f"Qwen3.5 DeltaNet StateCache mode must be one of {allowed}")
    return resolved


def _resolve_qwen35_deltanet_statecache_renorm_interval(
    layer_id: int,
    *,
    default_interval: int = 0,
    interval_overrides: dict[int, int] | None = None,
) -> int:
    if interval_overrides is None:
        return int(default_interval)
    return int(interval_overrides.get(int(layer_id), int(default_interval)))


def _should_apply_qwen35_deltanet_statecache_renorm(
    layer_id: int,
    *,
    step_index: int | None,
    default_interval: int = 0,
    interval_overrides: dict[int, int] | None = None,
) -> bool:
    if step_index is None:
        return False
    resolved_interval = _resolve_qwen35_deltanet_statecache_renorm_interval(
        int(layer_id),
        default_interval=default_interval,
        interval_overrides=interval_overrides,
    )
    return bool(resolved_interval > 0 and (int(step_index) + 1) % int(resolved_interval) == 0)


def _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
    base_overrides: dict[int, int] | None,
    stage_overrides: dict[int, int] | None,
) -> dict[int, int] | None:
    if base_overrides is None and stage_overrides is None:
        return None
    merged: dict[int, int] = {}
    if base_overrides:
        merged.update({int(layer_id): int(interval) for layer_id, interval in base_overrides.items()})
    if stage_overrides:
        merged.update({int(layer_id): int(interval) for layer_id, interval in stage_overrides.items()})
    return merged


def _merge_qwen35_deltanet_statecache_mode_overrides(
    base_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None,
    stage_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None,
) -> dict[int, Qwen35DeltaNetStateCacheMode] | None:
    if base_overrides is None and stage_overrides is None:
        return None
    merged: dict[int, Qwen35DeltaNetStateCacheMode] = {}
    if base_overrides:
        merged.update({int(layer_id): mode for layer_id, mode in base_overrides.items()})
    if stage_overrides:
        merged.update({int(layer_id): mode for layer_id, mode in stage_overrides.items()})
    return merged


def _compressed_state_nbytes(
    tensor: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> int:
    if tensor is None:
        return 0
    if mode == "M3":
        return int(tensor.detach().nbytes)
    spec = StateTileSpec(
        state_rows=int(np.prod(tensor.shape[:-1])) if tensor.ndim > 1 else 1,
        state_cols=int(tensor.shape[-1]),
        group_size=int(max(min(group_size, int(tensor.shape[-1])), 1)),
        bits=int(bits),
        mode="M0",
    )
    _, payload_nbytes, metadata_nbytes = simulate_state_codec(
        tensor.detach().to(dtype=torch.float32).cpu().numpy(),
        spec,
    )
    return int(payload_nbytes + metadata_nbytes)


def _resolve_deltanet_statecache_bits(
    layer_id: int,
    *,
    default_bits: int,
    layer_bits_overrides: dict[int, int] | None = None,
) -> int:
    if not layer_bits_overrides:
        return int(default_bits)
    return int(layer_bits_overrides.get(int(layer_id), int(default_bits)))


def _resolve_deltanet_statecache_group_size(
    layer_id: int,
    *,
    default_group_size: int,
    layer_group_size_overrides: dict[int, int] | None = None,
) -> int:
    if not layer_group_size_overrides:
        return int(default_group_size)
    return int(layer_group_size_overrides.get(int(layer_id), int(default_group_size)))


def _prepare_qwen35_deltanet_statecache(
    cache: Any,
    *,
    layer_ids: list[int],
    recurrent_bits: int,
    conv_bits: int,
    group_size: int,
    renorm: bool = False,
    step_index: int | None = None,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    recurrent_renorm_interval: int = 0,
    conv_renorm_interval: int = 0,
    recurrent_layer_bits_overrides: dict[int, int] | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    recurrent_layer_group_size_overrides: dict[int, int] | None = None,
    conv_layer_group_size_overrides: dict[int, int] | None = None,
    recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    conv_renorm_interval_overrides: dict[int, int] | None = None,
    recurrent_default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    conv_default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    quantization_telemetry_layer_ids: set[int] | None = None,
    quantization_telemetry_records: list[dict[str, Any]] | None = None,
    quantization_telemetry_phase: str = "statecache",
) -> None:
    scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    recurrent_states = getattr(cache, "recurrent_states", None)
    conv_states = getattr(cache, "conv_states", None)
    if _statecache_scope_includes_recurrent(scope) and recurrent_states is None:
        raise ValueError("Qwen3.5 DeltaNet StateCache path requires recurrent_states on past_key_values")
    if _statecache_scope_includes_conv(scope) and conv_states is None:
        raise ValueError("Qwen3.5 DeltaNet StateCache path requires conv_states on past_key_values")

    for layer_id in layer_ids:
        if _statecache_scope_includes_recurrent(scope) and recurrent_states is not None and layer_id < len(recurrent_states):
            recurrent_state = recurrent_states[layer_id]
            if recurrent_state is not None:
                recurrent_renorm_applied = renorm or _should_apply_qwen35_deltanet_statecache_renorm(
                    int(layer_id),
                    step_index=step_index,
                    default_interval=int(recurrent_renorm_interval),
                    interval_overrides=recurrent_renorm_interval_overrides,
                )
                if recurrent_renorm_applied:
                    recurrent_state = _renorm_state_rows_tensor(recurrent_state)
                recurrent_mode = _resolve_qwen35_deltanet_statecache_mode(
                    int(layer_id),
                    default_mode=recurrent_default_mode,
                    mode_overrides=recurrent_mode_overrides,
                )
                resolved_recurrent_group_size = _resolve_deltanet_statecache_group_size(
                    int(layer_id),
                    default_group_size=int(group_size),
                    layer_group_size_overrides=recurrent_layer_group_size_overrides,
                )
                quantized_recurrent_state = _quantize_state_tensor(
                    recurrent_state,
                    bits=_resolve_deltanet_statecache_bits(
                        int(layer_id),
                        default_bits=int(recurrent_bits),
                        layer_bits_overrides=recurrent_layer_bits_overrides,
                    ),
                    group_size=resolved_recurrent_group_size,
                    mode=recurrent_mode,
                )
                recurrent_states[layer_id] = quantized_recurrent_state
                if (
                    quantization_telemetry_records is not None
                    and quantization_telemetry_layer_ids is not None
                    and int(layer_id) in quantization_telemetry_layer_ids
                ):
                    record = _build_state_quantization_telemetry_record(
                        recurrent_state,
                        quantized_recurrent_state,
                        layer_id=int(layer_id),
                        state_family="recurrent",
                        phase=str(quantization_telemetry_phase),
                        step_index=step_index,
                        bits=_resolve_deltanet_statecache_bits(
                            int(layer_id),
                            default_bits=int(recurrent_bits),
                            layer_bits_overrides=recurrent_layer_bits_overrides,
                        ),
                        group_size=resolved_recurrent_group_size,
                        mode=recurrent_mode,
                        renorm_applied=bool(recurrent_renorm_applied),
                    )
                    if record is not None:
                        quantization_telemetry_records.append(record)
        if _statecache_scope_includes_conv(scope) and conv_states is not None and layer_id < len(conv_states):
            conv_state = conv_states[layer_id]
            if conv_state is not None:
                conv_renorm_applied = renorm or _should_apply_qwen35_deltanet_statecache_renorm(
                    int(layer_id),
                    step_index=step_index,
                    default_interval=int(conv_renorm_interval),
                    interval_overrides=conv_renorm_interval_overrides,
                )
                if conv_renorm_applied:
                    conv_state = _renorm_state_rows_tensor(conv_state)
                conv_mode = _resolve_qwen35_deltanet_statecache_mode(
                    int(layer_id),
                    default_mode=conv_default_mode,
                    mode_overrides=conv_mode_overrides,
                )
                resolved_conv_group_size = _resolve_deltanet_statecache_group_size(
                    int(layer_id),
                    default_group_size=int(group_size),
                    layer_group_size_overrides=conv_layer_group_size_overrides,
                )
                quantized_conv_state = _quantize_state_tensor(
                    conv_state,
                    bits=_resolve_deltanet_statecache_bits(
                        int(layer_id),
                        default_bits=int(conv_bits),
                        layer_bits_overrides=conv_layer_bits_overrides,
                    ),
                    group_size=resolved_conv_group_size,
                    mode=conv_mode,
                )
                conv_states[layer_id] = quantized_conv_state
                if (
                    quantization_telemetry_records is not None
                    and quantization_telemetry_layer_ids is not None
                    and int(layer_id) in quantization_telemetry_layer_ids
                ):
                    record = _build_state_quantization_telemetry_record(
                        conv_state,
                        quantized_conv_state,
                        layer_id=int(layer_id),
                        state_family="conv",
                        phase=str(quantization_telemetry_phase),
                        step_index=step_index,
                        bits=_resolve_deltanet_statecache_bits(
                            int(layer_id),
                            default_bits=int(conv_bits),
                            layer_bits_overrides=conv_layer_bits_overrides,
                        ),
                        group_size=resolved_conv_group_size,
                        mode=conv_mode,
                        renorm_applied=bool(conv_renorm_applied),
                    )
                    if record is not None:
                        quantization_telemetry_records.append(record)


def _quantize_qwen35_deltanet_recurrent_state_in_cache(
    cache: Any,
    *,
    layer_ids: list[int],
    bits: int,
    group_size: int,
    layer_bits_overrides: dict[int, int] | None = None,
    default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
) -> None:
    _prepare_qwen35_deltanet_statecache(
        cache,
        layer_ids=layer_ids,
        recurrent_bits=int(bits),
        conv_bits=int(bits),
        group_size=int(group_size),
        statecache_scope="recurrent_only",
        recurrent_layer_bits_overrides=layer_bits_overrides,
        recurrent_default_mode=default_mode,
        recurrent_mode_overrides=mode_overrides,
    )


def _renorm_state_rows_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    value = tensor.detach().to(dtype=torch.float32)
    flat = value.reshape(-1, value.shape[-1])
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min_(1e-8)
    renormed = (flat / norms).reshape(value.shape)
    return renormed.to(dtype=tensor.dtype)


def _prepare_qwen35_deltanet_recurrent_statecache(
    cache: Any,
    *,
    layer_ids: list[int],
    bits: int,
    group_size: int,
    renorm: bool = False,
    step_index: int | None = None,
    renorm_interval: int = 0,
    layer_bits_overrides: dict[int, int] | None = None,
    layer_group_size_overrides: dict[int, int] | None = None,
    renorm_interval_overrides: dict[int, int] | None = None,
    default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
) -> None:
    _prepare_qwen35_deltanet_statecache(
        cache,
        layer_ids=layer_ids,
        recurrent_bits=int(bits),
        conv_bits=int(bits),
        group_size=int(group_size),
        renorm=renorm,
        step_index=step_index,
        statecache_scope="recurrent_only",
        recurrent_renorm_interval=int(renorm_interval),
        recurrent_layer_bits_overrides=layer_bits_overrides,
        recurrent_layer_group_size_overrides=layer_group_size_overrides,
        recurrent_renorm_interval_overrides=renorm_interval_overrides,
        recurrent_default_mode=default_mode,
        recurrent_mode_overrides=mode_overrides,
    )


def _summarize_qwen35_deltanet_statecache_bytes(
    prefill_partition: Qwen35HybridStatePartition,
    *,
    group_size: int,
    statecache_scope: Qwen35DeltaNetStateCacheScope,
    recurrent_bits: int,
    conv_bits: int,
    recurrent_layer_bits_overrides: dict[int, int] | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    recurrent_layer_group_size_overrides: dict[int, int] | None = None,
    conv_layer_group_size_overrides: dict[int, int] | None = None,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
) -> dict[str, Any]:
    scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    dense_conv_bytes = 0
    dense_recurrent_bytes = 0
    statecache_conv_bytes = 0
    statecache_recurrent_bytes = 0
    per_layer_dense_conv_bytes: dict[str, int] = {}
    per_layer_dense_recurrent_bytes: dict[str, int] = {}
    per_layer_statecache_conv_bytes: dict[str, int] = {}
    per_layer_statecache_recurrent_bytes: dict[str, int] = {}
    per_layer_conv_bits: dict[str, int] = {}
    per_layer_recurrent_bits: dict[str, int] = {}
    per_layer_conv_group_size: dict[str, int] = {}
    per_layer_recurrent_group_size: dict[str, int] = {}
    per_layer_conv_modes: dict[str, str] = {}
    per_layer_recurrent_modes: dict[str, str] = {}

    for layer in prefill_partition.fixed_resident_layers:
        layer_key = str(int(layer.layer_id))

        dense_conv = int(layer.conv_state_bytes)
        dense_recurrent = int(layer.recurrent_state_bytes)
        dense_conv_bytes += dense_conv
        dense_recurrent_bytes += dense_recurrent
        per_layer_dense_conv_bytes[layer_key] = dense_conv
        per_layer_dense_recurrent_bytes[layer_key] = dense_recurrent

        resolved_conv_bits = _resolve_deltanet_statecache_bits(
            int(layer.layer_id),
            default_bits=int(conv_bits),
            layer_bits_overrides=conv_layer_bits_overrides,
        )
        resolved_recurrent_bits = _resolve_deltanet_statecache_bits(
            int(layer.layer_id),
            default_bits=int(recurrent_bits),
            layer_bits_overrides=recurrent_layer_bits_overrides,
        )
        resolved_conv_mode = _resolve_qwen35_deltanet_statecache_mode(
            int(layer.layer_id),
            default_mode="M0",
            mode_overrides=conv_mode_overrides,
        )
        resolved_recurrent_mode = _resolve_qwen35_deltanet_statecache_mode(
            int(layer.layer_id),
            default_mode="M0",
            mode_overrides=recurrent_mode_overrides,
        )
        resolved_conv_group_size = _resolve_deltanet_statecache_group_size(
            int(layer.layer_id),
            default_group_size=int(group_size),
            layer_group_size_overrides=conv_layer_group_size_overrides,
        )
        resolved_recurrent_group_size = _resolve_deltanet_statecache_group_size(
            int(layer.layer_id),
            default_group_size=int(group_size),
            layer_group_size_overrides=recurrent_layer_group_size_overrides,
        )
        per_layer_conv_bits[layer_key] = int(resolved_conv_bits)
        per_layer_recurrent_bits[layer_key] = int(resolved_recurrent_bits)
        per_layer_conv_group_size[layer_key] = int(resolved_conv_group_size)
        per_layer_recurrent_group_size[layer_key] = int(resolved_recurrent_group_size)
        per_layer_conv_modes[layer_key] = resolved_conv_mode
        per_layer_recurrent_modes[layer_key] = resolved_recurrent_mode

        compressed_conv = dense_conv
        if _statecache_scope_includes_conv(scope) and layer.conv_state is not None:
            compressed_conv = _compressed_state_nbytes(
                layer.conv_state,
                bits=resolved_conv_bits,
                group_size=resolved_conv_group_size,
                mode=resolved_conv_mode,
            )

        compressed_recurrent = dense_recurrent
        if _statecache_scope_includes_recurrent(scope) and layer.recurrent_state is not None:
            compressed_recurrent = _compressed_state_nbytes(
                layer.recurrent_state,
                bits=resolved_recurrent_bits,
                group_size=resolved_recurrent_group_size,
                mode=resolved_recurrent_mode,
            )

        statecache_conv_bytes += int(compressed_conv)
        statecache_recurrent_bytes += int(compressed_recurrent)
        per_layer_statecache_conv_bytes[layer_key] = int(compressed_conv)
        per_layer_statecache_recurrent_bytes[layer_key] = int(compressed_recurrent)

    dense_fixed_resident_bytes = int(dense_conv_bytes + dense_recurrent_bytes)
    statecache_fixed_resident_bytes = int(statecache_conv_bytes + statecache_recurrent_bytes)
    return {
        "deltanet_conv_state_bytes": int(dense_conv_bytes),
        "deltanet_recurrent_state_bytes": int(dense_recurrent_bytes),
        "deltanet_statecache_conv_state_bytes": int(statecache_conv_bytes),
        "deltanet_statecache_recurrent_state_bytes": int(statecache_recurrent_bytes),
        "deltanet_dense_fixed_resident_bytes": dense_fixed_resident_bytes,
        "deltanet_statecache_fixed_resident_bytes": statecache_fixed_resident_bytes,
        "deltanet_statecache_effective_conv_compression_ratio": (
            float(dense_conv_bytes / max(statecache_conv_bytes, 1)) if dense_conv_bytes > 0 else 1.0
        ),
        "deltanet_statecache_effective_recurrent_compression_ratio": (
            float(dense_recurrent_bytes / max(statecache_recurrent_bytes, 1)) if dense_recurrent_bytes > 0 else 1.0
        ),
        "deltanet_statecache_effective_fixed_resident_compression_ratio": (
            float(dense_fixed_resident_bytes / max(statecache_fixed_resident_bytes, 1))
            if dense_fixed_resident_bytes > 0
            else 1.0
        ),
        "deltanet_statecache_per_layer_dense_conv_bytes": dict(sorted(per_layer_dense_conv_bytes.items())),
        "deltanet_statecache_per_layer_dense_recurrent_bytes": dict(sorted(per_layer_dense_recurrent_bytes.items())),
        "deltanet_statecache_per_layer_conv_bytes": dict(sorted(per_layer_statecache_conv_bytes.items())),
        "deltanet_statecache_per_layer_recurrent_bytes": dict(sorted(per_layer_statecache_recurrent_bytes.items())),
        "deltanet_statecache_per_layer_conv_bits": dict(sorted(per_layer_conv_bits.items())),
        "deltanet_statecache_per_layer_recurrent_bits": dict(sorted(per_layer_recurrent_bits.items())),
        "deltanet_statecache_per_layer_conv_group_size": dict(sorted(per_layer_conv_group_size.items())),
        "deltanet_statecache_per_layer_recurrent_group_size": dict(sorted(per_layer_recurrent_group_size.items())),
        "deltanet_statecache_per_layer_conv_mode": dict(sorted(per_layer_conv_modes.items())),
        "deltanet_statecache_per_layer_recurrent_mode": dict(sorted(per_layer_recurrent_modes.items())),
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
    weight_quantization: str = "none",
):
    _require_qwen35_model_class()
    dtype = getattr(torch, torch_dtype)
    resolved_device = _default_model_device() if device is None else device
    auth_kwargs = resolve_hf_auth_kwargs()
    from_pretrained_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        **auth_kwargs,
    }
    if weight_quantization == "none":
        from_pretrained_kwargs["torch_dtype"] = dtype
    elif weight_quantization == "bnb_8bit":
        if BitsAndBytesConfig is None:
            raise RuntimeError("bnb_8bit requires transformers BitsAndBytesConfig and bitsandbytes to be installed")
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if str(resolved_device).startswith("cuda"):
            from_pretrained_kwargs["device_map"] = {"": str(resolved_device)}
        else:
            raise RuntimeError("bnb_8bit Qwen3.5 loading is currently only supported on CUDA devices")
    else:
        raise ValueError(f"Unsupported weight_quantization={weight_quantization!r}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        **from_pretrained_kwargs,
    )
    if weight_quantization == "none":
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


class CaptureQwen35DeltaNet(nn.Module):
    def __init__(self, base_linear_attn: nn.Module, adapter: "Qwen35DeltaNetStateModelAdapter") -> None:
        super().__init__()
        self.base_linear_attn = base_linear_attn
        self.adapter = adapter
        self.layer_idx = int(base_linear_attn.layer_idx)

    def _forward_base_linear_attn(
        self,
        *,
        hidden_states: torch.Tensor,
        cache_params,
        cache_position: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
    ):
        try:
            return self.base_linear_attn(
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            return self.base_linear_attn(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if not self.adapter.capture_enabled or tuple(hidden_states.shape[:2]) != (1, 1) or cache_params is None:
            return self._forward_base_linear_attn(
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        pre_conv_state = _clone_state_tensor(cache_params.conv_states[self.layer_idx])
        pre_recurrent_state = _clone_state_tensor(cache_params.recurrent_states[self.layer_idx])
        output = self._forward_base_linear_attn(
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        token_index = self.adapter.current_token_index(cache_position)
        self.adapter.record_state(
            Qwen35DeltaNetStateRecord(
                step_index=self.adapter.capture_step_index,
                layer_id=self.layer_idx,
                token_index=token_index,
                hidden_states=hidden_states.detach().to(dtype=torch.float32).cpu().clone(),
                output_states=output.detach().to(dtype=torch.float32).cpu().clone(),
                pre_conv_state=pre_conv_state,
                post_conv_state=_clone_state_tensor(cache_params.conv_states[self.layer_idx]),
                pre_recurrent_state=pre_recurrent_state,
                post_recurrent_state=_clone_state_tensor(cache_params.recurrent_states[self.layer_idx]),
            )
        )
        return output


@dataclass(slots=True)
class Qwen35DeltaNetStateModelAdapter(Qwen35TextModelAdapter):
    capture_enabled: bool = False
    capture_step_index: int = -1
    _pending_records: list[Qwen35DeltaNetStateRecord] = field(default_factory=list, init=False, repr=False)
    _wrappers: list[CaptureQwen35DeltaNet] = field(default_factory=list, init=False, repr=False)
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
            if layer_types[layer_id] != "linear_attention" or not hasattr(layer, "linear_attn"):
                continue
            base_linear_attn = layer.linear_attn
            if isinstance(base_linear_attn, CaptureQwen35DeltaNet):
                base_linear_attn = base_linear_attn.base_linear_attn
            wrapper = CaptureQwen35DeltaNet(base_linear_attn, self)
            layer.linear_attn = wrapper
            self._wrappers.append(wrapper)

    def begin_capture_step(self, step_index: int) -> None:
        self.capture_step_index = int(step_index)
        self.capture_enabled = True
        self._pending_records = []

    def end_capture_step(self) -> list[Qwen35DeltaNetStateRecord]:
        records = list(self._pending_records)
        self.capture_step_index = -1
        self.capture_enabled = False
        self._pending_records = []
        return records

    def record_state(self, record: Qwen35DeltaNetStateRecord) -> None:
        if self.capture_step_index < 0:
            return
        self._pending_records.append(record)

    def current_token_index(self, cache_position) -> int:
        if self._current_token_index_override is not None:
            return self._current_token_index_override
        if cache_position is None:
            raise ValueError("cache_position is required for Qwen3.5 DeltaNet capture")
        token_positions = cache_position.reshape(-1)
        if token_positions.numel() != 1:
            raise ValueError("Qwen3.5 DeltaNet capture requires a single cache_position per decode step")
        return int(token_positions.item())

    def set_current_token_index(self, token_index: int | None) -> None:
        self._current_token_index_override = None if token_index is None else int(token_index)

    def deltanet_layer_ids(self) -> list[int]:
        return [record["layer_id"] for record in self.hybrid_layer_summary() if record["layer_type"] == "linear_attention"]


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
        self.adapter._record_layer_timing(self.adapter.qkv_projection_ms_total_by_layer, self.layer_idx, qkv_ms)

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
        self.adapter._record_layer_timing(self.adapter.append_runtime_ms_total_by_layer, self.layer_idx, append_ms)

        decode_trace = ExecutionTrace(capture_timings=self.adapter.profile_backend) if self.adapter.profile_backend else None
        force_grouped_batching = os.environ.get("DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        context_states, decode_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.decode_layer_torch(
                self.layer_idx,
                query_step,
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                # Qwen3.5 full-attention layers are a small fixed shape on CUDA:
                # 8 query heads, 2 KV heads, 4 pages per KV head at exact-64 with
                # the current profile. The grouped batched decode path is slower
                # than the per-KV-head fallback for this workload.
                prefer_grouped_batching=force_grouped_batching or hidden_states.device.type != "cuda",
                trace=decode_trace,
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
        self.adapter._record_layer_timing(self.adapter.decode_runtime_ms_total_by_layer, self.layer_idx, decode_ms)
        self.adapter.decode_call_count_by_layer[self.layer_idx] = self.adapter.decode_call_count_by_layer.get(self.layer_idx, 0) + 1
        if decode_trace is not None:
            self.adapter.decode_backend_trace.merge(decode_trace)

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
        self.adapter._record_layer_timing(
            self.adapter.output_projection_ms_total_by_layer,
            self.layer_idx,
            output_projection_ms,
        )

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
    q_head_to_kv_head: np.ndarray = field(init=False, repr=False)
    _pending_records: list[LlamaReplayRecord] = field(default_factory=list, init=False, repr=False)
    _wrappers: list[DotCacheQwen35AttentionSubset] = field(default_factory=list, init=False, repr=False)
    _current_token_index_override: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        Qwen35TextModelAdapter.__post_init__(self)
        text_config = _qwen35_text_config(self.model)
        self.q_head_to_kv_head = _default_q_head_to_kv_head(
            int(text_config.num_attention_heads),
            int(text_config.num_key_value_heads),
        )
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
            base_attention = layer.self_attn
            if isinstance(base_attention, DotCacheQwen35AttentionSubset):
                base_attention = base_attention.base_attention
            wrapper = DotCacheQwen35AttentionSubset(base_attention, self)
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
    profile_backend: bool = field(default=False, init=False, repr=False)
    decode_backend_trace: ExecutionTrace = field(default_factory=ExecutionTrace, init=False, repr=False)
    qkv_projection_ms_total_by_layer: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    append_runtime_ms_total_by_layer: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    decode_runtime_ms_total_by_layer: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    output_projection_ms_total_by_layer: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    decode_call_count_by_layer: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    native_hybrid_runtime_state: Qwen35NativeHybridRuntimeState | None = field(default=None, init=False, repr=False)
    hybrid_dotcache_runtime_state: Qwen35HybridDotCacheRuntimeState | None = field(default=None, init=False, repr=False)
    serving_shortlist_heuristic_applied: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        Qwen35AttentionSubsetModelAdapter.__post_init__(self)
        text_config = _qwen35_text_config(self.model)
        expected_head_dim = _qwen35_attention_head_dim(self.model)
        if self.dotcache_config.head_dim != expected_head_dim:
            self.dotcache_config = replace(self.dotcache_config, head_dim=expected_head_dim)
        self._rebuild_model_kv_cache()

    def _rebuild_model_kv_cache(self) -> None:
        text_config = _qwen35_text_config(self.model)
        self.model_kv_cache = ModelPagedKVCache(
            config=self.dotcache_config,
            num_hidden_layers=int(text_config.num_hidden_layers),
            num_attention_heads=int(text_config.num_attention_heads),
            num_key_value_heads=int(text_config.num_key_value_heads),
            backend=self.backend,
            cache=self.cache,
        )
        self.q_head_to_kv_head = self.model_kv_cache.default_q_head_to_kv_head.copy()

    def maybe_apply_mps_serving_shortlist_heuristic(self, *, prompt_length: int) -> bool:
        updated_config, applied = _qwen35_mps_serving_shortlist_heuristic(
            self.dotcache_config,
            backend=self.backend,
            prompt_length=int(prompt_length),
        )
        if applied:
            self.dotcache_config = updated_config
            self._rebuild_model_kv_cache()
        self.serving_shortlist_heuristic_applied = bool(
            self.backend == "torch_mps"
            and int(prompt_length) >= 4096
            and int(self.dotcache_config.execution_recent_window) == 1024
            and int(self.dotcache_config.execution_sink_window) == 256
            and int(self.dotcache_config.execution_relevance_top_k) == 4
            and str(self.dotcache_config.execution_relevance_mode) == "envelope"
            and tuple(self.dotcache_config.execution_relevance_top_k_overrides) == ()
            and tuple(self.dotcache_config.execution_relevance_top_k_context_overrides) == ("layer:23:min_ctx:8192=8",)
        )
        return bool(applied)

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
        self.decode_backend_trace = ExecutionTrace(capture_timings=self.profile_backend)
        self.qkv_projection_ms_total_by_layer = {}
        self.append_runtime_ms_total_by_layer = {}
        self.decode_runtime_ms_total_by_layer = {}
        self.output_projection_ms_total_by_layer = {}
        self.decode_call_count_by_layer = {}
        self.native_hybrid_runtime_state = None
        self.hybrid_dotcache_runtime_state = None

    def set_backend_profiling(self, enabled: bool) -> None:
        self.profile_backend = bool(enabled)
        self.decode_backend_trace = ExecutionTrace(capture_timings=self.profile_backend)

    def _record_layer_timing(self, store: dict[int, float], layer_id: int, ms: float) -> None:
        store[int(layer_id)] = float(store.get(int(layer_id), 0.0) + float(ms))

    def per_layer_runtime_summary(self) -> dict[str, Any]:
        return {
            "dotcache_decode_call_count_by_layer": {
                str(layer_id): int(count) for layer_id, count in sorted(self.decode_call_count_by_layer.items())
            },
            "dotcache_qkv_projection_ms_total_by_layer": {
                str(layer_id): float(total) for layer_id, total in sorted(self.qkv_projection_ms_total_by_layer.items())
            },
            "dotcache_append_runtime_ms_total_by_layer": {
                str(layer_id): float(total) for layer_id, total in sorted(self.append_runtime_ms_total_by_layer.items())
            },
            "dotcache_decode_runtime_ms_total_by_layer": {
                str(layer_id): float(total) for layer_id, total in sorted(self.decode_runtime_ms_total_by_layer.items())
            },
            "dotcache_output_projection_ms_total_by_layer": {
                str(layer_id): float(total) for layer_id, total in sorted(self.output_projection_ms_total_by_layer.items())
            },
        }

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

    def deltanet_layer_ids(self) -> list[int]:
        return [
            record["layer_id"]
            for record in self.hybrid_layer_summary()
            if record["layer_type"] == "linear_attention"
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
        weight_quantization: str = "none",
    ) -> "Qwen35TextHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            weight_quantization=weight_quantization,
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
class Qwen35DeltaNetStateHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35DeltaNetStateModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        torch_dtype: str = "float16",
        weight_quantization: str = "none",
    ) -> "Qwen35DeltaNetStateHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            weight_quantization=weight_quantization,
        )
        adapter = Qwen35DeltaNetStateModelAdapter(model=model)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        helper = Qwen35TextHarness(model=self.model, tokenizer=self.tokenizer, adapter=self.adapter)
        return helper.tokenize_prompt(prompt, multimodal_inputs=multimodal_inputs)

    def inspect_deltanet_state(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return inspect_qwen35_deltanet_state(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            multimodal_inputs=multimodal_inputs,
        )

    def run_deltanet_ablation(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        group_size: int = 32,
        bits: tuple[int, ...] = (8, 4),
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_deltanet_state_ablation_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            group_size=group_size,
            bits=bits,
            multimodal_inputs=multimodal_inputs,
        )

    def run_deltanet_statecache_readout(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        group_size: int = 32,
        bits: int = 8,
        layer_bits_overrides: dict[int, int] | None = None,
        statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
        conv_bits: int | None = None,
        conv_layer_bits_overrides: dict[int, int] | None = None,
        state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
        renorm_interval: int = 0,
        recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        conv_renorm_interval_overrides: dict[int, int] | None = None,
        recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
        recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
        readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
        readout_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        readout_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        multimodal_inputs: Any | None = None,
        ) -> dict[str, Any]:
        return run_qwen35_deltanet_statecache_readout_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            statecache_scope=statecache_scope,
            conv_bits=conv_bits,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
            readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
            post_update_recurrent_renorm_interval_overrides=post_update_recurrent_renorm_interval_overrides,
            post_update_recurrent_mode_overrides=post_update_recurrent_mode_overrides,
            conv_mode_overrides=conv_mode_overrides,
            multimodal_inputs=multimodal_inputs,
        )

    def run_deltanet_statecache_serving(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        group_size: int = 32,
        bits: int = 8,
        layer_bits_overrides: dict[int, int] | None = None,
        recurrent_group_size_policy: Qwen35DeltaNetStateCacheRecurrentGroupSizePolicy | str | None = None,
        recurrent_layer_group_size_overrides: dict[int, int] | None = None,
        state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
        renorm_interval: int = 0,
        recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
        recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
        readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
        readout_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        readout_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_deltanet_statecache_serving_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            recurrent_group_size_policy=recurrent_group_size_policy,
            recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
            readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
            post_update_recurrent_renorm_interval_overrides=post_update_recurrent_renorm_interval_overrides,
            post_update_recurrent_mode_overrides=post_update_recurrent_mode_overrides,
            multimodal_inputs=multimodal_inputs,
        )

    def evaluate_deltanet_statecache_loss(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
        group_size: int = 32,
        bits: int = 8,
        layer_bits_overrides: dict[int, int] | None = None,
        statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
        conv_bits: int | None = None,
        conv_layer_bits_overrides: dict[int, int] | None = None,
        state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
        renorm_interval: int = 0,
        recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        conv_renorm_interval_overrides: dict[int, int] | None = None,
        recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
        recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
        readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
        post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
        post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_deltanet_statecache_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            statecache_scope=statecache_scope,
            conv_bits=conv_bits,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            post_update_recurrent_renorm_interval_overrides=post_update_recurrent_renorm_interval_overrides,
            post_update_recurrent_mode_overrides=post_update_recurrent_mode_overrides,
            conv_mode_overrides=conv_mode_overrides,
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
        weight_quantization: str = "none",
    ) -> "Qwen35AttentionSubsetHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            weight_quantization=weight_quantization,
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

    def capture_attention_subset_page_traces(
        self,
        *,
        output_dir: str | Path,
        tokens_per_page: int,
        kinds: tuple[str, ...] = ("K", "V"),
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_page_trace_capture_harness(
            self.model,
            self.adapter,
            output_dir=output_dir,
            tokens_per_page=tokens_per_page,
            kinds=kinds,
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
        weight_quantization: str = "none",
    ) -> "Qwen35AttentionSubsetDotCacheHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            weight_quantization=weight_quantization,
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
        profile_backend: bool = False,
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
            profile_backend=profile_backend,
            multimodal_inputs=multimodal_inputs,
        )

    def run_attention_subset_dotcache_serving(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        profile_backend: bool = False,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_serving_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            profile_backend=profile_backend,
            multimodal_inputs=multimodal_inputs,
        )

    def run_attention_subset_dotcache_serving_quality(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        profile_backend: bool = False,
        trace_python_allocations: bool = False,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_serving_quality_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            profile_backend=profile_backend,
            trace_python_allocations=trace_python_allocations,
            multimodal_inputs=multimodal_inputs,
        )

    def run_attention_subset_dotcache_serving_recall_analysis(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        profile_backend: bool = False,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_serving_recall_analysis_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            profile_backend=profile_backend,
            multimodal_inputs=multimodal_inputs,
        )

    def run_attention_subset_dotcache_serving_scorer_diagnostic(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        profile_backend: bool = False,
        trace_python_allocations: bool = False,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            profile_backend=profile_backend,
            trace_python_allocations=trace_python_allocations,
            multimodal_inputs=multimodal_inputs,
        )

    def evaluate_attention_subset_dotcache_loss(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
        profile_backend: bool = False,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_dotcache_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            profile_backend=profile_backend,
            multimodal_inputs=multimodal_inputs,
        )

    def run_hybrid_combined_localization(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
        profile_backend: bool = False,
        statecache_group_size: int = 32,
        statecache_bits: int = 8,
        statecache_layer_bits_overrides: dict[int, int] | None = None,
        statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
        statecache_conv_bits: int | None = None,
        statecache_conv_layer_bits_overrides: dict[int, int] | None = None,
        statecache_stage: Qwen35DeltaNetStateCacheStage = "post_update_m0",
        statecache_renorm_interval: int = 0,
        statecache_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        statecache_conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_hybrid_combined_localization_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            profile_backend=profile_backend,
            statecache_group_size=statecache_group_size,
            statecache_bits=statecache_bits,
            statecache_layer_bits_overrides=statecache_layer_bits_overrides,
            statecache_scope=statecache_scope,
            statecache_conv_bits=statecache_conv_bits,
            statecache_conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
            statecache_stage=statecache_stage,
            statecache_renorm_interval=statecache_renorm_interval,
            statecache_recurrent_mode_overrides=statecache_recurrent_mode_overrides,
            statecache_conv_mode_overrides=statecache_conv_mode_overrides,
            multimodal_inputs=multimodal_inputs,
        )

    def run_attention_subset_dotcache_statecache(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        profile_backend: bool = False,
        group_size: int = 32,
        bits: int = 8,
        state_stage: Qwen35DeltaNetStateCacheStage = "post_update_m0",
        renorm_interval: int = 0,
        recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_statecache_dotcache_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
            profile_backend=profile_backend,
            group_size=group_size,
            bits=bits,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_mode_overrides=recurrent_mode_overrides,
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


def _run_dense_prefill(model, *, input_ids, attention_mask, logits_to_keep: int | torch.Tensor = 1):
    return _run_inference(
        lambda: model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
    )


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

    device = input_ids.device
    prefill_cuda_memory_baseline = _begin_cuda_memory_region(device)
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    prefill_cuda_memory = _end_cuda_memory_region(device, prefill_cuda_memory_baseline)
    prefill_cache_bytes = _hybrid_cache_nbytes(prefill_outputs.past_key_values)
    generated_ids: list[int] = []
    dense_decode_ms_total = 0.0
    final_past_key_values = prefill_outputs.past_key_values
    decode_cuda_memory: dict[str, int] = {}

    if max_new_tokens > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(current_input_ids.item()))
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        past_key_values = prefill_outputs.past_key_values
        decode_cuda_memory_baseline = _begin_cuda_memory_region(device)

        for _ in range(max(max_new_tokens - 1, 0)):
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
                device=device,
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
        decode_cuda_memory = _end_cuda_memory_region(device, decode_cuda_memory_baseline)

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
    result.update({f"dense_prefill_{key}": value for key, value in prefill_cuda_memory.items()})
    result.update({f"dense_decode_{key}": value for key, value in decode_cuda_memory.items()})
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


def _run_qwen35_deltanet_dense_capture(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    input_ids,
    attention_mask,
    decode_steps: int,
) -> dict[str, Any]:
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    per_step_records: list[list[Qwen35DeltaNetStateRecord]] = []
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
        "capture_records": per_step_records,
        "decode_inputs": decode_inputs,
        "step_logits": step_logits,
        "decode_ms_total": float(dense_decode_ms_total),
    }


def _run_qwen35_deltanet_dense_teacher_forced_capture(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prefix_input_ids,
    prefix_attention_mask,
    continuation_ids,
) -> dict[str, Any]:
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=prefix_input_ids.device,
    )
    per_step_records: list[list[Qwen35DeltaNetStateRecord]] = []
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()]
    past_key_values = prefill_outputs.past_key_values
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=prefix_input_ids.device)
    dense_decode_ms_total = 0.0

    for step_index in range(max(int(continuation_ids.shape[1]) - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]
        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(prefix_input_ids.shape[1] + step_index))
        try:
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
                device=prefix_input_ids.device,
            )
        finally:
            adapter.set_current_token_index(None)
        dense_decode_ms_total += step_ms
        per_step_records.append(adapter.end_capture_step())
        logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        past_key_values = outputs.past_key_values
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    return {
        "prefill_outputs": prefill_outputs,
        "prefill_ms": float(prefill_ms),
        "capture_records": per_step_records,
        "step_logits": logits_list,
        "decode_ms_total": float(dense_decode_ms_total),
    }


def _first_drift_step(
    dense_logits: list[np.ndarray],
    approx_logits: list[np.ndarray],
) -> int | None:
    for step_index, (dense_step, approx_step) in enumerate(zip(dense_logits, approx_logits)):
        dense_argmax = np.argmax(dense_step, axis=-1)
        approx_argmax = np.argmax(approx_step, axis=-1)
        if not np.array_equal(dense_argmax, approx_argmax):
            return int(step_index)
    return None


def _first_layer_over_threshold(
    per_layer_error: dict[str, float],
    *,
    threshold: float = 1e-6,
) -> int | None:
    for layer_key in sorted(per_layer_error, key=lambda value: int(value)):
        if float(per_layer_error[layer_key]) > threshold:
            return int(layer_key)
    return None


def _summarize_deltanet_state_capture(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    input_ids,
    decode_steps: int,
    prefill_outputs,
    prefill_ms: float,
    dense_decode_ms_total: float,
    per_step_records: list[list[Qwen35DeltaNetStateRecord]],
) -> dict[str, Any]:
    hybrid_summary = summarize_qwen35_hybrid_state(prefill_outputs.past_key_values, model)
    linear_records = [record for record in hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"]
    record_map: dict[int, list[Qwen35DeltaNetStateRecord]] = {int(record["layer_id"]): [] for record in linear_records}
    for step_records in per_step_records:
        for record in step_records:
            record_map.setdefault(int(record.layer_id), []).append(record)

    deltanet_layers: list[dict[str, Any]] = []
    for layer_record in linear_records:
        layer_id = int(layer_record["layer_id"])
        captured = record_map.get(layer_id, [])
        state_shapes: dict[str, list[int]] = {}
        if captured:
            first = captured[0]
            if first.pre_conv_state is not None:
                state_shapes["conv_state"] = list(first.pre_conv_state.shape)
            if first.pre_recurrent_state is not None:
                state_shapes["recurrent_state"] = list(first.pre_recurrent_state.shape)
            state_shapes["hidden_states"] = list(first.hidden_states.shape)
            state_shapes["output_states"] = list(first.output_states.shape)

        step_delta_norms: list[dict[str, float | int]] = []
        for record in captured:
            conv_delta = None
            if record.pre_conv_state is not None and record.post_conv_state is not None:
                conv_delta = record.post_conv_state - record.pre_conv_state
            recurrent_delta = None
            if record.pre_recurrent_state is not None and record.post_recurrent_state is not None:
                recurrent_delta = record.post_recurrent_state - record.pre_recurrent_state
            step_delta_norms.append(
                {
                    "step_index": int(record.step_index),
                    "token_index": int(record.token_index),
                    "conv_state_before_rms": _tensor_rms(record.pre_conv_state),
                    "conv_state_after_rms": _tensor_rms(record.post_conv_state),
                    "conv_state_delta_rms": _tensor_rms(conv_delta),
                    "conv_state_delta_max_abs": _tensor_max_abs(conv_delta),
                    "recurrent_state_before_rms": _tensor_rms(record.pre_recurrent_state),
                    "recurrent_state_after_rms": _tensor_rms(record.post_recurrent_state),
                    "recurrent_state_delta_rms": _tensor_rms(recurrent_delta),
                    "recurrent_state_delta_max_abs": _tensor_max_abs(recurrent_delta),
                    "output_state_rms": _tensor_rms(record.output_states),
                    "output_state_max_abs": _tensor_max_abs(record.output_states),
                }
            )

        deltanet_layers.append(
            StateLayerRecord(
                layer_id=layer_id,
                layer_type=str(layer_record["layer_type"]),
                state_family="linear_recurrent",
                conv_state_bytes=int(layer_record["conv_state_bytes"]),
                recurrent_state_bytes=int(layer_record["recurrent_state_bytes"]),
                layer_state_bytes=int(layer_record["conv_state_bytes"] + layer_record["recurrent_state_bytes"]),
                state_shapes=state_shapes,
                state_delta_norms=step_delta_norms,
            ).to_dict()
        )

    deltanet_conv_bytes = int(sum(layer["conv_state_bytes"] for layer in linear_records))
    deltanet_recurrent_bytes = int(sum(layer["recurrent_state_bytes"] for layer in linear_records))
    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "text_only": True,
        "dotcache_ready": False,
        "deltanet_state_ready": True,
        "runtime_mode": "dense_deltanet_state_capture",
        "uses_native_qwen35_class": True,
        "deltanet_conv_state_bytes": deltanet_conv_bytes,
        "deltanet_recurrent_state_bytes": deltanet_recurrent_bytes,
        "deltanet_total_state_bytes": int(deltanet_conv_bytes + deltanet_recurrent_bytes),
        "deltanet_state_layers": deltanet_layers,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    result.update(hybrid_summary)
    return result


def _replay_deltanet_linear_step(
    adapter: Qwen35DeltaNetStateModelAdapter,
    record: Qwen35DeltaNetStateRecord,
    *,
    conv_state: torch.Tensor | None,
    recurrent_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    layer_count = int(_qwen35_text_config(adapter.model).num_hidden_layers)
    module = None
    wrapper_module = None
    for wrapper in adapter._wrappers:
        if wrapper.layer_idx == int(record.layer_id):
            module = wrapper.base_linear_attn
            wrapper_module = wrapper
            break
    if module is None:
        raise ValueError(f"missing DeltaNet module for layer {record.layer_id}")
    hidden_states = record.hidden_states.to(device=adapter.device, dtype=next(module.parameters()).dtype)
    conv_state_input = None
    if conv_state is not None:
        conv_state_input = conv_state.detach().to(device=adapter.device, dtype=next(module.parameters()).dtype).clone()
    recurrent_state_input = None
    if recurrent_state is not None:
        recurrent_state_input = recurrent_state.detach().to(device=adapter.device, dtype=next(module.parameters()).dtype).clone()
    cache_stub = _Qwen35DeltaNetCacheStub(
        layer_count=layer_count,
        target_layer_id=int(record.layer_id),
        conv_state=conv_state_input,
        recurrent_state=recurrent_state_input,
        has_previous_state=True,
    )
    cache_position = torch.tensor([int(record.token_index)], dtype=torch.long, device=adapter.device)
    with torch.no_grad():
        output = wrapper_module._forward_base_linear_attn(
            hidden_states=hidden_states,
            cache_params=cache_stub,
            cache_position=cache_position,
            attention_mask=None,
        )
    post_conv_state = cache_stub.conv_states[int(record.layer_id)]
    post_recurrent_state = cache_stub.recurrent_states[int(record.layer_id)]
    return (
        output.detach().to(dtype=torch.float32).cpu().clone(),
        _clone_state_tensor(post_conv_state),
        _clone_state_tensor(post_recurrent_state),
    )


def _run_deltanet_ablation_stage(
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]],
    stage_name: str,
    bits: int | None,
    group_size: int,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    conv_bits: int | None = None,
    recurrent_layer_bits_overrides: dict[int, int] | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    recurrent_layer_group_size_overrides: dict[int, int] | None = None,
    conv_layer_group_size_overrides: dict[int, int] | None = None,
    recurrent_default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_default_mode: Qwen35DeltaNetStateCacheMode = "M0",
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
) -> StateAblationResult:
    scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    per_layer_max_abs_error: dict[str, float] = {}
    per_layer_max_rel_error: dict[str, float] = {}
    per_layer_output_max_abs_error: dict[str, float] = {}
    per_step_output_max_abs_error: list[float] = []

    for layer_id, records in sorted(records_by_layer.items()):
        recurrent_mode = _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode=recurrent_default_mode,
            mode_overrides=recurrent_mode_overrides,
        )
        conv_mode = _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode=conv_default_mode,
            mode_overrides=conv_mode_overrides,
        )
        resolved_recurrent_bits = _resolve_deltanet_statecache_bits(
            int(layer_id),
            default_bits=int(bits or 8),
            layer_bits_overrides=recurrent_layer_bits_overrides,
        )
        resolved_conv_bits = _resolve_deltanet_statecache_bits(
            int(layer_id),
            default_bits=int(conv_bits if conv_bits is not None else (bits or 8)),
            layer_bits_overrides=conv_layer_bits_overrides,
        )
        resolved_recurrent_group_size = _resolve_deltanet_statecache_group_size(
            int(layer_id),
            default_group_size=int(group_size),
            layer_group_size_overrides=recurrent_layer_group_size_overrides,
        )
        resolved_conv_group_size = _resolve_deltanet_statecache_group_size(
            int(layer_id),
            default_group_size=int(group_size),
            layer_group_size_overrides=conv_layer_group_size_overrides,
        )
        carried_conv: torch.Tensor | None = None
        carried_recurrent: torch.Tensor | None = None
        layer_step_output_errors: list[float] = []
        layer_max_abs = 0.0
        layer_max_rel = 0.0
        layer_output_max = 0.0

        for step_index, record in enumerate(records):
            dense_pre_conv = record.pre_conv_state
            dense_pre_recurrent = record.pre_recurrent_state
            dense_post_conv = record.post_conv_state
            dense_post_recurrent = record.post_recurrent_state
            dense_output = record.output_states

            if stage_name == "dense_baseline":
                replay_output = dense_output
                replay_post_conv = dense_post_conv
                replay_post_recurrent = dense_post_recurrent
            elif stage_name == "escape_m3":
                replay_output, replay_post_conv, replay_post_recurrent = _replay_deltanet_linear_step(
                    adapter,
                    record,
                    conv_state=dense_pre_conv,
                    recurrent_state=dense_pre_recurrent,
                )
            elif stage_name == "readout_only_m0":
                replay_output, replay_post_conv, replay_post_recurrent = _replay_deltanet_linear_step(
                    adapter,
                    record,
                    conv_state=(
                        _quantize_state_tensor(
                            dense_pre_conv,
                            bits=resolved_conv_bits,
                            group_size=resolved_conv_group_size,
                            mode=conv_mode,
                        )
                        if _statecache_scope_includes_conv(scope)
                        else dense_pre_conv
                    ),
                    recurrent_state=(
                        _quantize_state_tensor(
                            dense_pre_recurrent,
                            bits=resolved_recurrent_bits,
                            group_size=resolved_recurrent_group_size,
                            mode=recurrent_mode,
                        )
                        if _statecache_scope_includes_recurrent(scope)
                        else dense_pre_recurrent
                    ),
                )
            elif stage_name == "pre_update_m0":
                replay_output, replay_post_conv, replay_post_recurrent = _replay_deltanet_linear_step(
                    adapter,
                    record,
                    conv_state=(
                        _quantize_state_tensor(
                            dense_pre_conv,
                            bits=resolved_conv_bits,
                            group_size=resolved_conv_group_size,
                            mode=conv_mode,
                        )
                        if _statecache_scope_includes_conv(scope)
                        else dense_pre_conv
                    ),
                    recurrent_state=(
                        _quantize_state_tensor(
                            dense_pre_recurrent,
                            bits=resolved_recurrent_bits,
                            group_size=resolved_recurrent_group_size,
                            mode=recurrent_mode,
                        )
                        if _statecache_scope_includes_recurrent(scope)
                        else dense_pre_recurrent
                    ),
                )
            elif stage_name == "post_update_m0":
                replay_output = dense_output
                replay_post_conv = (
                    _quantize_state_tensor(
                        dense_post_conv,
                        bits=resolved_conv_bits,
                        group_size=resolved_conv_group_size,
                        mode=conv_mode,
                    )
                    if _statecache_scope_includes_conv(scope)
                    else dense_post_conv
                )
                replay_post_recurrent = (
                    _quantize_state_tensor(
                        dense_post_recurrent,
                        bits=resolved_recurrent_bits,
                        group_size=resolved_recurrent_group_size,
                        mode=recurrent_mode,
                    )
                    if _statecache_scope_includes_recurrent(scope)
                    else dense_post_recurrent
                )
                if step_index > 0:
                    replay_output, replay_post_conv_dense, replay_post_recurrent_dense = _replay_deltanet_linear_step(
                        adapter,
                        record,
                        conv_state=carried_conv,
                        recurrent_state=carried_recurrent,
                    )
                    replay_post_conv = (
                        _quantize_state_tensor(
                            replay_post_conv_dense,
                            bits=resolved_conv_bits,
                            group_size=resolved_conv_group_size,
                            mode=conv_mode,
                        )
                        if _statecache_scope_includes_conv(scope)
                        else replay_post_conv_dense
                    )
                    replay_post_recurrent = (
                        _quantize_state_tensor(
                            replay_post_recurrent_dense,
                            bits=resolved_recurrent_bits,
                            group_size=resolved_recurrent_group_size,
                            mode=recurrent_mode,
                        )
                        if _statecache_scope_includes_recurrent(scope)
                        else replay_post_recurrent_dense
                    )
                carried_conv = replay_post_conv
                carried_recurrent = replay_post_recurrent
            elif stage_name == "full_state_path_m0":
                input_conv = (
                    (
                        _quantize_state_tensor(
                            dense_pre_conv,
                            bits=resolved_conv_bits,
                            group_size=resolved_conv_group_size,
                            mode=conv_mode,
                        )
                        if _statecache_scope_includes_conv(scope)
                        else dense_pre_conv
                    )
                    if carried_conv is None
                    else carried_conv
                )
                input_recurrent = (
                    (
                        _quantize_state_tensor(
                            dense_pre_recurrent,
                            bits=resolved_recurrent_bits,
                            group_size=resolved_recurrent_group_size,
                            mode=recurrent_mode,
                        )
                        if _statecache_scope_includes_recurrent(scope)
                        else dense_pre_recurrent
                    )
                    if carried_recurrent is None
                    else carried_recurrent
                )
                replay_output, replay_post_conv_dense, replay_post_recurrent_dense = _replay_deltanet_linear_step(
                    adapter,
                    record,
                    conv_state=input_conv,
                    recurrent_state=input_recurrent,
                )
                replay_post_conv = (
                    _quantize_state_tensor(
                        replay_post_conv_dense,
                        bits=resolved_conv_bits,
                        group_size=resolved_conv_group_size,
                        mode=conv_mode,
                    )
                    if _statecache_scope_includes_conv(scope)
                    else replay_post_conv_dense
                )
                replay_post_recurrent = (
                    _quantize_state_tensor(
                        replay_post_recurrent_dense,
                        bits=resolved_recurrent_bits,
                        group_size=resolved_recurrent_group_size,
                        mode=recurrent_mode,
                    )
                    if _statecache_scope_includes_recurrent(scope)
                    else replay_post_recurrent_dense
                )
                carried_conv = replay_post_conv
                carried_recurrent = replay_post_recurrent
            else:
                raise ValueError(f"unsupported DeltaNet ablation stage {stage_name!r}")

            state_abs_error = max(
                _max_abs_error(replay_post_conv, dense_post_conv),
                _max_abs_error(replay_post_recurrent, dense_post_recurrent),
            )
            state_rel_error = max(
                _max_rel_error(replay_post_conv, dense_post_conv),
                _max_rel_error(replay_post_recurrent, dense_post_recurrent),
            )
            output_abs_error = _max_abs_error(replay_output, dense_output)
            layer_max_abs = max(layer_max_abs, state_abs_error)
            layer_max_rel = max(layer_max_rel, state_rel_error)
            layer_output_max = max(layer_output_max, output_abs_error)
            layer_step_output_errors.append(output_abs_error)

        per_layer_max_abs_error[str(layer_id)] = layer_max_abs
        per_layer_max_rel_error[str(layer_id)] = layer_max_rel
        per_layer_output_max_abs_error[str(layer_id)] = layer_output_max
        if len(per_step_output_max_abs_error) < len(layer_step_output_errors):
            per_step_output_max_abs_error.extend([0.0] * (len(layer_step_output_errors) - len(per_step_output_max_abs_error)))
        for step_index, value in enumerate(layer_step_output_errors):
            per_step_output_max_abs_error[step_index] = max(per_step_output_max_abs_error[step_index], value)

    error_grows = False
    if per_step_output_max_abs_error:
        error_grows = per_step_output_max_abs_error[-1] > (per_step_output_max_abs_error[0] + 1e-6)

    return StateAblationResult(
        stage_name=stage_name,
        bits=bits,
        max_abs_error=max(per_layer_max_abs_error.values(), default=0.0),
        max_rel_error=max(per_layer_max_rel_error.values(), default=0.0),
        output_max_abs_error=max(per_layer_output_max_abs_error.values(), default=0.0),
        error_grows_step_to_step=bool(error_grows),
        per_layer_max_abs_error=per_layer_max_abs_error,
        per_layer_max_rel_error=per_layer_max_rel_error,
        per_layer_output_max_abs_error=per_layer_output_max_abs_error,
        per_step_output_max_abs_error=per_step_output_max_abs_error,
    )


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


def inspect_qwen35_deltanet_state(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
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
    dense_capture = _run_qwen35_deltanet_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    return _summarize_deltanet_state_capture(
        model,
        adapter,
        input_ids=input_ids,
        decode_steps=decode_steps,
        prefill_outputs=dense_capture["prefill_outputs"],
        prefill_ms=float(dense_capture["prefill_ms"]),
        dense_decode_ms_total=float(dense_capture["decode_ms_total"]),
        per_step_records=dense_capture["capture_records"],
    )


def build_qwen35_deltanet_state_sample(
    per_step_records: list[list[Qwen35DeltaNetStateRecord]],
    *,
    prompt_length: int,
    layer_id: int | None = None,
    state_kind: Literal["recurrent", "conv"] = "recurrent",
) -> dict[str, Any]:
    records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]] = {}
    for step_records in per_step_records:
        for record in step_records:
            records_by_layer.setdefault(int(record.layer_id), []).append(record)
    if not records_by_layer:
        raise ValueError("no DeltaNet state records were captured")
    selected_layer_id = min(records_by_layer) if layer_id is None else int(layer_id)
    if selected_layer_id not in records_by_layer:
        raise ValueError(f"requested DeltaNet sample layer {selected_layer_id} was not captured")
    records = sorted(records_by_layer[selected_layer_id], key=lambda item: int(item.step_index))
    if state_kind == "recurrent":
        initial_state = records[0].pre_recurrent_state
        pre_states = [record.pre_recurrent_state for record in records]
        post_states = [record.post_recurrent_state for record in records]
    else:
        initial_state = records[0].pre_conv_state
        pre_states = [record.pre_conv_state for record in records]
        post_states = [record.post_conv_state for record in records]
    if initial_state is None:
        raise ValueError(f"captured DeltaNet {state_kind} state is unavailable for layer {selected_layer_id}")
    if any(state is None for state in pre_states) or any(state is None for state in post_states):
        raise ValueError(f"incomplete DeltaNet {state_kind} state history for layer {selected_layer_id}")

    token_indices: list[int] = []
    update_arrays: list[np.ndarray] = []
    for record, pre_state, post_state in zip(records, pre_states, post_states):
        assert pre_state is not None and post_state is not None
        token_indices.append(int(record.token_index))
        update_arrays.append((post_state - pre_state).detach().to(dtype=torch.float32).cpu().numpy())

    return {
        "source": "qwen35_deltanet_capture",
        "state_kind": state_kind,
        "layer_id": selected_layer_id,
        "prompt_length": int(prompt_length),
        "token_indices": token_indices,
        "initial_state": initial_state.detach().to(dtype=torch.float32).cpu().numpy(),
        "update_deltas": np.stack(update_arrays, axis=0),
    }


def save_qwen35_deltanet_state_sample(path: str | Path, sample: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        source=np.asarray(sample["source"]),
        state_kind=np.asarray(sample["state_kind"]),
        layer_id=np.asarray(int(sample["layer_id"]), dtype=np.int64),
        prompt_length=np.asarray(int(sample["prompt_length"]), dtype=np.int64),
        token_indices=np.asarray(sample["token_indices"], dtype=np.int64),
        initial_state=np.asarray(sample["initial_state"], dtype=np.float32),
        update_deltas=np.asarray(sample["update_deltas"], dtype=np.float32),
    )


def capture_qwen35_deltanet_state_sample(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    layer_id: int | None = None,
    state_kind: Literal["recurrent", "conv"] = "recurrent",
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
    dense_capture = _run_qwen35_deltanet_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    sample = build_qwen35_deltanet_state_sample(
        dense_capture["capture_records"],
        prompt_length=int(input_ids.shape[1]),
        layer_id=layer_id,
        state_kind=state_kind,
    )
    sample["decode_steps"] = int(decode_steps)
    return sample


def run_qwen35_deltanet_state_ablation_harness(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    group_size: int = 32,
    bits: tuple[int, ...] = (8, 4),
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
    dense_capture = _run_qwen35_deltanet_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    linear_records = [
        record
        for step_records in dense_capture["capture_records"]
        for record in step_records
    ]
    records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]] = {}
    for record in linear_records:
        records_by_layer.setdefault(int(record.layer_id), []).append(record)

    stage_runs: list[dict[str, Any]] = []
    dominant_failure_stage = "dense_baseline"
    dominant_failure_error = -1.0
    for bit_width in bits:
        for stage_name in ("dense_baseline", "readout_only_m0", "post_update_m0", "pre_update_m0", "full_state_path_m0", "escape_m3"):
            stage_result = _run_deltanet_ablation_stage(
                adapter,
                records_by_layer=records_by_layer,
                stage_name=stage_name,
                bits=None if stage_name in {"dense_baseline", "escape_m3"} else int(bit_width),
                group_size=int(group_size),
            )
            stage_runs.append(stage_result.to_dict())
            if stage_name in {"readout_only_m0", "post_update_m0", "pre_update_m0", "full_state_path_m0"}:
                if stage_result.output_max_abs_error > dominant_failure_error:
                    dominant_failure_error = float(stage_result.output_max_abs_error)
                    dominant_failure_stage = stage_name

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(dense_capture["prefill_ms"]),
        "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "text_only": True,
        "dotcache_ready": False,
        "deltanet_state_ready": True,
        "deltanet_state_ablation_ready": True,
        "runtime_mode": "dense_deltanet_state_ablation",
        "uses_native_qwen35_class": True,
        "deltanet_ablation_group_size": int(group_size),
        "deltanet_ablation_bits": [int(bit) for bit in bits],
        "deltanet_ablation_results": stage_runs,
        "deltanet_dominant_failure_stage": dominant_failure_stage,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    hybrid_summary = summarize_qwen35_hybrid_state(dense_capture["prefill_outputs"].past_key_values, model)
    result.update(hybrid_summary)
    result["deltanet_conv_state_bytes"] = int(sum(record["conv_state_bytes"] for record in hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"))
    result["deltanet_recurrent_state_bytes"] = int(sum(record["recurrent_state_bytes"] for record in hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"))
    result["deltanet_total_state_bytes"] = int(result["deltanet_conv_state_bytes"] + result["deltanet_recurrent_state_bytes"])
    result["deltanet_state_layers"] = [record for record in hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"]
    return result


def run_qwen35_deltanet_statecache_readout_harness(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    group_size: int = 32,
    bits: int = 8,
    layer_bits_overrides: dict[int, int] | None = None,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    conv_bits: int | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
    renorm_interval: int = 0,
    recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    conv_renorm_interval_overrides: dict[int, int] | None = None,
    recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
    readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
    readout_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    readout_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
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
    device = input_ids.device
    dense_capture_cuda_memory_baseline = _begin_cuda_memory_region(device)
    dense_capture = _run_qwen35_deltanet_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    dense_capture_cuda_memory = _end_cuda_memory_region(device, dense_capture_cuda_memory_baseline)
    linear_records = [
        record
        for step_records in dense_capture["capture_records"]
        for record in step_records
    ]
    records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]] = {}
    for record in linear_records:
        records_by_layer.setdefault(int(record.layer_id), []).append(record)

    prefill_partition = adapter.partition_hybrid_state(dense_capture["prefill_outputs"].past_key_values)
    deltanet_layer_ids = adapter.deltanet_layer_ids()
    resolved_scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    resolved_conv_bits = int(bits if conv_bits is None else conv_bits)
    if recurrent_mode_policy is not None and recurrent_mode_overrides:
        raise ValueError("recurrent_mode_policy cannot be combined with explicit recurrent mode overrides")
    resolved_recurrent_mode_policy_overrides, resolved_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=recurrent_mode_policy,
        )
    )
    resolved_base_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_recurrent_mode_policy_overrides,
        recurrent_mode_overrides,
    )
    if readout_recurrent_policy is not None and readout_recurrent_renorm_interval_overrides:
        raise ValueError("readout_recurrent_policy cannot be combined with explicit readout recurrent renorm overrides")
    if readout_recurrent_mode_policy is not None and readout_recurrent_mode_overrides:
        raise ValueError("readout_recurrent_mode_policy cannot be combined with explicit readout recurrent mode overrides")
    resolved_readout_recurrent_policy_overrides, resolved_readout_recurrent_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=readout_recurrent_policy,
        )
    )
    resolved_readout_recurrent_mode_policy_overrides, resolved_readout_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_mode_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=readout_recurrent_mode_policy,
        )
    )
    has_explicit_readout_recurrent_policy = (
        bool(resolved_readout_recurrent_policy_overrides)
        or bool(resolved_readout_recurrent_mode_policy_overrides)
        or readout_recurrent_renorm_interval_overrides is not None
        or readout_recurrent_mode_overrides is not None
    )
    resolved_readout_recurrent_renorm_interval_overrides = _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
        recurrent_renorm_interval_overrides,
        (
            readout_recurrent_renorm_interval_overrides
            if readout_recurrent_renorm_interval_overrides is not None
            else resolved_readout_recurrent_policy_overrides
        ),
    )
    resolved_readout_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        (
            readout_recurrent_mode_overrides
            if readout_recurrent_mode_overrides is not None
            else resolved_readout_recurrent_mode_policy_overrides
        ),
    )
    resolved_post_update_recurrent_renorm_interval_overrides = (
        _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
            recurrent_renorm_interval_overrides,
            post_update_recurrent_renorm_interval_overrides,
        )
    )
    resolved_post_update_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        post_update_recurrent_mode_overrides,
    )
    resolved_recurrent_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=resolved_base_recurrent_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    resolved_conv_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=conv_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    byte_summary = _summarize_qwen35_deltanet_statecache_bytes(
        prefill_partition,
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        recurrent_bits=int(bits),
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_policy
            else (
            resolved_post_update_recurrent_mode_overrides
            if state_stage == "post_update_m0"
            else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )

    dense_generated_ids = [
        int(decode_input[0, 0].item())
        for decode_input in dense_capture.get("decode_inputs", [])
    ]

    statecache_generated_ids: list[int] = []
    statecache_decode_ms_total = 0.0
    statecache_prefill_cuda_memory: dict[str, int] = {}
    statecache_decode_cuda_memory: dict[str, int] = {}
    if decode_steps > 0:
        # Run the StateCache decode from a fresh prefill. The dense capture path
        # performs additional decode steps for record collection, and the native
        # Qwen3.5 linear-attention stack does not behave as a purely stateless
        # function of `past_key_values` across those extra calls.
        statecache_prefill_cuda_memory_baseline = _begin_cuda_memory_region(device)
        statecache_prefill_outputs = _run_dense_prefill(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        statecache_prefill_cuda_memory = _end_cuda_memory_region(device, statecache_prefill_cuda_memory_baseline)
        statecache_past_key_values = _clone_qwen35_past_key_values(statecache_prefill_outputs.past_key_values)
        if state_stage == "post_update_m0":
            _prepare_qwen35_deltanet_statecache(
                statecache_past_key_values,
                layer_ids=deltanet_layer_ids,
                recurrent_bits=int(bits),
                conv_bits=resolved_conv_bits,
                group_size=int(group_size),
                renorm=False,
                statecache_scope=resolved_scope,
                recurrent_renorm_interval=int(renorm_interval),
                conv_renorm_interval=int(renorm_interval),
                recurrent_layer_bits_overrides=layer_bits_overrides,
                conv_layer_bits_overrides=conv_layer_bits_overrides,
                recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
                conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                recurrent_default_mode="M0",
                conv_default_mode="M0",
                recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
                conv_mode_overrides=conv_mode_overrides,
            )
        current_input_ids = statecache_prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        statecache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
        for step_index in range(decode_steps):
            statecache_generated_ids.append(int(current_input_ids.item()))
            def _run_statecache_decode():
                if state_stage == "readout_only_m0" or has_explicit_readout_recurrent_policy:
                    _prepare_qwen35_deltanet_statecache(
                        statecache_past_key_values,
                        layer_ids=deltanet_layer_ids,
                        recurrent_bits=int(bits),
                        conv_bits=resolved_conv_bits,
                        group_size=int(group_size),
                        renorm=False,
                        statecache_scope=resolved_scope,
                        recurrent_renorm_interval=int(renorm_interval),
                        conv_renorm_interval=int(renorm_interval),
                        recurrent_layer_bits_overrides=layer_bits_overrides,
                        conv_layer_bits_overrides=conv_layer_bits_overrides,
                        recurrent_renorm_interval_overrides=resolved_readout_recurrent_renorm_interval_overrides,
                        conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                        recurrent_default_mode="M0",
                        conv_default_mode="M0",
                        recurrent_mode_overrides=resolved_readout_recurrent_mode_overrides,
                        conv_mode_overrides=conv_mode_overrides,
                    )
                return _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=statecache_past_key_values,
                    cache_position=cache_position,
                )

            outputs, step_ms = _timed_call(
                _run_statecache_decode,
                device=input_ids.device,
            )
            statecache_decode_ms_total += step_ms
            statecache_past_key_values = outputs.past_key_values
            if state_stage == "post_update_m0":
                _prepare_qwen35_deltanet_statecache(
                    statecache_past_key_values,
                    layer_ids=deltanet_layer_ids,
                    recurrent_bits=int(bits),
                    conv_bits=resolved_conv_bits,
                    group_size=int(group_size),
                    step_index=step_index,
                    statecache_scope=resolved_scope,
                    recurrent_renorm_interval=int(renorm_interval),
                    conv_renorm_interval=int(renorm_interval),
                    recurrent_layer_bits_overrides=layer_bits_overrides,
                    conv_layer_bits_overrides=conv_layer_bits_overrides,
                    recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
                    conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                    recurrent_default_mode="M0",
                    conv_default_mode="M0",
                    recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
                    conv_mode_overrides=conv_mode_overrides,
                )
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1
        statecache_decode_cuda_memory = _end_cuda_memory_region(device, statecache_decode_cuda_memory_baseline)

    greedy_matches = sum(
        1 for dense_token, statecache_token in zip(dense_generated_ids, statecache_generated_ids) if dense_token == statecache_token
    )
    greedy_token_agreement_rate = (
        float(greedy_matches / max(len(dense_generated_ids), 1))
        if dense_generated_ids
        else 1.0
    )
    first_divergence_step = next(
        (
            int(step_index)
            for step_index, (dense_token, statecache_token) in enumerate(zip(dense_generated_ids, statecache_generated_ids))
            if dense_token != statecache_token
        ),
        None,
    )

    # Keep the benchmarked generation path on a clean model/cache state. The
    # ablation replay uses the same wrapped DeltaNet modules and is only needed
    # for summary metrics, so compute it after the real decode loop.
    statecache_result = _run_deltanet_ablation_stage(
        adapter,
        records_by_layer=records_by_layer,
        stage_name=str(state_stage),
        bits=int(bits),
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_policy
            else (
            resolved_post_update_recurrent_mode_overrides
            if state_stage == "post_update_m0"
            else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(dense_capture["prefill_ms"]),
        "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "deltanet_dense_generated_ids": dense_generated_ids,
        "deltanet_statecache_generated_ids": statecache_generated_ids,
        "deltanet_statecache_decode_ms_per_step": float(statecache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "deltanet_statecache_greedy_token_agreement_rate": greedy_token_agreement_rate,
        "text_only": True,
        "dotcache_ready": False,
        "deltanet_state_ready": True,
        "deltanet_statecache_ready": True,
        "runtime_mode": "dense_deltanet_statecache_readout",
        "uses_native_qwen35_class": True,
        "deltanet_statecache_scope": resolved_scope,
        "deltanet_statecache_stage_name": str(state_stage),
        "deltanet_statecache_group_size": int(group_size),
        "deltanet_statecache_bits": int(bits),
        "deltanet_statecache_conv_bits": int(resolved_conv_bits),
        "deltanet_statecache_readout_recurrent_policy": (
            str(readout_recurrent_policy) if readout_recurrent_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_policy_band": resolved_readout_recurrent_policy_band,
        "deltanet_statecache_recurrent_mode_policy": (
            str(recurrent_mode_policy) if recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_recurrent_mode_policy_band": resolved_recurrent_mode_policy_band,
        "deltanet_statecache_readout_recurrent_mode_policy": (
            str(readout_recurrent_mode_policy) if readout_recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_mode_policy_band": resolved_readout_recurrent_mode_policy_band,
        "deltanet_statecache_layer_bits": byte_summary["deltanet_statecache_per_layer_recurrent_bits"],
        "deltanet_statecache_conv_layer_bits": byte_summary["deltanet_statecache_per_layer_conv_bits"],
        "deltanet_statecache_mode": "M0",
        "deltanet_statecache_renorm_interval": int(renorm_interval),
        "deltanet_statecache_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_readout_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_readout_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_post_update_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_post_update_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_conv_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((conv_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_recurrent_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted(resolved_recurrent_mode_overrides.items()) if mode != "M0"
        },
        "deltanet_statecache_readout_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_readout_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_post_update_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_post_update_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_conv_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted(resolved_conv_mode_overrides.items()) if mode != "M0"
        },
        "deltanet_statecache_result": statecache_result.to_dict(),
        "deltanet_statecache_output_max_abs_error": float(statecache_result.output_max_abs_error),
        "deltanet_statecache_max_abs_error": float(statecache_result.max_abs_error),
        "deltanet_statecache_error_grows_step_to_step": bool(statecache_result.error_grows_step_to_step),
        "deltanet_statecache_per_layer_recurrent_mode": byte_summary["deltanet_statecache_per_layer_recurrent_mode"],
        "deltanet_statecache_per_layer_conv_mode": byte_summary["deltanet_statecache_per_layer_conv_mode"],
    }
    result.update(byte_summary)
    if first_divergence_step is not None:
        result["deltanet_statecache_first_divergence_step"] = first_divergence_step
    for key, value in dense_capture_cuda_memory.items():
        result[f"deltanet_dense_capture_{key}"] = value
    for key, value in statecache_prefill_cuda_memory.items():
        result[f"deltanet_statecache_prefill_{key}"] = value
    for key, value in statecache_decode_cuda_memory.items():
        result[f"deltanet_statecache_decode_{key}"] = value
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    hybrid_summary = summarize_qwen35_hybrid_state(dense_capture["prefill_outputs"].past_key_values, model)
    result.update(hybrid_summary)
    result["deltanet_total_state_bytes"] = int(result["deltanet_conv_state_bytes"] + result["deltanet_recurrent_state_bytes"])
    result["deltanet_state_layers"] = [record for record in hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"]
    dense_text = _decode_text(tokenizer, dense_generated_ids)
    if dense_text is not None:
        result["deltanet_dense_text"] = dense_text
    statecache_text = _decode_text(tokenizer, statecache_generated_ids)
    if statecache_text is not None:
        result["deltanet_statecache_text"] = statecache_text
    return result


def run_qwen35_deltanet_statecache_serving_harness(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    group_size: int = 32,
    bits: int = 8,
    layer_bits_overrides: dict[int, int] | None = None,
    recurrent_group_size_policy: Qwen35DeltaNetStateCacheRecurrentGroupSizePolicy | str | None = None,
    recurrent_layer_group_size_overrides: dict[int, int] | None = None,
    state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
    renorm_interval: int = 0,
    recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
    readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
    readout_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    readout_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
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
    device = input_ids.device
    deltanet_layer_ids = adapter.deltanet_layer_ids()
    if recurrent_mode_policy is not None and recurrent_mode_overrides:
        raise ValueError("recurrent_mode_policy cannot be combined with explicit recurrent mode overrides")
    if recurrent_group_size_policy is not None and recurrent_layer_group_size_overrides:
        raise ValueError(
            "recurrent_group_size_policy cannot be combined with explicit recurrent layer group-size overrides"
        )
    resolved_recurrent_mode_policy_overrides, resolved_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=recurrent_mode_policy,
        )
    )
    resolved_recurrent_group_size_policy_overrides, resolved_recurrent_group_size_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_group_size_policy(
            prompt_length=int(input_ids.shape[1]),
            decode_steps=int(decode_steps),
            policy=recurrent_group_size_policy,
        )
    )
    resolved_recurrent_layer_group_size_overrides = dict(
        recurrent_layer_group_size_overrides
        if recurrent_layer_group_size_overrides
        else resolved_recurrent_group_size_policy_overrides
    )
    resolved_base_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_recurrent_mode_policy_overrides,
        recurrent_mode_overrides,
    )
    if readout_recurrent_policy is not None and readout_recurrent_renorm_interval_overrides:
        raise ValueError("readout_recurrent_policy cannot be combined with explicit readout recurrent renorm overrides")
    if readout_recurrent_mode_policy is not None and readout_recurrent_mode_overrides:
        raise ValueError("readout_recurrent_mode_policy cannot be combined with explicit readout recurrent mode overrides")
    resolved_readout_recurrent_policy_overrides, resolved_readout_recurrent_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=readout_recurrent_policy,
        )
    )
    resolved_readout_recurrent_mode_policy_overrides, resolved_readout_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_mode_policy(
            prompt_length=int(input_ids.shape[1]),
            policy=readout_recurrent_mode_policy,
        )
    )
    has_explicit_readout_recurrent_policy = (
        bool(resolved_readout_recurrent_policy_overrides)
        or bool(resolved_readout_recurrent_mode_policy_overrides)
        or readout_recurrent_renorm_interval_overrides is not None
        or readout_recurrent_mode_overrides is not None
    )
    resolved_readout_recurrent_renorm_interval_overrides = _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
        recurrent_renorm_interval_overrides,
        (
            readout_recurrent_renorm_interval_overrides
            if readout_recurrent_renorm_interval_overrides is not None
            else resolved_readout_recurrent_policy_overrides
        ),
    )
    resolved_readout_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        (
            readout_recurrent_mode_overrides
            if readout_recurrent_mode_overrides is not None
            else resolved_readout_recurrent_mode_policy_overrides
        ),
    )
    resolved_post_update_recurrent_renorm_interval_overrides = (
        _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
            recurrent_renorm_interval_overrides,
            post_update_recurrent_renorm_interval_overrides,
        )
    )
    resolved_post_update_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        post_update_recurrent_mode_overrides,
    )

    prefill_cuda_memory_baseline = _begin_cuda_memory_region(device)
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    prefill_cuda_memory = _end_cuda_memory_region(device, prefill_cuda_memory_baseline)

    prefill_partition = adapter.partition_hybrid_state(prefill_outputs.past_key_values)
    resolved_recurrent_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=(
                resolved_readout_recurrent_mode_overrides
                if has_explicit_readout_recurrent_policy
                else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else resolved_base_recurrent_mode_overrides
                )
            ),
        )
        for layer_id in deltanet_layer_ids
    }
    recurrent_dense_bytes = 0
    recurrent_statecache_bytes = 0
    per_layer_dense_recurrent_bytes: dict[str, int] = {}
    per_layer_statecache_recurrent_bytes: dict[str, int] = {}
    per_layer_statecache_bits: dict[str, int] = {}
    per_layer_statecache_modes: dict[str, str] = {}
    for layer in prefill_partition.fixed_resident_layers:
        if layer.recurrent_state is None:
            continue
        layer_id = str(int(layer.layer_id))
        recurrent_mode = resolved_recurrent_mode_overrides.get(int(layer.layer_id), "M0")
        dense_bytes = int(layer.recurrent_state_bytes)
        layer_bits = _resolve_deltanet_statecache_bits(
            int(layer.layer_id),
            default_bits=int(bits),
            layer_bits_overrides=layer_bits_overrides,
        )
        layer_group_size = _resolve_deltanet_statecache_group_size(
            int(layer.layer_id),
            default_group_size=int(group_size),
            layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
        )
        compressed_bytes = _compressed_state_nbytes(
            layer.recurrent_state,
            bits=layer_bits,
            group_size=layer_group_size,
            mode=recurrent_mode,
        )
        recurrent_dense_bytes += dense_bytes
        recurrent_statecache_bytes += compressed_bytes
        per_layer_dense_recurrent_bytes[layer_id] = dense_bytes
        per_layer_statecache_recurrent_bytes[layer_id] = compressed_bytes
        per_layer_statecache_bits[layer_id] = int(layer_bits)
        per_layer_statecache_modes[layer_id] = recurrent_mode

    conv_state_bytes = int(sum(layer.conv_state_bytes for layer in prefill_partition.fixed_resident_layers))
    dense_fixed_resident_bytes = int(sum(layer.fixed_resident_state_bytes for layer in prefill_partition.fixed_resident_layers))
    statecache_fixed_resident_bytes = int(conv_state_bytes + recurrent_statecache_bytes)

    statecache_past_key_values = prefill_outputs.past_key_values
    if state_stage == "post_update_m0":
        _prepare_qwen35_deltanet_recurrent_statecache(
            statecache_past_key_values,
            layer_ids=deltanet_layer_ids,
            bits=int(bits),
            group_size=int(group_size),
            renorm=False,
            renorm_interval=int(renorm_interval),
            layer_bits_overrides=layer_bits_overrides,
            layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
            renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
            default_mode="M0",
            mode_overrides=resolved_post_update_recurrent_mode_overrides,
        )

    statecache_generated_ids: list[int] = []
    statecache_decode_ms_total = 0.0
    statecache_per_step_decode_ms: list[float] = []
    statecache_decode_cuda_memory: dict[str, int] = {}
    if decode_steps > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        statecache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
        for step_index in range(decode_steps):
            statecache_generated_ids.append(int(current_input_ids.item()))

            def _run_statecache_decode():
                if state_stage == "readout_only_m0" or has_explicit_readout_recurrent_policy:
                    _prepare_qwen35_deltanet_recurrent_statecache(
                        statecache_past_key_values,
                        layer_ids=deltanet_layer_ids,
                        bits=int(bits),
                        group_size=int(group_size),
                        renorm=False,
                        renorm_interval=int(renorm_interval),
                        layer_bits_overrides=layer_bits_overrides,
                        layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
                        renorm_interval_overrides=resolved_readout_recurrent_renorm_interval_overrides,
                        default_mode="M0",
                        mode_overrides=resolved_readout_recurrent_mode_overrides,
                    )
                return _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=statecache_past_key_values,
                    cache_position=cache_position,
                )

            outputs, step_ms = _timed_call(_run_statecache_decode, device=device)
            statecache_decode_ms_total += step_ms
            statecache_per_step_decode_ms.append(float(step_ms))
            statecache_past_key_values = outputs.past_key_values
            if state_stage == "post_update_m0":
                _prepare_qwen35_deltanet_recurrent_statecache(
                    statecache_past_key_values,
                    layer_ids=deltanet_layer_ids,
                    bits=int(bits),
                    group_size=int(group_size),
                    step_index=step_index,
                    renorm_interval=int(renorm_interval),
                    layer_bits_overrides=layer_bits_overrides,
                    layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
                    renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
                    default_mode="M0",
                    mode_overrides=resolved_post_update_recurrent_mode_overrides,
                )
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1
        statecache_decode_cuda_memory = _end_cuda_memory_region(device, statecache_decode_cuda_memory_baseline)

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(prefill_ms),
        "deltanet_statecache_generated_ids": statecache_generated_ids,
        "deltanet_statecache_per_step_decode_ms": statecache_per_step_decode_ms,
        "deltanet_statecache_decode_ms_per_step": float(statecache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "text_only": True,
        "dotcache_ready": False,
        "deltanet_state_ready": True,
        "deltanet_statecache_ready": True,
        "runtime_mode": "statecache_serving_only",
        "uses_native_qwen35_class": True,
        "deltanet_statecache_stage_name": str(state_stage),
        "deltanet_statecache_group_size": int(group_size),
        "deltanet_statecache_bits": int(bits),
        "deltanet_statecache_recurrent_group_size_policy": (
            str(recurrent_group_size_policy) if recurrent_group_size_policy is not None else None
        ),
        "deltanet_statecache_recurrent_group_size_policy_band": resolved_recurrent_group_size_policy_band,
        "deltanet_statecache_recurrent_layer_group_size_overrides": {
            str(layer_id): int(group)
            for layer_id, group in sorted((resolved_recurrent_layer_group_size_overrides or {}).items())
            if int(group) != int(group_size)
        },
        "deltanet_statecache_readout_recurrent_policy": (
            str(readout_recurrent_policy) if readout_recurrent_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_policy_band": resolved_readout_recurrent_policy_band,
        "deltanet_statecache_recurrent_mode_policy": (
            str(recurrent_mode_policy) if recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_recurrent_mode_policy_band": resolved_recurrent_mode_policy_band,
        "deltanet_statecache_readout_recurrent_mode_policy": (
            str(readout_recurrent_mode_policy) if readout_recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_mode_policy_band": resolved_readout_recurrent_mode_policy_band,
        "deltanet_statecache_layer_bits": dict(sorted(per_layer_statecache_bits.items())),
        "deltanet_statecache_mode": "M0",
        "deltanet_statecache_renorm_interval": int(renorm_interval),
        "deltanet_statecache_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_readout_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_readout_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_post_update_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_post_update_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_recurrent_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted(resolved_recurrent_mode_overrides.items()) if mode != "M0"
        },
        "deltanet_statecache_readout_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_readout_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_post_update_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_post_update_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_conv_state_bytes": conv_state_bytes,
        "deltanet_recurrent_state_bytes": recurrent_dense_bytes,
        "deltanet_statecache_recurrent_state_bytes": int(recurrent_statecache_bytes),
        "deltanet_dense_fixed_resident_bytes": dense_fixed_resident_bytes,
        "deltanet_statecache_fixed_resident_bytes": statecache_fixed_resident_bytes,
        "deltanet_statecache_effective_recurrent_compression_ratio": (
            float(recurrent_dense_bytes / max(recurrent_statecache_bytes, 1)) if recurrent_dense_bytes > 0 else 1.0
        ),
        "deltanet_statecache_effective_fixed_resident_compression_ratio": (
            float(dense_fixed_resident_bytes / max(statecache_fixed_resident_bytes, 1)) if dense_fixed_resident_bytes > 0 else 1.0
        ),
        "deltanet_statecache_per_layer_dense_recurrent_bytes": per_layer_dense_recurrent_bytes,
        "deltanet_statecache_per_layer_recurrent_bytes": per_layer_statecache_recurrent_bytes,
        "deltanet_statecache_per_layer_recurrent_group_size": {
            str(layer.layer_id): _resolve_deltanet_statecache_group_size(
                int(layer.layer_id),
                default_group_size=int(group_size),
                layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
            )
            for layer in prefill_partition.fixed_resident_layers
            if layer.recurrent_state is not None
        },
        "deltanet_statecache_per_layer_recurrent_mode": per_layer_statecache_modes,
    }
    for key, value in prefill_cuda_memory.items():
        result[f"deltanet_statecache_prefill_{key}"] = value
    for key, value in statecache_decode_cuda_memory.items():
        result[f"deltanet_statecache_decode_{key}"] = value
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    prefill_hybrid_summary = summarize_qwen35_hybrid_state(prefill_outputs.past_key_values, model)
    result.update(prefill_hybrid_summary)
    result["deltanet_total_state_bytes"] = int(result["deltanet_conv_state_bytes"] + result["deltanet_recurrent_state_bytes"])
    result["deltanet_state_layers"] = [record for record in prefill_hybrid_summary["hybrid_state_layers"] if record["layer_type"] == "linear_attention"]
    statecache_text = _decode_text(tokenizer, statecache_generated_ids)
    if statecache_text is not None:
        result["deltanet_statecache_text"] = statecache_text
    return result


def run_qwen35_deltanet_statecache_loss_harness(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    group_size: int = 32,
    bits: int = 8,
    layer_bits_overrides: dict[int, int] | None = None,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    conv_bits: int | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    state_stage: Qwen35DeltaNetStateCacheStage = "readout_only_m0",
    renorm_interval: int = 0,
    recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    conv_renorm_interval_overrides: dict[int, int] | None = None,
    recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    readout_recurrent_policy: Qwen35DeltaNetStateCacheReadoutPolicy | str | None = None,
    readout_recurrent_mode_policy: Qwen35DeltaNetStateCacheReadoutModePolicy | str | None = None,
    post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
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

    dense_result = run_qwen35_text_loss_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=prefix_length,
        eval_steps=eval_steps,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=input_ids.device,
    )
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu()]
    statecache_past_key_values = _clone_qwen35_past_key_values(prefill_outputs.past_key_values)
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    statecache_decode_ms_total = 0.0
    deltanet_layer_ids = adapter.deltanet_layer_ids()
    resolved_scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    resolved_conv_bits = int(bits if conv_bits is None else conv_bits)
    if recurrent_mode_policy is not None and recurrent_mode_overrides:
        raise ValueError("recurrent_mode_policy cannot be combined with explicit recurrent mode overrides")
    resolved_recurrent_mode_policy_overrides, resolved_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
            prompt_length=int(prefix_length),
            policy=recurrent_mode_policy,
        )
    )
    resolved_base_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_recurrent_mode_policy_overrides,
        recurrent_mode_overrides,
    )
    resolved_readout_recurrent_policy_overrides, resolved_readout_recurrent_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_policy(
            prompt_length=int(prefix_length),
            policy=readout_recurrent_policy,
        )
    )
    resolved_readout_recurrent_mode_policy_overrides, resolved_readout_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_readout_mode_policy(
            prompt_length=int(prefix_length),
            policy=readout_recurrent_mode_policy,
        )
    )
    resolved_post_update_recurrent_renorm_interval_overrides = (
        _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
            recurrent_renorm_interval_overrides,
            post_update_recurrent_renorm_interval_overrides,
        )
    )
    resolved_readout_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        resolved_readout_recurrent_mode_policy_overrides,
    )
    resolved_post_update_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        post_update_recurrent_mode_overrides,
    )
    statecache_prefill_partition = adapter.partition_hybrid_state(prefill_outputs.past_key_values)
    byte_summary = _summarize_qwen35_deltanet_statecache_bytes(
        statecache_prefill_partition,
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        recurrent_bits=int(bits),
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_mode_overrides
            else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    if state_stage == "post_update_m0":
        _prepare_qwen35_deltanet_statecache(
            statecache_past_key_values,
            layer_ids=deltanet_layer_ids,
            recurrent_bits=int(bits),
            conv_bits=resolved_conv_bits,
            group_size=int(group_size),
            renorm=False,
            statecache_scope=resolved_scope,
            recurrent_renorm_interval=int(renorm_interval),
            conv_renorm_interval=int(renorm_interval),
            recurrent_layer_bits_overrides=layer_bits_overrides,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_default_mode="M0",
            conv_default_mode="M0",
            recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
            conv_mode_overrides=conv_mode_overrides,
        )

    for step_index in range(max(eval_steps - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]

        def _run_statecache_decode():
            if (
                state_stage == "readout_only_m0"
                or bool(resolved_readout_recurrent_policy_overrides)
                or bool(resolved_readout_recurrent_mode_policy_overrides)
            ):
                _prepare_qwen35_deltanet_statecache(
                    statecache_past_key_values,
                    layer_ids=deltanet_layer_ids,
                    recurrent_bits=int(bits),
                    conv_bits=resolved_conv_bits,
                    group_size=int(group_size),
                    renorm=False,
                    statecache_scope=resolved_scope,
                    recurrent_renorm_interval=int(renorm_interval),
                    conv_renorm_interval=int(renorm_interval),
                    recurrent_layer_bits_overrides=layer_bits_overrides,
                    conv_layer_bits_overrides=conv_layer_bits_overrides,
                    recurrent_renorm_interval_overrides=(
                        _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
                            recurrent_renorm_interval_overrides,
                            resolved_readout_recurrent_policy_overrides,
                        )
                    ),
                    conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                    recurrent_default_mode="M0",
                    conv_default_mode="M0",
                    recurrent_mode_overrides=resolved_readout_recurrent_mode_overrides,
                    conv_mode_overrides=conv_mode_overrides,
                )
            return _run_dense_decode_step(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=statecache_past_key_values,
                cache_position=cache_position,
            )

        outputs, step_ms = _timed_call(
            _run_statecache_decode,
            device=input_ids.device,
        )
        statecache_decode_ms_total += step_ms
        statecache_past_key_values = outputs.past_key_values
        if state_stage == "post_update_m0":
            _prepare_qwen35_deltanet_statecache(
                statecache_past_key_values,
                layer_ids=deltanet_layer_ids,
                recurrent_bits=int(bits),
                conv_bits=resolved_conv_bits,
                group_size=int(group_size),
                step_index=step_index,
                statecache_scope=resolved_scope,
                recurrent_renorm_interval=int(renorm_interval),
                conv_renorm_interval=int(renorm_interval),
                recurrent_layer_bits_overrides=layer_bits_overrides,
                conv_layer_bits_overrides=conv_layer_bits_overrides,
                recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
                conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                recurrent_default_mode="M0",
                conv_default_mode="M0",
                recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
                conv_mode_overrides=conv_mode_overrides,
            )
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

    statecache_prefill_partition = adapter.partition_hybrid_state(prefill_outputs.past_key_values)
    resolved_recurrent_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=resolved_base_recurrent_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    resolved_conv_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=conv_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    byte_summary = _summarize_qwen35_deltanet_statecache_bytes(
        statecache_prefill_partition,
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        recurrent_bits=int(bits),
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_mode_overrides
            else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )

    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_result["dense_decode_ms_per_step"]),
        "deltanet_statecache_decode_ms_per_step": float(statecache_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "dense_teacher_forced_loss": float(dense_result["dense_teacher_forced_loss"]),
        "dense_teacher_forced_perplexity": float(dense_result["dense_teacher_forced_perplexity"]),
        "dense_teacher_forced_target_match_rate": float(dense_result["dense_teacher_forced_target_match_rate"]),
        "deltanet_statecache_teacher_forced_loss": mean_loss,
        "deltanet_statecache_teacher_forced_perplexity": perplexity,
        "deltanet_statecache_teacher_forced_target_match_rate": float((predictions == target_tokens).mean()),
        "teacher_forced_loss_delta": float(mean_loss - dense_result["dense_teacher_forced_loss"]),
        "teacher_forced_perplexity_ratio": float(perplexity / max(float(dense_result["dense_teacher_forced_perplexity"]), 1e-8)),
        "teacher_forced_token_agreement_rate": float((predictions == target_tokens).mean()),
        "deltanet_statecache_scope": resolved_scope,
        "deltanet_statecache_bits": int(bits),
        "deltanet_statecache_conv_bits": int(resolved_conv_bits),
        "deltanet_statecache_layer_bits": byte_summary["deltanet_statecache_per_layer_recurrent_bits"],
        "deltanet_statecache_conv_layer_bits": byte_summary["deltanet_statecache_per_layer_conv_bits"],
        "deltanet_statecache_group_size": int(group_size),
        "deltanet_statecache_stage_name": str(state_stage),
        "deltanet_statecache_renorm_interval": int(renorm_interval),
        "deltanet_statecache_readout_recurrent_policy": (
            str(readout_recurrent_policy) if readout_recurrent_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_policy_band": resolved_readout_recurrent_policy_band,
        "deltanet_statecache_recurrent_mode_policy": (
            str(recurrent_mode_policy) if recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_recurrent_mode_policy_band": resolved_recurrent_mode_policy_band,
        "deltanet_statecache_readout_recurrent_mode_policy": (
            str(readout_recurrent_mode_policy) if readout_recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_readout_recurrent_mode_policy_band": resolved_readout_recurrent_mode_policy_band,
        "deltanet_statecache_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_readout_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted(
                _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
                    recurrent_renorm_interval_overrides,
                    resolved_readout_recurrent_policy_overrides,
                ).items()
            )
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_post_update_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_post_update_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_conv_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((conv_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_mode": "M0",
        "deltanet_statecache_recurrent_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted(resolved_recurrent_mode_overrides.items()) if mode != "M0"
        },
        "deltanet_statecache_readout_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_readout_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_post_update_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_post_update_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_conv_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted(resolved_conv_mode_overrides.items()) if mode != "M0"
        },
        "text_only": True,
        "dotcache_ready": False,
        "deltanet_state_ready": True,
        "deltanet_statecache_ready": True,
        "runtime_mode": "dense_deltanet_statecache_loss",
        "uses_native_qwen35_class": True,
    }
    result.update(byte_summary)
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_deltanet_statecache_localization_harness(
    model,
    adapter: Qwen35DeltaNetStateModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    prefix_length: int,
    eval_steps: int,
    group_size: int = 32,
    bits: int = 8,
    layer_bits_overrides: dict[int, int] | None = None,
    recurrent_group_size_policy: Qwen35DeltaNetStateCacheRecurrentGroupSizePolicy | str | None = None,
    recurrent_layer_group_size_overrides: dict[int, int] | None = None,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    conv_bits: int | None = None,
    conv_layer_bits_overrides: dict[int, int] | None = None,
    state_stage: Qwen35DeltaNetStateCacheStage = "post_update_m0",
    renorm_interval: int = 0,
    recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    conv_renorm_interval_overrides: dict[int, int] | None = None,
    recurrent_mode_policy: Qwen35DeltaNetStateCacheRecurrentModePolicy | str | None = None,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    readout_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    readout_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    post_update_recurrent_renorm_interval_overrides: dict[int, int] | None = None,
    post_update_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    quantization_telemetry_layer_ids: set[int] | list[int] | tuple[int, ...] | None = None,
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
    if recurrent_mode_policy is not None and recurrent_mode_overrides:
        raise ValueError("recurrent_mode_policy cannot be combined with explicit recurrent mode overrides")
    if recurrent_group_size_policy is not None and recurrent_layer_group_size_overrides:
        raise ValueError(
            "recurrent_group_size_policy cannot be combined with explicit recurrent layer group-size overrides"
        )
    resolved_recurrent_mode_policy_overrides, resolved_recurrent_mode_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
            prompt_length=int(prefix_length),
            policy=recurrent_mode_policy,
        )
    )
    resolved_recurrent_group_size_policy_overrides, resolved_recurrent_group_size_policy_band = (
        resolve_qwen35_deltanet_statecache_recurrent_group_size_policy(
            prompt_length=int(prefix_length),
            decode_steps=int(max(eval_steps - 1, 1)),
            policy=recurrent_group_size_policy,
        )
    )
    resolved_recurrent_layer_group_size_overrides = dict(
        recurrent_layer_group_size_overrides
        if recurrent_layer_group_size_overrides
        else resolved_recurrent_group_size_policy_overrides
    )
    resolved_base_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_recurrent_mode_policy_overrides,
        recurrent_mode_overrides,
    )
    resolved_readout_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        readout_recurrent_mode_overrides,
    )
    resolved_readout_recurrent_renorm_interval_overrides = _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
        recurrent_renorm_interval_overrides,
        readout_recurrent_renorm_interval_overrides,
    )
    has_explicit_readout_recurrent_overrides = bool(
        readout_recurrent_mode_overrides or readout_recurrent_renorm_interval_overrides
    )

    dense_capture = _run_qwen35_deltanet_dense_teacher_forced_capture(
        model,
        adapter,
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        continuation_ids=continuation_ids,
    )
    linear_records = [
        record
        for step_records in dense_capture["capture_records"]
        for record in step_records
    ]
    records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]] = {}
    for record in linear_records:
        records_by_layer.setdefault(int(record.layer_id), []).append(record)

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=input_ids.device,
    )
    statecache_past_key_values = _clone_qwen35_past_key_values(prefill_outputs.past_key_values)
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()]
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    statecache_decode_ms_total = 0.0
    statecache_per_step_decode_ms: list[float] = []
    deltanet_layer_ids = adapter.deltanet_layer_ids()
    resolved_scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    resolved_conv_bits = int(bits if conv_bits is None else conv_bits)
    resolved_post_update_recurrent_renorm_interval_overrides = (
        _merge_qwen35_deltanet_statecache_renorm_interval_overrides(
            recurrent_renorm_interval_overrides,
            post_update_recurrent_renorm_interval_overrides,
        )
    )
    resolved_post_update_recurrent_mode_overrides = _merge_qwen35_deltanet_statecache_mode_overrides(
        resolved_base_recurrent_mode_overrides,
        post_update_recurrent_mode_overrides,
    )
    telemetry_layer_ids = (
        {int(layer_id) for layer_id in quantization_telemetry_layer_ids}
        if quantization_telemetry_layer_ids is not None
        else None
    )
    quantization_telemetry_records: list[dict[str, Any]] = []
    statecache_prefill_partition = adapter.partition_hybrid_state(prefill_outputs.past_key_values)
    byte_summary = _summarize_qwen35_deltanet_statecache_bytes(
        statecache_prefill_partition,
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        recurrent_bits=int(bits),
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
        recurrent_mode_overrides=(
            resolved_post_update_recurrent_mode_overrides
            if state_stage == "post_update_m0"
            else resolved_base_recurrent_mode_overrides
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    if state_stage == "post_update_m0":
        _prepare_qwen35_deltanet_statecache(
            statecache_past_key_values,
            layer_ids=deltanet_layer_ids,
            recurrent_bits=int(bits),
            conv_bits=resolved_conv_bits,
            group_size=int(group_size),
            renorm=False,
            statecache_scope=resolved_scope,
            recurrent_renorm_interval=int(renorm_interval),
            conv_renorm_interval=int(renorm_interval),
            recurrent_layer_bits_overrides=layer_bits_overrides,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
            recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_default_mode="M0",
            conv_default_mode="M0",
            recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
            conv_mode_overrides=conv_mode_overrides,
            quantization_telemetry_layer_ids=telemetry_layer_ids,
            quantization_telemetry_records=quantization_telemetry_records,
            quantization_telemetry_phase="prefill_post_update",
        )

    for step_index in range(max(eval_steps - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]

        def _run_statecache_decode():
            if state_stage == "readout_only_m0" or has_explicit_readout_recurrent_overrides:
                _prepare_qwen35_deltanet_statecache(
                    statecache_past_key_values,
                    layer_ids=deltanet_layer_ids,
                    recurrent_bits=int(bits),
                    conv_bits=resolved_conv_bits,
                    group_size=int(group_size),
                    renorm=False,
                    statecache_scope=resolved_scope,
                    recurrent_renorm_interval=int(renorm_interval),
                    conv_renorm_interval=int(renorm_interval),
                    recurrent_layer_bits_overrides=layer_bits_overrides,
                    conv_layer_bits_overrides=conv_layer_bits_overrides,
                    recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
                    recurrent_renorm_interval_overrides=resolved_readout_recurrent_renorm_interval_overrides,
                    conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                    recurrent_default_mode="M0",
                    conv_default_mode="M0",
                    recurrent_mode_overrides=resolved_readout_recurrent_mode_overrides,
                    conv_mode_overrides=conv_mode_overrides,
                    quantization_telemetry_layer_ids=telemetry_layer_ids,
                    quantization_telemetry_records=quantization_telemetry_records,
                    quantization_telemetry_phase="readout",
                )
            return _run_dense_decode_step(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=statecache_past_key_values,
                cache_position=cache_position,
            )

        outputs, step_ms = _timed_call(_run_statecache_decode, device=input_ids.device)
        statecache_decode_ms_total += step_ms
        statecache_per_step_decode_ms.append(float(step_ms))
        statecache_past_key_values = outputs.past_key_values
        if state_stage == "post_update_m0":
            _prepare_qwen35_deltanet_statecache(
                statecache_past_key_values,
                layer_ids=deltanet_layer_ids,
                recurrent_bits=int(bits),
                conv_bits=resolved_conv_bits,
                group_size=int(group_size),
                step_index=step_index,
                statecache_scope=resolved_scope,
                recurrent_renorm_interval=int(renorm_interval),
                conv_renorm_interval=int(renorm_interval),
                recurrent_layer_bits_overrides=layer_bits_overrides,
                conv_layer_bits_overrides=conv_layer_bits_overrides,
                recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
                recurrent_renorm_interval_overrides=resolved_post_update_recurrent_renorm_interval_overrides,
                conv_renorm_interval_overrides=conv_renorm_interval_overrides,
                recurrent_default_mode="M0",
                conv_default_mode="M0",
                recurrent_mode_overrides=resolved_post_update_recurrent_mode_overrides,
                conv_mode_overrides=conv_mode_overrides,
                quantization_telemetry_layer_ids=telemetry_layer_ids,
                quantization_telemetry_records=quantization_telemetry_records,
                quantization_telemetry_phase="post_update",
            )
        logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    recurrent_only_result = _run_deltanet_ablation_stage(
        adapter,
        records_by_layer=records_by_layer,
        stage_name=str(state_stage),
        bits=int(bits),
        group_size=int(group_size),
        statecache_scope="recurrent_only",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_overrides
            else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    conv_only_result = _run_deltanet_ablation_stage(
        adapter,
        records_by_layer=records_by_layer,
        stage_name=str(state_stage),
        bits=int(bits),
        group_size=int(group_size),
        statecache_scope="conv_only",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_overrides
            else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else resolved_base_recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    combined_state_result = _run_deltanet_ablation_stage(
        adapter,
        records_by_layer=records_by_layer,
        stage_name=str(state_stage),
        bits=int(bits),
        group_size=int(group_size),
        statecache_scope="conv_plus_recurrent",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_layer_group_size_overrides=resolved_recurrent_layer_group_size_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_overrides
            else (
                resolved_post_update_recurrent_mode_overrides
                if state_stage == "post_update_m0"
                else recurrent_mode_overrides
            )
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    statecache_result = _run_deltanet_ablation_stage(
        adapter,
        records_by_layer=records_by_layer,
        stage_name=str(state_stage),
        bits=int(bits),
        group_size=int(group_size),
        statecache_scope=resolved_scope,
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=layer_bits_overrides,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        recurrent_mode_overrides=(
            resolved_readout_recurrent_mode_overrides
            if has_explicit_readout_recurrent_overrides
            else resolved_base_recurrent_mode_overrides
        ),
        conv_mode_overrides=conv_mode_overrides,
    )
    per_step_logit_max_abs_error: list[float] = []
    for dense_step, approx_step in zip(dense_capture["step_logits"], logits_list):
        per_step_logit_max_abs_error.append(float(np.max(np.abs(approx_step - dense_step))))

    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "deltanet_statecache_decode_ms_per_step": float(statecache_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "deltanet_statecache_per_step_decode_ms": statecache_per_step_decode_ms,
        "deltanet_statecache_ready": True,
        "deltanet_state_ready": True,
        "runtime_mode": "dense_deltanet_statecache_localization",
        "deltanet_statecache_scope": resolved_scope,
        "deltanet_statecache_stage_name": str(state_stage),
        "deltanet_statecache_recurrent_mode_policy": (
            str(recurrent_mode_policy) if recurrent_mode_policy is not None else None
        ),
        "deltanet_statecache_recurrent_mode_policy_band": resolved_recurrent_mode_policy_band,
        "deltanet_statecache_recurrent_group_size_policy": (
            str(recurrent_group_size_policy) if recurrent_group_size_policy is not None else None
        ),
        "deltanet_statecache_recurrent_group_size_policy_band": resolved_recurrent_group_size_policy_band,
        "deltanet_statecache_bits": int(bits),
        "deltanet_statecache_conv_bits": int(resolved_conv_bits),
        "deltanet_statecache_group_size": int(group_size),
        "deltanet_statecache_recurrent_layer_group_size_overrides": {
            str(layer_id): int(group)
            for layer_id, group in sorted((resolved_recurrent_layer_group_size_overrides or {}).items())
            if int(group) != int(group_size)
        },
        "deltanet_statecache_layer_bits": {
            str(layer_id): _resolve_deltanet_statecache_bits(
                int(layer_id),
                default_bits=int(bits),
                layer_bits_overrides=layer_bits_overrides,
            )
            for layer_id in deltanet_layer_ids
        },
        "deltanet_statecache_conv_layer_bits": {
            str(layer_id): _resolve_deltanet_statecache_bits(
                int(layer_id),
                default_bits=resolved_conv_bits,
                layer_bits_overrides=conv_layer_bits_overrides,
            )
            for layer_id in deltanet_layer_ids
        },
        "deltanet_statecache_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted(resolved_base_recurrent_mode_overrides.items())
            if mode != "M0"
        },
        "deltanet_statecache_readout_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted(resolved_readout_recurrent_mode_overrides.items())
            if mode != "M0"
        },
        "deltanet_statecache_readout_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_readout_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_post_update_recurrent_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((resolved_post_update_recurrent_mode_overrides or {}).items())
            if mode != "M0"
        },
        "deltanet_statecache_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_post_update_recurrent_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((resolved_post_update_recurrent_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_conv_mode_overrides": {
            str(layer_id): mode
            for layer_id, mode in sorted((conv_mode_overrides or {}).items())
        },
        "deltanet_statecache_conv_renorm_interval_overrides": {
            str(layer_id): int(interval)
            for layer_id, interval in sorted((conv_renorm_interval_overrides or {}).items())
            if int(interval) != int(renorm_interval)
        },
        "deltanet_statecache_quantization_telemetry_tracked_layers": (
            sorted(int(layer_id) for layer_id in telemetry_layer_ids) if telemetry_layer_ids is not None else []
        ),
        "deltanet_statecache_quantization_telemetry_records": list(quantization_telemetry_records),
        "deltanet_statecache_quantization_telemetry_summary": _summarize_state_quantization_telemetry_records(
            quantization_telemetry_records
        ),
        "deltanet_statecache_per_step_logit_max_abs_error": per_step_logit_max_abs_error,
        "deltanet_statecache_first_divergence_step": _first_drift_step(dense_capture["step_logits"], logits_list),
        "deltanet_statecache_first_failure_layer": _first_layer_over_threshold(
            statecache_result.per_layer_output_max_abs_error
        ),
        "deltanet_statecache_first_recurrent_state_failure_layer": _first_layer_over_threshold(
            recurrent_only_result.per_layer_max_abs_error
        ),
        "deltanet_statecache_first_recurrent_failure_layer": _first_layer_over_threshold(
            recurrent_only_result.per_layer_output_max_abs_error
        ),
        "deltanet_statecache_first_conv_failure_layer": _first_layer_over_threshold(
            conv_only_result.per_layer_output_max_abs_error
        ),
        "deltanet_statecache_first_combined_failure_layer": _first_layer_over_threshold(
            combined_state_result.per_layer_output_max_abs_error
        ),
        "deltanet_statecache_result": statecache_result.to_dict(),
        "deltanet_statecache_recurrent_result": recurrent_only_result.to_dict(),
        "deltanet_statecache_recurrent_state_max_abs_error_by_layer": dict(
            sorted(recurrent_only_result.per_layer_max_abs_error.items())
        ),
        "deltanet_statecache_recurrent_output_max_abs_error_by_layer": dict(
            sorted(recurrent_only_result.per_layer_output_max_abs_error.items())
        ),
        "deltanet_statecache_conv_result": conv_only_result.to_dict(),
        "deltanet_statecache_combined_result": combined_state_result.to_dict(),
    }
    result.update(byte_summary)
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_hybrid_combined_localization_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    prefix_length: int,
    eval_steps: int,
    profile_backend: bool = False,
    statecache_group_size: int = 32,
    statecache_bits: int = 8,
    statecache_layer_bits_overrides: dict[int, int] | None = None,
    statecache_scope: Qwen35DeltaNetStateCacheScope = "recurrent_only",
    statecache_conv_bits: int | None = None,
    statecache_conv_layer_bits_overrides: dict[int, int] | None = None,
    statecache_stage: Qwen35DeltaNetStateCacheStage = "post_update_m0",
    statecache_renorm_interval: int = 0,
    statecache_recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    statecache_conv_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_backend_profiling(profile_backend)
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
    if prefix_length <= 0 or prefix_length >= int(input_ids.shape[1]):
        raise ValueError("prefix_length must be in [1, sequence_length)")
    available_eval_steps = int(input_ids.shape[1]) - prefix_length
    if eval_steps <= 0 or eval_steps > available_eval_steps:
        raise ValueError("eval_steps must be positive and fit inside the provided sequence after prefix_length")

    prefix_input_ids = input_ids[:, :prefix_length]
    prefix_attention_mask = attention_mask[:, :prefix_length]
    continuation_ids = input_ids[:, prefix_length : prefix_length + eval_steps]
    resolved_scope = _resolve_qwen35_deltanet_statecache_scope(statecache_scope)
    resolved_conv_bits = int(statecache_bits if statecache_conv_bits is None else statecache_conv_bits)

    dense_capture = _run_qwen35_attention_subset_dense_teacher_forced_capture(
        model,
        adapter,
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        continuation_ids=continuation_ids,
    )
    delta_adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    deltanet_dense_capture = _run_qwen35_deltanet_dense_teacher_forced_capture(
        model,
        delta_adapter,
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        continuation_ids=continuation_ids,
    )
    delta_records_by_layer: dict[int, list[Qwen35DeltaNetStateRecord]] = {}
    for record in [
        record
        for step_records in deltanet_dense_capture["capture_records"]
        for record in step_records
    ]:
        delta_records_by_layer.setdefault(int(record.layer_id), []).append(record)

    combined_prefill_outputs, combined_prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=input_ids.device,
    )
    adapter.clear()
    adapter.load_attention_subset_prefill_cache(combined_prefill_outputs.past_key_values)
    adapter.set_mode("dotcache_attention_subset")
    runtime_state = adapter.require_hybrid_dotcache_runtime_state()
    deltanet_layer_ids = [
        layer_id
        for layer_id, layer_type in enumerate(_hybrid_layer_types(model))
        if layer_type == "linear_attention"
    ]
    resolved_recurrent_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=statecache_recurrent_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    resolved_conv_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=statecache_conv_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    combined_prefill_partition = delta_adapter.partition_hybrid_state(combined_prefill_outputs.past_key_values)
    byte_summary = _summarize_qwen35_deltanet_statecache_bytes(
        combined_prefill_partition,
        group_size=int(statecache_group_size),
        statecache_scope=resolved_scope,
        recurrent_bits=int(statecache_bits),
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
        conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
        recurrent_mode_overrides=statecache_recurrent_mode_overrides,
        conv_mode_overrides=statecache_conv_mode_overrides,
    )

    if statecache_stage == "post_update_m0":
        _prepare_qwen35_deltanet_statecache(
            runtime_state.model_past_key_values,
            layer_ids=deltanet_layer_ids,
            recurrent_bits=int(statecache_bits),
            conv_bits=resolved_conv_bits,
            group_size=int(statecache_group_size),
            renorm=False,
            statecache_scope=resolved_scope,
            recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
            conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
            recurrent_default_mode="M0",
            conv_default_mode="M0",
            recurrent_mode_overrides=statecache_recurrent_mode_overrides,
            conv_mode_overrides=statecache_conv_mode_overrides,
        )

    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    combined_step_logits: list[np.ndarray] = [combined_prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()]
    combined_records: list[list[LlamaReplayRecord]] = []
    combined_decode_ms_total = 0.0

    for step_index in range(max(eval_steps - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]

        def _run_combined_decode():
            if statecache_stage == "readout_only_m0":
                _prepare_qwen35_deltanet_statecache(
                    runtime_state.model_past_key_values,
                    layer_ids=deltanet_layer_ids,
                    recurrent_bits=int(statecache_bits),
                    conv_bits=resolved_conv_bits,
                    group_size=int(statecache_group_size),
                    renorm=False,
                    statecache_scope=resolved_scope,
                    recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
                    conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
                    recurrent_default_mode="M0",
                    conv_default_mode="M0",
                    recurrent_mode_overrides=statecache_recurrent_mode_overrides,
                    conv_mode_overrides=statecache_conv_mode_overrides,
                )
            return _run_dense_decode_step(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=runtime_state.model_past_key_values,
                cache_position=cache_position,
            )

        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(prefix_input_ids.shape[1] + step_index))
        try:
            outputs, step_ms = _timed_call(_run_combined_decode, device=input_ids.device)
        finally:
            adapter.set_current_token_index(None)
        combined_decode_ms_total += step_ms
        combined_records.append(adapter.end_capture_step())
        combined_step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        runtime_state.advance(outputs.past_key_values)
        if statecache_stage == "post_update_m0":
            _prepare_qwen35_deltanet_statecache(
                runtime_state.model_past_key_values,
                layer_ids=deltanet_layer_ids,
                recurrent_bits=int(statecache_bits),
                conv_bits=resolved_conv_bits,
                group_size=int(statecache_group_size),
                renorm=bool(statecache_renorm_interval > 0 and (step_index + 1) % int(statecache_renorm_interval) == 0),
                statecache_scope=resolved_scope,
                recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
                conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
                recurrent_default_mode="M0",
                conv_default_mode="M0",
                recurrent_mode_overrides=statecache_recurrent_mode_overrides,
                conv_mode_overrides=statecache_conv_mode_overrides,
            )
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
    combined_record_map = {
        (record.step_index, record.layer_id): record
        for step_records in combined_records
        for record in step_records
    }
    per_layer_attention_output_max_abs: dict[str, float] = {}
    for replay_key, dense_record in dense_record_map.items():
        combined_record = combined_record_map.get(replay_key)
        if combined_record is None:
            continue
        output_delta = np.abs(combined_record.output_states - dense_record.output_states)
        layer_key = str(dense_record.layer_id)
        per_layer_attention_output_max_abs[layer_key] = max(
            per_layer_attention_output_max_abs.get(layer_key, 0.0),
            float(np.max(output_delta)),
        )

    per_step_logit_max_abs_error: list[float] = []
    for dense_step, combined_step in zip(dense_capture["step_logits"], combined_step_logits):
        per_step_logit_max_abs_error.append(float(np.max(np.abs(combined_step - dense_step))))

    recurrent_only_result = _run_deltanet_ablation_stage(
        delta_adapter,
        records_by_layer=delta_records_by_layer,
        stage_name=str(statecache_stage),
        bits=int(statecache_bits),
        group_size=int(statecache_group_size),
        statecache_scope="recurrent_only",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
        conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
        recurrent_mode_overrides=statecache_recurrent_mode_overrides,
        conv_mode_overrides=statecache_conv_mode_overrides,
    )
    conv_only_result = _run_deltanet_ablation_stage(
        delta_adapter,
        records_by_layer=delta_records_by_layer,
        stage_name=str(statecache_stage),
        bits=int(statecache_bits),
        group_size=int(statecache_group_size),
        statecache_scope="conv_only",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
        conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
        recurrent_mode_overrides=statecache_recurrent_mode_overrides,
        conv_mode_overrides=statecache_conv_mode_overrides,
    )
    deltanet_combined_result = _run_deltanet_ablation_stage(
        delta_adapter,
        records_by_layer=delta_records_by_layer,
        stage_name=str(statecache_stage),
        bits=int(statecache_bits),
        group_size=int(statecache_group_size),
        statecache_scope="conv_plus_recurrent",
        conv_bits=resolved_conv_bits,
        recurrent_layer_bits_overrides=statecache_layer_bits_overrides,
        conv_layer_bits_overrides=statecache_conv_layer_bits_overrides,
        recurrent_mode_overrides=statecache_recurrent_mode_overrides,
        conv_mode_overrides=statecache_conv_mode_overrides,
    )

    attention_first_failure_layer = _first_layer_over_threshold(per_layer_attention_output_max_abs)
    recurrent_first_failure_layer = _first_layer_over_threshold(recurrent_only_result.per_layer_output_max_abs_error)
    conv_first_failure_layer = _first_layer_over_threshold(conv_only_result.per_layer_output_max_abs_error)
    if attention_first_failure_layer is not None and (
        recurrent_first_failure_layer is not None or conv_first_failure_layer is not None
    ):
        combined_first_failure_family = "mixed"
    elif attention_first_failure_layer is not None:
        combined_first_failure_family = "attention"
    elif recurrent_first_failure_layer is not None and conv_first_failure_layer is not None:
        combined_first_failure_family = "mixed"
    elif recurrent_first_failure_layer is not None:
        combined_first_failure_family = "recurrent"
    elif conv_first_failure_layer is not None:
        combined_first_failure_family = "conv"
    else:
        combined_first_failure_family = None

    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "combined_prefill_ms": float(combined_prefill_ms),
        "combined_decode_ms_per_step": float(combined_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "hybrid_combined_ready": True,
        "runtime_mode": "qwen35_hybrid_combined_localization",
        "statecache_scope": resolved_scope,
        "statecache_bits": int(statecache_bits),
        "statecache_conv_bits": int(resolved_conv_bits),
        "statecache_group_size": int(statecache_group_size),
        "statecache_stage_name": str(statecache_stage),
        "statecache_renorm_interval": int(statecache_renorm_interval),
        "statecache_layer_bits_overrides": {
            str(layer_id): bits for layer_id, bits in sorted((statecache_layer_bits_overrides or {}).items())
        },
        "statecache_conv_layer_bits_overrides": {
            str(layer_id): bits for layer_id, bits in sorted((statecache_conv_layer_bits_overrides or {}).items())
        },
        "statecache_recurrent_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted((statecache_recurrent_mode_overrides or {}).items())
        },
        "statecache_conv_mode_overrides": {
            str(layer_id): mode for layer_id, mode in sorted((statecache_conv_mode_overrides or {}).items())
        },
        "combined_per_step_logit_max_abs_error": per_step_logit_max_abs_error,
        "combined_first_divergence_step": _first_drift_step(dense_capture["step_logits"], combined_step_logits),
        "combined_attention_output_max_abs_error_by_layer": dict(sorted(per_layer_attention_output_max_abs.items())),
        "combined_first_attention_failure_layer": attention_first_failure_layer,
        "combined_deltanet_recurrent_output_max_abs_error_by_layer": dict(sorted(recurrent_only_result.per_layer_output_max_abs_error.items())),
        "combined_deltanet_conv_output_max_abs_error_by_layer": dict(sorted(conv_only_result.per_layer_output_max_abs_error.items())),
        "combined_deltanet_output_max_abs_error_by_layer": dict(sorted(deltanet_combined_result.per_layer_output_max_abs_error.items())),
        "combined_first_recurrent_failure_layer": recurrent_first_failure_layer,
        "combined_first_conv_failure_layer": conv_first_failure_layer,
        "combined_first_deltanet_failure_layer": _first_layer_over_threshold(deltanet_combined_result.per_layer_output_max_abs_error),
        "combined_first_failure_family": combined_first_failure_family,
        "combined_deltanet_recurrent_result": recurrent_only_result.to_dict(),
        "combined_deltanet_conv_result": conv_only_result.to_dict(),
        "combined_deltanet_combined_result": deltanet_combined_result.to_dict(),
    }
    result.update(byte_summary)
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(byte_summary)
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
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
    prefill_tensors = _clone_attention_subset_prefill_tensors(
        _extract_attention_subset_prefill_tensors(prefill_outputs.past_key_values, adapter.attention_subset_layer_ids())
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
        "prefill_tensors": prefill_tensors,
        "prefill_ms": float(prefill_ms),
        "decode_ms_total": float(dense_decode_ms_total),
        "decode_inputs": decode_inputs,
        "step_logits": step_logits,
        "capture_records": per_step_records,
    }


def _decode_input_id_sequence(decode_inputs: list[Any]) -> list[int]:
    generated_ids: list[int] = []
    for decode_input_ids in decode_inputs:
        if torch is not None and torch.is_tensor(decode_input_ids):
            generated_ids.extend(int(token_id) for token_id in decode_input_ids.detach().view(-1).tolist())
        else:
            generated_ids.extend(int(token_id) for token_id in np.asarray(decode_input_ids).reshape(-1).tolist())
    return generated_ids


def _run_qwen35_attention_subset_dense_teacher_forced_capture(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    prefix_input_ids,
    prefix_attention_mask,
    continuation_ids,
) -> dict[str, Any]:
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=prefix_input_ids.device,
    )
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()]
    per_step_records: list[list[LlamaReplayRecord]] = []
    past_key_values = prefill_outputs.past_key_values
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=prefix_input_ids.device)
    dense_decode_ms_total = 0.0

    for step_index in range(max(int(continuation_ids.shape[1]) - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]
        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(prefix_input_ids.shape[1] + step_index))
        try:
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
                device=prefix_input_ids.device,
            )
        finally:
            adapter.set_current_token_index(None)
        dense_decode_ms_total += step_ms
        per_step_records.append(adapter.end_capture_step())
        logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        past_key_values = outputs.past_key_values
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    return {
        "prefill_outputs": prefill_outputs,
        "prefill_ms": float(prefill_ms),
        "decode_ms_total": float(dense_decode_ms_total),
        "step_logits": logits_list,
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


def _aggregate_query_states_by_kv_head(
    query_states: np.ndarray,
    q_head_to_kv_head: np.ndarray,
    *,
    num_key_value_heads: int,
) -> np.ndarray:
    queries = np.asarray(query_states, dtype=np.float32)
    mapping = np.asarray(q_head_to_kv_head, dtype=np.int32)
    if queries.ndim != 2:
        raise ValueError("query_states must have shape [query_heads, head_dim]")
    if mapping.ndim != 1 or mapping.shape[0] != queries.shape[0]:
        raise ValueError("q_head_to_kv_head must have shape [query_heads]")
    kv_queries = np.zeros((int(num_key_value_heads), int(queries.shape[1])), dtype=np.float32)
    counts = np.zeros((int(num_key_value_heads),), dtype=np.int32)
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        if kv_head_id < 0 or kv_head_id >= int(num_key_value_heads):
            raise ValueError("q_head_to_kv_head contains an out-of-range kv head")
        kv_queries[kv_head_id] += queries[q_head_id]
        counts[kv_head_id] += 1
    counts = np.maximum(counts, 1)
    kv_queries /= counts[:, None].astype(np.float32)
    return kv_queries


def _build_page_traces_from_streams(
    streams: dict[tuple[int, int, str], list[tuple[int, np.ndarray, np.ndarray | None]]],
    *,
    max_token_index: int,
    tokens_per_page: int,
    source: str,
    stage: str,
) -> list[PageTraceRecord]:
    page_traces: list[PageTraceRecord] = []
    for (layer_id, kv_head_id, kind), entries in sorted(streams.items()):
        entries.sort(key=lambda item: item[0])
        for offset in range(0, len(entries), int(tokens_per_page)):
            chunk = entries[offset : offset + int(tokens_per_page)]
            token_indices = [token_index for token_index, _, _ in chunk]
            values = np.stack([value for _, value, _ in chunk], axis=0).astype(np.float32, copy=False)
            queries = [query_vector for _, _, query_vector in chunk if query_vector is not None]
            query = None
            if queries:
                query = np.mean(np.stack(queries, axis=0), axis=0, dtype=np.float32).astype(np.float32, copy=False)
            page_traces.append(
                PageTraceRecord(
                    source=source,
                    kind=kind,  # type: ignore[arg-type]
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=int(token_indices[0]),
                    token_age=max(max_token_index - int(token_indices[-1]), 0),
                    values=values,
                    query=query,
                    notes=[
                        f"stage={stage}",
                        "query_aggregation=mean_mapped_q_heads" if query is not None else "query_aggregation=none",
                        f"token_indices={token_indices[0]}..{token_indices[-1]}",
                    ],
                )
            )
    return page_traces


def build_attention_subset_page_trace_records(
    per_step_records: list[list[LlamaReplayRecord]],
    *,
    q_head_to_kv_head: np.ndarray,
    tokens_per_page: int,
    kinds: tuple[str, ...] = ("K", "V"),
    source: str = "qwen35_attention_subset_dense_capture",
) -> list[PageTraceRecord]:
    if int(tokens_per_page) <= 0:
        raise ValueError("tokens_per_page must be positive")

    normalized_kinds = tuple(str(kind).upper() for kind in kinds)
    invalid_kinds = [kind for kind in normalized_kinds if kind not in {"K", "V"}]
    if invalid_kinds:
        raise ValueError(f"unsupported capture kinds: {invalid_kinds}")

    streams: dict[tuple[int, int, str], list[tuple[int, np.ndarray, np.ndarray | None]]] = {}
    max_token_index = -1

    for step_records in per_step_records:
        for record in step_records:
            max_token_index = max(max_token_index, int(record.token_index))
            kv_head_count = int(record.key_states.shape[0])
            kv_queries = _aggregate_query_states_by_kv_head(
                record.query_states,
                q_head_to_kv_head,
                num_key_value_heads=kv_head_count,
            )
            for kv_head_id in range(kv_head_count):
                if "K" in normalized_kinds:
                    streams.setdefault((int(record.layer_id), kv_head_id, "K"), []).append(
                        (
                            int(record.token_index),
                            np.asarray(record.key_states[kv_head_id], dtype=np.float32),
                            np.asarray(kv_queries[kv_head_id], dtype=np.float32),
                        )
                    )
                if "V" in normalized_kinds:
                    streams.setdefault((int(record.layer_id), kv_head_id, "V"), []).append(
                        (
                            int(record.token_index),
                            np.asarray(record.value_states[kv_head_id], dtype=np.float32),
                            np.asarray(kv_queries[kv_head_id], dtype=np.float32),
                        )
                    )

    return _build_page_traces_from_streams(
        streams,
        max_token_index=max_token_index,
        tokens_per_page=tokens_per_page,
        source=source,
        stage="decode",
    )


def build_attention_subset_prefill_page_trace_records(
    prefill_tensors: dict[int, tuple[Any, Any]],
    *,
    tokens_per_page: int,
    kinds: tuple[str, ...] = ("K", "V"),
    source: str = "qwen35_attention_subset_dense_capture",
    max_token_index: int | None = None,
) -> list[PageTraceRecord]:
    if int(tokens_per_page) <= 0:
        raise ValueError("tokens_per_page must be positive")

    normalized_kinds = tuple(str(kind).upper() for kind in kinds)
    invalid_kinds = [kind for kind in normalized_kinds if kind not in {"K", "V"}]
    if invalid_kinds:
        raise ValueError(f"unsupported capture kinds: {invalid_kinds}")

    streams: dict[tuple[int, int, str], list[tuple[int, np.ndarray, np.ndarray | None]]] = {}
    resolved_max_token_index = -1 if max_token_index is None else int(max_token_index)
    for layer_id, (layer_keys, layer_values) in sorted(prefill_tensors.items()):
        key_array = _tensor_to_float32_numpy(layer_keys)
        value_array = _tensor_to_float32_numpy(layer_values)
        if key_array.ndim != 4 or value_array.ndim != 4 or key_array.shape[0] != 1 or value_array.shape[0] != 1:
            raise ValueError("prefill tensors must have shape [1, kv_heads, seq_len, head_dim]")
        if key_array.shape[:3] != value_array.shape[:3]:
            raise ValueError("prefill key and value tensors must align on batch, kv_head, and seq_len")
        _, kv_head_count, seq_len, _ = key_array.shape
        resolved_max_token_index = max(resolved_max_token_index, int(seq_len) - 1)
        for kv_head_id in range(int(kv_head_count)):
            for token_index in range(int(seq_len)):
                if "K" in normalized_kinds:
                    streams.setdefault((int(layer_id), kv_head_id, "K"), []).append(
                        (
                            int(token_index),
                            np.asarray(key_array[0, kv_head_id, token_index], dtype=np.float32),
                            None,
                        )
                    )
                if "V" in normalized_kinds:
                    streams.setdefault((int(layer_id), kv_head_id, "V"), []).append(
                        (
                            int(token_index),
                            np.asarray(value_array[0, kv_head_id, token_index], dtype=np.float32),
                            None,
                        )
                    )
    return _build_page_traces_from_streams(
        streams,
        max_token_index=resolved_max_token_index,
        tokens_per_page=tokens_per_page,
        source=source,
        stage="prefill",
    )


def export_attention_subset_page_traces(
    per_step_records: list[list[LlamaReplayRecord]],
    *,
    q_head_to_kv_head: np.ndarray,
    output_dir: str | Path,
    tokens_per_page: int,
    kinds: tuple[str, ...] = ("K", "V"),
    source: str = "qwen35_attention_subset_dense_capture",
    prefill_tensors: dict[int, tuple[Any, Any]] | None = None,
    prefill_token_count: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    page_traces = build_attention_subset_page_trace_records(
        per_step_records,
        q_head_to_kv_head=q_head_to_kv_head,
        tokens_per_page=tokens_per_page,
        kinds=kinds,
        source=source,
    )
    if prefill_tensors:
        prefill_length = max(int(prefill_token_count or 0), 0)
        max_token_index = max(prefill_length - 1 + len(per_step_records), 0)
        page_traces = build_attention_subset_prefill_page_trace_records(
            prefill_tensors,
            tokens_per_page=tokens_per_page,
            kinds=kinds,
            source=source,
            max_token_index=max_token_index,
        ) + page_traces
    trace_paths: list[str] = []
    counts_by_kind: dict[str, int] = {}
    counts_by_layer: dict[str, int] = {}
    counts_by_stage: dict[str, int] = {}
    for index, trace in enumerate(page_traces):
        stage = "unknown"
        for note in trace.notes:
            if note.startswith("stage="):
                stage = note.split("=", 1)[1]
                break
        trace_name = (
            f"{stage}_layer{trace.layer_id:02d}_kv{trace.kv_head_id:02d}_{trace.kind.lower()}_"
            f"t{trace.token_start:06d}_n{trace.token_count:03d}_{index:04d}.npz"
        )
        target = output_path / trace_name
        save_page_trace(trace, target)
        trace_paths.append(str(target))
        counts_by_kind[trace.kind] = counts_by_kind.get(trace.kind, 0) + 1
        counts_by_stage[stage] = counts_by_stage.get(stage, 0) + 1
        layer_key = str(trace.layer_id)
        counts_by_layer[layer_key] = counts_by_layer.get(layer_key, 0) + 1
    manifest = {
        "output_dir": str(output_path),
        "page_trace_count": len(page_traces),
        "page_trace_paths": trace_paths,
        "page_trace_counts_by_kind": dict(sorted(counts_by_kind.items())),
        "page_trace_counts_by_stage": dict(sorted(counts_by_stage.items())),
        "page_trace_counts_by_layer": dict(sorted(counts_by_layer.items())),
        "tokens_per_page": int(tokens_per_page),
        "kinds": list(kinds),
        "source": source,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_qwen35_attention_subset_page_trace_capture_harness(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    output_dir: str | Path,
    tokens_per_page: int,
    kinds: tuple[str, ...] = ("K", "V"),
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
    prefill_tensors = dense_capture["prefill_tensors"]
    generated_ids = _decode_input_id_sequence(dense_capture["decode_inputs"])
    result = _summarize_attention_subset_capture(
        adapter,
        input_ids=input_ids,
        decode_steps=decode_steps,
        prefill_ms=float(dense_capture["prefill_ms"]),
        dense_decode_ms_total=float(dense_capture["decode_ms_total"]),
        per_step_records=dense_capture["capture_records"],
    )
    result.update(
        export_attention_subset_page_traces(
            dense_capture["capture_records"],
            q_head_to_kv_head=adapter.q_head_to_kv_head,
            output_dir=output_dir,
            tokens_per_page=tokens_per_page,
            kinds=kinds,
            prefill_tensors=prefill_tensors,
            prefill_token_count=int(input_ids.shape[1]),
        )
    )
    result["runtime_mode"] = "dense_attention_subset_page_trace_capture"
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
    profile_backend: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_backend_profiling(profile_backend)
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
    generated_ids = _decode_input_id_sequence(dense_capture["decode_inputs"])
    generated_ids = _decode_input_id_sequence(dense_capture["decode_inputs"])
    generated_ids = _decode_input_id_sequence(dense_capture["decode_inputs"])
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
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(runtime_state.summary())
    return result


def _prepare_qwen35_attention_subset_dotcache_runtime(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    profile_backend: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_backend_profiling(profile_backend)
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
    adapter.maybe_apply_mps_serving_shortlist_heuristic(prompt_length=int(input_ids.shape[1]))
    device = input_ids.device
    dotcache_prefill_cuda_memory_baseline = _begin_cuda_memory_region(device)
    dotcache_prefill_outputs, dotcache_prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    adapter.clear()
    adapter.load_attention_subset_prefill_cache(dotcache_prefill_outputs.past_key_values)
    adapter.set_mode("dotcache_attention_subset")
    dotcache_prefill_cuda_memory = _end_cuda_memory_region(device, dotcache_prefill_cuda_memory_baseline)
    runtime_state = adapter.require_hybrid_dotcache_runtime_state()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dotcache_prefill_outputs": dotcache_prefill_outputs,
        "dotcache_prefill_ms": float(dotcache_prefill_ms),
        "dotcache_prefill_cuda_memory": dotcache_prefill_cuda_memory,
        "runtime_state": runtime_state,
        "serving_shortlist_heuristic_applied": bool(adapter.serving_shortlist_heuristic_applied),
    }


def run_qwen35_attention_subset_dotcache_serving_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    profile_backend: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        profile_backend=profile_backend,
        multimodal_inputs=multimodal_inputs,
    )
    input_ids = prepared["input_ids"]
    attention_mask = prepared["attention_mask"]
    dotcache_prefill_outputs = prepared["dotcache_prefill_outputs"]
    dotcache_prefill_ms = float(prepared["dotcache_prefill_ms"])
    dotcache_prefill_cuda_memory = prepared["dotcache_prefill_cuda_memory"]
    runtime_state = prepared["runtime_state"]
    serving_shortlist_heuristic_applied = bool(prepared["serving_shortlist_heuristic_applied"])
    device = input_ids.device

    generated_ids: list[int] = []
    dotcache_decode_ms_total = 0.0
    dotcache_decode_cuda_memory: dict[str, int] = {}
    if decode_steps > 0:
        current_input_ids = dotcache_prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        dotcache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
        for _ in range(decode_steps):
            generated_ids.append(int(current_input_ids.item()))
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                ),
                device=device,
            )
            dotcache_decode_ms_total += step_ms
            runtime_state.advance(outputs.past_key_values)
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = cache_position + 1
        dotcache_decode_cuda_memory = _end_cuda_memory_region(device, dotcache_decode_cuda_memory_baseline)

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "dotcache_prefill_ms": float(dotcache_prefill_ms),
        "dotcache_generated_ids": generated_ids,
        "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "dotcache_attention_subset_ready": True,
        "dotcache_ready": False,
        "runtime_mode": "dotcache_attention_subset_serving",
        "uses_native_qwen35_class": True,
        "text_only": True,
        "attention_subset_layer_ids": adapter.attention_subset_layer_ids(),
        "attention_subset_capture_layer_count": len(adapter.attention_subset_layer_ids()),
        "num_attention_heads": int(adapter.model_kv_cache.num_attention_heads),
        "num_key_value_heads": int(adapter.model_kv_cache.num_key_value_heads),
        "query_heads_per_kv_head": int(adapter.model_kv_cache.num_attention_heads // max(adapter.model_kv_cache.num_key_value_heads, 1)),
        "head_dim": int(adapter.dotcache_config.head_dim),
        "group_size": int(adapter.dotcache_config.group_size),
        "num_groups": int(adapter.dotcache_config.num_groups),
        "padded_head_dim": int(adapter.dotcache_config.padded_head_dim),
        "tokens_per_page": int(adapter.dotcache_config.tokens_per_page),
        "execution_recent_window": int(adapter.dotcache_config.execution_recent_window),
        "execution_sink_window": int(adapter.dotcache_config.execution_sink_window),
        "execution_recent_window_overrides": list(adapter.dotcache_config.execution_recent_window_overrides),
        "execution_recent_window_context_overrides": list(
            adapter.dotcache_config.execution_recent_window_context_overrides
        ),
        "execution_relevance_top_k": int(adapter.dotcache_config.execution_relevance_top_k),
        "execution_relevance_top_k_overrides": list(adapter.dotcache_config.execution_relevance_top_k_overrides),
        "execution_relevance_top_k_context_overrides": list(adapter.dotcache_config.execution_relevance_top_k_context_overrides),
        "execution_full_context_layers": list(adapter.dotcache_config.execution_full_context_layers),
        "execution_disable_grouped_batching_layers": list(
            adapter.dotcache_config.execution_disable_grouped_batching_layers
        ),
        "execution_recent_old_bonus_window": int(adapter.dotcache_config.execution_recent_old_bonus_window),
        "execution_recent_old_bonus_strength": float(adapter.dotcache_config.execution_recent_old_bonus_strength),
        "execution_recent_old_bonus_layers": list(adapter.dotcache_config.execution_recent_old_bonus_layers),
        "execution_relevance_mode": str(adapter.dotcache_config.execution_relevance_mode),
        "execution_secondary_relevance_mode": str(adapter.dotcache_config.execution_secondary_relevance_mode),
        "execution_secondary_relevance_top_k": int(adapter.dotcache_config.execution_secondary_relevance_top_k),
        "execution_secondary_relevance_min_overlap": float(adapter.dotcache_config.execution_secondary_relevance_min_overlap),
        "execution_secondary_relevance_layers": list(adapter.dotcache_config.execution_secondary_relevance_layers),
        "execution_recent_neighbor_rescue_top_k": int(adapter.dotcache_config.execution_recent_neighbor_rescue_top_k),
        "execution_recent_neighbor_rescue_anchor_window": int(
            adapter.dotcache_config.execution_recent_neighbor_rescue_anchor_window
        ),
        "execution_recent_neighbor_rescue_min_anchor_pages": int(
            adapter.dotcache_config.execution_recent_neighbor_rescue_min_anchor_pages
        ),
        "execution_recent_neighbor_rescue_layers": list(adapter.dotcache_config.execution_recent_neighbor_rescue_layers),
        "execution_exact_promote_top_k": int(adapter.dotcache_config.execution_exact_promote_top_k),
        "execution_exact_promote_min_margin_threshold": float(
            adapter.dotcache_config.execution_exact_promote_min_margin_threshold
        ),
        "execution_exact_promote_max_context": int(adapter.dotcache_config.execution_exact_promote_max_context),
        "execution_exact_promote_margin_threshold": float(adapter.dotcache_config.execution_exact_promote_margin_threshold),
        "execution_exact_promote_layers": list(adapter.dotcache_config.execution_exact_promote_layers),
        "execution_exact_promote_union_rescue_top_k": int(
            adapter.dotcache_config.execution_exact_promote_union_rescue_top_k
        ),
        "execution_grouped_decode_compact": bool(adapter.dotcache_config.execution_grouped_decode_compact),
        "execution_grouped_mix_compact": bool(adapter.dotcache_config.execution_grouped_mix_compact),
        "execution_grouped_mix_disable_packed_cuda": bool(adapter.dotcache_config.execution_grouped_mix_disable_packed_cuda),
        "execution_freeze_chunk_budget_during_decode": bool(
            adapter.dotcache_config.execution_freeze_chunk_budget_during_decode
        ),
        "execution_builtin_selector_cache": bool(adapter.dotcache_config.execution_builtin_selector_cache),
        "execution_builtin_selector_score_all_pages": bool(
            adapter.dotcache_config.execution_builtin_selector_score_all_pages
        ),
        "execution_builtin_selector_candidate_only": bool(
            adapter.dotcache_config.execution_builtin_selector_candidate_only
        ),
        "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
            adapter.dotcache_config.execution_builtin_selector_score_all_pages_min_candidate_fraction
        ),
        "serving_shortlist_heuristic_applied": serving_shortlist_heuristic_applied,
        "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
        "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
        "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
        "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
    }
    result.update({f"dotcache_prefill_{key}": value for key, value in dotcache_prefill_cuda_memory.items()})
    result.update({f"dotcache_decode_{key}": value for key, value in dotcache_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(adapter.per_layer_runtime_summary())
    result.update(adapter.model_kv_cache.decode_path_summary())
    result.update(adapter.model_kv_cache.decode_stage_summary())
    result.update(adapter.model_kv_cache.builtin_selector_summary())
    result.update(adapter.model_kv_cache.chunk_budget_summary())
    result.update(adapter.model_kv_cache.execution_value_escape_summary())
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


_BACKEND_TRACE_TIMING_KEYS = (
    "prepare_ms_total",
    "score_ms_total",
    "mix_ms_total",
    "softmax_ms_total",
    "unpack_ms_total",
    "fwht_ms_total",
    "chunk_assembly_ms_total",
)

_MODEL_KV_CACHE_DECODE_STAGE_KEYS = (
    "execution_decode_prepare_pages_with_tail_ms_total",
    "execution_decode_prepare_layout_build_ms_total",
    "execution_decode_m2_prefilter_ms_total",
    "execution_decode_query_export_ms_total",
    "execution_decode_shortlist_selection_ms_total",
    "execution_decode_shortlist_base_window_ms_total",
    "execution_decode_shortlist_candidate_scoring_ms_total",
    "execution_decode_shortlist_candidate_approx_scoring_ms_total",
    "execution_decode_shortlist_candidate_ranking_ms_total",
    "execution_decode_shortlist_candidate_secondary_scoring_ms_total",
    "execution_decode_shortlist_candidate_neighbor_rescue_ms_total",
    "execution_decode_shortlist_candidate_builtin_selection_ms_total",
    "execution_decode_shortlist_candidate_builtin_candidate_index_build_ms_total",
    "execution_decode_shortlist_candidate_builtin_sidecar_stack_ms_total",
    "execution_decode_shortlist_candidate_builtin_score_compute_ms_total",
    "execution_decode_shortlist_candidate_builtin_ranking_ms_total",
    "execution_decode_shortlist_exact_selection_ms_total",
    "execution_decode_shortlist_union_rescue_ms_total",
    "execution_decode_shortlist_materialization_ms_total",
    "execution_decode_grouping_validation_ms_total",
    "execution_decode_chunk_budget_sync_ms_total",
    "execution_decode_backend_call_wall_ms_total",
    "execution_decode_backend_call_non_backend_ms_total",
)

_MODEL_KV_CACHE_CHUNK_BUDGET_COUNTER_KEYS = (
    "execution_chunk_budget_dirty_marks",
    "execution_chunk_budget_dirty_transitions",
    "execution_chunk_budget_sync_invocations",
    "execution_chunk_budget_sync_clean_skips",
    "execution_chunk_budget_sync_dirty_invocations",
    "execution_chunk_budget_override_calls",
    "execution_chunk_budget_override_budget_change_calls",
    "execution_chunk_budget_override_same_budget_calls",
    "execution_chunk_budget_freeze_override_calls",
)

_MODEL_KV_CACHE_BUILTIN_SELECTOR_COUNTER_KEYS = (
    "execution_builtin_selector_score_all_pages_calls",
    "execution_builtin_selector_candidate_only_calls",
    "execution_builtin_selector_candidate_pages",
    "execution_builtin_selector_total_pages",
    "execution_builtin_selector_candidate_fraction_sum",
    "execution_builtin_selector_candidate_fraction_max",
    "execution_builtin_selector_cache_hits",
    "execution_builtin_selector_cache_builds",
    "execution_builtin_selector_cache_build_bytes",
    "execution_builtin_selector_cache_build_bytes_max",
)

_MODEL_KV_CACHE_VALUE_ESCAPE_COUNTER_KEYS = (
    "execution_value_escape_cache_hits",
    "execution_value_escape_source_registrations",
    "execution_value_escape_prepared_page_builds",
    "execution_value_escape_builds",
    "execution_value_escape_applied_pages",
)


def _adapter_runtime_snapshot(adapter: Qwen35AttentionSubsetDotCacheModelAdapter) -> dict[str, float]:
    snapshot = {
        "qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
        "append_runtime_ms_total": float(adapter.append_runtime_ms_total),
        "decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
        "output_projection_ms_total": float(adapter.output_projection_ms_total),
    }
    snapshot.update(adapter.model_kv_cache.decode_stage_runtime_totals())
    chunk_budget_summary = adapter.model_kv_cache.chunk_budget_summary()
    snapshot.update(
        {
            key: float(chunk_budget_summary.get(key, 0))
            for key in _MODEL_KV_CACHE_CHUNK_BUDGET_COUNTER_KEYS
        }
    )
    builtin_selector_summary = adapter.model_kv_cache.builtin_selector_summary()
    snapshot.update(
        {
            key: float(builtin_selector_summary.get(key, 0))
            for key in _MODEL_KV_CACHE_BUILTIN_SELECTOR_COUNTER_KEYS
        }
    )
    value_escape_summary = adapter.model_kv_cache.execution_value_escape_summary()
    snapshot.update(
        {
            key: float(value_escape_summary.get(key, 0))
            for key in _MODEL_KV_CACHE_VALUE_ESCAPE_COUNTER_KEYS
        }
    )
    return snapshot


def _chunk_budget_reason_counts_snapshot(
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
) -> dict[str, int]:
    return dict(
        adapter.model_kv_cache.chunk_budget_summary().get("execution_chunk_budget_dirty_reason_counts", {})
    )


def _backend_trace_snapshot(adapter: Qwen35AttentionSubsetDotCacheModelAdapter) -> dict[str, int | float]:
    return dict(adapter.decode_backend_trace.to_dict())


def _ensure_python_allocation_tracing(enabled: bool) -> bool:
    if not enabled or tracemalloc.is_tracing():
        return False
    tracemalloc.start()
    return True


def _python_allocation_snapshot(enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    allocated_blocks_getter = getattr(sys, "getallocatedblocks", None)
    allocated_blocks = int(allocated_blocks_getter()) if callable(allocated_blocks_getter) else 0
    gc_counts = gc.get_count()
    return {
        "current_bytes": int(current_bytes),
        "peak_bytes": int(peak_bytes),
        "allocated_blocks": int(allocated_blocks),
        "gc_counts": [int(gc_counts[0]), int(gc_counts[1]), int(gc_counts[2])],
    }


def _numeric_delta_dict(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> dict[str, int | float]:
    delta: dict[str, int | float] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        if isinstance(after_value, float) or isinstance(before_value, float):
            delta[key] = float(after_value) - float(before_value)
        else:
            delta[key] = int(after_value) - int(before_value)
    return delta


def _reason_count_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
        if int(after.get(key, 0)) - int(before.get(key, 0)) != 0
    }


def _summarize_step_runtime_breakdown(
    *,
    step_index: int,
    step_ms: float,
    adapter_before: dict[str, float],
    adapter_after: dict[str, float],
    chunk_budget_reason_counts_before: dict[str, int],
    chunk_budget_reason_counts_after: dict[str, int],
    trace_before: dict[str, int | float],
    trace_after: dict[str, int | float],
    python_before: dict[str, Any] | None = None,
    python_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter_delta = {key: float(value) for key, value in _numeric_delta_dict(adapter_before, adapter_after).items()}
    trace_delta = _numeric_delta_dict(trace_before, trace_after)
    chunk_budget_reason_delta = _reason_count_delta(
        chunk_budget_reason_counts_before,
        chunk_budget_reason_counts_after,
    )
    python_current_bytes_delta = 0
    python_peak_bytes = 0
    python_allocated_blocks_delta = 0
    python_gc_count_delta = [0, 0, 0]
    if python_before is not None and python_after is not None:
        python_current_bytes_delta = int(python_after["current_bytes"]) - int(python_before["current_bytes"])
        python_peak_bytes = int(python_after["peak_bytes"])
        python_allocated_blocks_delta = int(python_after["allocated_blocks"]) - int(python_before["allocated_blocks"])
        python_gc_count_delta = [
            int(after_count) - int(before_count)
            for before_count, after_count in zip(
                python_before["gc_counts"],
                python_after["gc_counts"],
                strict=True,
            )
        ]
    backend_ms_total = float(sum(float(trace_delta.get(key, 0.0)) for key in _BACKEND_TRACE_TIMING_KEYS))
    decode_runtime_ms = float(adapter_delta["decode_runtime_ms_total"])
    accounted_model_ms = float(
        adapter_delta["qkv_projection_ms_total"]
        + adapter_delta["append_runtime_ms_total"]
        + decode_runtime_ms
        + adapter_delta["output_projection_ms_total"]
    )
    stage_totals = {
        key: float(adapter_delta.get(key, 0.0))
        for key in _MODEL_KV_CACHE_DECODE_STAGE_KEYS
    }
    decode_non_backend_ms_total = float(decode_runtime_ms - backend_ms_total)
    decode_pre_backend_ms_total = float(
        stage_totals["execution_decode_prepare_pages_with_tail_ms_total"]
        + stage_totals["execution_decode_m2_prefilter_ms_total"]
        + stage_totals["execution_decode_query_export_ms_total"]
        + stage_totals["execution_decode_shortlist_selection_ms_total"]
        + stage_totals["execution_decode_shortlist_union_rescue_ms_total"]
        + stage_totals["execution_decode_shortlist_materialization_ms_total"]
        + stage_totals["execution_decode_grouping_validation_ms_total"]
        + stage_totals["execution_decode_chunk_budget_sync_ms_total"]
    )
    return {
        "step_index": int(step_index),
        "step_ms_total": float(step_ms),
        "qkv_projection_ms_total": float(adapter_delta["qkv_projection_ms_total"]),
        "append_runtime_ms_total": float(adapter_delta["append_runtime_ms_total"]),
        "decode_runtime_ms_total": decode_runtime_ms,
        "output_projection_ms_total": float(adapter_delta["output_projection_ms_total"]),
        "backend_prepare_ms_total": float(trace_delta.get("prepare_ms_total", 0.0)),
        "backend_score_ms_total": float(trace_delta.get("score_ms_total", 0.0)),
        "backend_mix_ms_total": float(trace_delta.get("mix_ms_total", 0.0)),
        "backend_softmax_ms_total": float(trace_delta.get("softmax_ms_total", 0.0)),
        "backend_unpack_ms_total": float(trace_delta.get("unpack_ms_total", 0.0)),
        "backend_fwht_ms_total": float(trace_delta.get("fwht_ms_total", 0.0)),
        "backend_chunk_assembly_ms_total": float(trace_delta.get("chunk_assembly_ms_total", 0.0)),
        "backend_decode_ms_total": backend_ms_total,
        "decode_non_backend_ms_total": decode_non_backend_ms_total,
        "decode_prepare_pages_with_tail_ms_total": stage_totals["execution_decode_prepare_pages_with_tail_ms_total"],
        "decode_prepare_layout_build_ms_total": stage_totals["execution_decode_prepare_layout_build_ms_total"],
        "decode_m2_prefilter_ms_total": stage_totals["execution_decode_m2_prefilter_ms_total"],
        "decode_query_export_ms_total": stage_totals["execution_decode_query_export_ms_total"],
        "decode_shortlist_selection_ms_total": stage_totals["execution_decode_shortlist_selection_ms_total"],
        "decode_shortlist_base_window_ms_total": stage_totals["execution_decode_shortlist_base_window_ms_total"],
        "decode_shortlist_candidate_scoring_ms_total": stage_totals["execution_decode_shortlist_candidate_scoring_ms_total"],
        "decode_shortlist_candidate_approx_scoring_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_approx_scoring_ms_total"
        ],
        "decode_shortlist_candidate_ranking_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_ranking_ms_total"
        ],
        "decode_shortlist_candidate_secondary_scoring_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_secondary_scoring_ms_total"
        ],
        "decode_shortlist_candidate_neighbor_rescue_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_neighbor_rescue_ms_total"
        ],
        "decode_shortlist_candidate_builtin_selection_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_builtin_selection_ms_total"
        ],
        "decode_shortlist_candidate_builtin_candidate_index_build_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_builtin_candidate_index_build_ms_total"
        ],
        "decode_shortlist_candidate_builtin_sidecar_stack_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_builtin_sidecar_stack_ms_total"
        ],
        "decode_shortlist_candidate_builtin_score_compute_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_builtin_score_compute_ms_total"
        ],
        "decode_shortlist_candidate_builtin_ranking_ms_total": stage_totals[
            "execution_decode_shortlist_candidate_builtin_ranking_ms_total"
        ],
        "decode_shortlist_exact_selection_ms_total": stage_totals["execution_decode_shortlist_exact_selection_ms_total"],
        "decode_shortlist_union_rescue_ms_total": stage_totals["execution_decode_shortlist_union_rescue_ms_total"],
        "decode_shortlist_materialization_ms_total": stage_totals["execution_decode_shortlist_materialization_ms_total"],
        "decode_grouping_validation_ms_total": stage_totals["execution_decode_grouping_validation_ms_total"],
        "decode_chunk_budget_sync_ms_total": stage_totals["execution_decode_chunk_budget_sync_ms_total"],
        "decode_chunk_budget_dirty_marks": int(adapter_delta.get("execution_chunk_budget_dirty_marks", 0.0)),
        "decode_chunk_budget_dirty_transitions": int(
            adapter_delta.get("execution_chunk_budget_dirty_transitions", 0.0)
        ),
        "decode_chunk_budget_dirty_reason_counts": chunk_budget_reason_delta,
        "decode_chunk_budget_sync_invocations": int(
            adapter_delta.get("execution_chunk_budget_sync_invocations", 0.0)
        ),
        "decode_chunk_budget_sync_clean_skips": int(
            adapter_delta.get("execution_chunk_budget_sync_clean_skips", 0.0)
        ),
        "decode_chunk_budget_sync_dirty_invocations": int(
            adapter_delta.get("execution_chunk_budget_sync_dirty_invocations", 0.0)
        ),
        "decode_chunk_budget_override_calls": int(
            adapter_delta.get("execution_chunk_budget_override_calls", 0.0)
        ),
        "decode_chunk_budget_override_budget_change_calls": int(
            adapter_delta.get("execution_chunk_budget_override_budget_change_calls", 0.0)
        ),
        "decode_chunk_budget_override_same_budget_calls": int(
            adapter_delta.get("execution_chunk_budget_override_same_budget_calls", 0.0)
        ),
        "decode_chunk_budget_freeze_override_calls": int(
            adapter_delta.get("execution_chunk_budget_freeze_override_calls", 0.0)
        ),
        "decode_builtin_selector_score_all_pages_calls": int(
            adapter_delta.get("execution_builtin_selector_score_all_pages_calls", 0.0)
        ),
        "decode_builtin_selector_candidate_only_calls": int(
            adapter_delta.get("execution_builtin_selector_candidate_only_calls", 0.0)
        ),
        "decode_builtin_selector_candidate_pages": int(
            adapter_delta.get("execution_builtin_selector_candidate_pages", 0.0)
        ),
        "decode_builtin_selector_total_pages": int(
            adapter_delta.get("execution_builtin_selector_total_pages", 0.0)
        ),
        "decode_builtin_selector_candidate_fraction_sum": float(
            adapter_delta.get("execution_builtin_selector_candidate_fraction_sum", 0.0)
        ),
        "decode_builtin_selector_candidate_fraction_max": float(
            adapter_delta.get("execution_builtin_selector_candidate_fraction_max", 0.0)
        ),
        "decode_builtin_selector_cache_hits": int(
            adapter_delta.get("execution_builtin_selector_cache_hits", 0.0)
        ),
        "decode_builtin_selector_cache_builds": int(
            adapter_delta.get("execution_builtin_selector_cache_builds", 0.0)
        ),
        "decode_builtin_selector_cache_build_bytes": int(
            adapter_delta.get("execution_builtin_selector_cache_build_bytes", 0.0)
        ),
        "decode_builtin_selector_cache_build_bytes_max": int(
            adapter_delta.get("execution_builtin_selector_cache_build_bytes_max", 0.0)
        ),
        "decode_value_escape_cache_hits": int(
            adapter_delta.get("execution_value_escape_cache_hits", 0.0)
        ),
        "decode_value_escape_source_registrations": int(
            adapter_delta.get("execution_value_escape_source_registrations", 0.0)
        ),
        "decode_value_escape_prepared_page_builds": int(
            adapter_delta.get("execution_value_escape_prepared_page_builds", 0.0)
        ),
        "decode_value_escape_builds": int(
            adapter_delta.get("execution_value_escape_builds", 0.0)
        ),
        "decode_value_escape_applied_pages": int(
            adapter_delta.get("execution_value_escape_applied_pages", 0.0)
        ),
        "decode_backend_call_wall_ms_total": stage_totals["execution_decode_backend_call_wall_ms_total"],
        "decode_backend_call_non_backend_ms_total": stage_totals["execution_decode_backend_call_non_backend_ms_total"],
        "decode_non_backend_unattributed_ms_total": float(
            decode_non_backend_ms_total
            - decode_pre_backend_ms_total
            - stage_totals["execution_decode_backend_call_non_backend_ms_total"]
        ),
        "model_step_accounted_ms_total": accounted_model_ms,
        "model_step_non_adapter_ms_total": float(step_ms - accounted_model_ms),
        "python_tracemalloc_current_bytes_delta": int(python_current_bytes_delta),
        "python_tracemalloc_peak_bytes": int(python_peak_bytes),
        "python_allocated_blocks_delta": int(python_allocated_blocks_delta),
        "python_gc_count_delta": list(python_gc_count_delta),
    }


def run_qwen35_attention_subset_dotcache_serving_quality_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    profile_backend: bool = False,
    trace_python_allocations: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
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

    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        profile_backend=profile_backend,
        multimodal_inputs=multimodal_inputs,
    )
    dotcache_prefill_outputs = prepared["dotcache_prefill_outputs"]
    dotcache_prefill_ms = float(prepared["dotcache_prefill_ms"])
    dotcache_prefill_cuda_memory = prepared["dotcache_prefill_cuda_memory"]
    runtime_state = prepared["runtime_state"]
    serving_shortlist_heuristic_applied = bool(prepared["serving_shortlist_heuristic_applied"])
    device = input_ids.device

    dotcache_step_logits: list[np.ndarray] = []
    dotcache_records: list[list[LlamaReplayRecord]] = []
    dotcache_decode_ms_total = 0.0
    dotcache_step_runtime_breakdown: list[dict[str, Any]] = []
    dotcache_decode_cuda_memory: dict[str, int] = {}
    managed_python_allocation_tracing = _ensure_python_allocation_tracing(trace_python_allocations)
    try:
        if decode_steps > 0:
            current_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
            dotcache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
            for step_index, decode_input_ids in enumerate(dense_capture["decode_inputs"]):
                adapter.begin_capture_step(step_index)
                adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
                adapter_runtime_before = _adapter_runtime_snapshot(adapter)
                chunk_budget_reason_counts_before = _chunk_budget_reason_counts_snapshot(adapter)
                trace_before = _backend_trace_snapshot(adapter)
                if trace_python_allocations:
                    tracemalloc.reset_peak()
                python_before = _python_allocation_snapshot(trace_python_allocations)
                try:
                    outputs, step_ms = _timed_call(
                        lambda: _run_dense_decode_step(
                            model,
                            decode_input_ids=decode_input_ids,
                            attention_mask=current_attention_mask,
                            past_key_values=runtime_state.model_past_key_values,
                            cache_position=cache_position,
                        ),
                        device=device,
                    )
                finally:
                    adapter.set_current_token_index(None)
                adapter_runtime_after = _adapter_runtime_snapshot(adapter)
                chunk_budget_reason_counts_after = _chunk_budget_reason_counts_snapshot(adapter)
                trace_after = _backend_trace_snapshot(adapter)
                python_after = _python_allocation_snapshot(trace_python_allocations)
                dotcache_decode_ms_total += step_ms
                dotcache_step_runtime_breakdown.append(
                    _summarize_step_runtime_breakdown(
                        step_index=step_index,
                        step_ms=step_ms,
                        adapter_before=adapter_runtime_before,
                        adapter_after=adapter_runtime_after,
                        chunk_budget_reason_counts_before=chunk_budget_reason_counts_before,
                        chunk_budget_reason_counts_after=chunk_budget_reason_counts_after,
                        trace_before=trace_before,
                        trace_after=trace_after,
                        python_before=python_before,
                        python_after=python_after,
                    )
                )
                dotcache_records.append(adapter.end_capture_step())
                dotcache_step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
                runtime_state.advance(outputs.past_key_values)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                    dim=1,
                )
                cache_position = cache_position + 1
            dotcache_decode_cuda_memory = _end_cuda_memory_region(device, dotcache_decode_cuda_memory_baseline)
    finally:
        if managed_python_allocation_tracing:
            tracemalloc.stop()

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
            raise ValueError(f"missing DotCache serving replay record for step/layer {replay_key}")
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
        teacher_forced_mean_abs = 0.0
        teacher_forced_rmse = 0.0
        teacher_forced_token_agreement = 1.0
        teacher_forced_per_step_max_abs: list[float] = []
        dense_teacher_forced_loss = 0.0
        dense_teacher_forced_perplexity = 1.0
        dotcache_teacher_forced_loss = 0.0
        dotcache_teacher_forced_perplexity = 1.0
        teacher_forced_loss_delta = 0.0
        teacher_forced_perplexity_ratio = 1.0
    else:
        logit_delta = np.abs(dotcache_logits - dense_logits)
        logit_denom = np.maximum(np.abs(dense_logits), 1e-8)
        teacher_forced_max_abs = float(np.max(logit_delta))
        teacher_forced_max_rel = float(np.max(logit_delta / logit_denom))
        teacher_forced_mean_abs = float(np.mean(logit_delta))
        teacher_forced_rmse = float(np.sqrt(np.mean(np.square(dotcache_logits - dense_logits))))
        dense_target_tokens = np.argmax(dense_logits, axis=-1).astype(np.int64, copy=False)
        dense_stabilized = dense_logits - np.max(dense_logits, axis=-1, keepdims=True)
        dense_log_probs = dense_stabilized - np.log(np.sum(np.exp(dense_stabilized), axis=-1, keepdims=True))
        dense_token_losses = -dense_log_probs[np.arange(dense_target_tokens.shape[0]), dense_target_tokens]
        dense_teacher_forced_loss = float(np.mean(dense_token_losses))
        dense_teacher_forced_perplexity = float(np.exp(min(dense_teacher_forced_loss, 50.0)))
        dotcache_stabilized = dotcache_logits - np.max(dotcache_logits, axis=-1, keepdims=True)
        dotcache_log_probs = dotcache_stabilized - np.log(
            np.sum(np.exp(dotcache_stabilized), axis=-1, keepdims=True)
        )
        dotcache_token_losses = -dotcache_log_probs[np.arange(dense_target_tokens.shape[0]), dense_target_tokens]
        dotcache_teacher_forced_loss = float(np.mean(dotcache_token_losses))
        dotcache_teacher_forced_perplexity = float(np.exp(min(dotcache_teacher_forced_loss, 50.0)))
        teacher_forced_loss_delta = float(dotcache_teacher_forced_loss - dense_teacher_forced_loss)
        teacher_forced_perplexity_ratio = float(
            dotcache_teacher_forced_perplexity / max(dense_teacher_forced_perplexity, 1e-8)
        )
        teacher_forced_token_agreement = float(
            np.mean(
                (
                    np.argmax(dotcache_logits, axis=-1).astype(np.int64, copy=False)
                    == dense_target_tokens
                ).astype(np.float32)
            )
        )
        teacher_forced_per_step_max_abs = [
            float(np.max(np.abs(dotcache_step - dense_step)))
            for dense_step, dotcache_step in zip(dense_logits, dotcache_logits, strict=True)
        ]

    generated_ids = _decode_input_id_sequence(dense_capture["decode_inputs"])
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
            "runtime_mode": "dotcache_attention_subset_serving_quality",
            "dotcache_prefill_ms": float(dotcache_prefill_ms),
            "dense_generated_ids": list(generated_ids),
            "dotcache_generated_ids": list(generated_ids),
            "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "replay_context_max_abs_error": replay_context_max_abs,
            "replay_context_max_rel_error": replay_context_max_rel,
            "replay_output_max_abs_error": replay_output_max_abs,
            "replay_output_max_rel_error": replay_output_max_rel,
            "replay_context_max_abs_error_by_layer": dict(sorted(per_layer_context_max_abs.items())),
            "replay_output_max_abs_error_by_layer": dict(sorted(per_layer_output_max_abs.items())),
            "dense_teacher_forced_loss": dense_teacher_forced_loss,
            "dense_teacher_forced_perplexity": dense_teacher_forced_perplexity,
            "dotcache_teacher_forced_loss": dotcache_teacher_forced_loss,
            "dotcache_teacher_forced_perplexity": dotcache_teacher_forced_perplexity,
            "teacher_forced_loss_delta": teacher_forced_loss_delta,
            "teacher_forced_perplexity_ratio": teacher_forced_perplexity_ratio,
            "teacher_forced_logit_max_abs_error": teacher_forced_max_abs,
            "teacher_forced_logit_max_rel_error": teacher_forced_max_rel,
            "teacher_forced_logit_mean_abs_error": teacher_forced_mean_abs,
            "teacher_forced_logit_rmse": teacher_forced_rmse,
            "teacher_forced_token_agreement_rate": teacher_forced_token_agreement,
            "teacher_forced_per_step_logit_max_abs_error": teacher_forced_per_step_max_abs,
            "dotcache_step_runtime_breakdown": dotcache_step_runtime_breakdown,
            "dotcache_backend_decode_ms_total_from_trace": float(
                sum(step["backend_decode_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_decode_non_backend_ms_total": float(
                sum(step["decode_non_backend_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_model_step_non_adapter_ms_total": float(
                sum(step["model_step_non_adapter_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_allocation_tracing": bool(trace_python_allocations),
            "dotcache_python_tracemalloc_peak_bytes_max": int(
                max((int(step["python_tracemalloc_peak_bytes"]) for step in dotcache_step_runtime_breakdown), default=0)
            ),
            "dotcache_python_tracemalloc_current_bytes_delta_total": int(
                sum(int(step["python_tracemalloc_current_bytes_delta"]) for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_allocated_blocks_delta_total": int(
                sum(int(step["python_allocated_blocks_delta"]) for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_gc_count_delta_total": [
                int(
                    sum(
                        int(step["python_gc_count_delta"][generation_index])
                        for step in dotcache_step_runtime_breakdown
                    )
                )
                for generation_index in range(3)
            ],
            "execution_recent_window": int(adapter.dotcache_config.execution_recent_window),
            "execution_sink_window": int(adapter.dotcache_config.execution_sink_window),
            "execution_recent_window_overrides": list(adapter.dotcache_config.execution_recent_window_overrides),
            "execution_recent_window_context_overrides": list(
                adapter.dotcache_config.execution_recent_window_context_overrides
            ),
            "execution_relevance_top_k": int(adapter.dotcache_config.execution_relevance_top_k),
            "execution_relevance_top_k_overrides": list(adapter.dotcache_config.execution_relevance_top_k_overrides),
            "execution_relevance_top_k_context_overrides": list(adapter.dotcache_config.execution_relevance_top_k_context_overrides),
            "execution_full_context_layers": list(adapter.dotcache_config.execution_full_context_layers),
            "execution_disable_grouped_batching_layers": list(
                adapter.dotcache_config.execution_disable_grouped_batching_layers
            ),
            "execution_recent_old_bonus_window": int(adapter.dotcache_config.execution_recent_old_bonus_window),
            "execution_recent_old_bonus_strength": float(adapter.dotcache_config.execution_recent_old_bonus_strength),
            "execution_recent_old_bonus_layers": list(adapter.dotcache_config.execution_recent_old_bonus_layers),
            "execution_relevance_mode": str(adapter.dotcache_config.execution_relevance_mode),
            "execution_secondary_relevance_mode": str(adapter.dotcache_config.execution_secondary_relevance_mode),
            "execution_secondary_relevance_top_k": int(adapter.dotcache_config.execution_secondary_relevance_top_k),
            "execution_secondary_relevance_min_overlap": float(
                adapter.dotcache_config.execution_secondary_relevance_min_overlap
            ),
            "execution_secondary_relevance_layers": list(adapter.dotcache_config.execution_secondary_relevance_layers),
            "execution_recent_neighbor_rescue_top_k": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_top_k
            ),
            "execution_recent_neighbor_rescue_anchor_window": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_anchor_window
            ),
            "execution_recent_neighbor_rescue_min_anchor_pages": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_min_anchor_pages
            ),
            "execution_recent_neighbor_rescue_layers": list(
                adapter.dotcache_config.execution_recent_neighbor_rescue_layers
            ),
            "execution_exact_promote_top_k": int(adapter.dotcache_config.execution_exact_promote_top_k),
            "execution_exact_promote_min_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_min_margin_threshold
            ),
            "execution_exact_promote_max_context": int(adapter.dotcache_config.execution_exact_promote_max_context),
            "execution_exact_promote_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_margin_threshold
            ),
            "execution_exact_promote_layers": list(adapter.dotcache_config.execution_exact_promote_layers),
            "execution_exact_promote_union_rescue_top_k": int(
                adapter.dotcache_config.execution_exact_promote_union_rescue_top_k
            ),
            "execution_grouped_decode_compact": bool(adapter.dotcache_config.execution_grouped_decode_compact),
            "execution_grouped_mix_compact": bool(adapter.dotcache_config.execution_grouped_mix_compact),
            "execution_grouped_mix_disable_packed_cuda": bool(adapter.dotcache_config.execution_grouped_mix_disable_packed_cuda),
            "execution_freeze_chunk_budget_during_decode": bool(
                adapter.dotcache_config.execution_freeze_chunk_budget_during_decode
            ),
            "execution_builtin_selector_cache": bool(adapter.dotcache_config.execution_builtin_selector_cache),
            "execution_builtin_selector_score_all_pages": bool(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages
            ),
            "execution_builtin_selector_candidate_only": bool(
                adapter.dotcache_config.execution_builtin_selector_candidate_only
            ),
            "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages_min_candidate_fraction
            ),
            "serving_shortlist_heuristic_applied": serving_shortlist_heuristic_applied,
            "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
            "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
            "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
            "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
        }
    )
    result.update({f"dotcache_prefill_{key}": value for key, value in dotcache_prefill_cuda_memory.items()})
    result.update({f"dotcache_decode_{key}": value for key, value in dotcache_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(adapter.per_layer_runtime_summary())
    result.update(adapter.model_kv_cache.decode_path_summary())
    result.update(adapter.model_kv_cache.decode_stage_summary())
    result.update(adapter.model_kv_cache.builtin_selector_summary())
    result.update(adapter.model_kv_cache.chunk_budget_summary())
    result.update(adapter.model_kv_cache.execution_value_escape_summary())
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    decoded_text = _decode_text(tokenizer, generated_ids)
    if decoded_text is not None:
        result["dense_text"] = decoded_text
        result["dotcache_text"] = decoded_text
    return result


def run_qwen35_attention_subset_dotcache_serving_recall_analysis_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    profile_backend: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
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

    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        profile_backend=profile_backend,
        multimodal_inputs=multimodal_inputs,
    )
    dotcache_prefill_outputs = prepared["dotcache_prefill_outputs"]
    dotcache_prefill_ms = float(prepared["dotcache_prefill_ms"])
    dotcache_prefill_cuda_memory = prepared["dotcache_prefill_cuda_memory"]
    runtime_state = prepared["runtime_state"]
    serving_shortlist_heuristic_applied = bool(prepared["serving_shortlist_heuristic_applied"])
    device = input_ids.device

    recall_records: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    dotcache_decode_ms_total = 0.0
    dotcache_decode_cuda_memory: dict[str, int] = {}
    if decode_steps > 0:
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        dotcache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
        for step_index, decode_input_ids in enumerate(dense_capture["decode_inputs"]):
            generated_ids.append(int(decode_input_ids.item()))
            for dense_record in dense_capture["capture_records"][step_index]:
                analysis_record = adapter.model_kv_cache.analyze_execution_shortlist_layer(
                    dense_record.layer_id,
                    dense_record.query_states,
                    adapter.q_head_to_kv_head,
                    trace=None,
                )
                recall_records.append(
                    {
                        "step_index": int(step_index),
                        "token_index": int(dense_record.token_index),
                        **analysis_record,
                    }
                )
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                ),
                device=device,
            )
            dotcache_decode_ms_total += step_ms
            runtime_state.advance(outputs.past_key_values)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = cache_position + 1
        dotcache_decode_cuda_memory = _end_cuda_memory_region(device, dotcache_decode_cuda_memory_baseline)

    recall_values: list[float] = []
    recall_hits_total = 0
    recall_budget_total = 0
    recall_union_added_pages_total = 0
    recall_age_bucket_totals = {"recent": 0, "middle": 0, "old": 0}
    per_layer_recalls: dict[str, list[float]] = {}
    per_layer_first_missed_ranks: dict[str, list[int]] = {}
    per_layer_union_added_pages: dict[str, int] = {}
    per_layer_age_buckets: dict[str, dict[str, int]] = {}
    for record in recall_records:
        layer_key = str(record["layer_id"])
        per_layer_recalls.setdefault(layer_key, [])
        per_layer_first_missed_ranks.setdefault(layer_key, [])
        per_layer_union_added_pages[layer_key] = int(per_layer_union_added_pages.get(layer_key, 0))
        per_layer_age_buckets.setdefault(layer_key, {"recent": 0, "middle": 0, "old": 0})
        for group in record["groups"]:
            recall_budget_total += int(group["exact_top_budget"])
            recall_hits_total += int(group["exact_top_overlap"])
            recall_union_added_pages_total += int(group["union_added_pages"])
            per_layer_union_added_pages[layer_key] += int(group["union_added_pages"])
            if int(group["exact_top_budget"]) > 0:
                per_layer_recalls[layer_key].append(float(group["exact_top_recall"]))
                recall_values.append(float(group["exact_top_recall"]))
            if group["first_missed_exact_rank"] is not None:
                per_layer_first_missed_ranks[layer_key].append(int(group["first_missed_exact_rank"]))
            for age_bucket, count in group["missed_exact_age_buckets"].items():
                recall_age_bucket_totals[str(age_bucket)] += int(count)
                per_layer_age_buckets[layer_key][str(age_bucket)] += int(count)

    shortlist_recall_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else 1.0
        for layer_id, values in sorted(per_layer_recalls.items())
    }
    shortlist_recall_min_by_layer = {
        layer_id: float(min(values)) if values else 1.0
        for layer_id, values in sorted(per_layer_recalls.items())
    }
    shortlist_recall_first_missed_rank_min_by_layer = {
        layer_id: (min(values) if values else None)
        for layer_id, values in sorted(per_layer_first_missed_ranks.items())
    }
    shortlist_recall_worst_layer_id = None
    if shortlist_recall_min_by_layer:
        shortlist_recall_worst_layer_id = min(
            shortlist_recall_min_by_layer.items(),
            key=lambda item: (item[1], shortlist_recall_mean_by_layer.get(item[0], 1.0), int(item[0])),
        )[0]

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
            "runtime_mode": "dotcache_attention_subset_serving_recall_analysis",
            "dotcache_prefill_ms": float(dotcache_prefill_ms),
            "dotcache_generated_ids": generated_ids,
            "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "shortlist_recall_ready": bool(recall_records),
            "shortlist_recall_record_count": int(len(recall_records)),
            "shortlist_recall_step_count": int(len({int(record["step_index"]) for record in recall_records})),
            "shortlist_recall_layer_count": int(len({int(record["layer_id"]) for record in recall_records})),
            "shortlist_recall_exact_top_budget_total": int(recall_budget_total),
            "shortlist_recall_exact_top_hits_total": int(recall_hits_total),
            "shortlist_recall_exact_top_recall_weighted": (
                float(recall_hits_total / max(recall_budget_total, 1)) if recall_budget_total > 0 else 1.0
            ),
            "shortlist_recall_exact_top_recall_mean": float(sum(recall_values) / len(recall_values)) if recall_values else 1.0,
            "shortlist_recall_exact_top_recall_min": float(min(recall_values)) if recall_values else 1.0,
            "shortlist_recall_union_added_pages_total": int(recall_union_added_pages_total),
            "shortlist_recall_missed_exact_age_buckets_total": dict(recall_age_bucket_totals),
            "shortlist_recall_mean_by_layer": shortlist_recall_mean_by_layer,
            "shortlist_recall_min_by_layer": shortlist_recall_min_by_layer,
            "shortlist_recall_first_missed_rank_min_by_layer": shortlist_recall_first_missed_rank_min_by_layer,
            "shortlist_recall_union_added_pages_by_layer": {
                layer_id: int(total) for layer_id, total in sorted(per_layer_union_added_pages.items())
            },
            "shortlist_recall_missed_exact_age_buckets_by_layer": {
                layer_id: dict(sorted(counts.items()))
                for layer_id, counts in sorted(per_layer_age_buckets.items())
            },
            "shortlist_recall_worst_layer_id": shortlist_recall_worst_layer_id,
            "shortlist_recall_layer_records": recall_records,
            "execution_recent_window": int(adapter.dotcache_config.execution_recent_window),
            "execution_sink_window": int(adapter.dotcache_config.execution_sink_window),
            "execution_recent_window_overrides": list(adapter.dotcache_config.execution_recent_window_overrides),
            "execution_recent_window_context_overrides": list(
                adapter.dotcache_config.execution_recent_window_context_overrides
            ),
            "execution_relevance_top_k": int(adapter.dotcache_config.execution_relevance_top_k),
            "execution_relevance_top_k_overrides": list(adapter.dotcache_config.execution_relevance_top_k_overrides),
            "execution_relevance_top_k_context_overrides": list(adapter.dotcache_config.execution_relevance_top_k_context_overrides),
            "execution_full_context_layers": list(adapter.dotcache_config.execution_full_context_layers),
            "execution_disable_grouped_batching_layers": list(
                adapter.dotcache_config.execution_disable_grouped_batching_layers
            ),
            "execution_recent_old_bonus_window": int(adapter.dotcache_config.execution_recent_old_bonus_window),
            "execution_recent_old_bonus_strength": float(adapter.dotcache_config.execution_recent_old_bonus_strength),
            "execution_recent_old_bonus_layers": list(adapter.dotcache_config.execution_recent_old_bonus_layers),
            "execution_relevance_mode": str(adapter.dotcache_config.execution_relevance_mode),
            "execution_secondary_relevance_mode": str(adapter.dotcache_config.execution_secondary_relevance_mode),
            "execution_secondary_relevance_top_k": int(adapter.dotcache_config.execution_secondary_relevance_top_k),
            "execution_secondary_relevance_min_overlap": float(
                adapter.dotcache_config.execution_secondary_relevance_min_overlap
            ),
            "execution_secondary_relevance_layers": list(adapter.dotcache_config.execution_secondary_relevance_layers),
            "execution_recent_neighbor_rescue_top_k": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_top_k
            ),
            "execution_recent_neighbor_rescue_anchor_window": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_anchor_window
            ),
            "execution_recent_neighbor_rescue_min_anchor_pages": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_min_anchor_pages
            ),
            "execution_recent_neighbor_rescue_layers": list(
                adapter.dotcache_config.execution_recent_neighbor_rescue_layers
            ),
            "execution_exact_promote_top_k": int(adapter.dotcache_config.execution_exact_promote_top_k),
            "execution_exact_promote_min_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_min_margin_threshold
            ),
            "execution_exact_promote_max_context": int(adapter.dotcache_config.execution_exact_promote_max_context),
            "execution_exact_promote_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_margin_threshold
            ),
            "execution_exact_promote_layers": list(adapter.dotcache_config.execution_exact_promote_layers),
            "execution_exact_promote_union_rescue_top_k": int(
                adapter.dotcache_config.execution_exact_promote_union_rescue_top_k
            ),
            "execution_grouped_decode_compact": bool(adapter.dotcache_config.execution_grouped_decode_compact),
            "execution_grouped_mix_compact": bool(adapter.dotcache_config.execution_grouped_mix_compact),
            "execution_grouped_mix_disable_packed_cuda": bool(adapter.dotcache_config.execution_grouped_mix_disable_packed_cuda),
            "execution_freeze_chunk_budget_during_decode": bool(
                adapter.dotcache_config.execution_freeze_chunk_budget_during_decode
            ),
            "execution_builtin_selector_cache": bool(adapter.dotcache_config.execution_builtin_selector_cache),
            "execution_builtin_selector_score_all_pages": bool(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages
            ),
            "execution_builtin_selector_candidate_only": bool(
                adapter.dotcache_config.execution_builtin_selector_candidate_only
            ),
            "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages_min_candidate_fraction
            ),
            "serving_shortlist_heuristic_applied": serving_shortlist_heuristic_applied,
            "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
            "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
            "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
            "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
        }
    )
    result.update({f"dotcache_prefill_{key}": value for key, value in dotcache_prefill_cuda_memory.items()})
    result.update({f"dotcache_decode_{key}": value for key, value in dotcache_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(adapter.per_layer_runtime_summary())
    result.update(adapter.model_kv_cache.decode_path_summary())
    result.update(adapter.model_kv_cache.decode_stage_summary())
    result.update(adapter.model_kv_cache.builtin_selector_summary())
    result.update(adapter.model_kv_cache.chunk_budget_summary())
    result.update(adapter.model_kv_cache.execution_value_escape_summary())
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    profile_backend: bool = False,
    trace_python_allocations: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
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

    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        profile_backend=profile_backend,
        multimodal_inputs=multimodal_inputs,
    )
    dotcache_prefill_outputs = prepared["dotcache_prefill_outputs"]
    dotcache_prefill_ms = float(prepared["dotcache_prefill_ms"])
    dotcache_prefill_cuda_memory = prepared["dotcache_prefill_cuda_memory"]
    runtime_state = prepared["runtime_state"]
    serving_shortlist_heuristic_applied = bool(prepared["serving_shortlist_heuristic_applied"])
    device = input_ids.device

    diagnostic_records: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    dotcache_decode_ms_total = 0.0
    dotcache_step_runtime_breakdown: list[dict[str, Any]] = []
    dotcache_decode_cuda_memory: dict[str, int] = {}
    managed_python_allocation_tracing = _ensure_python_allocation_tracing(trace_python_allocations)
    try:
        if decode_steps > 0:
            current_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
            dotcache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
            for step_index, decode_input_ids in enumerate(dense_capture["decode_inputs"]):
                generated_ids.append(int(decode_input_ids.item()))
                for dense_record in dense_capture["capture_records"][step_index]:
                    analysis_record = adapter.model_kv_cache.analyze_execution_shortlist_layer(
                        dense_record.layer_id,
                        dense_record.query_states,
                        adapter.q_head_to_kv_head,
                        trace=None,
                    )
                    diagnostic_records.append(
                        {
                            "step_index": int(step_index),
                            "token_index": int(dense_record.token_index),
                            **analysis_record,
                        }
                    )
                adapter_runtime_before = _adapter_runtime_snapshot(adapter)
                chunk_budget_reason_counts_before = _chunk_budget_reason_counts_snapshot(adapter)
                trace_before = _backend_trace_snapshot(adapter)
                if trace_python_allocations:
                    tracemalloc.reset_peak()
                python_before = _python_allocation_snapshot(trace_python_allocations)
                outputs, step_ms = _timed_call(
                    lambda: _run_dense_decode_step(
                        model,
                        decode_input_ids=decode_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=runtime_state.model_past_key_values,
                        cache_position=cache_position,
                    ),
                    device=device,
                )
                adapter_runtime_after = _adapter_runtime_snapshot(adapter)
                chunk_budget_reason_counts_after = _chunk_budget_reason_counts_snapshot(adapter)
                trace_after = _backend_trace_snapshot(adapter)
                python_after = _python_allocation_snapshot(trace_python_allocations)
                dotcache_decode_ms_total += step_ms
                dotcache_step_runtime_breakdown.append(
                    _summarize_step_runtime_breakdown(
                        step_index=step_index,
                        step_ms=step_ms,
                        adapter_before=adapter_runtime_before,
                        adapter_after=adapter_runtime_after,
                        chunk_budget_reason_counts_before=chunk_budget_reason_counts_before,
                        chunk_budget_reason_counts_after=chunk_budget_reason_counts_after,
                        trace_before=trace_before,
                        trace_after=trace_after,
                        python_before=python_before,
                        python_after=python_after,
                    )
                )
                runtime_state.advance(outputs.past_key_values)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                    dim=1,
                )
                cache_position = cache_position + 1
            dotcache_decode_cuda_memory = _end_cuda_memory_region(device, dotcache_decode_cuda_memory_baseline)
    finally:
        if managed_python_allocation_tracing:
            tracemalloc.stop()

    rank_corr_by_layer: dict[str, list[float]] = {}
    value_corr_by_layer: dict[str, list[float]] = {}
    approx_recall_by_layer: dict[str, list[float]] = {}
    exact_top1_approx_rank_by_layer: dict[str, list[int]] = {}
    approx_top1_exact_rank_by_layer: dict[str, list[int]] = {}
    mean_abs_rank_error_by_layer: dict[str, list[float]] = {}
    boundary_margin_by_layer: dict[str, list[float]] = {}
    secondary_primary_recall_by_layer: dict[str, list[float]] = {}
    secondary_exact_recall_by_layer: dict[str, list[float]] = {}
    secondary_trigger_count_by_layer: dict[str, int] = {}
    secondary_group_count_by_layer: dict[str, int] = {}
    recent_neighbor_trigger_count_by_layer: dict[str, int] = {}
    recent_neighbor_group_count_by_layer: dict[str, int] = {}
    scorer_missed_age_buckets_by_layer: dict[str, dict[str, int]] = {}
    for record in diagnostic_records:
        layer_key = str(record["layer_id"])
        rank_corr_by_layer.setdefault(layer_key, [])
        value_corr_by_layer.setdefault(layer_key, [])
        approx_recall_by_layer.setdefault(layer_key, [])
        exact_top1_approx_rank_by_layer.setdefault(layer_key, [])
        approx_top1_exact_rank_by_layer.setdefault(layer_key, [])
        mean_abs_rank_error_by_layer.setdefault(layer_key, [])
        boundary_margin_by_layer.setdefault(layer_key, [])
        secondary_primary_recall_by_layer.setdefault(layer_key, [])
        secondary_exact_recall_by_layer.setdefault(layer_key, [])
        secondary_trigger_count_by_layer.setdefault(layer_key, 0)
        secondary_group_count_by_layer.setdefault(layer_key, 0)
        recent_neighbor_trigger_count_by_layer.setdefault(layer_key, 0)
        recent_neighbor_group_count_by_layer.setdefault(layer_key, 0)
        scorer_missed_age_buckets_by_layer.setdefault(layer_key, {"recent": 0, "middle": 0, "old": 0})
        for group in record["groups"]:
            if group["score_rank_correlation"] is not None:
                rank_corr_by_layer[layer_key].append(float(group["score_rank_correlation"]))
            if group["score_value_correlation"] is not None:
                value_corr_by_layer[layer_key].append(float(group["score_value_correlation"]))
            approx_recall_by_layer[layer_key].append(float(group["approx_exact_top_recall"]))
            if group["exact_top1_approx_rank"] is not None:
                exact_top1_approx_rank_by_layer[layer_key].append(int(group["exact_top1_approx_rank"]))
            if group["approx_top1_exact_rank"] is not None:
                approx_top1_exact_rank_by_layer[layer_key].append(int(group["approx_top1_exact_rank"]))
            if group["mean_abs_rank_error"] is not None:
                mean_abs_rank_error_by_layer[layer_key].append(float(group["mean_abs_rank_error"]))
            if group["approx_boundary_margin_normalized"] is not None:
                boundary_margin_by_layer[layer_key].append(float(group["approx_boundary_margin_normalized"]))
            if group["secondary_relevance_mode"] is not None:
                secondary_group_count_by_layer[layer_key] += 1
                secondary_primary_recall_by_layer[layer_key].append(float(group["secondary_primary_top_recall"]))
                secondary_exact_recall_by_layer[layer_key].append(float(group["secondary_exact_top_recall"]))
                if bool(group["secondary_triggered"]):
                    secondary_trigger_count_by_layer[layer_key] += 1
            if "recent_neighbor_rescue_triggered" in group:
                recent_neighbor_group_count_by_layer[layer_key] += 1
                if bool(group["recent_neighbor_rescue_triggered"]):
                    recent_neighbor_trigger_count_by_layer[layer_key] += 1
            for age_bucket, count in group["scorer_missed_exact_age_buckets"].items():
                scorer_missed_age_buckets_by_layer[layer_key][str(age_bucket)] += int(count)

    scorer_rank_correlation_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(rank_corr_by_layer.items())
    }
    scorer_value_correlation_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(value_corr_by_layer.items())
    }
    scorer_approx_exact_top_recall_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else 1.0
        for layer_id, values in sorted(approx_recall_by_layer.items())
    }
    scorer_exact_top1_approx_rank_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(exact_top1_approx_rank_by_layer.items())
    }
    scorer_approx_top1_exact_rank_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(approx_top1_exact_rank_by_layer.items())
    }
    scorer_mean_abs_rank_error_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(mean_abs_rank_error_by_layer.items())
    }
    scorer_boundary_margin_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(boundary_margin_by_layer.items())
    }
    scorer_secondary_primary_top_recall_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(secondary_primary_recall_by_layer.items())
    }
    scorer_secondary_exact_top_recall_mean_by_layer = {
        layer_id: float(sum(values) / len(values)) if values else None
        for layer_id, values in sorted(secondary_exact_recall_by_layer.items())
    }
    scorer_secondary_trigger_rate_by_layer = {
        layer_id: (
            float(secondary_trigger_count_by_layer[layer_id] / max(secondary_group_count_by_layer[layer_id], 1))
            if secondary_group_count_by_layer.get(layer_id, 0) > 0
            else None
        )
        for layer_id in sorted(secondary_group_count_by_layer.keys())
    }
    scorer_recent_neighbor_rescue_trigger_rate_by_layer = {
        layer_id: (
            float(recent_neighbor_trigger_count_by_layer[layer_id] / max(recent_neighbor_group_count_by_layer[layer_id], 1))
            if recent_neighbor_group_count_by_layer.get(layer_id, 0) > 0
            else None
        )
        for layer_id in sorted(recent_neighbor_group_count_by_layer.keys())
    }
    scorer_worst_layer_id = None
    if scorer_approx_exact_top_recall_mean_by_layer:
        scorer_worst_layer_id = min(
            scorer_approx_exact_top_recall_mean_by_layer.items(),
            key=lambda item: (item[1], int(item[0])),
        )[0]

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
            "runtime_mode": "dotcache_attention_subset_serving_scorer_diagnostic",
            "dotcache_prefill_ms": float(dotcache_prefill_ms),
            "dotcache_generated_ids": generated_ids,
            "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
            "scorer_diagnostic_ready": bool(diagnostic_records),
            "scorer_diagnostic_record_count": int(len(diagnostic_records)),
            "scorer_diagnostic_layer_count": int(len({int(record["layer_id"]) for record in diagnostic_records})),
            "scorer_diagnostic_step_count": int(len({int(record["step_index"]) for record in diagnostic_records})),
            "dotcache_step_runtime_breakdown": dotcache_step_runtime_breakdown,
            "dotcache_backend_decode_ms_total_from_trace": float(
                sum(step["backend_decode_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_decode_non_backend_ms_total": float(
                sum(step["decode_non_backend_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_model_step_non_adapter_ms_total": float(
                sum(step["model_step_non_adapter_ms_total"] for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_allocation_tracing": bool(trace_python_allocations),
            "dotcache_python_tracemalloc_peak_bytes_max": int(
                max((int(step["python_tracemalloc_peak_bytes"]) for step in dotcache_step_runtime_breakdown), default=0)
            ),
            "dotcache_python_tracemalloc_current_bytes_delta_total": int(
                sum(int(step["python_tracemalloc_current_bytes_delta"]) for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_allocated_blocks_delta_total": int(
                sum(int(step["python_allocated_blocks_delta"]) for step in dotcache_step_runtime_breakdown)
            ),
            "dotcache_python_gc_count_delta_total": [
                int(
                    sum(
                        int(step["python_gc_count_delta"][generation_index])
                        for step in dotcache_step_runtime_breakdown
                    )
                )
                for generation_index in range(3)
            ],
            "scorer_rank_correlation_mean_by_layer": scorer_rank_correlation_mean_by_layer,
            "scorer_value_correlation_mean_by_layer": scorer_value_correlation_mean_by_layer,
            "scorer_approx_exact_top_recall_mean_by_layer": scorer_approx_exact_top_recall_mean_by_layer,
            "scorer_exact_top1_approx_rank_mean_by_layer": scorer_exact_top1_approx_rank_mean_by_layer,
            "scorer_approx_top1_exact_rank_mean_by_layer": scorer_approx_top1_exact_rank_mean_by_layer,
            "scorer_mean_abs_rank_error_by_layer": scorer_mean_abs_rank_error_by_layer,
            "scorer_boundary_margin_mean_by_layer": scorer_boundary_margin_mean_by_layer,
            "scorer_secondary_primary_top_recall_mean_by_layer": scorer_secondary_primary_top_recall_mean_by_layer,
            "scorer_secondary_exact_top_recall_mean_by_layer": scorer_secondary_exact_top_recall_mean_by_layer,
            "scorer_secondary_trigger_rate_by_layer": scorer_secondary_trigger_rate_by_layer,
            "scorer_recent_neighbor_rescue_trigger_rate_by_layer": scorer_recent_neighbor_rescue_trigger_rate_by_layer,
            "scorer_missed_exact_age_buckets_by_layer": {
                layer_id: dict(sorted(counts.items()))
                for layer_id, counts in sorted(scorer_missed_age_buckets_by_layer.items())
            },
            "scorer_worst_layer_id": scorer_worst_layer_id,
            "scorer_layer_records": diagnostic_records,
            "execution_recent_window": int(adapter.dotcache_config.execution_recent_window),
            "execution_sink_window": int(adapter.dotcache_config.execution_sink_window),
            "execution_recent_window_overrides": list(adapter.dotcache_config.execution_recent_window_overrides),
            "execution_recent_window_context_overrides": list(
                adapter.dotcache_config.execution_recent_window_context_overrides
            ),
            "execution_relevance_top_k": int(adapter.dotcache_config.execution_relevance_top_k),
            "execution_relevance_top_k_overrides": list(adapter.dotcache_config.execution_relevance_top_k_overrides),
            "execution_relevance_top_k_context_overrides": list(adapter.dotcache_config.execution_relevance_top_k_context_overrides),
            "execution_full_context_layers": list(adapter.dotcache_config.execution_full_context_layers),
            "execution_disable_grouped_batching_layers": list(
                adapter.dotcache_config.execution_disable_grouped_batching_layers
            ),
            "execution_recent_old_bonus_window": int(adapter.dotcache_config.execution_recent_old_bonus_window),
            "execution_recent_old_bonus_strength": float(adapter.dotcache_config.execution_recent_old_bonus_strength),
            "execution_recent_old_bonus_layers": list(adapter.dotcache_config.execution_recent_old_bonus_layers),
            "execution_relevance_mode": str(adapter.dotcache_config.execution_relevance_mode),
            "execution_secondary_relevance_mode": str(adapter.dotcache_config.execution_secondary_relevance_mode),
            "execution_secondary_relevance_top_k": int(adapter.dotcache_config.execution_secondary_relevance_top_k),
            "execution_secondary_relevance_min_overlap": float(
                adapter.dotcache_config.execution_secondary_relevance_min_overlap
            ),
            "execution_secondary_relevance_layers": list(adapter.dotcache_config.execution_secondary_relevance_layers),
            "execution_recent_neighbor_rescue_top_k": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_top_k
            ),
            "execution_recent_neighbor_rescue_anchor_window": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_anchor_window
            ),
            "execution_recent_neighbor_rescue_min_anchor_pages": int(
                adapter.dotcache_config.execution_recent_neighbor_rescue_min_anchor_pages
            ),
            "execution_recent_neighbor_rescue_layers": list(
                adapter.dotcache_config.execution_recent_neighbor_rescue_layers
            ),
            "execution_exact_promote_top_k": int(adapter.dotcache_config.execution_exact_promote_top_k),
            "execution_exact_promote_min_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_min_margin_threshold
            ),
            "execution_exact_promote_max_context": int(adapter.dotcache_config.execution_exact_promote_max_context),
            "execution_exact_promote_margin_threshold": float(
                adapter.dotcache_config.execution_exact_promote_margin_threshold
            ),
            "execution_exact_promote_layers": list(adapter.dotcache_config.execution_exact_promote_layers),
            "execution_exact_promote_union_rescue_top_k": int(
                adapter.dotcache_config.execution_exact_promote_union_rescue_top_k
            ),
            "execution_grouped_decode_compact": bool(adapter.dotcache_config.execution_grouped_decode_compact),
            "execution_grouped_mix_compact": bool(adapter.dotcache_config.execution_grouped_mix_compact),
            "execution_grouped_mix_disable_packed_cuda": bool(adapter.dotcache_config.execution_grouped_mix_disable_packed_cuda),
            "execution_freeze_chunk_budget_during_decode": bool(
                adapter.dotcache_config.execution_freeze_chunk_budget_during_decode
            ),
            "execution_builtin_selector_cache": bool(adapter.dotcache_config.execution_builtin_selector_cache),
            "execution_builtin_selector_score_all_pages": bool(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages
            ),
            "execution_builtin_selector_candidate_only": bool(
                adapter.dotcache_config.execution_builtin_selector_candidate_only
            ),
            "execution_builtin_selector_score_all_pages_min_candidate_fraction": float(
                adapter.dotcache_config.execution_builtin_selector_score_all_pages_min_candidate_fraction
            ),
            "serving_shortlist_heuristic_applied": serving_shortlist_heuristic_applied,
            "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
            "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
            "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
            "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
        }
    )
    result.update({f"dotcache_prefill_{key}": value for key, value in dotcache_prefill_cuda_memory.items()})
    result.update({f"dotcache_decode_{key}": value for key, value in dotcache_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(adapter.per_layer_runtime_summary())
    result.update(adapter.model_kv_cache.decode_path_summary())
    result.update(adapter.model_kv_cache.decode_stage_summary())
    result.update(adapter.model_kv_cache.builtin_selector_summary())
    result.update(adapter.model_kv_cache.chunk_budget_summary())
    result.update(adapter.model_kv_cache.execution_value_escape_summary())
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_attention_subset_dotcache_loss_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    tokenizer=None,
    profile_backend: bool = False,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_backend_profiling(profile_backend)
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
    if prefix_length <= 0 or prefix_length >= int(input_ids.shape[1]):
        raise ValueError("prefix_length must be in [1, sequence_length)")
    available_eval_steps = int(input_ids.shape[1]) - prefix_length
    if eval_steps <= 0 or eval_steps > available_eval_steps:
        raise ValueError("eval_steps must be positive and fit inside the provided sequence after prefix_length")

    prefix_input_ids = input_ids[:, :prefix_length]
    prefix_attention_mask = attention_mask[:, :prefix_length]
    continuation_ids = input_ids[:, prefix_length : prefix_length + eval_steps]
    dense_capture = _run_qwen35_attention_subset_dense_teacher_forced_capture(
        model,
        adapter,
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        continuation_ids=continuation_ids,
    )

    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        tokenizer=tokenizer,
        profile_backend=profile_backend,
        multimodal_inputs=multimodal_inputs,
    )
    dotcache_prefill_outputs = prepared["dotcache_prefill_outputs"]
    dotcache_prefill_ms = float(prepared["dotcache_prefill_ms"])
    dotcache_prefill_cuda_memory = prepared["dotcache_prefill_cuda_memory"]
    runtime_state = prepared["runtime_state"]
    device = prefix_input_ids.device

    dotcache_logits_list = [dotcache_prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy()]
    dotcache_decode_ms_total = 0.0
    dotcache_decode_cuda_memory: dict[str, int] = {}
    if eval_steps > 1:
        current_attention_mask = torch.cat(
            [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=device)],
            dim=1,
        )
        cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=device)
        dotcache_decode_cuda_memory_baseline = _begin_cuda_memory_region(device)
        for step_index in range(eval_steps - 1):
            decode_input_ids = continuation_ids[:, step_index : step_index + 1]
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                ),
                device=device,
            )
            dotcache_decode_ms_total += step_ms
            runtime_state.advance(outputs.past_key_values)
            dotcache_logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = cache_position + 1
        dotcache_decode_cuda_memory = _end_cuda_memory_region(device, dotcache_decode_cuda_memory_baseline)

    dense_logits = np.concatenate(
        [logits.astype(np.float32, copy=False) for logits in dense_capture["step_logits"]],
        axis=0,
    )
    dotcache_logits = np.concatenate(
        [logits.astype(np.float32, copy=False) for logits in dotcache_logits_list],
        axis=0,
    )
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
    target_match = float(np.mean((dotcache_predictions == target_tokens).astype(np.float32)))
    logit_delta = np.abs(dotcache_logits - dense_logits)
    logit_denom = np.maximum(np.abs(dense_logits), 1e-8)

    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "dotcache_prefill_ms": float(dotcache_prefill_ms),
        "dense_decode_ms_per_step": float(dense_capture["decode_ms_total"] / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "dotcache_decode_ms_per_step": float(dotcache_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "dense_teacher_forced_loss": dense_loss,
        "dense_teacher_forced_perplexity": dense_perplexity,
        "dotcache_teacher_forced_loss": dotcache_loss,
        "dotcache_teacher_forced_perplexity": dotcache_perplexity,
        "teacher_forced_loss_delta": float(dotcache_loss - dense_loss),
        "teacher_forced_perplexity_ratio": float(dotcache_perplexity / max(dense_perplexity, 1e-8)),
        "teacher_forced_token_agreement_rate": token_agreement,
        "teacher_forced_target_match_rate": target_match,
        "teacher_forced_logit_max_abs_error": float(np.max(logit_delta)),
        "teacher_forced_logit_max_rel_error": float(np.max(logit_delta / logit_denom)),
        "dotcache_attention_subset_ready": True,
        "dotcache_ready": False,
        "runtime_mode": "dotcache_attention_subset_loss",
        "uses_native_qwen35_class": True,
        "text_only": True,
        "attention_subset_layer_ids": adapter.attention_subset_layer_ids(),
        "attention_subset_capture_layer_count": len(adapter.attention_subset_layer_ids()),
        "dotcache_append_runtime_ms_total": float(adapter.append_runtime_ms_total),
        "dotcache_decode_runtime_ms_total": float(adapter.decode_runtime_ms_total),
        "dotcache_qkv_projection_ms_total": float(adapter.qkv_projection_ms_total),
        "dotcache_output_projection_ms_total": float(adapter.output_projection_ms_total),
    }
    result.update({f"dotcache_prefill_{key}": value for key, value in dotcache_prefill_cuda_memory.items()})
    result.update({f"dotcache_decode_{key}": value for key, value in dotcache_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(runtime_state.summary())
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


def run_qwen35_attention_subset_statecache_dotcache_harness(
    model,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
    profile_backend: bool = False,
    group_size: int = 32,
    bits: int = 8,
    state_stage: Qwen35DeltaNetStateCacheStage = "post_update_m0",
    renorm_interval: int = 0,
    recurrent_mode_overrides: dict[int, Qwen35DeltaNetStateCacheMode] | None = None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_backend_profiling(profile_backend)
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
    device = input_ids.device
    dense_capture_cuda_memory_baseline = _begin_cuda_memory_region(device)
    dense_capture = _run_qwen35_attention_subset_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
    )
    dense_capture_cuda_memory = _end_cuda_memory_region(device, dense_capture_cuda_memory_baseline)

    hybrid_prefill_cuda_memory_baseline = _begin_cuda_memory_region(device)
    hybrid_prefill_outputs, hybrid_prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    hybrid_prefill_cuda_memory = _end_cuda_memory_region(device, hybrid_prefill_cuda_memory_baseline)
    adapter.clear()
    adapter.load_attention_subset_prefill_cache(hybrid_prefill_outputs.past_key_values)
    adapter.set_mode("dotcache_attention_subset")

    runtime_state = adapter.require_hybrid_dotcache_runtime_state()
    deltanet_layer_ids = adapter.deltanet_layer_ids()
    resolved_recurrent_mode_overrides = {
        int(layer_id): _resolve_qwen35_deltanet_statecache_mode(
            int(layer_id),
            default_mode="M0",
            mode_overrides=recurrent_mode_overrides,
        )
        for layer_id in deltanet_layer_ids
    }
    prefill_partition = runtime_state.native_state.prefill_partition
    recurrent_dense_bytes = 0
    recurrent_statecache_bytes = 0
    per_layer_dense_recurrent_bytes: dict[str, int] = {}
    per_layer_statecache_recurrent_bytes: dict[str, int] = {}
    per_layer_statecache_modes: dict[str, str] = {}
    for layer in prefill_partition.fixed_resident_layers:
        if layer.recurrent_state is None:
            continue
        layer_id = str(int(layer.layer_id))
        recurrent_mode = resolved_recurrent_mode_overrides.get(int(layer.layer_id), "M0")
        dense_bytes = int(layer.recurrent_state_bytes)
        compressed_bytes = _compressed_state_nbytes(
            layer.recurrent_state,
            bits=int(bits),
            group_size=int(group_size),
            mode=recurrent_mode,
        )
        recurrent_dense_bytes += dense_bytes
        recurrent_statecache_bytes += compressed_bytes
        per_layer_dense_recurrent_bytes[layer_id] = dense_bytes
        per_layer_statecache_recurrent_bytes[layer_id] = compressed_bytes
        per_layer_statecache_modes[layer_id] = recurrent_mode
    conv_state_bytes = int(sum(layer.conv_state_bytes for layer in prefill_partition.fixed_resident_layers))
    dense_fixed_resident_bytes = int(sum(layer.fixed_resident_state_bytes for layer in prefill_partition.fixed_resident_layers))
    statecache_fixed_resident_bytes = int(conv_state_bytes + recurrent_statecache_bytes)

    if state_stage == "post_update_m0":
        _prepare_qwen35_deltanet_recurrent_statecache(
            runtime_state.model_past_key_values,
            layer_ids=deltanet_layer_ids,
            bits=int(bits),
            group_size=int(group_size),
            renorm=False,
            mode_overrides=recurrent_mode_overrides,
        )

    hybrid_step_logits: list[np.ndarray] = []
    hybrid_records: list[list[LlamaReplayRecord]] = []
    hybrid_decode_ms_total = 0.0
    hybrid_decode_cuda_memory: dict[str, int] = {}
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
    hybrid_decode_cuda_memory_baseline = _begin_cuda_memory_region(device) if decode_steps > 0 else None
    for step_index, decode_input_ids in enumerate(dense_capture["decode_inputs"]):
        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
        try:
            def _run_hybrid_decode():
                if state_stage == "readout_only_m0":
                    _prepare_qwen35_deltanet_recurrent_statecache(
                        runtime_state.model_past_key_values,
                        layer_ids=deltanet_layer_ids,
                        bits=int(bits),
                        group_size=int(group_size),
                        renorm=False,
                        mode_overrides=recurrent_mode_overrides,
                    )
                return _run_dense_decode_step(
                    model,
                    decode_input_ids=decode_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=runtime_state.model_past_key_values,
                    cache_position=cache_position,
                )

            outputs, step_ms = _timed_call(
                _run_hybrid_decode,
                device=input_ids.device,
            )
        finally:
            adapter.set_current_token_index(None)
        hybrid_decode_ms_total += step_ms
        hybrid_records.append(adapter.end_capture_step())
        hybrid_step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        next_past_key_values = outputs.past_key_values
        if state_stage == "post_update_m0":
            _prepare_qwen35_deltanet_recurrent_statecache(
                next_past_key_values,
                layer_ids=deltanet_layer_ids,
                bits=int(bits),
                group_size=int(group_size),
                renorm=bool(renorm_interval > 0 and (step_index + 1) % int(renorm_interval) == 0),
                mode_overrides=recurrent_mode_overrides,
            )
        runtime_state.advance(next_past_key_values)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1
    if hybrid_decode_cuda_memory_baseline is not None:
        hybrid_decode_cuda_memory = _end_cuda_memory_region(device, hybrid_decode_cuda_memory_baseline)

    dense_record_map = {
        (record.step_index, record.layer_id): record
        for step_records in dense_capture["capture_records"]
        for record in step_records
    }
    hybrid_record_map = {
        (record.step_index, record.layer_id): record
        for step_records in hybrid_records
        for record in step_records
    }
    replay_context_max_abs = 0.0
    replay_context_max_rel = 0.0
    replay_output_max_abs = 0.0
    replay_output_max_rel = 0.0
    per_layer_context_max_abs: dict[str, float] = {}
    per_layer_output_max_abs: dict[str, float] = {}
    for replay_key, dense_record in dense_record_map.items():
        hybrid_record = hybrid_record_map.get(replay_key)
        if hybrid_record is None:
            raise ValueError(f"missing combined replay record for step/layer {replay_key}")
        context_delta = np.abs(hybrid_record.context_states - dense_record.context_states)
        context_denom = np.maximum(np.abs(dense_record.context_states), 1e-8)
        replay_context_max_abs = max(replay_context_max_abs, float(np.max(context_delta)))
        replay_context_max_rel = max(replay_context_max_rel, float(np.max(context_delta / context_denom)))
        layer_key = str(dense_record.layer_id)
        per_layer_context_max_abs[layer_key] = max(
            per_layer_context_max_abs.get(layer_key, 0.0),
            float(np.max(context_delta)),
        )
        output_delta = np.abs(hybrid_record.output_states - dense_record.output_states)
        output_denom = np.maximum(np.abs(dense_record.output_states), 1e-8)
        replay_output_max_abs = max(replay_output_max_abs, float(np.max(output_delta)))
        replay_output_max_rel = max(replay_output_max_rel, float(np.max(output_delta / output_denom)))
        per_layer_output_max_abs[layer_key] = max(
            per_layer_output_max_abs.get(layer_key, 0.0),
            float(np.max(output_delta)),
        )

    dense_logits = np.stack(dense_capture["step_logits"], axis=0) if dense_capture["step_logits"] else np.zeros((0, 1))
    hybrid_logits = np.stack(hybrid_step_logits, axis=0) if hybrid_step_logits else np.zeros((0, 1))
    if dense_logits.size == 0:
        teacher_forced_max_abs = 0.0
        teacher_forced_max_rel = 0.0
    else:
        logit_delta = np.abs(hybrid_logits - dense_logits)
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
            "deltanet_statecache_ready": True,
            "hybrid_dotcache_statecache_ready": True,
            "dotcache_ready": False,
            "runtime_mode": "dotcache_attention_subset_deltanet_statecache",
            "dotcache_prefill_ms": float(hybrid_prefill_ms),
            "dotcache_decode_ms_per_step": float(hybrid_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
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
            "deltanet_statecache_stage_name": str(state_stage),
            "deltanet_statecache_group_size": int(group_size),
            "deltanet_statecache_bits": int(bits),
            "deltanet_statecache_mode": "M0",
            "deltanet_statecache_renorm_interval": int(renorm_interval),
            "deltanet_statecache_recurrent_mode_overrides": {
                str(layer_id): mode for layer_id, mode in sorted(resolved_recurrent_mode_overrides.items()) if mode != "M0"
            },
            "deltanet_conv_state_bytes": conv_state_bytes,
            "deltanet_recurrent_state_bytes": recurrent_dense_bytes,
            "deltanet_statecache_recurrent_state_bytes": int(recurrent_statecache_bytes),
            "deltanet_dense_fixed_resident_bytes": dense_fixed_resident_bytes,
            "deltanet_statecache_fixed_resident_bytes": statecache_fixed_resident_bytes,
            "deltanet_statecache_effective_recurrent_compression_ratio": (
                float(recurrent_dense_bytes / max(recurrent_statecache_bytes, 1)) if recurrent_dense_bytes > 0 else 1.0
            ),
            "deltanet_statecache_effective_fixed_resident_compression_ratio": (
                float(dense_fixed_resident_bytes / max(statecache_fixed_resident_bytes, 1)) if dense_fixed_resident_bytes > 0 else 1.0
            ),
            "deltanet_statecache_per_layer_dense_recurrent_bytes": per_layer_dense_recurrent_bytes,
            "deltanet_statecache_per_layer_recurrent_bytes": per_layer_statecache_recurrent_bytes,
            "deltanet_statecache_per_layer_recurrent_mode": per_layer_statecache_modes,
        }
    )
    result.update({f"dense_capture_{key}": value for key, value in dense_capture_cuda_memory.items()})
    result.update({f"hybrid_prefill_{key}": value for key, value in hybrid_prefill_cuda_memory.items()})
    result.update({f"hybrid_decode_{key}": value for key, value in hybrid_decode_cuda_memory.items()})
    if profile_backend:
        result["decode_backend_trace"] = adapter.decode_backend_trace.to_dict()
    result.update(runtime_state.summary())
    result["hybrid_runtime_state_kind"] = "qwen35_attention_subset_statecache"
    return result


__all__ = [
    "Qwen35AttentionSubsetDotCacheHarness",
    "Qwen35AttentionSubsetDotCacheModelAdapter",
    "Qwen35AttentionSubsetHarness",
    "Qwen35AttentionSubsetModelAdapter",
    "Qwen35DeltaNetStateRecord",
    "Qwen35DeltaNetStateHarness",
    "Qwen35DeltaNetStateModelAdapter",
    "build_qwen35_deltanet_state_sample",
    "capture_qwen35_deltanet_state_sample",
    "Qwen35TextHarness",
    "Qwen35TextModelAdapter",
    "inspect_qwen35_deltanet_state",
    "inspect_qwen35_hybrid_state",
    "load_qwen35_text_only_from_pretrained",
    "run_qwen35_attention_subset_prefill_ablation_harness",
    "run_qwen35_hybrid_combined_localization_harness",
    "run_qwen35_attention_subset_dotcache_harness",
    "run_qwen35_attention_subset_dotcache_serving_harness",
    "run_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_harness",
    "run_qwen35_attention_subset_dotcache_serving_recall_analysis_harness",
    "run_qwen35_attention_subset_dotcache_serving_quality_harness",
    "run_qwen35_attention_subset_dotcache_loss_harness",
    "run_qwen35_attention_subset_statecache_dotcache_harness",
    "run_qwen35_attention_subset_replay_harness",
    "run_qwen35_deltanet_state_ablation_harness",
    "run_qwen35_deltanet_statecache_localization_harness",
    "run_qwen35_deltanet_statecache_readout_harness",
    "run_qwen35_deltanet_statecache_serving_harness",
    "run_qwen35_deltanet_statecache_loss_harness",
    "run_qwen35_text_generation_harness",
    "run_qwen35_text_loss_harness",
    "save_qwen35_deltanet_state_sample",
    "summarize_qwen35_dotcache_fit",
    "summarize_qwen35_hybrid_state",
    "summarize_qwen35_hybrid_state_growth",
]
