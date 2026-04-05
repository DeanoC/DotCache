from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, replace
from typing import Any

from ..config import DotCacheConfig
from ..model_kv_cache import ModelPagedKVCache, PreparedPageCache
from .llama import _require_transformers, resolve_hf_auth_kwargs, transformers_available

if transformers_available():
    from transformers import AutoConfig
    import torch
else:  # pragma: no cover - exercised in environments without transformers
    AutoConfig = None
    torch = None


_NEMOTRON_H_PATTERN_TO_BLOCK_TYPE: dict[str, str] = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
}


def _nemotron_h_config(model_or_config: Any) -> Any:
    return getattr(model_or_config, "config", model_or_config)


def parse_nemotron_h_hybrid_pattern(pattern: str) -> tuple[str, ...]:
    normalized_pattern = str(pattern)
    if not normalized_pattern:
        return ()
    block_types: list[str] = []
    for char in normalized_pattern:
        try:
            block_types.append(_NEMOTRON_H_PATTERN_TO_BLOCK_TYPE[char])
        except KeyError as exc:
            raise ValueError(
                f"unsupported Nemotron-H hybrid pattern character {char!r}; "
                f"expected one of {tuple(_NEMOTRON_H_PATTERN_TO_BLOCK_TYPE)}"
            ) from exc
    return tuple(block_types)


def nemotron_h_layer_types(model_or_config: Any) -> tuple[str, ...]:
    config = _nemotron_h_config(model_or_config)
    layer_types = getattr(config, "layers_block_type", None)
    if layer_types is not None:
        return tuple(str(layer_type) for layer_type in layer_types)
    pattern = getattr(config, "hybrid_override_pattern", None)
    if pattern is None:
        return ()
    return parse_nemotron_h_hybrid_pattern(str(pattern))


def nemotron_h_attention_layers(model_or_config: Any) -> tuple[int, ...]:
    return tuple(layer_idx for layer_idx, layer_type in enumerate(nemotron_h_layer_types(model_or_config)) if layer_type == "attention")


def _hybrid_cache_nbytes(value: Any | None) -> int:
    if value is None:
        return 0
    if hasattr(value, "element_size") and hasattr(value, "numel"):
        return int(value.element_size() * value.numel())
    return 0


def _nemotron_h_cache_layer(cache: Any, layer_id: int) -> Any | None:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list | tuple) and layer_id < len(layers):
        return layers[layer_id]
    return None


def _nemotron_h_cache_component_value(cache: Any, attr_name: str, layer_id: int) -> Any | None:
    layer = _nemotron_h_cache_layer(cache, layer_id)
    if layer is not None and hasattr(layer, attr_name):
        return getattr(layer, attr_name)
    values = getattr(cache, attr_name, None)
    if not isinstance(values, list | tuple) or layer_id >= len(values):
        return None
    return values[layer_id]


def _first_present(*values: Any | None) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


@dataclass(slots=True)
class NemotronHLayerStateSlice:
    layer_id: int
    layer_type: str
    state_growth_family: str
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
        return int(self.key_cache_bytes + self.value_cache_bytes + self.conv_state_bytes + self.recurrent_state_bytes)

    @property
    def fixed_resident_state_bytes(self) -> int:
        return int(self.layer_state_bytes) if self.state_growth_family == "fixed_resident" else 0

    @property
    def token_growing_state_bytes(self) -> int:
        return int(self.layer_state_bytes) if self.state_growth_family == "token_growing" else 0

    def summary_record(self) -> dict[str, Any]:
        return {
            "layer_id": int(self.layer_id),
            "layer_type": str(self.layer_type),
            "state_growth_family": str(self.state_growth_family),
            "key_cache_bytes": int(self.key_cache_bytes),
            "value_cache_bytes": int(self.value_cache_bytes),
            "conv_state_bytes": int(self.conv_state_bytes),
            "recurrent_state_bytes": int(self.recurrent_state_bytes),
            "layer_state_bytes": int(self.layer_state_bytes),
            "fixed_resident_state_bytes": int(self.fixed_resident_state_bytes),
            "token_growing_state_bytes": int(self.token_growing_state_bytes),
        }


@dataclass(slots=True)
class NemotronHHybridStatePartition:
    fixed_resident_layers: tuple[NemotronHLayerStateSlice, ...]
    token_growing_layers: tuple[NemotronHLayerStateSlice, ...]
    no_cache_layers: tuple[NemotronHLayerStateSlice, ...]

    @property
    def all_layers(self) -> tuple[NemotronHLayerStateSlice, ...]:
        return tuple(
            sorted(self.fixed_resident_layers + self.token_growing_layers + self.no_cache_layers, key=lambda record: record.layer_id)
        )

    @property
    def fixed_resident_layer_ids(self) -> list[int]:
        return [int(record.layer_id) for record in self.fixed_resident_layers]

    @property
    def token_growing_layer_ids(self) -> list[int]:
        return [int(record.layer_id) for record in self.token_growing_layers]

    @property
    def no_cache_layer_ids(self) -> list[int]:
        return [int(record.layer_id) for record in self.no_cache_layers]

    def to_summary(self) -> dict[str, Any]:
        all_layers = self.all_layers
        attention_kv_bytes = sum(record.key_cache_bytes + record.value_cache_bytes for record in all_layers)
        mamba_conv_bytes = sum(record.conv_state_bytes for record in all_layers)
        mamba_recurrent_bytes = sum(record.recurrent_state_bytes for record in all_layers)
        fixed_resident_bytes = sum(record.fixed_resident_state_bytes for record in self.fixed_resident_layers)
        token_growing_bytes = sum(record.token_growing_state_bytes for record in self.token_growing_layers)
        return {
            "hybrid_state_total_bytes": int(attention_kv_bytes + mamba_conv_bytes + mamba_recurrent_bytes),
            "hybrid_attention_kv_bytes": int(attention_kv_bytes),
            "hybrid_mamba_conv_state_bytes": int(mamba_conv_bytes),
            "hybrid_mamba_recurrent_state_bytes": int(mamba_recurrent_bytes),
            "hybrid_fixed_resident_bytes": int(fixed_resident_bytes),
            "hybrid_token_growing_bytes": int(token_growing_bytes),
            "hybrid_fixed_resident_layer_ids": self.fixed_resident_layer_ids,
            "hybrid_token_growing_layer_ids": self.token_growing_layer_ids,
            "hybrid_no_cache_layer_ids": self.no_cache_layer_ids,
            "hybrid_state_layers": [record.summary_record() for record in all_layers],
        }


def _nemotron_h_layer_records(model_or_config: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer_id, layer_type in enumerate(nemotron_h_layer_types(model_or_config)):
        growth_family = (
            "token_growing"
            if layer_type == "attention"
            else "fixed_resident"
            if layer_type == "mamba"
            else "no_cache"
        )
        records.append(
            {
                "layer_id": int(layer_id),
                "layer_type": str(layer_type),
                "dotcache_candidate": bool(layer_type == "attention"),
                "requires_hybrid_state": bool(layer_type == "mamba"),
                "state_growth_family": str(growth_family),
            }
        )
    return records


def summarize_nemotron_h_dotcache_fit(model_or_config: Any) -> dict[str, Any]:
    layer_records = _nemotron_h_layer_records(model_or_config)
    attention_candidate_layers = [record["layer_id"] for record in layer_records if record["dotcache_candidate"]]
    hybrid_only_layers = [record["layer_id"] for record in layer_records if record["requires_hybrid_state"]]
    return {
        "attention_candidate_layer_ids": attention_candidate_layers,
        "attention_candidate_layer_count": len(attention_candidate_layers),
        "hybrid_only_layer_ids": hybrid_only_layers,
        "hybrid_only_layer_count": len(hybrid_only_layers),
        "requires_hybrid_state_abstraction": bool(hybrid_only_layers),
        "suggested_next_step": "attention_subset_plus_mamba_state",
    }


def partition_nemotron_h_hybrid_state(cache: Any, model_or_config: Any) -> NemotronHHybridStatePartition:
    fixed_resident_layers: list[NemotronHLayerStateSlice] = []
    token_growing_layers: list[NemotronHLayerStateSlice] = []
    no_cache_layers: list[NemotronHLayerStateSlice] = []
    for layer_record in _nemotron_h_layer_records(model_or_config):
        layer_id = int(layer_record["layer_id"])
        layer_type = str(layer_record["layer_type"])
        state_slice = NemotronHLayerStateSlice(
            layer_id=layer_id,
            layer_type=layer_type,
            state_growth_family=str(layer_record["state_growth_family"]),
            key_cache=_first_present(
                _nemotron_h_cache_component_value(cache, "keys", layer_id),
                _nemotron_h_cache_component_value(cache, "key_cache", layer_id),
            ),
            value_cache=_first_present(
                _nemotron_h_cache_component_value(cache, "values", layer_id),
                _nemotron_h_cache_component_value(cache, "value_cache", layer_id),
            ),
            conv_state=_first_present(
                _nemotron_h_cache_component_value(cache, "conv_states", layer_id),
                _nemotron_h_cache_component_value(cache, "conv_state", layer_id),
            ),
            recurrent_state=_first_present(
                _nemotron_h_cache_component_value(cache, "recurrent_states", layer_id),
                _nemotron_h_cache_component_value(cache, "recurrent_state", layer_id),
            ),
        )
        if state_slice.state_growth_family == "fixed_resident":
            fixed_resident_layers.append(state_slice)
        elif state_slice.state_growth_family == "token_growing":
            token_growing_layers.append(state_slice)
        else:
            no_cache_layers.append(state_slice)
    return NemotronHHybridStatePartition(
        fixed_resident_layers=tuple(fixed_resident_layers),
        token_growing_layers=tuple(token_growing_layers),
        no_cache_layers=tuple(no_cache_layers),
    )


def summarize_nemotron_h_hybrid_state(cache: Any, model_or_config: Any) -> dict[str, Any]:
    return partition_nemotron_h_hybrid_state(cache, model_or_config).to_summary()


@dataclass(slots=True)
class NemotronHNativeHybridRuntimeState:
    model_or_config: Any
    past_key_values: Any
    prefill_partition: NemotronHHybridStatePartition
    current_partition: NemotronHHybridStatePartition

    @classmethod
    def from_post_handoff_cache(cls, past_key_values: Any, model_or_config: Any) -> "NemotronHNativeHybridRuntimeState":
        partition = partition_nemotron_h_hybrid_state(past_key_values, model_or_config)
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

    @property
    def no_cache_layer_ids(self) -> list[int]:
        return self.current_partition.no_cache_layer_ids

    def refresh(self, past_key_values: Any) -> None:
        self.past_key_values = past_key_values
        self.current_partition = partition_nemotron_h_hybrid_state(past_key_values, self.model_or_config)

    def prefill_summary(self) -> dict[str, Any]:
        return self.prefill_partition.to_summary()

    def current_summary(self) -> dict[str, Any]:
        return self.current_partition.to_summary()

    def summary(self) -> dict[str, Any]:
        prefill_summary = self.prefill_summary()
        final_summary = self.current_summary()
        result = {
            "hybrid_state_partition_ready": True,
            "native_hybrid_fixed_resident_layer_ids": self.fixed_resident_layer_ids,
            "native_hybrid_token_growing_layer_ids": self.token_growing_layer_ids,
            "native_hybrid_no_cache_layer_ids": self.no_cache_layer_ids,
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
class NemotronHHybridDotCacheRuntimeState:
    native_state: NemotronHNativeHybridRuntimeState
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
                "hybrid_runtime_state_kind": "nemotron_h_attention_subset",
                "hybrid_runtime_token_growing_layer_ids": self.native_state.token_growing_layer_ids,
                "hybrid_runtime_fixed_resident_layer_ids": self.native_state.fixed_resident_layer_ids,
                "hybrid_runtime_no_cache_layer_ids": self.native_state.no_cache_layer_ids,
            }
        )
        result.update(self.model_kv_cache.resident_byte_summary())
        result.update(self.model_kv_cache.page_mode_summary())
        result.update(self.model_kv_cache.execution_shortlist_summary())
        return result


def _nemotron_h_attention_head_dim(model_or_config: Any) -> int:
    config = _nemotron_h_config(model_or_config)
    return int(getattr(config, "head_dim", getattr(config, "attention_head_dim", 0)) or 0)


def _nemotron_h_model_device(model: Any):
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            pass
    if torch is not None:
        return torch.device("cpu")
    return "cpu"


@dataclass(slots=True)
class NemotronHTextModelAdapter:
    model: Any
    dotcache_config: DotCacheConfig = field(
        default_factory=lambda: DotCacheConfig(head_dim=128, group_size=32, bits_k=4, bits_v=4, tokens_per_page=16)
    )
    backend: str = "cpu_ref"
    cache: PreparedPageCache = field(default_factory=PreparedPageCache)
    model_kv_cache: ModelPagedKVCache = field(init=False, repr=False)
    native_hybrid_runtime_state: NemotronHNativeHybridRuntimeState | None = field(default=None, init=False, repr=False)
    hybrid_dotcache_runtime_state: NemotronHHybridDotCacheRuntimeState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        expected_head_dim = _nemotron_h_attention_head_dim(self.model)
        if expected_head_dim > 0 and self.dotcache_config.head_dim != expected_head_dim:
            self.dotcache_config = replace(self.dotcache_config, head_dim=expected_head_dim)
        self._rebuild_model_kv_cache()

    @property
    def device(self):
        return _nemotron_h_model_device(self.model)

    def _rebuild_model_kv_cache(self) -> None:
        config = _nemotron_h_config(self.model)
        self.model_kv_cache = ModelPagedKVCache(
            config=self.dotcache_config,
            num_hidden_layers=int(getattr(config, "num_hidden_layers", len(nemotron_h_layer_types(config)))),
            num_attention_heads=int(getattr(config, "num_attention_heads", 0) or 0),
            num_key_value_heads=int(getattr(config, "num_key_value_heads", 0) or 0),
            backend=self.backend,
            cache=self.cache,
        )

    def hybrid_block_summary(self) -> dict[str, Any]:
        return nemotron_h_block_summary(self.model)

    def hybrid_layer_summary(self) -> list[dict[str, Any]]:
        return _nemotron_h_layer_records(self.model)

    def hybrid_fit_summary(self) -> dict[str, Any]:
        return summarize_nemotron_h_dotcache_fit(self.model)

    def attention_subset_layer_ids(self) -> list[int]:
        return self.hybrid_fit_summary()["attention_candidate_layer_ids"]

    def fixed_resident_layer_ids(self) -> list[int]:
        if self.native_hybrid_runtime_state is not None:
            return list(self.native_hybrid_runtime_state.fixed_resident_layer_ids)
        return [record["layer_id"] for record in self.hybrid_layer_summary() if record["state_growth_family"] == "fixed_resident"]

    def token_growing_layer_ids(self) -> list[int]:
        if self.native_hybrid_runtime_state is not None:
            return list(self.native_hybrid_runtime_state.token_growing_layer_ids)
        return self.attention_subset_layer_ids()

    def no_cache_layer_ids(self) -> list[int]:
        if self.native_hybrid_runtime_state is not None:
            return list(self.native_hybrid_runtime_state.no_cache_layer_ids)
        return [record["layer_id"] for record in self.hybrid_layer_summary() if record["state_growth_family"] == "no_cache"]

    def partition_hybrid_state(self, cache: Any) -> NemotronHHybridStatePartition:
        return partition_nemotron_h_hybrid_state(cache, self.model)

    def clear(self) -> None:
        self.model_kv_cache.clear()
        self.native_hybrid_runtime_state = None
        self.hybrid_dotcache_runtime_state = None

    def refresh_native_hybrid_runtime_state(self, past_key_values: Any) -> None:
        if self.hybrid_dotcache_runtime_state is not None:
            self.hybrid_dotcache_runtime_state.refresh_native(past_key_values)
            self.native_hybrid_runtime_state = self.hybrid_dotcache_runtime_state.native_state
            return
        self.native_hybrid_runtime_state = NemotronHNativeHybridRuntimeState.from_post_handoff_cache(
            past_key_values,
            self.model,
        )
        self.hybrid_dotcache_runtime_state = NemotronHHybridDotCacheRuntimeState(
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

    def require_hybrid_dotcache_runtime_state(self) -> NemotronHHybridDotCacheRuntimeState:
        if self.hybrid_dotcache_runtime_state is None:
            raise ValueError("Nemotron-H DotCache runtime state is not initialized")
        return self.hybrid_dotcache_runtime_state


def nemotron_h_block_summary(model_or_config: Any) -> dict[str, Any]:
    config = _nemotron_h_config(model_or_config)
    layer_types = nemotron_h_layer_types(config)
    return {
        "hybrid_family": "nemotron_h",
        "model_type": str(getattr(config, "model_type", "")),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", len(layer_types)) or len(layer_types)),
        "hybrid_layer_count": len(layer_types),
        "hybrid_layer_types": list(layer_types),
        "hybrid_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "attention"),
        "hybrid_mamba_layer_count": sum(1 for layer_type in layer_types if layer_type == "mamba"),
        "hybrid_mlp_layer_count": sum(1 for layer_type in layer_types if layer_type == "mlp"),
        "hybrid_other_layer_type_count": sum(
            1 for layer_type in layer_types if layer_type not in {"attention", "mamba", "mlp"}
        ),
        "attention_layers": list(nemotron_h_attention_layers(config)),
        "head_dim": int(getattr(config, "head_dim", getattr(config, "attention_head_dim", 0)) or 0),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0) or 0),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", 0) or 0),
        "mamba_num_heads": int(getattr(config, "mamba_num_heads", 0) or 0),
        "mamba_head_dim": int(getattr(config, "mamba_head_dim", 0) or 0),
        "ssm_state_size": int(getattr(config, "ssm_state_size", 0) or 0),
        "chunk_size": int(getattr(config, "chunk_size", 0) or 0),
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0) or 0),
    }


def nemotron_h_environment_summary() -> dict[str, Any]:
    _require_transformers()
    import torch
    import transformers

    return {
        "transformers_version": str(transformers.__version__),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "mamba_ssm_installed": bool(importlib.util.find_spec("mamba_ssm") is not None),
        "nemotron_h_config_available": bool(hasattr(transformers, "NemotronHConfig")),
        "nemotron_h_model_available": bool(hasattr(transformers, "NemotronHForCausalLM")),
    }


def load_nemotron_h_remote_config(model_id: str) -> Any:
    _require_transformers()
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True, **resolve_hf_auth_kwargs())


def inspect_nemotron_h_native_config_compatibility(model_id: str) -> dict[str, Any]:
    _require_transformers()
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    except Exception as exc:
        return {
            "native_autoconfig_ok": False,
            "native_autoconfig_error_type": type(exc).__name__,
            "native_autoconfig_error_message": str(exc),
        }
    return {
        "native_autoconfig_ok": True,
        "native_autoconfig_class": type(config).__name__,
    }


__all__ = [
    "inspect_nemotron_h_native_config_compatibility",
    "load_nemotron_h_remote_config",
    "NemotronHHybridStatePartition",
    "NemotronHLayerStateSlice",
    "NemotronHTextModelAdapter",
    "nemotron_h_attention_layers",
    "nemotron_h_block_summary",
    "nemotron_h_environment_summary",
    "nemotron_h_layer_types",
    "partition_nemotron_h_hybrid_state",
    "parse_nemotron_h_hybrid_pattern",
    "summarize_nemotron_h_dotcache_fit",
    "summarize_nemotron_h_hybrid_state",
]
