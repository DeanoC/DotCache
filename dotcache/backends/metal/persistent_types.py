from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PersistentServingConfig:
    block_size: int = 16
    dense_only: bool = True
    enable_full_attention_persistent_compute: bool = True
    enable_linear_attention_persistent_compute: bool = False
    enable_priority: bool = False
    enable_early_exit: bool = False
    enable_compression: bool = False
    fp32_accumulation: bool = True


@dataclass(slots=True)
class PersistentLayerTelemetry:
    decode_ms_total: float = 0.0
    append_ms_total: float = 0.0
    linear_ms_total: float = 0.0
    update_ms_total: float = 0.0
    mutation_count: int = 0


@dataclass(slots=True)
class PersistentStepTelemetry:
    backend_kind: str = "torch_exact_fallback"
    host_to_device_bytes_after_prefill: int = 0
    full_attention_step_ms_total: float = 0.0
    linear_attention_step_ms_total: float = 0.0
    append_update_ms_total: float = 0.0
    layer_telemetry: dict[int, PersistentLayerTelemetry] = field(default_factory=dict)

    def require_layer(self, layer_id: int) -> PersistentLayerTelemetry:
        if int(layer_id) not in self.layer_telemetry:
            self.layer_telemetry[int(layer_id)] = PersistentLayerTelemetry()
        return self.layer_telemetry[int(layer_id)]


@dataclass(slots=True)
class PersistentFullAttentionLayerState:
    layer_id: int
    key_cache: Any
    value_cache: Any
    block_token_starts: Any
    block_token_counts: Any
    metadata_valid: Any
    append_count: int = 0


@dataclass(slots=True)
class PersistentLinearAttentionLayerState:
    layer_id: int
    conv_state: Any | None
    recurrent_state: Any | None
    has_previous_state: bool = False
    sync_into_cache_count: int = 0
    sync_from_cache_count: int = 0
    direct_compute_count: int = 0
