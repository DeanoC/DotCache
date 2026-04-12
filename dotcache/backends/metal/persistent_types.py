from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PersistentServingConfig:
    block_size: int = 16
    dense_only: bool = True
    enable_full_attention_persistent_compute: bool = True
    enable_linear_attention_persistent_compute: bool = False
    enable_full_attention_mixed_mode_execution: bool = False
    full_attention_mixed_mode_execution_strategy: str = "cached_reconstruct"
    full_attention_mixed_mode_execution_allow_value_m0: bool = False
    full_attention_mixed_mode_execution_max_k_comp_error: float | None = 0.10
    full_attention_mixed_mode_score_dtype: str = "auto"
    full_attention_mixed_mode_detailed_timing: bool = False
    enable_priority: bool = False
    enable_early_exit: bool = False
    enable_compression: bool = False
    fp32_accumulation: bool = True
    full_attention_mode_cost_weight: float = 0.10
    full_attention_fallback_widen_step: int = 16
    full_attention_fallback_max_optional_top_k: int = 0
    full_attention_dense_fallback_on_invalid_compression_metadata: bool = True
    full_attention_shortlist_policy_path: str | None = None
    full_attention_shortlist_policy_mode: str = "bias"
    full_attention_shortlist_policy_bias_weight: float = 0.10
    full_attention_shortlist_policy_min_safe_rate: float = 1.0
    full_attention_shortlist_policy_min_matched_oracle_rate: float = 1.0
    full_attention_shortlist_policy_min_vote_count: int = 4
    full_attention_shortlist_policy_min_step_index: int | None = None
    full_attention_shortlist_policy_max_step_index: int | None = None
    full_attention_check_interval: int = 1
    full_attention_streaming_order_mode: str = "shortlist"
    full_attention_streaming_proxy_token_weight: float = 1.0
    full_attention_streaming_proxy_value_weight: float = 1.0
    full_attention_streaming_proxy_value_weight_by_layer: dict[int, float] | None = None
    full_attention_streaming_priority_value_upper_weight: float = 0.25
    full_attention_streaming_exact_value_rerank_layers: list[int] | None = None
    full_attention_streaming_exact_value_rerank_max_remaining_blocks: int | None = None
    full_attention_streaming_refine_top_k: int = 0
    full_attention_mass_eps: float = 1e-3
    full_attention_value_eps: float = 1e-3
    full_attention_min_processed_blocks: int = 1
    full_attention_bound_eps: float = 1e-3
    full_attention_region_residual_caps: bool = False
    full_attention_residual_cluster_count: int = 0
    full_attention_key_centroid_count: int = 1
    full_attention_key_centroid_count_by_layer: dict[int, int] | None = None
    full_attention_value_centroid_count: int = 1
    full_attention_value_centroid_count_by_layer: dict[int, int] | None = None
    full_attention_refine_top_k: int = 0
    full_attention_refine_top_k_by_layer: dict[int, int] | None = None
    full_attention_probe_refine_top_k: int = 0
    full_attention_probe_sample_count: int = 4
    full_attention_sink_block_count: int = 1
    full_attention_recent_block_count: int = 1
    full_attention_mandatory_recent_block_count: int | None = None
    full_attention_exploration_blocks_per_region: int = 1
    full_attention_optional_top_k: int = 0
    full_attention_optional_use_upper_bounds_first: bool = True
    full_attention_optional_upper_bound_quota: int = 0
    full_attention_optional_far_anchor_quota: int = 0
    full_attention_optional_far_anchor_priority_margin: float = 0.0
    full_attention_optional_far_anchor_upper_bound_margin: float = 0.0
    full_attention_optional_far_quota: int = 0
    full_attention_optional_mid_quota: int = 0
    full_attention_optional_near_quota: int = 0
    full_attention_optional_bootstrap_far_anchor_quota: int | None = None
    full_attention_optional_bootstrap_far_quota: int | None = None
    full_attention_optional_bootstrap_mid_quota: int | None = None
    full_attention_optional_bootstrap_near_quota: int | None = None
    full_attention_optional_diversity_weight: float = 0.0
    full_attention_optional_diversity_radius: int = 0
    full_attention_optional_diversity_strategy: str = "window_suppress"
    full_attention_optional_diversity_requires_history: bool = False
    full_attention_optional_diversity_min_history_count: int = 0
    full_attention_optional_diversity_max_history_count: int | None = None
    full_attention_priority_prev_attention_weight: float = 1.0
    full_attention_priority_recency_weight: float = 0.05
    full_attention_priority_recency_decay_blocks: float = 32.0
    full_attention_priority_value_norm_weight: float = 0.05


@dataclass(slots=True)
class PersistentLayerTelemetry:
    decode_ms_total: float = 0.0
    append_ms_total: float = 0.0
    linear_ms_total: float = 0.0
    update_ms_total: float = 0.0
    score_ms_total: float = 0.0
    selection_ms_total: float = 0.0
    optional_selection_ms_total: float = 0.0
    diverse_selection_ms_total: float = 0.0
    compression_selection_ms_total: float = 0.0
    policy_bias_ms_total: float = 0.0
    mixed_execution_prepare_ms_total: float = 0.0
    mixed_execution_cache_refresh_ms_total: float = 0.0
    mixed_execution_direct_m0_assembly_ms_total: float = 0.0
    mixed_execution_direct_m0_query_prep_ms_total: float = 0.0
    mixed_execution_direct_m0_gather_ms_total: float = 0.0
    mixed_execution_direct_m0_score_ms_total: float = 0.0
    mixed_execution_exact_m3_score_ms_total: float = 0.0
    mixed_execution_final_mix_ms_total: float = 0.0
    mixed_execution_final_mix_logits_ms_total: float = 0.0
    mixed_execution_final_mix_softmax_ms_total: float = 0.0
    mixed_execution_final_mix_value_ms_total: float = 0.0
    selected_m0_metadata_block_count_total: int = 0
    selected_m3_metadata_block_count_total: int = 0
    executed_m0_block_count_total: int = 0
    executed_m3_block_count_total: int = 0
    fallback_process_more_count: int = 0
    fallback_widen_count: int = 0
    fallback_disable_compression_count: int = 0
    fallback_disable_pruning_count: int = 0
    dense_fallback_count: int = 0
    compression_rerank_count: int = 0
    last_fallback_rung: int = 0
    mutation_count: int = 0


@dataclass(slots=True)
class PersistentStepTelemetry:
    backend_kind: str = "torch_exact_fallback"
    host_to_device_bytes_after_prefill: int = 0
    full_attention_step_ms_total: float = 0.0
    linear_attention_step_ms_total: float = 0.0
    append_update_ms_total: float = 0.0
    shortlist_policy_load_ms_total: float = 0.0
    shortlist_policy_resolve_ms_total: float = 0.0
    shortlist_policy_load_count: int = 0
    shortlist_policy_resolve_count: int = 0
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
    mixed_key_cache: Any | None
    mixed_value_cache: Any | None
    mixed_key_score_cache: Any | None
    mixed_key_fused_scaled_cache: Any | None
    mixed_key_bias_cache: Any | None
    mixed_key_fused_scaled_score_cache: Any | None
    mixed_key_bias_score_cache: Any | None
    mixed_key_fused_with_bias_score_cache: Any | None
    mixed_key_packed_payload_cache: Any | None
    mixed_key_packed_scales_cache: Any | None
    mixed_key_packed_bias_cache: Any | None
    mixed_value_fused_scaled_cache: Any | None
    mixed_value_bias_cache: Any | None
    block_token_starts: Any
    block_token_counts: Any
    block_k_center: Any
    block_k_radius: Any
    block_k_subcenters: Any
    block_k_subradii: Any
    block_v_center: Any
    block_v_radius: Any
    block_v_subcenters: Any
    block_v_subradii: Any
    block_v_sub_norm_max: Any
    block_v_subtoken_counts: Any
    block_v_norm_max: Any
    block_v_pos_sum: Any
    block_v_neg_sum: Any
    block_prev_attention_ema: Any
    block_region_ids: Any
    block_k_mode: Any
    block_v_mode: Any
    block_k_comp_error: Any
    block_compression_metadata_valid: Any
    metadata_valid: Any
    last_residual_certificate: dict[str, Any] | None = None
    last_first_certified_stop: dict[str, Any] | None = None
    last_checkpoint_count: int = 0
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
