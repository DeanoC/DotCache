use super::backend_buffer_api;
use super::backend_buffer_api::Qwen35BackendBufferApi;
use super::decoder::{DecoderLayer, LayerKind, ModelForCausalLM, TextModel};
use super::frontend::{max_abs_delta, profile_elapsed, profile_start};
use super::types::{CacheState, LayerCacheState, RuntimeProfile, StateBuffer};
use crate::PreparedQwen35DirectMetadata;
use candle::Result;
use candle_core as candle;
use candle_core::DType;

struct DirectDecodePhaseContext {
    backend: &'static dyn Qwen35BackendBufferApi,
    output_dtype: DType,
}

impl DirectDecodePhaseContext {
    fn from_hidden_state(xs: &StateBuffer) -> Result<Self> {
        let backend = backend_buffer_api::for_device(xs.device());
        let output_dtype = xs.dtype();
        let _ = xs.dims3()?;
        Ok(Self {
            backend,
            output_dtype,
        })
    }
}

fn trace_direct_decode_layer_deltas_enabled() -> bool {
    std::env::var("DOTCACHE_QWEN35_TRACE_DIRECT_DECODE_LAYERS")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn trace_direct_prefill_enabled() -> bool {
    std::env::var("DOTCACHE_QWEN35_TRACE_DIRECT_PREFILL")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn trace_direct_finalize_enabled() -> bool {
    std::env::var("DOTCACHE_QWEN35_TRACE_DIRECT_FINALIZE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn trace_direct_prefill_cache_enabled() -> bool {
    std::env::var("DOTCACHE_QWEN35_TRACE_DIRECT_PREFILL_CACHE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn trace_direct_decode_layer_delta(
    layer_idx: usize,
    layer_type: &str,
    reference: &StateBuffer,
    direct: &StateBuffer,
) -> Result<()> {
    if !trace_direct_decode_layer_deltas_enabled() {
        return Ok(());
    }
    let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
    eprintln!(
        "direct-decode-layer-delta layer={} type={} max_abs_delta={:.6}",
        layer_idx, layer_type, delta
    );
    Ok(())
}

fn trace_direct_prefill_delta(reference: &StateBuffer, direct: &StateBuffer) -> Result<()> {
    if !trace_direct_prefill_enabled() {
        return Ok(());
    }
    let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
    eprintln!("direct-prefill-logits-delta max_abs_delta={:.6}", delta);
    Ok(())
}

fn trace_direct_finalize_deltas(
    reference_hidden: &StateBuffer,
    direct_hidden: &StateBuffer,
    reference_logits: &StateBuffer,
    direct_logits: &StateBuffer,
) -> Result<()> {
    if !trace_direct_finalize_enabled() {
        return Ok(());
    }
    let hidden_delta = max_abs_delta(reference_hidden.tensor(), direct_hidden.tensor())?;
    let logits_delta = max_abs_delta(reference_logits.tensor(), direct_logits.tensor())?;
    eprintln!(
        "direct-finalize-delta hidden_max_abs_delta={:.6} logits_max_abs_delta={:.6}",
        hidden_delta, logits_delta
    );
    Ok(())
}

fn trace_direct_prefill_cache_delta(reference: &CacheState, direct: &CacheState) -> Result<()> {
    if !trace_direct_prefill_cache_enabled() {
        return Ok(());
    }
    for (layer_idx, (reference_layer, direct_layer)) in
        reference.layers.iter().zip(direct.layers.iter()).enumerate()
    {
        match (reference_layer, direct_layer) {
            (LayerCacheState::Linear(reference), LayerCacheState::Linear(direct)) => {
                match (&reference.conv_state, &direct.conv_state) {
                    (Some(reference), Some(direct)) => {
                        let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=linear conv_state_max_abs_delta={:.6}",
                            layer_idx, delta
                        );
                    }
                    (None, None) => {}
                    _ => {
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=linear conv_state_presence_mismatch",
                            layer_idx
                        );
                    }
                }
                match (&reference.recurrent_state, &direct.recurrent_state) {
                    (Some(reference), Some(direct)) => {
                        let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=linear recurrent_state_max_abs_delta={:.6}",
                            layer_idx, delta
                        );
                    }
                    (None, None) => {}
                    _ => {
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=linear recurrent_state_presence_mismatch",
                            layer_idx
                        );
                    }
                }
            }
            (LayerCacheState::Full(reference), LayerCacheState::Full(direct)) => {
                match (&reference.kv_cache, &direct.kv_cache) {
                    (Some((reference_k, reference_v)), Some((direct_k, direct_v))) => {
                        let key_delta = max_abs_delta(reference_k.tensor(), direct_k.tensor())?;
                        let value_delta = max_abs_delta(reference_v.tensor(), direct_v.tensor())?;
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=full key_max_abs_delta={:.6} value_max_abs_delta={:.6}",
                            layer_idx, key_delta, value_delta
                        );
                    }
                    (None, None) => {}
                    _ => {
                        eprintln!(
                            "direct-prefill-cache-delta layer={} kind=full kv_presence_mismatch",
                            layer_idx
                        );
                    }
                }
            }
            _ => {
                eprintln!(
                    "direct-prefill-cache-delta layer={} layer_kind_mismatch",
                    layer_idx
                );
            }
        }
    }
    Ok(())
}

fn execute_linear_decode_layer(
    layer_idx: usize,
    layer: &mut DecoderLayer,
    xs: &StateBuffer,
    phase_context: &DirectDecodePhaseContext,
    seqlen_offset: usize,
    phase_output_scratch: &StateBuffer,
    linear_mixed_qkv: &StateBuffer,
    linear_z: &StateBuffer,
    linear_beta_raw: &StateBuffer,
    linear_a: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = xs.device();
    let mut profile = RuntimeProfile::default();
    let reference_cache_state = if trace_direct_decode_layer_deltas_enabled() {
        Some(layer.cache_state())
    } else {
        None
    };
    let residual = xs.clone();
    let layer_input = xs.clone();
    let xs_norm = layer.input_layernorm.forward_buffer(xs)?;
    let linear_attn = match &mut layer.token_mixer {
        LayerKind::Linear(linear_attn) => linear_attn,
        LayerKind::Full(_) => {
            candle::bail!("direct-hip-v1 linear decode expected linear-attention layer")
        }
    };
    let (mixed_qkv, z, beta_raw, a, projection_profile) =
        linear_attn.project_direct_decode_inputs_into_scratch(
            &xs_norm,
            linear_mixed_qkv,
            linear_z,
            linear_beta_raw,
            linear_a,
        )?;
    profile.add_assign(&projection_profile);
    let (xs, recurrent_state, linear_profile) = linear_attn.run_direct_decode_core(
        phase_context.output_dtype,
        &xs_norm,
        &mixed_qkv,
        &z,
        &beta_raw,
        &a,
    )?;
    linear_attn.commit_direct_decode_recurrent_state(recurrent_state);
    profile.add_assign(&linear_profile);
    let xs = phase_context.backend.add(&residual, &xs)?;
    let residual = xs.clone();
    let xs = layer.post_attention_layernorm.forward_buffer(&xs)?;
    let mlp_start = profile_start(device)?;
    let xs = layer.mlp.forward_buffer(&xs)?;
    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
    let xs = phase_context.backend.add(&residual, &xs)?;
    let xs = phase_context
        .backend
        .copy_state_into_scratch(&xs, phase_output_scratch)?;
    if let Some(cache_state) = reference_cache_state {
        let direct_cache_state = layer.cache_state();
        layer.restore_cache_state(&cache_state)?;
        let (reference, _) = layer.forward_profiled(&layer_input, None, seqlen_offset)?;
        layer.restore_cache_state(&direct_cache_state)?;
        trace_direct_decode_layer_delta(layer_idx, "linear_attention", &reference, &xs)?;
    }
    Ok((xs, profile))
}

fn execute_full_decode_layer(
    layer_idx: usize,
    layer: &mut DecoderLayer,
    xs: &StateBuffer,
    phase_context: &DirectDecodePhaseContext,
    seqlen_offset: usize,
    full_attention_gate: &StateBuffer,
    full_attention_qkv: &StateBuffer,
    full_attention_key: &StateBuffer,
    full_attention_value: &StateBuffer,
    full_attention_output_scratch: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = xs.device();
    let mut profile = RuntimeProfile::default();
    let reference_cache_state = if trace_direct_decode_layer_deltas_enabled() {
        Some(layer.cache_state())
    } else {
        None
    };
    let residual = xs.clone();
    let layer_input = xs.clone();
    let xs_norm = layer.input_layernorm.forward_buffer(xs)?;
    let self_attn = match &mut layer.token_mixer {
        LayerKind::Full(self_attn) => self_attn,
        LayerKind::Linear(_) => {
            candle::bail!("direct-hip-v1 full decode expected full-attention layer")
        }
    };
    let context = self_attn.direct_decode_context(
        &xs_norm,
        seqlen_offset,
        full_attention_gate,
        full_attention_qkv,
        full_attention_key,
        full_attention_value,
    )?;
    let (
        query_states,
        key_states,
        value_states,
        gate,
        appended_k,
        appended_v,
        input_profile,
    ) = self_attn.project_direct_decode_inputs_with_context(&xs_norm, &context)?;
    profile.add_assign(&input_profile);
    let attn_output = self_attn.run_direct_decode_core_with_context(
        &context,
        &query_states,
        &key_states,
        &value_states,
        &gate,
    )?;
    let xs = self_attn.project_direct_decode_output(&attn_output)?;
    let xs = phase_context
        .backend
        .copy_state_into_scratch(&xs, full_attention_output_scratch)?;
    drop(context);
    self_attn.commit_direct_decode_kv_cache(appended_k, appended_v);
    let xs = phase_context.backend.add(&residual, &xs)?;
    let residual = xs.clone();
    let xs = layer.post_attention_layernorm.forward_buffer(&xs)?;
    let mlp_start = profile_start(device)?;
    let xs = layer.mlp.forward_buffer(&xs)?;
    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
    let xs = phase_context.backend.add(&residual, &xs)?;
    let xs = phase_context
        .backend
        .copy_state_into_scratch(&xs, full_attention_output_scratch)?;
    if let Some(cache_state) = reference_cache_state {
        let direct_cache_state = layer.cache_state();
        layer.restore_cache_state(&cache_state)?;
        let (reference, _) = layer.forward_profiled(&layer_input, None, seqlen_offset)?;
        layer.restore_cache_state(&direct_cache_state)?;
        trace_direct_decode_layer_delta(layer_idx, "full_attention", &reference, &xs)?;
    }
    Ok((xs, profile))
}

fn execute_direct_decode_linear_phase_unchecked(
    model: &mut TextModel,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    phase_output_scratch: &StateBuffer,
    linear_mixed_qkv: &StateBuffer,
    linear_z: &StateBuffer,
    linear_beta_raw: &StateBuffer,
    linear_a: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let mut profile = RuntimeProfile::default();
    let mut xs = xs.clone();
    let phase_context = DirectDecodePhaseContext::from_hidden_state(&xs)?;
    let layer_count = model.layers.len();
    for layer_idx in start_layer_idx..end_layer_idx {
        let layer = model.layers.get_mut(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 linear_attention decode layer index {} out of range for {} layers",
                layer_idx,
                layer_count
            ))
        })?;
        if layer.layer_type() != "linear_attention" {
            candle::bail!(
                "direct-hip-v1 linear_attention decode layer type mismatch at layer {}: model={}",
                layer_idx,
                layer.layer_type(),
            );
        }
        let (next_xs, layer_profile) =
            execute_linear_decode_layer(
                layer_idx,
                layer,
                &xs,
                &phase_context,
                0,
                phase_output_scratch,
                linear_mixed_qkv,
                linear_z,
                linear_beta_raw,
                linear_a,
            )?;
        profile.add_assign(&layer_profile);
        xs = next_xs;
    }
    Ok((xs, profile))
}

fn execute_direct_decode_full_phase_unchecked(
    model: &mut TextModel,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
    full_attention_gate: &StateBuffer,
    full_attention_qkv: &StateBuffer,
    full_attention_key: &StateBuffer,
    full_attention_value: &StateBuffer,
    full_attention_output_scratch: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let mut profile = RuntimeProfile::default();
    let mut xs = xs.clone();
    let phase_context = DirectDecodePhaseContext::from_hidden_state(&xs)?;
    let layer_count = model.layers.len();
    for layer_idx in start_layer_idx..end_layer_idx {
        let layer = model.layers.get_mut(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 full_attention decode layer index {} out of range for {} layers",
                layer_idx,
                layer_count
            ))
        })?;
        if layer.layer_type() != "full_attention" {
            candle::bail!(
                "direct-hip-v1 full_attention decode layer type mismatch at layer {}: model={}",
                layer_idx,
                layer.layer_type(),
            );
        }
        let (next_xs, layer_profile) =
            execute_full_decode_layer(
                layer_idx,
                layer,
                &xs,
                &phase_context,
                seqlen_offset,
                full_attention_gate,
                full_attention_qkv,
                full_attention_key,
                full_attention_value,
                full_attention_output_scratch,
            )?;
        profile.add_assign(&layer_profile);
        xs = next_xs;
    }
    Ok((xs, profile))
}

pub(super) fn text_model_forward_hidden_states_profiled_direct_hip_v1(
    model: &mut TextModel,
    metadata: &PreparedQwen35DirectMetadata,
    hidden_states: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    validate_text_model_direct_hip_metadata(model, metadata)?;
    let device = hidden_states.device();
    let (b_size, seq_len, _) = hidden_states.dims3()?;
    let mut profile = RuntimeProfile::default();
    let scheduler_start = super::frontend::profile_start(device)?;
    let attention_mask = if seq_len > 1 {
        Some(model.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
    } else {
        None
    };
    profile.scheduler_planning_millis += super::frontend::profile_elapsed(scheduler_start, device)?;
    let mut xs = hidden_states.clone();
    for (layer_idx, (layer, layer_meta)) in model.layers.iter_mut().zip(metadata.layers.iter()).enumerate() {
        if layer_meta.layer_idx != layer_idx {
            candle::bail!(
                "direct-hip-v1 metadata index mismatch at layer {}: got {}",
                layer_idx,
                layer_meta.layer_idx
            );
        }
        if layer.layer_type() != layer_meta.layer_type {
            candle::bail!(
                "direct-hip-v1 layer type mismatch at layer {}: model={} metadata={}",
                layer_idx,
                layer.layer_type(),
                layer_meta.layer_type
            );
        }
        let mask = if layer.layer_type() == "full_attention" {
            attention_mask.as_ref()
        } else {
            None
        };
        let (next_xs, layer_profile) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
        profile.add_assign(&layer_profile);
        xs = next_xs;
    }
    Ok((model.norm.forward_buffer(&xs)?, profile))
}

pub(super) fn validate_text_model_direct_hip_metadata(
    model: &TextModel,
    metadata: &PreparedQwen35DirectMetadata,
) -> Result<()> {
    if model.layers.len() != metadata.num_hidden_layers {
        candle::bail!(
            "direct-hip-v1 decode layer count mismatch: model={} metadata={}",
            model.layers.len(),
            metadata.num_hidden_layers
        );
    }
    if metadata.layers.len() != model.layers.len() {
        candle::bail!(
            "direct-hip-v1 decode metadata layer schedule mismatch: metadata={} model={}",
            metadata.layers.len(),
            model.layers.len()
        );
    }
    Ok(())
}

pub(super) fn text_model_direct_decode_linear_phase_profiled_hip_v1_unchecked(
    model: &mut TextModel,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    phase_output_scratch: &StateBuffer,
    linear_mixed_qkv: &StateBuffer,
    linear_z: &StateBuffer,
    linear_beta_raw: &StateBuffer,
    linear_a: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    execute_direct_decode_linear_phase_unchecked(
        model,
        start_layer_idx,
        end_layer_idx,
        xs,
        phase_output_scratch,
        linear_mixed_qkv,
        linear_z,
        linear_beta_raw,
        linear_a,
    )
}

pub(super) fn text_model_direct_decode_full_phase_profiled_hip_v1_unchecked(
    model: &mut TextModel,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
    full_attention_gate: &StateBuffer,
    full_attention_qkv: &StateBuffer,
    full_attention_key: &StateBuffer,
    full_attention_value: &StateBuffer,
    full_attention_output_scratch: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    execute_direct_decode_full_phase_unchecked(
        model,
        start_layer_idx,
        end_layer_idx,
        xs,
        seqlen_offset,
        full_attention_gate,
        full_attention_qkv,
        full_attention_key,
        full_attention_value,
        full_attention_output_scratch,
    )
}

pub(super) fn text_model_finalize_direct_decode_hidden_hip_v1(
    model: &mut TextModel,
    xs: &StateBuffer,
) -> Result<StateBuffer> {
    model.norm.forward_buffer(xs)
}

pub(super) fn model_forward_hidden_states_profiled_direct_hip_v1(
    model: &mut ModelForCausalLM,
    metadata: &PreparedQwen35DirectMetadata,
    hidden_states: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let reference_cache_state = if trace_direct_prefill_enabled() || trace_direct_prefill_cache_enabled() {
        Some(model.cache_state())
    } else {
        None
    };
    let device = hidden_states.device();
    let backend = backend_buffer_api::for_device(device);
    let (direct_hidden_states, mut profile) = text_model_forward_hidden_states_profiled_direct_hip_v1(
        &mut model.language_model,
        metadata,
        hidden_states,
        seqlen_offset,
    )?;
    let output_start = super::frontend::profile_start(device)?;
    let logits = backend.slice_last_token(&direct_hidden_states)?;
    let logits = model.lm_head.forward_buffer(&logits)?;
    profile.output_projection_millis += super::frontend::profile_elapsed(output_start, device)?;
    if let Some(cache_state) = reference_cache_state {
        let direct_cache_state = model.cache_state();
        model.restore_cache_state(&cache_state)?;
        let (reference_logits, _) = model.forward_hidden_states_profiled(hidden_states, seqlen_offset)?;
        let reference_post_prefill_cache_state = model.cache_state();
        model.restore_cache_state(&direct_cache_state)?;
        trace_direct_prefill_delta(&reference_logits, &logits)?;
        trace_direct_prefill_cache_delta(&reference_post_prefill_cache_state, &direct_cache_state)?;
    }
    Ok((logits, profile))
}

pub(super) fn model_validate_direct_hip_metadata(
    model: &ModelForCausalLM,
    metadata: &PreparedQwen35DirectMetadata,
) -> Result<()> {
    validate_text_model_direct_hip_metadata(&model.language_model, metadata)
}

pub(super) fn model_direct_decode_linear_phase_profiled_hip_v1_unchecked(
    model: &mut ModelForCausalLM,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    phase_output_scratch: &StateBuffer,
    linear_mixed_qkv: &StateBuffer,
    linear_z: &StateBuffer,
    linear_beta_raw: &StateBuffer,
    linear_a: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    text_model_direct_decode_linear_phase_profiled_hip_v1_unchecked(
        &mut model.language_model,
        start_layer_idx,
        end_layer_idx,
        xs,
        phase_output_scratch,
        linear_mixed_qkv,
        linear_z,
        linear_beta_raw,
        linear_a,
    )
}

pub(super) fn model_direct_decode_full_phase_profiled_hip_v1_unchecked(
    model: &mut ModelForCausalLM,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
    full_attention_gate: &StateBuffer,
    full_attention_qkv: &StateBuffer,
    full_attention_key: &StateBuffer,
    full_attention_value: &StateBuffer,
    full_attention_output_scratch: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    text_model_direct_decode_full_phase_profiled_hip_v1_unchecked(
        &mut model.language_model,
        start_layer_idx,
        end_layer_idx,
        xs,
        seqlen_offset,
        full_attention_gate,
        full_attention_qkv,
        full_attention_key,
        full_attention_value,
        full_attention_output_scratch,
    )
}

pub(super) fn model_finalize_direct_decode_logits_hip_v1(
    model: &mut ModelForCausalLM,
    hidden_states: &StateBuffer,
    logits_scratch: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = hidden_states.device();
    let backend = backend_buffer_api::for_device(device);
    let output_start = super::frontend::profile_start(device)?;
    let direct_hidden_states =
        text_model_finalize_direct_decode_hidden_hip_v1(&mut model.language_model, hidden_states)?;
    let logits = backend.slice_last_token(&direct_hidden_states)?;
    let logits = model.lm_head.forward_buffer_into_scratch(&logits, logits_scratch)?;
    let mut profile = RuntimeProfile::default();
    profile.output_projection_millis += super::frontend::profile_elapsed(output_start, device)?;
    if trace_direct_finalize_enabled() {
        let reference_hidden = model.language_model.norm.forward_buffer(hidden_states)?;
        let reference_logits = backend.slice_last_token(&reference_hidden)?;
        let reference_logits = model.lm_head.forward_buffer(&reference_logits)?;
        trace_direct_finalize_deltas(
            &reference_hidden,
            &direct_hidden_states,
            &reference_logits,
            &logits,
        )?;
    }
    Ok((logits, profile))
}
