use super::backend_buffer_api;
use super::decoder::{DecoderLayer, LayerKind, ModelForCausalLM, TextModel};
use super::frontend::{profile_elapsed, profile_start};
use super::types::{RuntimeProfile, StateBuffer};
use crate::PreparedQwen35DirectMetadata;
use candle::Result;
use candle_core as candle;

fn execute_linear_decode_layer(
    layer: &mut DecoderLayer,
    xs: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = xs.device();
    let backend = backend_buffer_api::for_device(device);
    let mut profile = RuntimeProfile::default();
    let residual = xs.clone();
    let xs_norm = layer.input_layernorm.forward_buffer(xs)?;
    let linear_attn = match &mut layer.token_mixer {
        LayerKind::Linear(linear_attn) => linear_attn,
        LayerKind::Full(_) => {
            candle::bail!("direct-hip-v1 linear decode expected linear-attention layer")
        }
    };
    let (mixed_qkv, z, beta_raw, a, projection_profile) =
        linear_attn.project_direct_decode_inputs(&xs_norm)?;
    profile.add_assign(&projection_profile);
    let (xs, recurrent_state, linear_profile) = linear_attn.run_direct_decode_core(
        xs.tensor().dtype(),
        &xs_norm,
        &mixed_qkv,
        &z,
        &beta_raw,
        &a,
    )?;
    linear_attn.commit_direct_decode_recurrent_state(recurrent_state);
    profile.add_assign(&linear_profile);
    let xs = backend.add(&residual, &xs)?;
    let residual = xs.clone();
    let xs = layer.post_attention_layernorm.forward_buffer(&xs)?;
    let mlp_start = profile_start(device)?;
    let xs = layer.mlp.forward_buffer(&xs)?;
    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
    Ok((backend.add(&residual, &xs)?, profile))
}

fn execute_full_decode_layer(
    layer: &mut DecoderLayer,
    layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = xs.device();
    let backend = backend_buffer_api::for_device(device);
    let mut profile = RuntimeProfile::default();
    let residual = xs.clone();
    let xs_norm = layer.input_layernorm.forward_buffer(xs)?;
    let self_attn = match &mut layer.token_mixer {
        LayerKind::Full(self_attn) => self_attn,
        LayerKind::Linear(_) => {
            candle::bail!("direct-hip-v1 full decode expected full-attention layer")
        }
    };
    let (xs, layer_profile) =
        self_attn.forward_profiled_direct_decode_v1(&xs_norm, seqlen_offset, layer_idx)?;
    profile.add_assign(&layer_profile);
    let xs = backend.add(&residual, &xs)?;
    let residual = xs.clone();
    let xs = layer.post_attention_layernorm.forward_buffer(&xs)?;
    let mlp_start = profile_start(device)?;
    let xs = layer.mlp.forward_buffer(&xs)?;
    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
    Ok((backend.add(&residual, &xs)?, profile))
}

fn execute_direct_decode_linear_phase(
    model: &mut TextModel,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    validate_text_model_direct_hip_metadata(model, metadata)?;
    let mut profile = RuntimeProfile::default();
    let mut xs = xs.clone();
    let layer_count = model.layers.len();
    for layer_idx in start_layer_idx..end_layer_idx {
        let layer_meta = metadata.layers.get(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 decode metadata missing linear_attention layer {}",
                layer_idx
            ))
        })?;
        if layer_meta.layer_idx != layer_idx {
            candle::bail!(
                "direct-hip-v1 decode metadata index mismatch at linear_attention layer {}: got {}",
                layer_idx,
                layer_meta.layer_idx
            );
        }
        if layer_meta.layer_type != "linear_attention" {
            candle::bail!(
                "direct-hip-v1 linear_attention decode phase expected linear_attention at layer {}, got {}",
                layer_idx,
                layer_meta.layer_type
            );
        }
        let layer = model.layers.get_mut(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 linear_attention decode layer index {} out of range for {} layers",
                layer_idx,
                layer_count
            ))
        })?;
        if layer.layer_type() != layer_meta.layer_type {
            candle::bail!(
                "direct-hip-v1 linear_attention decode layer type mismatch at layer {}: model={} metadata={}",
                layer_idx,
                layer.layer_type(),
                layer_meta.layer_type
            );
        }
        let (next_xs, layer_profile) = execute_linear_decode_layer(layer, &xs)?;
        profile.add_assign(&layer_profile);
        xs = next_xs;
    }
    Ok((xs, profile))
}

fn execute_direct_decode_full_phase(
    model: &mut TextModel,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    validate_text_model_direct_hip_metadata(model, metadata)?;
    let mut profile = RuntimeProfile::default();
    let mut xs = xs.clone();
    let layer_count = model.layers.len();
    for layer_idx in start_layer_idx..end_layer_idx {
        let layer_meta = metadata.layers.get(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 decode metadata missing full_attention layer {}",
                layer_idx
            ))
        })?;
        if layer_meta.layer_idx != layer_idx {
            candle::bail!(
                "direct-hip-v1 decode metadata index mismatch at full_attention layer {}: got {}",
                layer_idx,
                layer_meta.layer_idx
            );
        }
        if layer_meta.layer_type != "full_attention" {
            candle::bail!(
                "direct-hip-v1 full_attention decode phase expected full_attention at layer {}, got {}",
                layer_idx,
                layer_meta.layer_type
            );
        }
        let layer = model.layers.get_mut(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "direct-hip-v1 full_attention decode layer index {} out of range for {} layers",
                layer_idx,
                layer_count
            ))
        })?;
        if layer.layer_type() != layer_meta.layer_type {
            candle::bail!(
                "direct-hip-v1 full_attention decode layer type mismatch at layer {}: model={} metadata={}",
                layer_idx,
                layer.layer_type(),
                layer_meta.layer_type
            );
        }
        let (next_xs, layer_profile) =
            execute_full_decode_layer(layer, layer_idx, &xs, seqlen_offset)?;
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

pub(super) fn text_model_direct_decode_linear_phase_profiled_hip_v1(
    model: &mut TextModel,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let _ = seqlen_offset;
    execute_direct_decode_linear_phase(model, metadata, start_layer_idx, end_layer_idx, xs)
}

pub(super) fn text_model_direct_decode_full_phase_profiled_hip_v1(
    model: &mut TextModel,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    execute_direct_decode_full_phase(
        model,
        metadata,
        start_layer_idx,
        end_layer_idx,
        xs,
        seqlen_offset,
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
    let device = hidden_states.device();
    let backend = backend_buffer_api::for_device(device);
    let (hidden_states, mut profile) = text_model_forward_hidden_states_profiled_direct_hip_v1(
        &mut model.language_model,
        metadata,
        hidden_states,
        seqlen_offset,
    )?;
    let output_start = super::frontend::profile_start(device)?;
    let logits = backend.slice_last_token(&hidden_states)?;
    let logits = model.lm_head.forward_buffer(&logits)?;
    profile.output_projection_millis += super::frontend::profile_elapsed(output_start, device)?;
    Ok((logits, profile))
}

pub(super) fn model_validate_direct_hip_metadata(
    model: &ModelForCausalLM,
    metadata: &PreparedQwen35DirectMetadata,
) -> Result<()> {
    validate_text_model_direct_hip_metadata(&model.language_model, metadata)
}

pub(super) fn model_direct_decode_linear_phase_profiled_hip_v1(
    model: &mut ModelForCausalLM,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    text_model_direct_decode_linear_phase_profiled_hip_v1(
        &mut model.language_model,
        metadata,
        start_layer_idx,
        end_layer_idx,
        xs,
        seqlen_offset,
    )
}

pub(super) fn model_direct_decode_full_phase_profiled_hip_v1(
    model: &mut ModelForCausalLM,
    metadata: &PreparedQwen35DirectMetadata,
    start_layer_idx: usize,
    end_layer_idx: usize,
    xs: &StateBuffer,
    seqlen_offset: usize,
) -> Result<(StateBuffer, RuntimeProfile)> {
    text_model_direct_decode_full_phase_profiled_hip_v1(
        &mut model.language_model,
        metadata,
        start_layer_idx,
        end_layer_idx,
        xs,
        seqlen_offset,
    )
}

pub(super) fn model_finalize_direct_decode_logits_hip_v1(
    model: &mut ModelForCausalLM,
    hidden_states: &StateBuffer,
) -> Result<(StateBuffer, RuntimeProfile)> {
    let device = hidden_states.device();
    let backend = backend_buffer_api::for_device(device);
    let output_start = super::frontend::profile_start(device)?;
    let hidden_states = text_model_finalize_direct_decode_hidden_hip_v1(&mut model.language_model, hidden_states)?;
    let logits = backend.slice_last_token(&hidden_states)?;
    let logits = model.lm_head.forward_buffer(&logits)?;
    let mut profile = RuntimeProfile::default();
    profile.output_projection_millis += super::frontend::profile_elapsed(output_start, device)?;
    Ok((logits, profile))
}
