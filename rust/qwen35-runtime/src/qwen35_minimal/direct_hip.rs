use super::{
    direct_decoder, DirectHipDecodePhaseKind, MinimalQwen35DirectRuntime, MinimalQwen35KvCache,
    MinimalQwen35RuntimeProfile, MinimalQwen35StateBuffer, ModelForCausalLM, Result,
};
use super::frontend::max_abs_delta;
use super::types::{CacheState, LayerCacheState};

fn trace_direct_whole_decode_enabled() -> bool {
    std::env::var("DOTCACHE_QWEN35_TRACE_DIRECT_WHOLE_DECODE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn trace_cache_delta(reference: &CacheState, direct: &CacheState) -> Result<()> {
    for (layer_idx, (reference_layer, direct_layer)) in
        reference.layers.iter().zip(direct.layers.iter()).enumerate()
    {
        match (reference_layer, direct_layer) {
            (LayerCacheState::Linear(reference), LayerCacheState::Linear(direct)) => {
                if let (Some(reference), Some(direct)) = (&reference.conv_state, &direct.conv_state)
                {
                    let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
                    eprintln!(
                        "direct-whole-decode-cache-delta layer={} kind=linear conv_state_max_abs_delta={:.6}",
                        layer_idx, delta
                    );
                }
                if let (Some(reference), Some(direct)) =
                    (&reference.recurrent_state, &direct.recurrent_state)
                {
                    let delta = max_abs_delta(reference.tensor(), direct.tensor())?;
                    eprintln!(
                        "direct-whole-decode-cache-delta layer={} kind=linear recurrent_state_max_abs_delta={:.6}",
                        layer_idx, delta
                    );
                }
            }
            (LayerCacheState::Full(reference), LayerCacheState::Full(direct)) => {
                if let (Some((reference_k, reference_v)), Some((direct_k, direct_v))) =
                    (&reference.kv_cache, &direct.kv_cache)
                {
                    let key_delta = max_abs_delta(reference_k.tensor(), direct_k.tensor())?;
                    let value_delta = max_abs_delta(reference_v.tensor(), direct_v.tensor())?;
                    eprintln!(
                        "direct-whole-decode-cache-delta layer={} kind=full key_max_abs_delta={:.6} value_max_abs_delta={:.6}",
                        layer_idx, key_delta, value_delta
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn decode_phase_from_hidden_state(
    model: &mut ModelForCausalLM,
    xs: &MinimalQwen35StateBuffer,
    seqlen_offset: usize,
    phase_kind: DirectHipDecodePhaseKind,
    start_layer_idx: usize,
    end_layer_idx: usize,
    full_attention_gate: &MinimalQwen35StateBuffer,
    full_attention_qkv: &MinimalQwen35StateBuffer,
    full_attention_key: &MinimalQwen35StateBuffer,
    full_attention_value: &MinimalQwen35StateBuffer,
    full_attention_output_scratch: &MinimalQwen35StateBuffer,
    phase_output_scratch: &MinimalQwen35StateBuffer,
    linear_mixed_qkv: &MinimalQwen35StateBuffer,
    linear_z: &MinimalQwen35StateBuffer,
    linear_beta_raw: &MinimalQwen35StateBuffer,
    linear_a: &MinimalQwen35StateBuffer,
) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
    match phase_kind {
        DirectHipDecodePhaseKind::LinearAttention => Ok(
            direct_decoder::model_direct_decode_linear_phase_profiled_hip_v1_unchecked(
                model,
                start_layer_idx,
                end_layer_idx,
                xs,
                phase_output_scratch,
                linear_mixed_qkv,
                linear_z,
                linear_beta_raw,
                linear_a,
            )?,
        ),
        DirectHipDecodePhaseKind::FullAttention => Ok(
            direct_decoder::model_direct_decode_full_phase_profiled_hip_v1_unchecked(
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
            )?,
        ),
    }
}

struct DirectHipDecodeWorkspace<'a> {
    ping: &'a mut MinimalQwen35StateBuffer,
    pong: &'a mut MinimalQwen35StateBuffer,
    active_slot_is_ping: bool,
}

impl<'a> DirectHipDecodeWorkspace<'a> {
    fn new(
        ping: &'a mut MinimalQwen35StateBuffer,
        pong: &'a mut MinimalQwen35StateBuffer,
        active_slot_is_ping: bool,
    ) -> Self {
        Self {
            ping,
            pong,
            active_slot_is_ping,
        }
    }

    fn seed_input(&mut self, hidden_state_t: &MinimalQwen35StateBuffer) {
        if self.active_slot_is_ping {
            *self.ping = hidden_state_t.clone();
        } else {
            *self.pong = hidden_state_t.clone();
        }
    }

    fn active(&self) -> &MinimalQwen35StateBuffer {
        if self.active_slot_is_ping {
            self.ping
        } else {
            self.pong
        }
    }

    fn inactive(&self) -> &MinimalQwen35StateBuffer {
        if self.active_slot_is_ping {
            self.pong
        } else {
            self.ping
        }
    }

    fn store_next(&mut self, next_xs: MinimalQwen35StateBuffer) {
        if self.active_slot_is_ping {
            *self.pong = next_xs;
        } else {
            *self.ping = next_xs;
        }
    }

    fn advance(&mut self) {
        self.active_slot_is_ping = !self.active_slot_is_ping;
    }

    fn active_slot_is_ping(&self) -> bool {
        self.active_slot_is_ping
    }
}

pub(crate) struct DirectHipQwen35V1Executor<'a> {
    model: &'a mut ModelForCausalLM,
    runtime: &'a mut MinimalQwen35DirectRuntime,
}

impl<'a> DirectHipQwen35V1Executor<'a> {
    pub(crate) fn new(
        model: &'a mut ModelForCausalLM,
        runtime: &'a mut MinimalQwen35DirectRuntime,
    ) -> Self {
        Self { model, runtime }
    }

    pub(crate) fn prefill_from_hidden_states(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(
        MinimalQwen35StateBuffer,
        MinimalQwen35KvCache,
        MinimalQwen35RuntimeProfile,
    )> {
        let (_, seq_len, _) = hidden_states.dims3()?;
        let (logits, profile) = self.model.forward_hidden_states_profiled_direct_hip_v1(
            self.runtime.metadata(),
            hidden_states,
            0,
        )?;
        let cache = self.model.cache_state();
        self.runtime.decode_logits = logits.clone();
        self.runtime.last_prefill_sequence_length = seq_len;
        Ok((logits, cache, profile))
    }

    pub(crate) fn decode_from_hidden_state(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
        let reference_cache_state = if trace_direct_whole_decode_enabled() {
            Some(cache.clone())
        } else {
            None
        };
        self.model.restore_cache_state(cache)?;
        let seqlen_offset = cache.sequence_length();
        self.model.validate_direct_hip_metadata(self.runtime.metadata())?;
        let (_, seq_len, _) = hidden_state_t.dims3()?;
        if seq_len != 1 {
            return Err(crate::RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct-hip-v1 decode expects a single-token hidden state, got seq_len={seq_len}"
                ),
            });
        }
        self.validate_decode_phases()?;
        let phase_specs = self.runtime.decode_phase_specs().to_vec();
        let mut profile = MinimalQwen35RuntimeProfile::default();
        let runtime = &mut self.runtime;
        let full_attention_gate = &runtime.full_attention_gate;
        let full_attention_qkv = &runtime.full_attention_qkv;
        let full_attention_key = &runtime.full_attention_key;
        let full_attention_value = &runtime.full_attention_value;
        let linear_mixed_qkv = &runtime.linear_mixed_qkv;
        let linear_z = &runtime.linear_z;
        let linear_beta_raw = &runtime.linear_beta_raw;
        let linear_a = &runtime.linear_a;
        let mut workspace = DirectHipDecodeWorkspace::new(
            &mut runtime.decode_hidden_ping,
            &mut runtime.decode_hidden_pong,
            runtime.next_hidden_slot_is_ping,
        );
        workspace.seed_input(hidden_state_t);
        for phase in phase_specs.iter().copied() {
            let (next_xs, phase_profile) = decode_phase_from_hidden_state(
                self.model,
                workspace.active(),
                seqlen_offset,
                phase.kind,
                phase.start_layer_idx,
                phase.end_layer_idx,
                full_attention_gate,
                full_attention_qkv,
                full_attention_key,
                full_attention_value,
                workspace.inactive(),
                workspace.inactive(),
                linear_mixed_qkv,
                linear_z,
                linear_beta_raw,
                linear_a,
            )?;
            profile.add_assign(&phase_profile);
            workspace.store_next(next_xs);
            workspace.advance();
        }
        let active_hidden = workspace.active();
        let (logits, finalize_profile) = self
            .model
            .finalize_direct_decode_logits_hip_v1(active_hidden, &runtime.decode_logits)?;
        profile.add_assign(&finalize_profile);
        *cache = self.model.cache_state();
        runtime.next_hidden_slot_is_ping = workspace.active_slot_is_ping();
        runtime.decode_logits = logits;
        runtime.last_decode_sequence_length = seqlen_offset + 1;
        if let Some(reference_cache_state) = reference_cache_state {
            let direct_cache_state = cache.clone();
            self.model.restore_cache_state(&reference_cache_state)?;
            let (reference_logits, _) =
                self.model.forward_hidden_states_profiled(hidden_state_t, seqlen_offset)?;
            let reference_post_decode_cache = self.model.cache_state();
            self.model.restore_cache_state(&direct_cache_state)?;
            let logits_delta =
                max_abs_delta(reference_logits.tensor(), runtime.decode_logits.tensor())?;
            eprintln!(
                "direct-whole-decode-delta logits_max_abs_delta={:.6}",
                logits_delta
            );
            trace_cache_delta(&reference_post_decode_cache, &direct_cache_state)?;
        }
        Ok((runtime.decode_logits.clone(), profile))
    }

    fn validate_decode_phases(&self) -> Result<()> {
        let metadata = self.runtime.metadata();
        if metadata.layers.len() != metadata.num_hidden_layers {
            return Err(crate::RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct-hip-v1 metadata layer count mismatch: layers={} num_hidden_layers={}",
                    metadata.layers.len(),
                    metadata.num_hidden_layers
                ),
            });
        }
        for phase in metadata.decode_phases.iter() {
            match phase.layer_type.as_str() {
                "linear_attention" | "full_attention" => {}
                other => {
                    return Err(crate::RuntimeError::External {
                        context: "qwen35-hip-direct",
                        message: format!(
                            "unsupported direct-hip-v1 decode phase type in metadata: {other}"
                        ),
                    });
                }
            }
        }
        if metadata.decode_phases.is_empty() {
            return Err(crate::RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct-hip-v1 metadata has no decode phases".to_string(),
            });
        }
        if self.runtime.decode_phase_specs().is_empty() {
            return Err(crate::RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct-hip-v1 runtime has no compiled decode phase specs".to_string(),
            });
        }
        Ok(())
    }
}
