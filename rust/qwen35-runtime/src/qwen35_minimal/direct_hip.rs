use super::{
    direct_decoder, DirectHipDecodePhaseKind, MinimalQwen35DirectRuntime, MinimalQwen35KvCache,
    MinimalQwen35RuntimeProfile, MinimalQwen35StateBuffer, ModelForCausalLM, Result,
};

fn decode_phase_from_hidden_state(
    model: &mut ModelForCausalLM,
    xs: &MinimalQwen35StateBuffer,
    seqlen_offset: usize,
    phase_kind: DirectHipDecodePhaseKind,
    start_layer_idx: usize,
    end_layer_idx: usize,
) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
    match phase_kind {
        DirectHipDecodePhaseKind::LinearAttention => Ok(
            direct_decoder::model_direct_decode_linear_phase_profiled_hip_v1_unchecked(
                model,
                start_layer_idx,
                end_layer_idx,
                xs,
            )?,
        ),
        DirectHipDecodePhaseKind::FullAttention => Ok(
            direct_decoder::model_direct_decode_full_phase_profiled_hip_v1_unchecked(
                model,
                start_layer_idx,
                end_layer_idx,
                xs,
                seqlen_offset,
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
        let mut workspace = DirectHipDecodeWorkspace::new(
            &mut self.runtime.decode_hidden_ping,
            &mut self.runtime.decode_hidden_pong,
            self.runtime.next_hidden_slot_is_ping,
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
            )?;
            profile.add_assign(&phase_profile);
            workspace.store_next(next_xs);
            workspace.advance();
        }
        let active_hidden = workspace.active();
        let (logits, finalize_profile) =
            self.model.finalize_direct_decode_logits_hip_v1(active_hidden)?;
        profile.add_assign(&finalize_profile);
        *cache = self.model.cache_state();
        self.runtime.next_hidden_slot_is_ping = workspace.active_slot_is_ping();
        self.runtime.decode_logits = logits.clone();
        self.runtime.last_decode_sequence_length = seqlen_offset + 1;
        Ok((logits, profile))
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
