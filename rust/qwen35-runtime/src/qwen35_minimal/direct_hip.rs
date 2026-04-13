use super::{
    MinimalQwen35DirectRuntime, MinimalQwen35KvCache, MinimalQwen35RuntimeProfile,
    MinimalQwen35StateBuffer, ModelForCausalLM, Result,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectHipDecodePhase {
    layer_type: &'static str,
    start_layer_idx: usize,
    end_layer_idx: usize,
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
        let mut profile = MinimalQwen35RuntimeProfile::default();
        let mut active_slot_is_ping = self.runtime.next_hidden_slot_is_ping;
        if active_slot_is_ping {
            self.runtime.decode_hidden_ping = hidden_state_t.clone();
        } else {
            self.runtime.decode_hidden_pong = hidden_state_t.clone();
        }
        let phases = self.decode_phases()?;
        for phase in phases {
            let current_xs = if active_slot_is_ping {
                self.runtime.decode_hidden_ping.clone()
            } else {
                self.runtime.decode_hidden_pong.clone()
            };
            let (next_xs, phase_profile) =
                self.decode_phase_from_hidden_state(&current_xs, seqlen_offset, phase)?;
            profile.add_assign(&phase_profile);
            if active_slot_is_ping {
                self.runtime.decode_hidden_pong = next_xs;
            } else {
                self.runtime.decode_hidden_ping = next_xs;
            }
            active_slot_is_ping = !active_slot_is_ping;
        }
        let active_hidden = if active_slot_is_ping {
            &self.runtime.decode_hidden_ping
        } else {
            &self.runtime.decode_hidden_pong
        };
        let (logits, finalize_profile) =
            self.model.finalize_direct_decode_logits_hip_v1(active_hidden)?;
        profile.add_assign(&finalize_profile);
        *cache = self.model.cache_state();
        self.runtime.next_hidden_slot_is_ping = active_slot_is_ping;
        self.runtime.decode_logits = logits.clone();
        self.runtime.last_decode_sequence_length = seqlen_offset + 1;
        Ok((logits, profile))
    }

    fn decode_phase_from_hidden_state(
        &mut self,
        xs: &MinimalQwen35StateBuffer,
        seqlen_offset: usize,
        phase: DirectHipDecodePhase,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
        match phase.layer_type {
            "linear_attention" | "full_attention" => {}
            other => {
                return Err(crate::RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!("unsupported direct-hip-v1 decode phase type: {other}"),
                });
            }
        }
        let mut profile = MinimalQwen35RuntimeProfile::default();
        let xs = xs.clone();
        let (xs, phase_profile) = match phase.layer_type {
            "linear_attention" => self.model.direct_decode_linear_phase_profiled_hip_v1(
                self.runtime.metadata(),
                phase.start_layer_idx,
                phase.end_layer_idx,
                &xs,
                seqlen_offset,
            )?,
            "full_attention" => self.model.direct_decode_full_phase_profiled_hip_v1(
                self.runtime.metadata(),
                phase.start_layer_idx,
                phase.end_layer_idx,
                &xs,
                seqlen_offset,
            )?,
            other => {
                return Err(crate::RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!("unsupported direct-hip-v1 decode phase type: {other}"),
                });
            }
        };
        profile.add_assign(&phase_profile);
        Ok((xs, profile))
    }

    fn decode_phases(&self) -> Result<Vec<DirectHipDecodePhase>> {
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
        let mut phases = Vec::with_capacity(metadata.decode_phases.len());
        for phase in metadata.decode_phases.iter() {
            let layer_type = match phase.layer_type.as_str() {
                "linear_attention" => "linear_attention",
                "full_attention" => "full_attention",
                other => {
                    return Err(crate::RuntimeError::External {
                        context: "qwen35-hip-direct",
                        message: format!(
                            "unsupported direct-hip-v1 decode phase type in metadata: {other}"
                        ),
                    });
                }
            };
            phases.push(DirectHipDecodePhase {
                layer_type,
                start_layer_idx: phase.start_layer_idx,
                end_layer_idx: phase.end_layer_idx,
            });
        }
        if phases.is_empty() {
            return Err(crate::RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct-hip-v1 metadata has no decode phases".to_string(),
            });
        }
        Ok(phases)
    }
}
