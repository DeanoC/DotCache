use super::{
    MinimalQwen35DirectRuntime, MinimalQwen35KvCache, MinimalQwen35RuntimeProfile,
    MinimalQwen35StateBuffer, ModelForCausalLM, Result,
};

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
        let (logits, profile) = self.model.forward_hidden_states_profiled_direct_hip_v1(
            self.runtime.metadata(),
            hidden_state_t,
            seqlen_offset,
        )?;
        *cache = self.model.cache_state();
        if self.runtime.next_hidden_slot_is_ping {
            self.runtime.decode_hidden_ping = hidden_state_t.clone();
        } else {
            self.runtime.decode_hidden_pong = hidden_state_t.clone();
        }
        self.runtime.next_hidden_slot_is_ping = !self.runtime.next_hidden_slot_is_ping;
        self.runtime.decode_logits = logits.clone();
        self.runtime.last_decode_sequence_length = seqlen_offset + 1;
        Ok((logits, profile))
    }
}
