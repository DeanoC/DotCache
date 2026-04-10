use std::time::Instant;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use crate::backend::CandlePageBackend;
use crate::session::{HybridCacheState, SessionId, SessionRuntime};
use crate::Result;

fn ensure_sealed_page_resident(
    sessions: &mut SessionRuntime,
    page_backend: &CandlePageBackend,
    page_id: usize,
    sealed_now: bool,
) -> Result<()> {
    if !sealed_now {
        return Ok(());
    }
    let page = sessions.cache().physical().store().page(page_id)?;
    let _ = page_backend.ensure_page_resident(page_id, page)?;
    if page_backend.can_promote_page_device_primary() {
        let promoted = sessions
            .cache_mut()
            .promote_physical_page_device_only(page_id)?;
        if promoted {
            page_backend.mark_page_device_primary(page_id);
        }
    }
    Ok(())
}

struct PagedFullAttention<'a> {
    sessions: &'a mut SessionRuntime,
    session_id: SessionId,
    page_backend: &'a CandlePageBackend,
}

impl candle_transformers::models::qwen3_5::ExternalFullAttention for PagedFullAttention<'_> {
    fn forward(
        &mut self,
        layer_id: usize,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        num_kv_groups: usize,
        head_dim: usize,
        seqlen_offset: usize,
    ) -> candle_core::Result<candle_transformers::models::qwen3_5::ExternalFullAttentionOutput>
    {
        let started = Instant::now();
        let (b_sz, q_heads, q_len, q_head_dim) = query_states.dims4()?;
        let (_, kv_heads, kv_len, kv_head_dim) = key_states.dims4()?;
        let (_, _, value_len, value_head_dim) = value_states.dims4()?;
        if b_sz != 1 {
            return Err(candle_core::Error::msg(format!(
                "instrumented_qwen35 paged full attention expects batch size 1, got {b_sz}"
            )));
        }
        if q_head_dim != head_dim || kv_head_dim != head_dim || value_head_dim != head_dim {
            return Err(candle_core::Error::msg(format!(
                "qwen35 paged full-attention head dim mismatch expected={head_dim} got={}",
                q_head_dim.max(kv_head_dim).max(value_head_dim)
            )));
        }
        if kv_len != q_len || value_len != q_len {
            return Err(candle_core::Error::msg(format!(
                "qwen35 paged full-attention step length mismatch expected={q_len} got={}",
                kv_len.max(value_len)
            )));
        }
        if q_heads != kv_heads * num_kv_groups {
            return Err(candle_core::Error::msg(format!(
                "qwen35 paged full-attention head grouping mismatch expected={} got={q_heads}",
                kv_heads * num_kv_groups
            )));
        }

        let k_values = key_states
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_values = value_states
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut token_outputs = Vec::with_capacity(q_len);
        for token_index in 0..q_len {
            let absolute_pos = (seqlen_offset + token_index) as u32;
            for kv_head in 0..kv_heads {
                let row_offset = (kv_head * q_len + token_index) * head_dim;
                let row_end = row_offset + head_dim;
                let append_result = self
                    .sessions
                    .append_kv_row_at(
                        self.session_id,
                        layer_id,
                        kv_head,
                        absolute_pos,
                        &k_values[row_offset..row_end],
                        &v_values[row_offset..row_end],
                    )
                    .map_err(|err| candle_core::Error::msg(format!("{err:?}")))?;
                ensure_sealed_page_resident(
                    self.sessions,
                    self.page_backend,
                    append_result.physical_page_id,
                    append_result.sealed_now,
                )
                .map_err(|err| candle_core::Error::msg(format!("{err:?}")))?;
            }

            let layer_plan = self
                .sessions
                .plan_layer_decode(self.session_id, layer_id)
                .map_err(|err| candle_core::Error::msg(format!("{err:?}")))?;
            let mut head_outputs = Vec::with_capacity(kv_heads);
            for kv_head in 0..kv_heads {
                let page_ids = layer_plan
                    .page_ids(kv_head)
                    .map_err(|err| candle_core::Error::msg(format!("{err:?}")))?;
                let q_head_start = kv_head * num_kv_groups;
                let query_batch = query_states
                    .narrow(1, q_head_start, num_kv_groups)?
                    .narrow(2, token_index, 1)?
                    .reshape((num_kv_groups, head_dim))?
                    .to_dtype(DType::F32)?;
                let decoded = self
                    .page_backend
                    .decode_tensor_fused(
                        self.sessions.cache().physical().store(),
                        page_ids,
                        &query_batch,
                    )
                    .map_err(|err| candle_core::Error::msg(format!("{err:?}")))?;
                head_outputs.push(decoded);
            }
            let head_output_refs = head_outputs.iter().collect::<Vec<_>>();
            token_outputs.push(Tensor::cat(&head_output_refs, 0)?);
        }

        let token_output_refs = token_outputs.iter().collect::<Vec<_>>();
        let attn_output = Tensor::cat(&token_output_refs, 0)?
            .reshape((1, q_len, q_heads, head_dim))?
            .transpose(1, 2)?;
        Ok(
            candle_transformers::models::qwen3_5::ExternalFullAttentionOutput {
                attn_output,
                profile: candle_transformers::models::qwen3_5::RuntimeProfile {
                    full_attention_millis: started.elapsed().as_secs_f64() * 1e3,
                    ..Default::default()
                },
            },
        )
    }
}

#[derive(Debug)]
pub struct InstrumentedQwen35 {
    model: candle_transformers::models::qwen3_5::ModelForCausalLM,
}

impl InstrumentedQwen35 {
    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::qwen3_5::Config,
    ) -> Result<Self> {
        Ok(Self {
            model: candle_transformers::models::qwen3_5::ModelForCausalLM::new(cfg, vb)?,
        })
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
        cache_state: Option<&HybridCacheState>,
    ) -> Result<(
        Tensor,
        HybridCacheState,
        candle_transformers::models::qwen3_5::RuntimeProfile,
    )> {
        self.model.clear_kv_cache();
        if let Some(state) = cache_state {
            match state {
                HybridCacheState::Qwen35(state) => self.model.restore_cache_state(state)?,
            }
        }
        let (logits, profile) = self.model.forward_profiled(input_ids, index_pos)?;
        let state = HybridCacheState::Qwen35(self.model.cache_state());
        let logits = logits.to_dtype(DType::F32)?;
        Ok((logits, state, profile))
    }

    pub fn forward_profiled_paged_full_attention(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
        cache_state: Option<&HybridCacheState>,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<(
        Tensor,
        HybridCacheState,
        candle_transformers::models::qwen3_5::RuntimeProfile,
    )> {
        self.model.clear_kv_cache();
        if let Some(state) = cache_state {
            match state {
                HybridCacheState::Qwen35(state) => self.model.restore_cache_state(state)?,
            }
        }
        let mut paged_full_attention = PagedFullAttention {
            sessions,
            session_id,
            page_backend,
        };
        let (logits, profile) = self.model.forward_profiled_with_external_full_attention(
            input_ids,
            index_pos,
            &mut paged_full_attention,
        )?;
        let state = HybridCacheState::Qwen35(self.model.cache_state());
        let logits = logits.to_dtype(DType::F32)?;
        Ok((logits, state, profile))
    }

    pub fn full_attention_layer_ids(&self) -> Vec<usize> {
        self.model.full_attention_layer_ids()
    }

    pub fn empty_cache_state(&self) -> candle_transformers::models::qwen3_5::CacheState {
        self.model.cache_state()
    }
}
