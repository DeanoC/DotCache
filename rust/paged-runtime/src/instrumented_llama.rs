use std::f32::consts::PI;

use crate::backend::CandlePageBackend;
use crate::decode::decode_query_batch_owned;
use crate::session::{SessionId, SessionRuntime};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    embedding, linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use crate::{Result, RuntimeError};

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

#[derive(Debug, Clone)]
pub struct LlamaCache {
    cos: Tensor,
    sin: Tensor,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl LlamaCache {
    pub fn new(
        _use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> Result<Self> {
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;
        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }
}

#[derive(Debug, Clone)]
struct InstrumentedCausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl InstrumentedCausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        Ok(Self {
            q_proj: linear_no_bias(size_in, size_q, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(size_in, size_kv, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(size_in, size_kv, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(size_q, size_in, vb.pp("o_proj"))?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        if b_sz != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_llama",
                message: format!("batch size {b_sz} is unsupported for paged decode"),
            });
        }

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        let q_values = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let k_values = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_values = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let q_heads_per_kv = self.num_attention_heads / self.num_key_value_heads;
        let mut outputs = Vec::with_capacity(seq_len * self.num_attention_heads * self.head_dim);

        for token_index in 0..seq_len {
            let absolute_pos = (index_pos + token_index) as u32;
            for kv_head in 0..self.num_key_value_heads {
                let row_offset = (kv_head * seq_len + token_index) * self.head_dim;
                let row_end = row_offset + self.head_dim;
                let append_result = sessions.append_kv_row_at(
                    session_id,
                    block_idx,
                    kv_head,
                    absolute_pos,
                    &k_values[row_offset..row_end],
                    &v_values[row_offset..row_end],
                )?;
                ensure_sealed_page_resident(
                    sessions,
                    page_backend,
                    append_result.physical_page_id,
                    append_result.sealed_now,
                )?;
            }

            let layer_plan = sessions.plan_layer_decode(session_id, block_idx)?;
            let mut head_outputs = vec![0.0; self.num_attention_heads * self.head_dim];
            for kv_head in 0..self.num_key_value_heads {
                let page_ids = layer_plan.page_ids(kv_head)?;
                let mut query_batch = Vec::with_capacity(q_heads_per_kv);
                let mut placements = Vec::with_capacity(q_heads_per_kv);
                for q_head_offset in 0..q_heads_per_kv {
                    let q_head = kv_head * q_heads_per_kv + q_head_offset;
                    let query_offset = (q_head * seq_len + token_index) * self.head_dim;
                    let query_end = query_offset + self.head_dim;
                    query_batch.push(&q_values[query_offset..query_end]);
                    placements.push(q_head);
                }

                let page_ids_by_query = vec![page_ids; query_batch.len()];
                let decoded = decode_query_batch_owned(
                    page_backend,
                    sessions.cache().physical().store(),
                    &page_ids_by_query,
                    &query_batch,
                )?;

                for (q_head, head_output) in placements.into_iter().zip(decoded.into_iter()) {
                    let output_offset = q_head * self.head_dim;
                    let output_end = output_offset + self.head_dim;
                    head_outputs[output_offset..output_end].copy_from_slice(&head_output);
                }
            }
            outputs.extend_from_slice(&head_outputs);
        }

        let y = Tensor::from_vec(
            outputs,
            (1, seq_len, self.num_attention_heads, self.head_dim),
            x.device(),
        )?
        .to_dtype(x.dtype())?
        .reshape((1, seq_len, hidden_size))?;
        self.o_proj.forward(&y).map_err(Into::into)
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        index_positions: &[usize],
        block_idx: usize,
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        if seq_len != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_llama",
                message: format!("batched decode expects seq_len=1, got {seq_len}"),
            });
        }
        if b_sz != index_positions.len() || b_sz != session_ids.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "batched decode session shape",
                expected: b_sz,
                got: index_positions.len().max(session_ids.len()),
            });
        }

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb_batch(&q, index_positions, cache)?;
        let k = self.apply_rotary_emb_batch(&k, index_positions, cache)?;

        let q_heads_per_kv = self.num_attention_heads / self.num_key_value_heads;
        let mut outputs = Vec::with_capacity(b_sz * self.num_attention_heads * self.head_dim);

        for (batch_idx, (&session_id, &index_pos)) in
            session_ids.iter().zip(index_positions.iter()).enumerate()
        {
            let k_batch = k.narrow(0, batch_idx, 1)?.squeeze(0)?.contiguous()?;
            let v_batch = v.narrow(0, batch_idx, 1)?.squeeze(0)?.contiguous()?;

            let k_values = k_batch
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let v_values = v_batch
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let absolute_pos = index_pos as u32;
            for kv_head in 0..self.num_key_value_heads {
                let row_offset = kv_head * self.head_dim;
                let row_end = row_offset + self.head_dim;
                let append_result = sessions.append_kv_row_at(
                    session_id,
                    block_idx,
                    kv_head,
                    absolute_pos,
                    &k_values[row_offset..row_end],
                    &v_values[row_offset..row_end],
                )?;
                ensure_sealed_page_resident(
                    sessions,
                    page_backend,
                    append_result.physical_page_id,
                    append_result.sealed_now,
                )?;
            }
        }

        let layer_plans = sessions.plan_sessions_layer_decode(session_ids, block_idx)?;
        let mut q_values_by_batch = Vec::with_capacity(b_sz);
        for batch_idx in 0..b_sz {
            let q_batch = q.narrow(0, batch_idx, 1)?.squeeze(0)?.contiguous()?;
            q_values_by_batch.push(
                q_batch
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }

        let mut head_outputs = vec![vec![0.0; self.num_attention_heads * self.head_dim]; b_sz];
        for kv_head in 0..self.num_key_value_heads {
            let mut query_batch = Vec::with_capacity(b_sz * q_heads_per_kv);
            let mut page_ids_by_query = Vec::with_capacity(b_sz * q_heads_per_kv);
            let mut placements = Vec::with_capacity(b_sz * q_heads_per_kv);

            for (batch_idx, layer_plan) in layer_plans.iter().enumerate() {
                let page_ids = layer_plan.page_ids(kv_head)?;
                let q_values = &q_values_by_batch[batch_idx];
                for q_head_offset in 0..q_heads_per_kv {
                    let q_head = kv_head * q_heads_per_kv + q_head_offset;
                    let query_offset = q_head * self.head_dim;
                    let query_end = query_offset + self.head_dim;
                    query_batch.push(&q_values[query_offset..query_end]);
                    page_ids_by_query.push(page_ids);
                    placements.push((batch_idx, q_head));
                }
            }

            let decoded = decode_query_batch_owned(
                page_backend,
                sessions.cache().physical().store(),
                &page_ids_by_query,
                &query_batch,
            )?;

            for ((batch_idx, q_head), head_output) in
                placements.into_iter().zip(decoded.into_iter())
            {
                let output_offset = q_head * self.head_dim;
                let output_end = output_offset + self.head_dim;
                head_outputs[batch_idx][output_offset..output_end].copy_from_slice(&head_output);
            }
        }

        for head_output in head_outputs {
            outputs.extend_from_slice(&head_output);
        }

        let y = Tensor::from_vec(
            outputs,
            (b_sz, seq_len, self.num_attention_heads, self.head_dim),
            x.device(),
        )?
        .to_dtype(x.dtype())?
        .reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y).map_err(Into::into)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &LlamaCache) -> Result<Tensor> {
        let (_b_sz, _heads, seq_len, _head_dim) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin).map_err(Into::into)
    }

    fn apply_rotary_emb_batch(
        &self,
        x: &Tensor,
        index_positions: &[usize],
        cache: &LlamaCache,
    ) -> Result<Tensor> {
        let (b_sz, _heads, _seq_len, _head_dim) = x.dims4()?;
        if b_sz != index_positions.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "rotary batch positions",
                expected: b_sz,
                got: index_positions.len(),
            });
        }
        if b_sz == 1 {
            return self.apply_rotary_emb(x, index_positions[0], cache);
        }

        let mut rotated = Vec::with_capacity(b_sz);
        for (batch_idx, &index_pos) in index_positions.iter().enumerate() {
            let batch = x.narrow(0, batch_idx, 1)?.contiguous()?;
            rotated.push(self.apply_rotary_emb(&batch, index_pos, cache)?);
        }
        let rotated_refs = rotated.iter().collect::<Vec<_>>();
        Tensor::cat(&rotated_refs, 0).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct InstrumentedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl InstrumentedMlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct InstrumentedBlock {
    rms_1: RmsNorm,
    attn: InstrumentedCausalSelfAttention,
    rms_2: RmsNorm,
    mlp: InstrumentedMlp,
}

impl InstrumentedBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            rms_1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: InstrumentedCausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            rms_2: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: InstrumentedMlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            index_pos,
            block_idx,
            cache,
            sessions,
            session_id,
            page_backend,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        index_positions: &[usize],
        block_idx: usize,
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward_decode_batch(
            &x,
            index_positions,
            block_idx,
            cache,
            sessions,
            session_ids,
            page_backend,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct InstrumentedLlama {
    wte: Embedding,
    blocks: Vec<InstrumentedBlock>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl InstrumentedLlama {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(wte.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            blocks.push(InstrumentedBlock::load(
                vb.pp(format!("model.layers.{layer_idx}")),
                cfg,
            )?);
        }
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let session_pos = sessions.current_position(session_id)?;
        if session_pos != index_pos as u32 {
            return Err(RuntimeError::PositionMismatch {
                expected: session_pos,
                got: index_pos as u32,
            });
        }

        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut x = self.wte.forward(input_ids)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                index_pos,
                block_idx,
                cache,
                sessions,
                session_id,
                page_backend,
            )?;
        }
        sessions.commit_positions(session_id, session_pos, seq_len)?;
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32).map_err(Into::into)
    }

    pub fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        index_positions: &[usize],
        cache: &mut LlamaCache,
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        if seq_len != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_llama",
                message: format!("batched decode expects seq_len=1, got {seq_len}"),
            });
        }
        if b_sz != session_ids.len() || b_sz != index_positions.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "batched llama decode shape",
                expected: b_sz,
                got: session_ids.len().max(index_positions.len()),
            });
        }

        for (&session_id, &index_pos) in session_ids.iter().zip(index_positions.iter()) {
            let session_pos = sessions.current_position(session_id)?;
            if session_pos != index_pos as u32 {
                return Err(RuntimeError::PositionMismatch {
                    expected: session_pos,
                    got: index_pos as u32,
                });
            }
        }

        let mut x = self.wte.forward(input_ids)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_decode_batch(
                &x,
                index_positions,
                block_idx,
                cache,
                sessions,
                session_ids,
                page_backend,
            )?;
        }
        for (&session_id, &index_pos) in session_ids.iter().zip(index_positions.iter()) {
            sessions.commit_positions(session_id, index_pos as u32, seq_len)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32).map_err(Into::into)
    }
}
