use std::sync::Arc;

use crate::backend::CandlePageBackend;
use crate::decode::decode_query_batch_owned;
use crate::session::{SessionId, SessionRuntime};
use crate::{Result, RuntimeError};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::Activation;
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_transformers::models::qwen2::Config;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};

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
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct InstrumentedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl InstrumentedMlp {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x)?;
        (lhs * rhs)?.apply(&self.down_proj).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct InstrumentedAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl InstrumentedAttention {
    fn load(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            q_proj: linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        if b_sz != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_qwen2",
                message: format!("batch size {b_sz} is unsupported for paged decode"),
            });
        }

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let q_values = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let k_values = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_values = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut outputs = Vec::with_capacity(seq_len * self.num_heads * self.head_dim);

        for token_index in 0..seq_len {
            let absolute_pos = (index_pos + token_index) as u32;
            for kv_head in 0..self.num_kv_heads {
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
            let mut head_outputs = vec![0.0; self.num_heads * self.head_dim];
            for kv_head in 0..self.num_kv_heads {
                let page_ids = layer_plan.page_ids(kv_head)?;
                let mut query_batch = Vec::with_capacity(self.num_kv_groups);
                let mut placements = Vec::with_capacity(self.num_kv_groups);

                for q_head_offset in 0..self.num_kv_groups {
                    let q_head = kv_head * self.num_kv_groups + q_head_offset;
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
            (1, seq_len, self.num_heads, self.head_dim),
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
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        if seq_len != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_qwen2",
                message: format!("batched decode expects seq_len=1, got {seq_len}"),
            });
        }
        if b_sz != index_positions.len() || b_sz != session_ids.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "batched qwen2 decode shape",
                expected: b_sz,
                got: index_positions.len().max(session_ids.len()),
            });
        }

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.apply_rotary_emb_batch(&q, &k, index_positions)?;

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
            for kv_head in 0..self.num_kv_heads {
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

        let mut head_outputs = vec![vec![0.0; self.num_heads * self.head_dim]; b_sz];
        for kv_head in 0..self.num_kv_heads {
            let mut query_batch = Vec::with_capacity(b_sz * self.num_kv_groups);
            let mut page_ids_by_query = Vec::with_capacity(b_sz * self.num_kv_groups);
            let mut placements = Vec::with_capacity(b_sz * self.num_kv_groups);

            for (batch_idx, layer_plan) in layer_plans.iter().enumerate() {
                let page_ids = layer_plan.page_ids(kv_head)?;
                let q_values = &q_values_by_batch[batch_idx];
                for q_head_offset in 0..self.num_kv_groups {
                    let q_head = kv_head * self.num_kv_groups + q_head_offset;
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

        let mut outputs = Vec::with_capacity(b_sz * self.num_heads * self.head_dim);
        for head_output in head_outputs {
            outputs.extend_from_slice(&head_output);
        }

        let y = Tensor::from_vec(
            outputs,
            (b_sz, seq_len, self.num_heads, self.head_dim),
            x.device(),
        )?
        .to_dtype(x.dtype())?
        .reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y).map_err(Into::into)
    }

    fn apply_rotary_emb_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, _heads, _seq_len, _head_dim) = q.dims4()?;
        if b_sz != index_positions.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "qwen2 rotary batch positions",
                expected: b_sz,
                got: index_positions.len(),
            });
        }
        if b_sz == 1 {
            return self
                .rotary_emb
                .apply_rotary_emb_qkv(q, k, index_positions[0]);
        }

        let mut rotated_q = Vec::with_capacity(b_sz);
        let mut rotated_k = Vec::with_capacity(b_sz);
        for (batch_idx, &index_pos) in index_positions.iter().enumerate() {
            let q_batch = q.narrow(0, batch_idx, 1)?.contiguous()?;
            let k_batch = k.narrow(0, batch_idx, 1)?.contiguous()?;
            let (q_rotated, k_rotated) = self
                .rotary_emb
                .apply_rotary_emb_qkv(&q_batch, &k_batch, index_pos)?;
            rotated_q.push(q_rotated);
            rotated_k.push(k_rotated);
        }

        let q_refs = rotated_q.iter().collect::<Vec<_>>();
        let k_refs = rotated_k.iter().collect::<Vec<_>>();
        Ok((Tensor::cat(&q_refs, 0)?, Tensor::cat(&k_refs, 0)?))
    }
}

#[derive(Debug, Clone)]
struct InstrumentedDecoderLayer {
    self_attn: InstrumentedAttention,
    mlp: InstrumentedMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl InstrumentedDecoderLayer {
    fn load(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: InstrumentedAttention::load(rotary_emb, cfg, vb.pp("self_attn"))?,
            mlp: InstrumentedMlp::load(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward(
            &x,
            index_pos,
            block_idx,
            sessions,
            session_id,
            page_backend,
        )? + residual)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + residual)?;
        Ok(x)
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        index_positions: &[usize],
        block_idx: usize,
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward_decode_batch(
            &x,
            index_positions,
            block_idx,
            sessions,
            session_ids,
            page_backend,
        )? + residual)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + residual)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct InstrumentedQwen2 {
    embed_tokens: Embedding,
    layers: Vec<InstrumentedDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl InstrumentedQwen2 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_model.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layer_builder = vb_model.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(InstrumentedDecoderLayer::load(
                rotary_emb.clone(),
                cfg,
                layer_builder.pp(layer_idx),
            )?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
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
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (block_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, index_pos, block_idx, sessions, session_id, page_backend)?;
        }
        sessions.commit_positions(session_id, session_pos, seq_len)?;
        let x = self.norm.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32).map_err(Into::into)
    }

    pub fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        index_positions: &[usize],
        sessions: &mut SessionRuntime,
        session_ids: &[SessionId],
        page_backend: &CandlePageBackend,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        if seq_len != 1 {
            return Err(RuntimeError::External {
                context: "instrumented_qwen2",
                message: format!("batched decode expects seq_len=1, got {seq_len}"),
            });
        }
        if b_sz != session_ids.len() || b_sz != index_positions.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "batched qwen2 session shape",
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

        let mut x = self.embed_tokens.forward(input_ids)?;
        for (block_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_decode_batch(
                &x,
                index_positions,
                block_idx,
                sessions,
                session_ids,
                page_backend,
            )?;
        }
        for (&session_id, &index_pos) in session_ids.iter().zip(index_positions.iter()) {
            sessions.commit_positions(session_id, index_pos as u32, seq_len)?;
        }
        let x = self.norm.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32).map_err(Into::into)
    }
}
