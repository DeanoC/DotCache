use super::backend_buffer_api;
use super::backend_buffer_api::Qwen35BackendBufferApi;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
use super::frontend::{
    debug_full_prefill_kernel_compare_enabled, max_abs_delta, prepared_linear_b,
    profile_elapsed, profile_start, Qwen35RmsNorm, RotaryEmbedding,
};
use super::model::{
    full_attention_blockwise_tiles, full_attention_sdpa_q_block, parse_usize_env, repeat_kv,
    use_full_attention_decode_megakernel, use_full_attention_prefill_megakernel,
    use_full_attention_torchlike_eager,
};
use super::ops;
use super::prepared::PreparedTensorSource;
use super::types::{
    ExternalFullAttention, FullAttentionCacheState, FullAttentionTrace, RuntimeProfile,
    StateBuffer, TextConfig,
};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::linear_b;
use super::with_tracing::Linear;
use candle::{DType, Device, DeviceLocation, Module, Result, Tensor, D};
use candle_core as candle;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attention_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(StateBuffer, StateBuffer)>,
}

pub(super) struct DirectFullAttentionDecodeContext<'a> {
    backend: &'static dyn Qwen35BackendBufferApi,
    output_dtype: DType,
    b_sz: usize,
    q_len: usize,
    seqlen_offset: usize,
    prev_k: Option<&'a StateBuffer>,
    prev_v: Option<&'a StateBuffer>,
    q_norm_weight: &'a Tensor,
    q_norm_eps: f64,
    k_norm_weight: &'a Tensor,
    k_norm_eps: f64,
    gate_workspace: &'a StateBuffer,
    query_workspace: &'a StateBuffer,
    key_workspace: &'a StateBuffer,
    value_workspace: &'a StateBuffer,
}

impl FullAttention {
    pub(super) fn cache_state(&self) -> FullAttentionCacheState {
        FullAttentionCacheState {
            kv_cache: self.kv_cache.clone(),
        }
    }

    pub(super) fn restore_cache_state(&mut self, state: &FullAttentionCacheState) {
        self.kv_cache = state.kv_cache.clone();
    }

    fn causal_block_mask(
        device: &Device,
        q_start: usize,
        q_len: usize,
        k_start: usize,
        k_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let query_base = seqlen_offset + q_start;
        let mut mask = Vec::with_capacity(q_len * k_len);
        for q_idx in 0..q_len {
            let q_abs = query_base + q_idx;
            for k_idx in 0..k_len {
                let k_abs = k_start + k_idx;
                mask.push(if k_abs > q_abs {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        Tensor::from_slice(&mask, (q_len, k_len), device)?.reshape((1, 1, q_len, k_len))
    }

    fn blockwise_attention_profiled(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        scale: f64,
        seqlen_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states_f.device();
        let (b_sz, q_heads, q_len, _head_dim) = query_states_f.dims4()?;
        let (_, _, kv_len, value_dim) = value_states_f.dims4()?;
        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let q_block = query_states_f.narrow(2, q_start, q_block_len)?;
            let mut running_max =
                Tensor::full(f32::NEG_INFINITY, (b_sz, q_heads, q_block_len, 1), device)?;
            let mut running_sum =
                Tensor::zeros((b_sz, q_heads, q_block_len, 1), DType::F32, device)?;
            let mut running_acc =
                Tensor::zeros((b_sz, q_heads, q_block_len, value_dim), DType::F32, device)?;

            for k_start in (0..kv_len).step_by(k_block_size) {
                let k_block_len = (kv_len - k_start).min(k_block_size);
                let q_abs_min = seqlen_offset + q_start;
                let q_abs_max = q_abs_min + q_block_len - 1;
                let k_abs_min = k_start;
                let k_abs_max = k_start + k_block_len - 1;
                if k_abs_min > q_abs_max {
                    break;
                }

                let score_start = profile_start(device)?;
                let k_block = key_states_f.narrow(2, k_start, k_block_len)?;
                let mut scores = (q_block.matmul(&k_block.transpose(2, 3)?)? * scale)?;
                let needs_partial_mask = !(k_abs_max <= q_abs_min);
                if needs_partial_mask {
                    let mask = Self::causal_block_mask(
                        device,
                        q_start,
                        q_block_len,
                        k_start,
                        k_block_len,
                        seqlen_offset,
                    )?;
                    scores = scores.broadcast_add(&mask)?;
                }
                profile.attention_score_millis += profile_elapsed(score_start, device)?;

                let softmax_start = profile_start(device)?;
                let block_max = scores.max_keepdim(D::Minus1)?;
                let new_max = running_max.maximum(&block_max)?;
                let prev_scale = running_max.broadcast_sub(&new_max)?.exp()?;
                let exp_scores = scores.broadcast_sub(&new_max)?.exp()?;
                let new_sum = running_sum
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.sum_keepdim(D::Minus1)?)?;
                profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

                let mix_start = profile_start(device)?;
                let v_block = value_states_f.narrow(2, k_start, k_block_len)?;
                let new_acc = running_acc
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.matmul(&v_block)?)?;
                running_max = new_max;
                running_sum = new_sum;
                running_acc = new_acc;
                profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            }

            q_outputs.push(running_acc.broadcast_div(&running_sum)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn blockwise_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f64,
        seqlen_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.blockwise_attention_profiled(
            query_states_f,
            key_states_f,
            value_states_f,
            scale,
            seqlen_offset,
            q_block_size,
            k_block_size,
            profile,
        )?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn sdpa_chunked_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        scale: f32,
        seqlen_offset: usize,
        q_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let (b_sz, q_heads, q_len, _) = query_states.dims4()?;
        let (_, kv_heads, kv_len, _) = key_states.dims4()?;
        let base_head_chunk = parse_usize_env("CANDLE_QWEN35_FULL_SDPA_Q_HEADS")
            .unwrap_or(q_heads)
            .min(q_heads)
            .max(self.num_kv_groups);
        let q_head_chunk =
            ((base_head_chunk / self.num_kv_groups).max(1) * self.num_kv_groups).min(q_heads);
        let kv_head_chunk = q_head_chunk / self.num_kv_groups;
        if kv_head_chunk == 0 || kv_head_chunk > kv_heads {
            candle::bail!("invalid sdpa q_head chunk for grouped attention")
        }

        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let mask_base =
                Self::causal_block_mask(device, q_start, q_block_len, 0, kv_len, seqlen_offset)?
                    .to_dtype(query_states.dtype())?;
            let mut head_outputs = Vec::with_capacity(q_heads.div_ceil(q_head_chunk));
            for q_head_start in (0..q_heads).step_by(q_head_chunk) {
                let q_head_len = (q_heads - q_head_start).min(q_head_chunk);
                let kv_head_start = q_head_start / self.num_kv_groups;
                let kv_head_len = q_head_len / self.num_kv_groups;
                let q_chunk = query_states
                    .narrow(1, q_head_start, q_head_len)?
                    .narrow(2, q_start, q_block_len)?
                    .contiguous()?;
                let k_chunk = key_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let v_chunk = value_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let mask = mask_base.broadcast_as((b_sz, q_head_len, q_block_len, kv_len))?;
                let fused_start = profile_start(device)?;
                let output =
                    ops::sdpa(&q_chunk, &k_chunk, &v_chunk, Some(&mask), false, scale, 1.0)?;
                let fused_elapsed = profile_elapsed(fused_start, device)?;
                profile.attention_mix_millis += fused_elapsed;
                head_outputs.push(output);
            }
            q_outputs.push(Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn sdpa_chunked_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f32,
        seqlen_offset: usize,
        q_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.sdpa_chunked_attention_profiled(
            query_states,
            key_states,
            value_states,
            scale,
            seqlen_offset,
            q_block_size,
            profile,
        )?
        .to_dtype(DType::F32)?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn grouped_torchlike_eager_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let compute_dtype = query_states.dtype();
        let (_, q_heads, _, _) = query_states.dims4()?;
        let (_, kv_heads, _, _) = key_states.dims4()?;
        let mask_start = profile_start(device)?;
        let mask = attention_mask
            .map(|mask| mask.to_dtype(compute_dtype))
            .transpose()?;
        profile.full_attention_mask_prepare_millis += profile_elapsed(mask_start, device)?;
        let mut outputs = Vec::with_capacity(kv_heads);

        for kv_head_idx in 0..kv_heads {
            let q_head_start = kv_head_idx * self.num_kv_groups;
            let q_head_len = (q_heads - q_head_start).min(self.num_kv_groups);
            if q_head_len == 0 {
                break;
            }
            let query_layout_start = profile_start(device)?;
            let q_chunk = query_states
                .narrow(1, q_head_start, q_head_len)?
                .contiguous()?;
            profile.full_attention_input_layout_millis +=
                profile_elapsed(query_layout_start, device)?;
            let kv_len = key_states.dim(2)?;
            let value_dim = value_states.dim(3)?;
            let kv_materialize_start = profile_start(device)?;
            let k_chunk = key_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                self.head_dim,
            ))?;
            let v_chunk = value_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                value_dim,
            ))?;
            let key_states_t = k_chunk.transpose(2, 3)?.contiguous()?;
            profile.full_attention_kv_materialize_millis +=
                profile_elapsed(kv_materialize_start, device)?;

            let score_start = profile_start(device)?;
            let mut attn_weights = (q_chunk.matmul(&key_states_t)? * scale)?;
            if let Some(mask) = &mask {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }
            profile.attention_score_millis += profile_elapsed(score_start, device)?;

            let softmax_start = profile_start(device)?;
            let attn_weights = ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
                .to_dtype(compute_dtype)?;
            profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

            let mix_start = profile_start(device)?;
            let attn_output = attn_weights.matmul(&v_chunk)?;
            profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            outputs.push(attn_output);
        }

        let collect_start = profile_start(device)?;
        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?;
        profile.full_attention_output_collect_millis += profile_elapsed(collect_start, device)?;
        Ok(output)
    }

    fn grouped_torchlike_eager_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        attention_mask: Option<&Tensor>,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f64,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.grouped_torchlike_eager_attention_profiled(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scale,
            profile,
        )?
        .to_dtype(DType::F32)?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn external_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        handler: &mut dyn ExternalFullAttention,
        layer_id: usize,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        seqlen_offset: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let device = query_states.device();
        let external_started = profile_start(device)?;
        let external = handler.forward(
            layer_id,
            query_states,
            key_states,
            value_states,
            self.num_kv_groups,
            self.head_dim,
            seqlen_offset,
        )?;
        let external_elapsed = profile_elapsed(external_started, device)?;
        profile.add_assign(&external.profile);
        if external.profile.full_attention_millis == 0.0 {
            profile.full_attention_millis += external_elapsed;
        }
        backend.prepare_full_attention_output(
            &external.attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    #[cfg(any(feature = "hf", test))]
    pub(super) fn new(cfg: &TextConfig, rotary_emb: Arc<RotaryEmbedding>, vb: WeightBuilder) -> Result<Self> {
        let q_proj = linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim * 2,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attention_size: cfg.num_attention_heads * cfg.head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    pub(super) fn from_prepared(
        cfg: &TextConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        source: &PreparedTensorSource,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: prepared_linear_b(&source.pp("q_proj"), cfg.attention_bias)?,
            k_proj: prepared_linear_b(&source.pp("k_proj"), cfg.attention_bias)?,
            v_proj: prepared_linear_b(&source.pp("v_proj"), cfg.attention_bias)?,
            o_proj: prepared_linear_b(&source.pp("o_proj"), cfg.attention_bias)?,
            q_norm: Qwen35RmsNorm::from_prepared(cfg.rms_norm_eps, &source.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::from_prepared(cfg.rms_norm_eps, &source.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attention_size: cfg.num_attention_heads * cfg.head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward_profiled_with_external(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_id: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;
        let qkv_start = profile_start(device)?;
        let q_and_gate = backend.tensor_to_buffer(self.q_proj.forward(xs)?)?;
        let k_proj = backend.tensor_to_buffer(self.k_proj.forward(xs)?)?;
        let v_proj = backend.tensor_to_buffer(self.v_proj.forward(xs)?)?;
        let (query_states, gate, key_states, value_states) = backend.prepare_full_attention_inputs(
            &q_and_gate,
            &k_proj,
            &v_proj,
            b_sz,
            q_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.weight(),
            self.k_norm.eps(),
        )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&query_states, &key_states, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let kv_append_start = profile_start(device)?;
        let appended_kv = if external_full_attention.is_none() {
            Some(backend.append_full_attention_kv_buffers(
                self.kv_cache.as_ref().map(|(k, _)| k),
                self.kv_cache.as_ref().map(|(_, v)| v),
                key_states.tensor(),
                value_states.tensor(),
            )?)
        } else {
            None
        };
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) = if let Some((key_states, value_states)) =
            appended_kv.as_ref()
        {
            backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                &query_states,
                key_states,
                value_states,
            )?
        } else {
            backend.prepare_full_attention_kernel_inputs(
                query_states.tensor(),
                key_states.tensor(),
                value_states.tensor(),
            )?
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let kv_len = key_states.dim(2)?;
        let prepared_attn_output = if let Some(handler) = external_full_attention.as_deref_mut() {
            self.external_attention_prepared_profiled(
                backend,
                handler,
                layer_id,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.dtype(),
                seqlen_offset,
                &mut profile,
            )?
        } else if use_full_attention_decode_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend
                .full_attention_decode(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.dtype(),
            )?
        } else if use_full_attention_prefill_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend
                .full_attention_prefill(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            if matches!(device.location(), DeviceLocation::Hip { .. })
                && debug_full_prefill_kernel_compare_enabled()
            {
                let key_states_ref =
                    repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
                let value_states_ref =
                    repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
                let query_states_f = query_states.to_dtype(DType::F32)?;
                let key_states_f = key_states_ref.to_dtype(DType::F32)?;
                let value_states_f = value_states_ref.to_dtype(DType::F32)?;
                let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
                let mut attn_weights = (query_states_f.matmul(&key_states_t)? * scale)?;
                if let Some(mask) = attention_mask {
                    attn_weights = attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?;
                }
                let attn_weights = ops::softmax_last_dim(&attn_weights)?;
                let fallback = attn_weights.matmul(&value_states_f)?;
                eprintln!(
                    "hip full-prefill layer={} q_len={} kv_len={} dtype={:?} max_delta={:.6}",
                    layer_id,
                    q_len,
                    kv_len,
                    query_states.dtype(),
                    max_abs_delta(&output.tensor(), &fallback)?,
                );
            }
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.dtype(),
            )?
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.dtype(),
                scale,
                &mut profile,
            )?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.dtype(),
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.dtype(),
                scale,
                &mut profile,
            )?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let (query_states_f, key_states_f, value_states_f) = backend
                .materialize_full_attention_dense_inputs(
                    &query_states,
                    &key_states,
                    &value_states,
                    self.num_kv_groups,
                )?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_prepared_profiled(
                    backend,
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    &gate,
                    b_sz,
                    q_len,
                    xs.dtype(),
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    attention_mask,
                    scale,
                    &gate,
                    b_sz,
                    q_len,
                    self.attention_size,
                    xs.dtype(),
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            }
        };

        let output_reshape_start = profile_start(device)?;
        let attn_output = prepared_attn_output;
        profile.full_attention_output_reshape_millis +=
            profile_elapsed(output_reshape_start, device)?;
        if external_full_attention.is_none() {
            self.kv_cache = appended_kv;
        } else {
            self.kv_cache = None;
        }
        let gate_start = profile_start(device)?;
        profile.full_attention_gate_millis += profile_elapsed(gate_start, device)?;
        let output_start = profile_start(device)?;
        let output = self.o_proj.forward_buffer(&attn_output)?.clone_tensor_as(xs.dtype())?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    pub(super) fn forward_profiled_with_external_buffer(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_id: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv_start = profile_start(device)?;
        let q_and_gate = self.q_proj.forward_buffer(xs)?;
        let k_proj = self.k_proj.forward_buffer(xs)?;
        let v_proj = self.v_proj.forward_buffer(xs)?;
        let (query_states, gate, key_states, value_states) = backend.prepare_full_attention_inputs(
            &q_and_gate,
            &k_proj,
            &v_proj,
            b_sz,
            q_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.weight(),
            self.k_norm.eps(),
        )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&query_states, &key_states, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let backend = backend_buffer_api::for_device(device);
        let kv_append_start = profile_start(device)?;
        let appended_kv = if external_full_attention.is_none() {
            Some(backend.append_full_attention_kv_buffers(
                self.kv_cache.as_ref().map(|(k, _)| k),
                self.kv_cache.as_ref().map(|(_, v)| v),
                key_states.tensor(),
                value_states.tensor(),
            )?)
        } else {
            None
        };
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) = if let Some((ref key_states, ref value_states)) = appended_kv {
            backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                &query_states,
                key_states,
                value_states,
            )?
        } else {
            backend.prepare_full_attention_kernel_inputs(
                query_states.tensor(),
                key_states.tensor(),
                value_states.tensor(),
            )?
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let kv_len = key_states.dim(2)?;
        let prepared_attn_output = if let Some(handler) = external_full_attention.as_deref_mut() {
            self.external_attention_prepared_profiled(
                backend,
                handler,
                layer_id,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                seqlen_offset,
                &mut profile,
            )?
        } else if use_full_attention_decode_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = self.full_attention_decode_projected(
                xs.tensor().dtype(),
                b_sz,
                q_len,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            output
        } else if use_full_attention_prefill_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend.full_attention_prefill(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.tensor().dtype(),
            )?
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let key_states = repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
            let value_states = repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;

            let query_states_f = query_states.to_dtype(DType::F32)?;
            let key_states_f = key_states.to_dtype(DType::F32)?;
            let value_states_f = value_states.to_dtype(DType::F32)?;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_prepared_profiled(
                    backend,
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    &gate,
                    b_sz,
                    q_len,
                    xs.tensor().dtype(),
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    attention_mask,
                    scale,
                    &gate,
                    b_sz,
                    q_len,
                    self.attention_size,
                    xs.tensor().dtype(),
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            }
        };

        let output_reshape_start = profile_start(device)?;
        let attn_output = prepared_attn_output;
        profile.full_attention_output_reshape_millis +=
            profile_elapsed(output_reshape_start, device)?;
        if external_full_attention.is_none() {
            self.kv_cache = appended_kv;
        } else {
            self.kv_cache = None;
        }
        let gate_start = profile_start(device)?;
        profile.full_attention_gate_millis += profile_elapsed(gate_start, device)?;
        let output_start = profile_start(device)?;
        let output = self.o_proj.forward_buffer(&attn_output)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    pub(super) fn trace_components_buffer(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(FullAttentionTrace, StateBuffer, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv_start = profile_start(device)?;
        let q_and_gate_output = self.q_proj.forward_buffer(xs)?;
        let k_proj_output = self.k_proj.forward_buffer(xs)?;
        let v_proj_output = self.v_proj.forward_buffer(xs)?;
        let (prepared_query, gate, prepared_key, prepared_value) = backend
            .prepare_full_attention_inputs(
                &q_and_gate_output,
                &k_proj_output,
                &v_proj_output,
                b_sz,
                q_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.weight(),
                self.q_norm.eps(),
                self.k_norm.weight(),
                self.k_norm.eps(),
            )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&prepared_query, &prepared_key, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let kv_append_start = profile_start(device)?;
        let appended_kv = backend.append_full_attention_kv_buffers(
            self.kv_cache.as_ref().map(|(k, _)| k),
            self.kv_cache.as_ref().map(|(_, v)| v),
            key_states.tensor(),
            prepared_value.tensor(),
        )?;
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) =
            backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                &query_states,
                &appended_kv.0,
                &appended_kv.1,
            )?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let prepared_attn_output = if use_full_attention_decode_megakernel(
            device,
            q_len,
            key_states.dim(2)?,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = self.full_attention_decode_projected(
                xs.tensor().dtype(),
                b_sz,
                q_len,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            output
        } else if use_full_attention_prefill_megakernel(
            device,
            q_len,
            key_states.dim(2)?,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend.full_attention_prefill(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.tensor().dtype(),
            )?
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let key_states = repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
            let value_states = repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;
            let query_states_f = query_states.to_dtype(DType::F32)?;
            let key_states_f = key_states.to_dtype(DType::F32)?;
            let value_states_f = value_states.to_dtype(DType::F32)?;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_prepared_profiled(
                    backend,
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    &gate,
                    b_sz,
                    q_len,
                    xs.tensor().dtype(),
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    attention_mask,
                    scale,
                    &gate,
                    b_sz,
                    q_len,
                    self.attention_size,
                    xs.tensor().dtype(),
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            }
        };

        self.kv_cache = Some(appended_kv);
        let output_start = profile_start(device)?;
        let output = self.o_proj.forward_buffer(&prepared_attn_output)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((
            FullAttentionTrace {
                q_and_gate_output,
                k_proj_output,
                v_proj_output,
                prepared_query,
                gate,
                prepared_key,
                prepared_value,
                attention_output: prepared_attn_output,
            },
            output,
            profile,
        ))
    }

    fn forward_profiled(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let mut no_external = None;
        self.forward_profiled_with_external(
            xs,
            attention_mask,
            seqlen_offset,
            usize::MAX,
            &mut no_external,
        )
    }

    fn full_attention_decode_projected(
        &self,
        output_dtype: DType,
        b_sz: usize,
        q_len: usize,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        let backend = backend_buffer_api::for_device(query_states.device());
        let output = backend.full_attention_decode(
            query_states,
            key_states,
            value_states,
            self.num_kv_groups,
            1f32 / f32::sqrt(self.head_dim as f32),
            seqlen_offset,
        )?;
        backend.prepare_full_attention_output_buffer(
            &output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn full_attention_decode_projected_buffer(
        &self,
        output_dtype: DType,
        b_sz: usize,
        q_len: usize,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
        gate: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        self.full_attention_decode_projected(
            output_dtype,
            b_sz,
            q_len,
            query_states.tensor(),
            key_states.tensor(),
            value_states.tensor(),
            gate,
            seqlen_offset,
        )
    }

    pub(super) fn direct_decode_context<'a>(
        &'a self,
        xs: &StateBuffer,
        seqlen_offset: usize,
        gate_workspace: &'a StateBuffer,
        query_workspace: &'a StateBuffer,
        key_workspace: &'a StateBuffer,
        value_workspace: &'a StateBuffer,
    ) -> Result<DirectFullAttentionDecodeContext<'a>> {
        let backend = backend_buffer_api::for_device(xs.device());
        let output_dtype = xs.dtype();
        let (b_sz, q_len, _) = xs.dims3()?;
        let prev_kv = self.kv_cache.as_ref();
        let gate_dims = gate_workspace.tensor().dims();
        let query_dims = query_workspace.tensor().dims();
        let key_dims = key_workspace.tensor().dims();
        let value_dims = value_workspace.tensor().dims();
        let expected_gate_dims = vec![1, 1, self.num_heads * self.head_dim];
        let expected_query_dims = vec![1, self.num_heads, 1, self.head_dim];
        let expected_kv_dims = vec![1, self.num_kv_heads, 1, self.head_dim];
        if gate_workspace.device().location() != xs.device().location()
            || query_workspace.device().location() != xs.device().location()
            || key_workspace.device().location() != xs.device().location()
            || value_workspace.device().location() != xs.device().location()
        {
            candle::bail!("direct-hip-v1 full-attention scratch must live on the hidden-state device");
        }
        if gate_workspace.dtype() != output_dtype
            || query_workspace.dtype() != output_dtype
            || key_workspace.dtype() != output_dtype
            || value_workspace.dtype() != output_dtype
        {
            candle::bail!(
                "direct-hip-v1 full-attention scratch dtype mismatch: hidden={:?} gate={:?} query={:?} key={:?} value={:?}",
                output_dtype,
                gate_workspace.dtype(),
                query_workspace.dtype(),
                key_workspace.dtype(),
                value_workspace.dtype(),
            );
        }
        if gate_dims != expected_gate_dims {
            candle::bail!(
                "direct-hip-v1 full-attention gate scratch dims mismatch: got {:?} expected {:?}",
                gate_dims,
                expected_gate_dims,
            );
        }
        if query_dims != expected_query_dims {
            candle::bail!(
                "direct-hip-v1 full-attention query scratch dims mismatch: got {:?} expected {:?}",
                query_dims,
                expected_query_dims,
            );
        }
        if key_dims != expected_kv_dims {
            candle::bail!(
                "direct-hip-v1 full-attention key scratch dims mismatch: got {:?} expected {:?}",
                key_dims,
                expected_kv_dims,
            );
        }
        if value_dims != expected_kv_dims {
            candle::bail!(
                "direct-hip-v1 full-attention value scratch dims mismatch: got {:?} expected {:?}",
                value_dims,
                expected_kv_dims,
            );
        }
        Ok(DirectFullAttentionDecodeContext {
            backend,
            output_dtype,
            b_sz,
            q_len,
            seqlen_offset,
            prev_k: prev_kv.map(|(k, _)| k),
            prev_v: prev_kv.map(|(_, v)| v),
            q_norm_weight: self.q_norm.weight(),
            q_norm_eps: self.q_norm.eps(),
            k_norm_weight: self.k_norm.weight(),
            k_norm_eps: self.k_norm.eps(),
            gate_workspace,
            query_workspace,
            key_workspace,
            value_workspace,
        })
    }

    pub(super) fn project_direct_decode_inputs_with_context(
        &self,
        xs: &StateBuffer,
        context: &DirectFullAttentionDecodeContext<'_>,
    ) -> Result<(
        StateBuffer,
        StateBuffer,
        StateBuffer,
        StateBuffer,
        StateBuffer,
        StateBuffer,
        RuntimeProfile,
    )> {
        let device = xs.device();
        let mut profile = RuntimeProfile::default();

        let qkv_start = profile_start(device)?;
        let _ = context.gate_workspace.tensor();
        let _ = context.query_workspace.tensor();
        let _ = context.key_workspace.tensor();
        let _ = context.value_workspace.tensor();
        let q_and_gate = self.q_proj.forward_buffer(xs)?;
        let k_proj = self.k_proj.forward_buffer(xs)?;
        let v_proj = self.v_proj.forward_buffer(xs)?;
        let (query_states, gate, key_states, value_states) = context
            .backend
            .prepare_full_attention_inputs_into_scratch(
                &q_and_gate,
                &k_proj,
                &v_proj,
                context.gate_workspace,
                context.query_workspace,
                context.key_workspace,
                context.value_workspace,
                context.b_sz,
                context.q_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                context.q_norm_weight,
                context.q_norm_eps,
                context.k_norm_weight,
                context.k_norm_eps,
            )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&query_states, &key_states, context.seqlen_offset)?;
        let query_states =
            context.backend.copy_state_into_scratch(&query_states, context.query_workspace)?;
        let key_states =
            context.backend.copy_state_into_scratch(&key_states, context.key_workspace)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let kv_append_start = profile_start(device)?;
        let appended_kv = context.backend.append_full_attention_kv_buffers(
            context.prev_k,
            context.prev_v,
            key_states.tensor(),
            value_states.tensor(),
        )?;
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) =
            context
                .backend
                .prepare_full_attention_kernel_input_buffers_with_buffer_kv(
                &query_states,
                &appended_kv.0,
                &appended_kv.1,
            )?;
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        Ok((
            query_states,
            key_states,
            value_states,
            gate,
            appended_kv.0,
            appended_kv.1,
            profile,
        ))
    }

    pub(super) fn run_direct_decode_core_with_context(
        &self,
        context: &DirectFullAttentionDecodeContext<'_>,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
        gate: &StateBuffer,
    ) -> Result<StateBuffer> {
        self.full_attention_decode_projected_buffer(
            context.output_dtype,
            context.b_sz,
            context.q_len,
            query_states,
            key_states,
            value_states,
            gate,
            context.seqlen_offset,
        )
    }

    pub(super) fn project_direct_decode_output(
        &self,
        attn_output: &StateBuffer,
    ) -> Result<StateBuffer> {
        self.o_proj.forward_buffer(attn_output)
    }

    pub(super) fn commit_direct_decode_kv_cache(
        &mut self,
        appended_k: StateBuffer,
        appended_v: StateBuffer,
    ) {
        self.kv_cache = Some((appended_k, appended_v));
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_profiled(xs, attention_mask, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub(super) fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}
