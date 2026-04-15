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
    /// Actual filled KV length (may be < tensor dim when using pre-allocated cache).
    /// None means use tensor dim as the length (standard path behavior).
    persistent_kv_filled: Option<usize>,
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
            persistent_kv_filled: self.persistent_kv_filled,
        }
    }

    pub(super) fn restore_cache_state(&mut self, state: &FullAttentionCacheState) {
        self.kv_cache = state.kv_cache.clone();
        self.persistent_kv_filled = state.persistent_kv_filled;
    }

    /// Output projection using the work-stealing megakernel when available,
    /// falling back to rocBLAS otherwise.
    fn o_proj_forward(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        #[cfg(feature = "qwen35-minimal-hip")]
        if let Ok((1, 1, _)) = xs.dims3() {
            if matches!(xs.device().location(), DeviceLocation::Hip { gpu_id: _ }) {
                if let Ok(result) = self.o_proj_megakernel(xs) {
                    return Ok(result);
                }
            }
        }
        self.o_proj.forward_buffer(xs)
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn o_proj_megakernel(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        use candle::Storage;
        use std::ffi::c_void;

        let xs = xs.contiguous()?;
        let ordinal = match xs.device().location() {
            DeviceLocation::Hip { gpu_id } => gpu_id,
            _ => candle::bail!("o_proj_megakernel: requires HIP"),
        };
        let dtype_code = super::hip::dtype_code(xs.tensor().dtype())
            .map_err(|e| candle::Error::Msg(format!("{e}")))?;

        let in_dim = xs.tensor().dim(2)?;
        let weight = self.o_proj.weight.contiguous()?;
        let out_dim = weight.dim(0)?;

        let out_tensor = Tensor::zeros((1, 1, out_dim), xs.tensor().dtype(), xs.device())?;

        let (xs_s, xs_l) = xs.tensor().storage_and_layout();
        let (w_s, w_l) = weight.storage_and_layout();
        let (o_s, o_l) = out_tensor.storage_and_layout();
        let (Storage::Hip(xs_hip), Storage::Hip(w_hip), Storage::Hip(o_hip)) =
            (&*xs_s, &*w_s, &*o_s)
        else {
            candle::bail!("o_proj_megakernel: tensors not on HIP");
        };

        let counter_ptr = super::decoder::megakernel_scratch::get_counter(ordinal)?;
        let status = unsafe {
            super::hip::ffi::dotcache_qwen35_hip_standalone_matvec(
                dtype_code, ordinal, in_dim, out_dim,
                xs_hip.raw_device_ptr_with_offset(xs_l.start_offset())? as *const c_void,
                w_hip.raw_device_ptr_with_offset(w_l.start_offset())? as *const c_void,
                o_hip.raw_device_ptr_with_offset(o_l.start_offset())? as *mut c_void,
                counter_ptr as *mut c_void,
            )
        };

        drop(xs_s); drop(w_s); drop(o_s);

        if status != 0 {
            candle::bail!("o_proj_megakernel: kernel failed with status {status}");
        }

        StateBuffer::from_tensor(out_tensor)
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
            persistent_kv_filled: None,
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
            persistent_kv_filled: None,
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
        let output = self.o_proj_forward(&attn_output)?.clone_tensor_as(xs.dtype())?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    pub(super) fn rotary_emb(&self) -> &RotaryEmbedding { &self.rotary_emb }

    /// Ensure the KV cache has room for position `seqlen_offset`.
    /// Pre-allocates in chunks of 256 to avoid per-token GPU allocation.
    #[cfg(feature = "qwen35-minimal-hip")]
    pub(super) fn ensure_kv_cache_capacity(
        &mut self, seqlen_offset: usize, device: &Device, dtype: DType,
    ) -> Result<()> {
        use candle::Tensor;
        const CHUNK: usize = 256;
        let needed = seqlen_offset + 1;
        if let Some((ref k_cache, ref v_cache)) = self.kv_cache {
            // KV cache is [batch, num_kv_heads, seq_len, head_dim] (4D) or [nkv, seq, hd] (3D)
            let (seq_dim, current_cap) = if k_cache.tensor().rank() == 4 {
                (2, k_cache.tensor().dim(2)?)
            } else {
                (1, k_cache.tensor().dim(1)?)
            };
            if current_cap >= needed { return Ok(()); }
            // Round up to next chunk boundary to amortize allocations
            let new_cap = ((needed + CHUNK - 1) / CHUNK) * CHUNK;
            let grow = new_cap - current_cap;
            // Create zero padding with same shape except seq_dim = grow
            let mut k_shape: Vec<usize> = k_cache.tensor().dims().to_vec();
            let mut v_shape: Vec<usize> = v_cache.tensor().dims().to_vec();
            k_shape[seq_dim] = grow;
            v_shape[seq_dim] = grow;
            let k_pad = Tensor::zeros(k_shape.as_slice(), dtype, device)?;
            let v_pad = Tensor::zeros(v_shape.as_slice(), dtype, device)?;
            let new_k = Tensor::cat(&[k_cache.tensor(), &k_pad], seq_dim)?;
            let new_v = Tensor::cat(&[v_cache.tensor(), &v_pad], seq_dim)?;
            let backend = super::backend_buffer_api::for_device(device);
            self.kv_cache = Some((
                backend.tensor_to_buffer(new_k)?,
                backend.tensor_to_buffer(new_v)?,
            ));
        }
        Ok(())
    }

    /// Record the filled KV length without narrowing the pre-allocated tensor.
    /// The kernel uses `kv_len` (filled) and `kv_max_t` (capacity) from the descriptor.
    #[cfg(feature = "qwen35-minimal-hip")]
    pub(super) fn trim_kv_cache_to(&mut self, len: usize) -> Result<()> {
        self.persistent_kv_filled = Some(len);
        Ok(())
    }

    /// Get the actual filled length of the KV cache (for sequence_length tracking).
    /// With pre-allocated cache, the tensor dim exceeds the filled length.
    #[cfg(feature = "qwen35-minimal-hip")]
    pub(super) fn kv_filled_len(&self) -> usize {
        if let Some(filled) = self.persistent_kv_filled {
            return filled;
        }
        if let Some((ref k_cache, _)) = self.kv_cache {
            let seq_dim = if k_cache.tensor().rank() == 4 { 2 } else { 1 };
            k_cache.tensor().dim(seq_dim).unwrap_or(0)
        } else {
            0
        }
    }

    /// Fill a DecodeLayerDesc with this layer's full attention pointers.
    #[cfg(feature = "qwen35-minimal-hip")]
    pub(super) fn fill_persistent_desc(
        &self,
        d: &mut super::DecodeLayerDesc,
        seqlen_offset: usize,
    ) -> Result<()> {
        use candle::Storage;
        use std::ffi::c_void;

        fn ptr(t: &Tensor) -> Result<*const c_void> {
            let t = t.contiguous()?;
            let (s, l) = t.storage_and_layout();
            let Storage::Hip(h) = &*s else { candle::bail!("not on HIP"); };
            Ok(h.raw_device_ptr_with_offset(l.start_offset())? as *const c_void)
        }

        d.q_proj_w = ptr(&self.q_proj.weight)?;
        d.q_out_dim = self.q_proj.weight.dim(0)? as i32;
        d.k_proj_w = ptr(&self.k_proj.weight)?;
        d.k_out_dim = self.k_proj.weight.dim(0)? as i32;
        d.v_proj_w = ptr(&self.v_proj.weight)?;
        d.o_proj_w = ptr(&self.o_proj.weight)?;
        d.attn_head_dim = self.head_dim as i32;
        d.attn_num_heads = self.num_heads as i32;
        d.attn_num_kv_heads = self.num_kv_heads as i32;
        d.q_norm_w = ptr(self.q_norm.weight())?;
        d.k_norm_w = ptr(self.k_norm.weight())?;
        d.q_norm_eps = self.q_norm.eps() as f32;
        d.k_norm_eps = self.k_norm.eps() as f32;
        d.kv_len = seqlen_offset as i32;

        if let Some((ref k_cache, ref v_cache)) = self.kv_cache {
            d.kv_cache_k = ptr(k_cache.tensor())? as *mut _;
            d.kv_cache_v = ptr(v_cache.tensor())? as *mut _;
            // Cache shape: [batch, num_kv_heads, seq_len, head_dim] or [nkv, seq, hd]
            let seq_dim = if k_cache.tensor().rank() == 4 { 2 } else { 1 };
            d.kv_max_t = k_cache.tensor().dim(seq_dim)? as i32;
        }
        Ok(())
    }

    /// Fused norm + Q/K/V projections using the megakernel.
    /// Returns (q_and_gate, k_proj, v_proj) or None if not available.
    pub(super) fn fused_norm_qkv_projections(
        &self,
        pre_norm_hidden: &StateBuffer,
        norm: &super::frontend::Qwen35RmsNorm,
    ) -> Result<Option<(StateBuffer, StateBuffer, StateBuffer)>> {
        #[cfg(not(feature = "qwen35-minimal-hip"))]
        { return Ok(None); }

        #[cfg(feature = "qwen35-minimal-hip")]
        {
            use candle::Storage;
            use std::ffi::c_void;

            let hidden = pre_norm_hidden.contiguous()?;
            let (b_sz, seq_len, hidden_dim) = hidden.dims3()?;
            if b_sz != 1 || seq_len != 1 { return Ok(None); }

            let ordinal = match hidden.device().location() {
                DeviceLocation::Hip { gpu_id } => gpu_id,
                _ => return Ok(None),
            };
            let dtype_code = match super::hip::dtype_code(hidden.tensor().dtype()) {
                Ok(c) => c,
                Err(_) => return Ok(None),
            };

            let norm_w = norm.weight().contiguous()?;
            let q_w = self.q_proj.weight.contiguous()?;
            let k_w = self.k_proj.weight.contiguous()?;
            let v_w = self.v_proj.weight.contiguous()?;

            let q_out_dim = q_w.dim(0)?;
            let k_out_dim = k_w.dim(0)?;
            let v_out_dim = v_w.dim(0)?;
            let total_rows = q_out_dim + k_out_dim + v_out_dim;

            let (q_storage, q_layout) = q_w.storage_and_layout();
            let (k_storage, k_layout) = k_w.storage_and_layout();
            let (v_storage, v_layout) = v_w.storage_and_layout();
            let (h_storage, h_layout) = hidden.tensor().storage_and_layout();
            let (nw_storage, nw_layout) = norm_w.storage_and_layout();

            let (
                Storage::Hip(q_s), Storage::Hip(k_s), Storage::Hip(v_s),
                Storage::Hip(h_s), Storage::Hip(nw_s),
            ) = (&*q_storage, &*k_storage, &*v_storage, &*h_storage, &*nw_storage)
            else { return Ok(None); };

            let proj_table = [
                super::ProjectionDesc {
                    weight: q_s.raw_device_ptr_with_offset(q_layout.start_offset())? as *const c_void,
                    out_dim: q_out_dim as i32, output_offset: 0,
                },
                super::ProjectionDesc {
                    weight: k_s.raw_device_ptr_with_offset(k_layout.start_offset())? as *const c_void,
                    out_dim: k_out_dim as i32, output_offset: q_out_dim as i32,
                },
                super::ProjectionDesc {
                    weight: v_s.raw_device_ptr_with_offset(v_layout.start_offset())? as *const c_void,
                    out_dim: v_out_dim as i32, output_offset: (q_out_dim + k_out_dim) as i32,
                },
            ];

            let table_bytes = std::mem::size_of_val(&proj_table);
            let table_ptr = super::decoder::megakernel_scratch::get_proj_table(ordinal, table_bytes)?;
            if let Err(e) = super::hip::copy_host_to_device(
                ordinal, table_ptr, proj_table.as_ptr() as *const c_void, table_bytes,
            ) {
                return Err(e.into());
            }

            // Allocate F32 output tensor on device (stays on GPU — no D2H/H2D)
            let dev = hidden.device();
            let output_tensor = Tensor::zeros(total_rows, DType::F32, dev)?;
            let (out_storage, out_layout) = output_tensor.storage_and_layout();
            let Storage::Hip(out_s) = &*out_storage else { return Ok(None); };
            let output_device_ptr =
                out_s.raw_device_ptr_with_offset(out_layout.start_offset())? as *mut c_void;

            let counter_ptr = super::decoder::megakernel_scratch::get_counter(ordinal)?;

            let status = unsafe {
                super::hip::ffi::dotcache_qwen35_hip_norm_multi_proj(
                    dtype_code, ordinal, hidden_dim, total_rows, norm.eps() as f32,
                    h_s.raw_device_ptr_with_offset(h_layout.start_offset())? as *const c_void,
                    nw_s.raw_device_ptr_with_offset(nw_layout.start_offset())? as *const c_void,
                    table_ptr as *const c_void, 3,
                    output_device_ptr,
                    counter_ptr as *mut c_void,
                )
            };

            // table_ptr and counter_ptr are cached — not freed here

            if status != 0 {
                candle::bail!("fused_norm_qkv: kernel failed with status {status}");
            }

            // Drop storage refs before narrowing
            drop(out_storage);

            // Split F32 tensor on device and cast to hidden dtype
            let dt = hidden.tensor().dtype();
            let q = output_tensor.narrow(0, 0, q_out_dim)?
                .reshape((1, 1, q_out_dim))?.to_dtype(dt)?;
            let k = output_tensor.narrow(0, q_out_dim, k_out_dim)?
                .reshape((1, 1, k_out_dim))?.to_dtype(dt)?;
            let v = output_tensor.narrow(0, q_out_dim + k_out_dim, v_out_dim)?
                .reshape((1, 1, v_out_dim))?.to_dtype(dt)?;

            Ok(Some((
                StateBuffer::from_tensor(q)?,
                StateBuffer::from_tensor(k)?,
                StateBuffer::from_tensor(v)?,
            )))
        }
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
        let output = self.o_proj_forward(&attn_output)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    /// Forward pass using pre-computed Q/K/V projections from fused megakernel.
    /// The projections are expected to have the same shapes as q_proj/k_proj/v_proj outputs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_from_projections_buffer(
        &mut self,
        q_and_gate: &StateBuffer,
        k_proj_out: &StateBuffer,
        v_proj_out: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        _layer_id: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
        hidden_dtype: DType,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = q_and_gate.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = q_and_gate.dims3()?;

        let qkv_start = profile_start(device)?;
        let (query_states, gate, key_states, value_states) = backend.prepare_full_attention_inputs(
            q_and_gate,
            k_proj_out,
            v_proj_out,
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
        let (query_states, key_states, value_states) =
            if let Some((ref key_states, ref value_states)) = appended_kv {
                backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                    &query_states, key_states, value_states,
                )?
            } else {
                backend.prepare_full_attention_kernel_inputs(
                    query_states.tensor(), key_states.tensor(), value_states.tensor(),
                )?
            };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let kv_len = key_states.dim(2)?;
        let prepared_attn_output =
            if use_full_attention_decode_megakernel(device, q_len, kv_len, self.head_dim, seqlen_offset)
            {
                let kernel_start = profile_start(device)?;
                let output = self.full_attention_decode_projected(
                    hidden_dtype, b_sz, q_len,
                    &query_states, &key_states, &value_states, &gate, seqlen_offset,
                )?;
                profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
                output
            } else if use_full_attention_prefill_megakernel(device, q_len, kv_len, self.head_dim, seqlen_offset)
            {
                let kernel_start = profile_start(device)?;
                let output = backend.full_attention_prefill(
                    &query_states, &key_states, &value_states,
                    self.num_kv_groups, scale as f32, seqlen_offset,
                )?;
                profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
                backend.prepare_full_attention_output_buffer(
                    &output, &gate, b_sz, q_len, self.attention_size, hidden_dtype,
                )?
            } else {
                let kv_materialize_start = profile_start(device)?;
                let key_states = repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
                let value_states = repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
                profile.layout_prepare_millis += profile_elapsed(kv_materialize_start, device)?;

                let query_states_f = query_states.to_dtype(DType::F32)?;
                let key_states_f = key_states.to_dtype(DType::F32)?;
                let value_states_f = value_states.to_dtype(DType::F32)?;
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f, &key_states_f, &value_states_f,
                    attention_mask, scale, &gate,
                    b_sz, q_len, self.attention_size, hidden_dtype,
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            };

        if external_full_attention.is_none() {
            self.kv_cache = appended_kv;
        }
        let output_start = profile_start(device)?;
        let output = self.o_proj_forward(&prepared_attn_output)?;
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
        let output = self.o_proj_forward(&prepared_attn_output)?;
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
        self.persistent_kv_filled = None;
    }
}
