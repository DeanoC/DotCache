use super::backend_buffer_api;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
use super::frontend::{
    build_prepared_linear_source_no_bias, deferred_in_proj_qkv_enabled, prepared_linear_no_bias,
    profile_elapsed, profile_start, repeat_heads, LinearSource, Qwen35RmsNormGated,
};
use super::hip_wrappers::{
    linear_attention_chunk_size, linear_attention_compute_dtype,
    use_delta_chunk_fused_kernel, use_delta_chunk_step_kernel, use_delta_full_scan_kernel,
    use_delta_recurrent_prefill_kernel, use_delta_state_kernel, use_delta_state_scan_kernel,
    use_hip_exact_multi_chunk_full_scan_prefill,
};
use super::model::{
    delta_chunk_step_raw, delta_chunk_step_raw_host_buffer, delta_chunk_step_windowed_raw,
    delta_chunk_step_windowed_raw_host_buffer, delta_net_compute_dtype,
    delta_net_execution_policy, use_delta_chunk_scan_kernel, use_delta_chunk_windowed_kernel,
    use_hip_chunk_single_prefill_kernel, use_hip_combined_linear_decode,
    use_hip_combined_linear_prefill, use_hip_multi_chunk_scan_prefill_kernel,
    use_hip_short_linear_prefill_recurrent, use_linear_prefill_packed_kernel, DeltaNetScanMode,
};
use super::prepared::PreparedTensorSource;
use super::types::{
    LinearAttentionCacheState, LinearAttentionCoreTrace, LinearAttentionProjectionTrace,
    RuntimeProfile, StateBuffer, TextConfig,
};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::linear_no_bias;
use super::with_tracing::Linear;
use candle::{DType, Device, DeviceLocation, IndexOp, Module, Result, Tensor, D};
use candle_core as candle;

#[derive(Debug, Clone)]
pub(crate) struct GatedDeltaNet {
    in_proj_qkv: LinearSource,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d_raw_weight: Option<Tensor>,
    conv1d_weight_squeezed: Option<Tensor>,
    dt_bias_raw: Option<Tensor>,
    a_log_raw: Option<Tensor>,
    dt_bias_prepared: Option<Tensor>,
    a_log_exp_prepared: Option<Tensor>,
    norm: Qwen35RmsNormGated,
    out_proj: Linear,
    pub(super) num_v_heads: usize,
    pub(super) num_k_heads: usize,
    pub(super) head_k_dim: usize,
    pub(super) head_v_dim: usize,
    pub(super) key_dim: usize,
    pub(super) value_dim: usize,
    pub(super) conv_kernel_size: usize,
    conv_state: Option<StateBuffer>,
    pub(super) recurrent_state: Option<StateBuffer>,
    chunk_cache: Option<LinearChunkCache>,
    value_cache: Option<LinearValueCache>,
}

#[derive(Debug, Clone)]
struct LinearChunkCache {
    chunk_size: usize,
    dtype: DType,
    device_location: DeviceLocation,
    lower: Tensor,
    eye: Tensor,
    strict_lower: Tensor,
    lower_2d: Tensor,
}

#[derive(Debug, Clone)]
struct LinearValueCache {
    dtype: DType,
    device_location: DeviceLocation,
    dt_bias: Tensor,
    a_log_exp: Tensor,
}

impl GatedDeltaNet {
    fn finalize_linear_output_buffer(
        &self,
        hidden_dtype: DType,
        batch_size: usize,
        seq_len: usize,
        z: &StateBuffer,
        core_attn_out: &Tensor,
    ) -> Result<StateBuffer> {
        let backend = backend_buffer_api::for_device(z.device());
        let core_attn_out = self
            .norm
            .forward_buffer(
                &backend.reshape_tensor_to_buffer(
                    core_attn_out,
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
                &backend.reshape_tensor_to_buffer(
                    z.tensor(),
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
            )?
            .tensor()
            .reshape((batch_size, seq_len, self.value_dim))?;
        let core_attn_out = if core_attn_out.dtype() == hidden_dtype {
            core_attn_out
        } else {
            core_attn_out.to_dtype(hidden_dtype)?
        };
        self.out_proj.forward_buffer(&backend.reshape_tensor_to_buffer(
            &core_attn_out,
            &[batch_size, seq_len, self.value_dim],
        )?)
    }

    pub(super) fn cache_state(&self) -> LinearAttentionCacheState {
        LinearAttentionCacheState {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }

    pub(super) fn restore_cache_state(&mut self, state: &LinearAttentionCacheState) {
        self.conv_state = state.conv_state.clone();
        self.recurrent_state = state.recurrent_state.clone();
    }

    pub(super) fn deferred_linear_count(&self) -> usize {
        usize::from(self.in_proj_qkv.is_deferred())
    }

#[cfg(any(feature = "hf", test))]
    pub(super) fn new(cfg: &TextConfig, vb: WeightBuilder) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d_raw_weight =
            vb.pp("conv1d")
                .get((conv_dim, 1, cfg.linear_conv_kernel_dim), "weight")?;
        Ok(Self {
            in_proj_qkv: LinearSource::Materialized(linear_no_bias(
                cfg.hidden_size,
                conv_dim,
                vb.pp("in_proj_qkv"),
            )?),
            in_proj_z: linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?,
            in_proj_a: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?,
            conv1d_raw_weight: Some(conv1d_raw_weight),
            conv1d_weight_squeezed: None,
            dt_bias_raw: Some(vb.get(cfg.linear_num_value_heads, "dt_bias")?),
            a_log_raw: Some(vb.get(cfg.linear_num_value_heads, "A_log")?),
            dt_bias_prepared: None,
            a_log_exp_prepared: None,
            norm: Qwen35RmsNormGated::new(
                cfg.linear_value_head_dim,
                cfg.rms_norm_eps,
                vb.pp("norm"),
            )?,
            out_proj: linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: None,
            recurrent_state: None,
            chunk_cache: None,
            value_cache: None,
        })
    }

    pub(super) fn from_prepared(cfg: &TextConfig, source: &PreparedTensorSource) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let dt_bias_prepared = source
            .contains_tensor("dt_bias.__dotcache_head_bias_reshaped")
            .then(|| source.get("dt_bias.__dotcache_head_bias_reshaped"))
            .transpose()?;
        let a_log_exp_prepared = source
            .contains_tensor("A_log.__dotcache_head_exp_reshaped")
            .then(|| source.get("A_log.__dotcache_head_exp_reshaped"))
            .transpose()?;
        let conv1d_weight_squeezed = source
            .contains_tensor("conv1d.weight.__dotcache_depthwise_squeezed")
            .then(|| source.get("conv1d.weight.__dotcache_depthwise_squeezed"))
            .transpose()?;
        Ok(Self {
            in_proj_qkv: build_prepared_linear_source_no_bias(
                &source.pp("in_proj_qkv"),
                cfg.hidden_size,
                key_dim * 2 + value_dim,
                deferred_in_proj_qkv_enabled(),
            )?,
            in_proj_z: prepared_linear_no_bias(&source.pp("in_proj_z"))?,
            in_proj_b: prepared_linear_no_bias(&source.pp("in_proj_b"))?,
            in_proj_a: prepared_linear_no_bias(&source.pp("in_proj_a"))?,
            conv1d_raw_weight: if conv1d_weight_squeezed.is_some() {
                None
            } else {
                Some(source.get("conv1d.weight")?)
            },
            conv1d_weight_squeezed,
            dt_bias_raw: if dt_bias_prepared.is_some() {
                None
            } else {
                Some(source.get("dt_bias")?)
            },
            a_log_raw: if a_log_exp_prepared.is_some() {
                None
            } else {
                Some(source.get("A_log")?)
            },
            dt_bias_prepared,
            a_log_exp_prepared,
            norm: Qwen35RmsNormGated::from_prepared(cfg.rms_norm_eps, &source.pp("norm"))?,
            out_proj: prepared_linear_no_bias(&source.pp("out_proj"))?,
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: None,
            recurrent_state: None,
            chunk_cache: None,
            value_cache: None,
        })
    }

    fn value_cache(&mut self, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let device_location = device.location();
        let rebuild = self
            .value_cache
            .as_ref()
            .map(|cache| cache.dtype != dtype || cache.device_location != device_location)
            .unwrap_or(true);
        if rebuild {
            let dt_bias_base = if let Some(dt_bias) = &self.dt_bias_prepared {
                dt_bias.clone()
            } else {
                let dt_bias = self.dt_bias_raw.as_ref().ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both prepared and raw dt_bias tensor".into(),
                    )
                })?;
                if dt_bias.dtype() == dtype {
                    dt_bias.clone()
                } else {
                    dt_bias.to_dtype(dtype)?
                }
            };
            let dt_bias = if dt_bias_base.rank() == 3 {
                if dt_bias_base.dtype() == dtype {
                    dt_bias_base
                } else {
                    dt_bias_base.to_dtype(dtype)?
                }
            } else {
                let dt_bias = if dt_bias_base.dtype() == dtype {
                    dt_bias_base
                } else {
                    dt_bias_base.to_dtype(dtype)?
                };
                dt_bias.reshape((1, 1, self.num_v_heads))?
            };
            let a_log_exp = if let Some(prepared) = &self.a_log_exp_prepared {
                if prepared.dtype() == dtype {
                    prepared.clone()
                } else {
                    prepared.to_dtype(dtype)?
                }
            } else {
                let a_log = self.a_log_raw.as_ref().ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both prepared and raw A_log tensor".into(),
                    )
                })?;
                let a_log = if a_log.dtype() == dtype {
                    a_log.clone()
                } else {
                    a_log.to_dtype(dtype)?
                };
                a_log.exp()?.reshape((1, 1, self.num_v_heads))?
            };
            self.value_cache = Some(LinearValueCache {
                dtype,
                device_location,
                dt_bias,
                a_log_exp,
            });
        }
        let cache = self
            .value_cache
            .as_ref()
            .expect("linear value cache must be initialized");
        Ok((cache.dt_bias.clone(), cache.a_log_exp.clone()))
    }

    fn conv1d_weight_squeezed(&self) -> Result<Tensor> {
        if let Some(weight) = &self.conv1d_weight_squeezed {
            Ok(weight.clone())
        } else {
            self.conv1d_raw_weight
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both squeezed and raw conv1d weight".into(),
                    )
                })?
                .squeeze(1)
        }
    }

    fn chunk_cache(
        &mut self,
        device: &Device,
        dtype: DType,
        chunk_size: usize,
    ) -> Result<LinearChunkCache> {
        let device_location = device.location();
        if let Some(cache) = &self.chunk_cache {
            if cache.chunk_size == chunk_size
                && cache.dtype == dtype
                && cache.device_location == device_location
            {
                return Ok(cache.clone());
            }
        }

        let lower =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let eye =
            Tensor::eye(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let strict_lower = lower.broadcast_sub(&eye)?;
        let lower_2d =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, chunk_size, chunk_size))?;
        let cache = LinearChunkCache {
            chunk_size,
            dtype,
            device_location,
            lower,
            eye,
            strict_lower,
            lower_2d,
        };
        self.chunk_cache = Some(cache.clone());
        Ok(cache)
    }

    fn chunk_gated_delta_rule_torch_like(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        _sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = 64usize;
        let compute_dtype = DType::F32;
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }
        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let prepare_start = profile_start(device)?;
        let batch_heads = batch_size * num_heads;
        let k_beta_start = profile_start(device)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let query = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let k_beta = k_beta.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let v_beta = v_beta.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        profile.linear_chunk_prepare_k_beta_millis += profile_elapsed(k_beta_start, device)?;
        let g_start = profile_start(device)?;
        let backend = backend_buffer_api::for_device(g.device());
        let g = {
            let g = g.reshape((batch_heads, num_chunks, chunk_size))?;
            if g.device().is_hip() {
                backend
                    .cumsum_last_dim(&backend.tensor_to_buffer(g.clone())?)?
                    .clone_tensor()
            } else {
                g.cumsum(D::Minus1)?
            }
        };
        profile.linear_chunk_prepare_g_millis += profile_elapsed(g_start, device)?;
        let cache_start = profile_start(device)?;
        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower_2d = cache.lower_2d.reshape((1, chunk_size, chunk_size))?;
        let eye_2d = Tensor::eye(chunk_size, compute_dtype, query.device())?
            .reshape((1, chunk_size, chunk_size))?;
        let strict_lower_2d = lower_2d.broadcast_sub(&eye_2d)?;
        profile.linear_chunk_prepare_cache_millis += profile_elapsed(cache_start, device)?;
        let base_attn_start = profile_start(device)?;
        let decay_deltas = g
            .unsqueeze(3)?
            .broadcast_sub(&g.unsqueeze(2)?)?
            .broadcast_mul(&lower_2d)?;
        let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower_2d)?;
        let exp_g = g.exp()?;
        profile.linear_chunk_prepare_base_attn_millis += profile_elapsed(base_attn_start, device)?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let solve_batch = batch_heads * num_chunks;
        let key_t = key.transpose(3, 2)?.contiguous()?;
        let base_attn = k_beta
            .matmul(&key_t)?
            .broadcast_mul(&decay_mask)?
            .neg()?
            .broadcast_mul(&strict_lower_2d)?
            .reshape((solve_batch, chunk_size, chunk_size))?;
        let mut rows = Vec::with_capacity(chunk_size);
        rows.push(Tensor::zeros(
            (solve_batch, 1, chunk_size),
            compute_dtype,
            query.device(),
        )?);
        for i in 1..chunk_size {
            let row = base_attn
                .narrow(1, i, 1)?
                .narrow(2, 0, i)?
                .reshape((solve_batch, i))?;
            let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
            let correction = row
                .unsqueeze(1)?
                .broadcast_mul(&sub)?
                .sum(1)?
                .reshape((solve_batch, i))?;
            let row = row.broadcast_add(&correction)?;
            let row =
                row.pad_with_zeros(1, 0, chunk_size - i)?
                    .reshape((solve_batch, 1, chunk_size))?;
            rows.push(row);
        }
        let attn = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
            .broadcast_add(&eye_2d)?
            .reshape((batch_heads, num_chunks, chunk_size, chunk_size))?;
        let solved_value = attn.matmul(&v_beta)?;
        let weighted_k = k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?;
        let attn_flat = attn
            .reshape((solve_batch, chunk_size, chunk_size))?
            .contiguous()?;
        let weighted_k_flat = weighted_k
            .reshape((solve_batch, chunk_size, k_head_dim))?
            .contiguous()?;
        let k_cumdecay = attn_flat.matmul(&weighted_k_flat)?.reshape((
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
        ))?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let scan_start = profile_start(device)?;
        let mut last_recurrent_state = backend_buffer_api::for_device(query.device()).zeros_tensor(
            query.device(),
            compute_dtype,
            &[batch_heads, k_head_dim, v_head_dim],
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let q_i = query.i((.., chunk_idx, .., ..))?;
            let k_i = key.i((.., chunk_idx, .., ..))?;
            let v_i = solved_value.i((.., chunk_idx, .., ..))?;
            let g_i = g.i((.., chunk_idx, ..))?;
            let decay_i = decay_mask.i((.., chunk_idx, .., ..))?;

            let recurrent_read_start = profile_start(device)?;
            let v_prime = k_cumdecay
                .i((.., chunk_idx, .., ..))?
                .matmul(&last_recurrent_state)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.exp()?.unsqueeze(D::Minus1)?)?
                .matmul(&last_recurrent_state)?;
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let v_new = v_i.broadcast_sub(&v_prime)?;

            let local_attn_start = profile_start(device)?;
            let k_i_t = k_i.transpose(2, 1)?.contiguous()?;
            let local_attn = q_i
                .matmul(&k_i_t)?
                .broadcast_mul(&decay_i)?
                .broadcast_mul(&lower_2d)?;
            let local_out = local_attn.matmul(&v_new)?;
            outputs.push(attn_inter.broadcast_add(&local_out)?.unsqueeze(1)?);
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let state_update_start = profile_start(device)?;
            let g_last = g_i.i((.., chunk_size - 1))?;
            let state_decay = g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let chunk_decay = g_last
                .unsqueeze(D::Minus1)?
                .broadcast_sub(&g_i)?
                .exp()?
                .unsqueeze(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(&state_decay)?
                .broadcast_add(
                    &k_i.broadcast_mul(&chunk_decay)?
                        .transpose(2, 1)?
                        .matmul(&v_new)?,
                )?;
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state, profile))
    }

    fn apply_mask_to_padding_states(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match attention_mask {
            Some(mask) if mask.dim(1)? > 1 && mask.dim(0)? > 1 => hidden_states
                .broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(hidden_states.dtype())?),
            None => Ok(hidden_states.clone()),
            Some(_) => Ok(hidden_states.clone()),
        }
    }

    fn prepare_depthwise_conv_input(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let backend = backend_buffer_api::for_device(mixed_qkv.device());
        let (mixed_qkv, next_state) =
            backend.prepare_depthwise_conv_input(self.conv_state.as_ref(), mixed_qkv, self.conv_kernel_size)?;
        self.conv_state = next_state;
        Ok(mixed_qkv)
    }

    fn update_depthwise_conv_state_from_raw(&mut self, mixed_qkv: &Tensor) -> Result<()> {
        let backend = backend_buffer_api::for_device(mixed_qkv.device());
        self.conv_state =
            backend.update_depthwise_conv_state(self.conv_state.as_ref(), mixed_qkv, self.conv_kernel_size)?;
        Ok(())
    }

    fn depthwise_conv_from_state(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let kernel = self.conv_kernel_size;
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?;
        let weights = self.conv1d_weight_squeezed()?;
        let mut output: Option<Tensor> = None;
        for tap in 0..kernel {
            let xs = mixed_qkv.narrow(2, tap, seq_len)?;
            let w = weights.i((.., tap))?.reshape((1, self.conv_dim(), 1))?;
            let contrib = xs.broadcast_mul(&w)?;
            output = Some(match output {
                Some(acc) => acc.broadcast_add(&contrib)?,
                None => contrib,
            });
        }
        output
            .expect("depthwise conv produced at least one tap")
            .silu()
    }

    fn run_depthwise_conv(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        if mixed_qkv.device().is_hip() {
            return self
                .run_depthwise_conv_packed_prefill(mixed_qkv)?
                .transpose(1, 2);
        }
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_update(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        if mixed_qkv.device().is_hip() {
            return self
                .run_depthwise_conv_materialized_pack(mixed_qkv)?
                .transpose(1, 2);
        }
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_materialized_pack(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?.contiguous()?;
        let weights = self.conv1d_weight_squeezed()?.contiguous()?;
        backend_buffer_api::for_device(mixed_qkv.device())
            .linear_prefill_conv(&mixed_qkv, &weights, seq_len, self.conv_kernel_size)
    }

    fn run_depthwise_conv_packed_prefill(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let weights = self.conv1d_weight_squeezed()?.contiguous()?;
        if mixed_qkv.device().is_hip() {
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(mixed_qkv.dtype())?,
                None => backend_buffer_api::for_device(mixed_qkv.device()).zeros_tensor(
                    mixed_qkv.device(),
                    mixed_qkv.dtype(),
                    &[mixed_qkv.dim(0)?, mixed_qkv.dim(1)?, state_len],
                )?,
            };
            let output = backend_buffer_api::for_device(mixed_qkv.device()).linear_stateful_conv(
                &mixed_qkv.contiguous()?,
                &prev_state,
                &weights,
                self.conv_kernel_size,
            )?;
            self.update_depthwise_conv_state_from_raw(mixed_qkv)?;
            return Ok(output);
        }

        self.run_depthwise_conv_materialized_pack(mixed_qkv)
    }

    pub(super) fn conv_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim
    }

    fn recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let compute_dtype = initial_dtype;
        let query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let mut recurrent_state = match initial_state {
            Some(state) => state.to_dtype(compute_dtype)?,
            None => backend_buffer_api::for_device(query.device()).zeros_tensor(
                query.device(),
                compute_dtype,
                &[batch_size, num_heads, k_head_dim, v_head_dim],
            )?,
        };

        if device.is_hip() && k_head_dim <= 256 {
            let batch_heads = batch_size * num_heads;
            let pack_start = profile_start(device)?;
            let initial_state = backend_buffer_api::for_device(device).reshape_tensor_to_buffer(
                &recurrent_state.contiguous()?,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let query_scan = query
                .reshape((batch_heads, seq_len, k_head_dim))?
                .contiguous()?;
            let key_scan = key
                .reshape((batch_heads, seq_len, k_head_dim))?
                .contiguous()?;
            let value_scan = value
                .reshape((batch_heads, seq_len, v_head_dim))?
                .contiguous()?;
            let beta_scan = beta.reshape((batch_heads, seq_len))?.contiguous()?;
            let g_scan = g.reshape((batch_heads, seq_len))?.contiguous()?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_recurrent_prefill(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let output = fused
                .tensor()
                .narrow(1, 0, seq_len)?
                .reshape((batch_size, num_heads, seq_len, v_head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            let recurrent_state = fused
                .tensor()
                .narrow(1, seq_len, k_head_dim)?
                .reshape((batch_size, num_heads, k_head_dim, v_head_dim))?
                .contiguous()?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_recurrent_loop_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, recurrent_state, profile));
        }

        let mut outputs = Vec::with_capacity(seq_len);
        let loop_start = profile_start(device)?;
        for step in 0..seq_len {
            let q_t = query.i((.., .., step, ..))?.contiguous()?;
            let k_t = key.i((.., .., step, ..))?.contiguous()?;
            let v_t = value.i((.., .., step, ..))?.contiguous()?;
            let beta_t = beta.i((.., .., step))?.unsqueeze(D::Minus1)?;
            let g_t = g
                .i((.., .., step))?
                .exp()?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?;

            recurrent_state = recurrent_state.broadcast_mul(&g_t)?;
            let kv_mem = recurrent_state
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            let delta = (v_t.broadcast_sub(&kv_mem)?).broadcast_mul(&beta_t)?;
            recurrent_state = recurrent_state.broadcast_add(
                &k_t.unsqueeze(D::Minus1)?
                    .broadcast_mul(&delta.unsqueeze(2)?)?,
            )?;
            let out_t = recurrent_state
                .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            outputs.push(out_t.unsqueeze(2)?);
        }
        profile.linear_recurrent_loop_millis += profile_elapsed(loop_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, recurrent_state, profile))
    }

    fn chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = linear_attention_chunk_size(query.device(), sequence_length);
        let estimated_pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let estimated_num_chunks = (sequence_length + estimated_pad_size) / chunk_size;
        let scan_policy =
            delta_net_execution_policy(query.device(), sequence_length, estimated_num_chunks);
        let scan_mode = scan_policy.scan_mode;
        if scan_mode == DeltaNetScanMode::TorchLike {
            return self.chunk_gated_delta_rule_torch_like(
                query,
                key,
                value,
                g,
                beta,
                sequence_length,
            );
        }
        let compute_dtype = delta_net_compute_dtype(scan_mode, initial_dtype);
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;

        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }

        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        if use_delta_recurrent_prefill_kernel(query.device(), sequence_length) {
            let batch_heads = batch_size * num_heads;
            let pack_start = profile_start(device)?;
            let query_scan = query
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let key_scan = key
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let value_scan = value
                .reshape((batch_heads, total_sequence_length, v_head_dim))?
                .contiguous()?;
            let beta_scan = beta
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let g_scan = g
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let backend = backend_buffer_api::for_device(device);
            let kernel_start = profile_start(device)?;
            let fused = backend.delta_recurrent_prefill(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), recurrent_state.clone_tensor(), profile));
        }

        let query = query.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let value = value.reshape((batch_size, num_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta = beta.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let g_raw = g.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let batch_heads = batch_size * num_heads;
        let query_scan = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key_scan = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let value_scan = value.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta_scan = beta.reshape((batch_heads, num_chunks, chunk_size))?;
        let g_scan = g_raw.reshape((batch_heads, num_chunks, chunk_size))?;

        if use_hip_chunk_single_prefill_kernel(
            query.device(),
            sequence_length,
            num_chunks,
            chunk_size,
        ) {
            let pack_start = profile_start(device)?;
            let query_i = query_scan.i((.., 0, .., ..))?.contiguous()?;
            let key_i = key_scan.i((.., 0, .., ..))?.contiguous()?;
            let value_i = value_scan.i((.., 0, .., ..))?.contiguous()?;
            let beta_i = beta_scan.i((.., 0, ..))?.contiguous()?;
            let g_i = g_scan.i((.., 0, ..))?.contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_chunk_single_prefill(
                &initial_state,
                &query_i,
                &key_i,
                &value_i,
                &beta_i,
                &g_i,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        if use_delta_chunk_scan_kernel(query.device(), scan_mode, sequence_length, chunk_size)
            || use_hip_multi_chunk_scan_prefill_kernel(
                query.device(),
                sequence_length,
                num_chunks,
                chunk_size,
            )
        {
            let pack_start = profile_start(device)?;
            let query_scan = query_scan.contiguous()?;
            let key_scan = key_scan.contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let beta_scan = beta_scan.contiguous()?;
            let g_scan = g_scan.contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_chunk_scan_raw(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        if use_delta_chunk_step_kernel(query.device(), scan_mode, sequence_length, chunk_size) {
            if use_delta_chunk_windowed_kernel(
                query.device(),
                scan_mode,
                sequence_length,
                chunk_size,
            ) {
                let pack_start = profile_start(device)?;
                let initial_state = backend_buffer_api::for_device(query.device()).zeros_tensor(
                    query.device(),
                    compute_dtype,
                    &[batch_heads, k_head_dim, v_head_dim],
                )?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let scan_start = profile_start(device)?;
                let (output, last_recurrent_state) = if let Some((bytes, shape)) =
                    delta_chunk_step_windowed_raw_host_buffer(
                        &initial_state,
                        &query_scan,
                        &key_scan,
                        &value_scan,
                        &beta_scan,
                        &g_scan,
                    )?
                {
                    let fused = crate::backends::hip::state_buffer_from_host_bytes(
                        bytes,
                        shape,
                        compute_dtype,
                        query.device(),
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(scan_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let backend = backend_buffer_api::for_device(device);
                    let (output, last_recurrent_state) =
                        backend.unpack_scan_fused_output_and_state(
                            &fused,
                            total_sequence_length,
                            sequence_length,
                            batch_size,
                            num_heads,
                            v_head_dim,
                            k_head_dim,
                            initial_dtype,
                        )?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                    (output.clone_tensor(), last_recurrent_state.clone_tensor())
                } else {
                    let fused = delta_chunk_step_windowed_raw(
                        &initial_state,
                        &query_scan,
                        &key_scan,
                        &value_scan,
                        &beta_scan,
                        &g_scan,
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(scan_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let output = fused
                        .narrow(1, 0, total_sequence_length)?
                        .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                        .narrow(2, 0, sequence_length)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(initial_dtype)?;
                    let last_recurrent_state = fused
                        .narrow(1, total_sequence_length, k_head_dim)?
                        .reshape((batch_heads, k_head_dim, v_head_dim))?
                        .contiguous()?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                    (output, last_recurrent_state)
                };
                profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                    + profile.linear_full_kernel_execute_millis
                    + profile.linear_full_kernel_unpack_millis;
                profile.linear_attention_millis += profile_elapsed(total_start, device)?;
                return Ok((output, last_recurrent_state, profile));
            }

            let mut last_recurrent_state =
                backend_buffer_api::for_device(query.device()).zeros_tensor(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let mut outputs = Vec::with_capacity(num_chunks);
            let scan_start = profile_start(device)?;
            for chunk_idx in 0..num_chunks {
                let pack_start = profile_start(device)?;
                let q_i = query_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let k_i = key_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let v_i = value_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let beta_i = beta_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let g_i = g_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let prev_state_i = last_recurrent_state.contiguous()?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let kernel_start = profile_start(device)?;
                if let Some((bytes, shape)) = delta_chunk_step_raw_host_buffer(
                    &prev_state_i,
                    &q_i,
                    &k_i,
                    &v_i,
                    &beta_i,
                    &g_i,
                )? {
                    let fused = crate::backends::hip::state_buffer_from_host_bytes(
                        bytes,
                        shape,
                        compute_dtype,
                        query.device(),
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(kernel_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let (output_i, recurrent_state) =
                        crate::backends::hip::unpack_delta_chunk_step_output(
                            &fused,
                            chunk_size,
                            k_head_dim,
                        )?;
                    outputs.push(output_i.clone_tensor().unsqueeze(1)?);
                    last_recurrent_state = recurrent_state.clone_tensor();
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                } else {
                    let fused =
                        delta_chunk_step_raw(&prev_state_i, &q_i, &k_i, &v_i, &beta_i, &g_i)?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(kernel_start, device)?;

                    let unpack_start = profile_start(device)?;
                    outputs.push(fused.narrow(1, 0, chunk_size)?.unsqueeze(1)?);
                    last_recurrent_state = fused
                        .narrow(1, chunk_size, k_head_dim)?
                        .reshape((batch_heads, k_head_dim, v_head_dim))?
                        .contiguous()?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                }
            }
            profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;
            let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, last_recurrent_state, profile));
        }
        let backend = backend_buffer_api::for_device(device);
        let prepare_start = profile_start(device)?;
        let k_beta_start = profile_start(device)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        profile.linear_chunk_prepare_k_beta_millis += profile_elapsed(k_beta_start, device)?;
        let g_start = profile_start(device)?;
        let g = backend
            .cumsum_last_dim(&backend.tensor_to_buffer(g_raw.clone())?)?
            .clone_tensor();
        let exp_g = g.exp()?;
        let exp_g_scan = exp_g.reshape((batch_heads, num_chunks, chunk_size))?;
        profile.linear_chunk_prepare_g_millis += profile_elapsed(g_start, device)?;

        let cache_start = profile_start(device)?;
        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower = cache.lower;
        let eye = cache.eye;
        let strict_lower = cache.strict_lower;
        let use_state_kernel = use_delta_state_kernel(query.device(), scan_mode, sequence_length);
        let use_state_scan_kernel =
            use_delta_state_scan_kernel(query.device(), scan_mode, sequence_length);
        let use_chunk_fused_kernel =
            use_delta_chunk_fused_kernel(query.device(), scan_mode, sequence_length);
        let use_full_scan_kernel = use_delta_full_scan_kernel(query.device(), scan_mode, sequence_length)
            || use_hip_exact_multi_chunk_full_scan_prefill(
                query.device(),
                scan_mode,
                sequence_length,
                num_chunks,
                chunk_size,
            );
        let hip_full_scan_fast_path = query.device().is_hip() && use_full_scan_kernel;
        let hip_base_attn_fast_path = query.device().is_hip() && !hip_full_scan_fast_path;
        profile.linear_chunk_prepare_cache_millis += profile_elapsed(cache_start, device)?;

        let solve_batch = batch_size * num_heads * num_chunks;
        let base_attn_start = profile_start(device)?;
        let (base_attn, decay_scan) = if hip_full_scan_fast_path {
            (None, None)
        } else if hip_base_attn_fast_path {
            let decay_mask_start = profile_start(device)?;
            let decay_deltas = g
                .unsqueeze(4)?
                .broadcast_sub(&g.unsqueeze(3)?)?
                .broadcast_mul(&lower)?;
            let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower)?;
            profile.linear_chunk_prepare_base_attn_decay_mask_millis +=
                profile_elapsed(decay_mask_start, device)?;

            let post_start = profile_start(device)?;
            let base_attn = backend.delta_base_attn_scan(
                &k_beta
                    .reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?
                    .contiguous()?,
                &key_scan.contiguous()?,
                &exp_g_scan.contiguous()?,
            )?;
            profile.linear_chunk_prepare_base_attn_post_millis +=
                profile_elapsed(post_start, device)?;
            (
                Some(base_attn),
                Some(decay_mask.reshape((batch_heads, num_chunks, chunk_size, chunk_size))?),
            )
        } else {
            let decay_mask_start = profile_start(device)?;
            let decay_deltas = g
                .unsqueeze(4)?
                .broadcast_sub(&g.unsqueeze(3)?)?
                .broadcast_mul(&lower)?;
            let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower)?;
            profile.linear_chunk_prepare_base_attn_decay_mask_millis +=
                profile_elapsed(decay_mask_start, device)?;

            let key_t_start = profile_start(device)?;
            let key_t = key.transpose(4, 3)?.contiguous()?;
            profile.linear_chunk_prepare_base_attn_key_t_millis +=
                profile_elapsed(key_t_start, device)?;

            let flatten_start = profile_start(device)?;
            let k_beta_flat = k_beta
                .reshape((solve_batch, chunk_size, k_head_dim))?
                .contiguous()?;
            let key_t_flat = key_t
                .reshape((solve_batch, k_head_dim, chunk_size))?
                .contiguous()?;
            let decay_mask_flat = decay_mask.reshape((solve_batch, chunk_size, chunk_size))?;
            profile.linear_chunk_prepare_base_attn_flatten_millis +=
                profile_elapsed(flatten_start, device)?;

            let matmul_start = profile_start(device)?;
            let raw_attn = k_beta_flat.matmul(&key_t_flat)?;
            profile.linear_chunk_prepare_base_attn_matmul_millis +=
                profile_elapsed(matmul_start, device)?;

            let post_start = profile_start(device)?;
            let result = (
                Some(backend.tensor_to_buffer(
                    raw_attn
                        .broadcast_mul(&decay_mask_flat)?
                        .neg()?
                        .broadcast_mul(&strict_lower.reshape((1, chunk_size, chunk_size))?)?
                        .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?,
                )?),
                Some(decay_mask.reshape((batch_heads, num_chunks, chunk_size, chunk_size))?),
            );
            profile.linear_chunk_prepare_base_attn_post_millis +=
                profile_elapsed(post_start, device)?;
            result
        };
        profile.linear_chunk_prepare_base_attn_millis += profile_elapsed(base_attn_start, device)?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let attn = if hip_full_scan_fast_path {
            backend.delta_attn_solve_from_inputs(
                &k_beta
                    .reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?
                    .contiguous()?,
                &key_scan.contiguous()?,
                &exp_g_scan.contiguous()?,
            )?
            .clone_tensor()
            .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
        } else if hip_base_attn_fast_path {
            let base_attn = base_attn.as_ref().expect("base_attn exists on HIP base-attn path");
            backend.delta_attn_solve_scan(&backend.reshape_tensor_to_buffer(
                &base_attn.clone_tensor().contiguous()?,
                &[batch_heads, num_chunks, chunk_size, chunk_size],
            )?)?
            .clone_tensor()
            .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
        } else if scan_policy.use_flattened_solve {
            let solve_batch = batch_size * num_heads * num_chunks;
            let base_attn = base_attn.as_ref().expect("base_attn exists on non-HIP solve path");
            let base_attn_flat = base_attn
                .tensor()
                .reshape((solve_batch, chunk_size, chunk_size))?;
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (solve_batch, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn_flat
                    .narrow(1, i, 1)?
                    .narrow(2, 0, i)?
                    .reshape((solve_batch, i))?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
                let correction = row
                    .unsqueeze(1)?
                    .broadcast_mul(&sub)?
                    .sum(1)?
                    .reshape((solve_batch, i))?;
                let row = row.broadcast_add(&correction)?;
                let row = row.pad_with_zeros(1, 0, chunk_size - i)?.reshape((
                    solve_batch,
                    1,
                    chunk_size,
                ))?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
                .broadcast_add(&eye)?
        } else {
            let base_attn = base_attn.as_ref().expect("base_attn exists on non-HIP solve path");
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (batch_size, num_heads, num_chunks, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn.tensor().narrow(3, i, 1)?.narrow(4, 0, i)?.squeeze(3)?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 3)?.narrow(4, 0, i)?;
                let correction = row.unsqueeze(4)?.broadcast_mul(&sub)?.sum(3)?;
                let row = (row + correction)?;
                let row = row.pad_with_zeros(3, 0, chunk_size - i)?.unsqueeze(3)?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 3)?.broadcast_add(&eye)?
        };
        let weighted_k = k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?;
        let attn_flat = attn
            .reshape((solve_batch, chunk_size, chunk_size))?
            .contiguous()?;
        let weighted_k_flat = weighted_k
            .reshape((solve_batch, chunk_size, k_head_dim))?
            .contiguous()?;
        let k_cumdecay = attn_flat
            .matmul(&weighted_k_flat)?
            .reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let lower_2d = cache.lower_2d;
        let k_cumdecay_scan =
            k_cumdecay.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let lower_2d_scan = lower_2d.reshape((1, 1, chunk_size, chunk_size))?;
        let local_attn_scan = match scan_mode {
            DeltaNetScanMode::PrebatchedLocal => Some({
                if hip_full_scan_fast_path {
                    backend.delta_local_attn_scan(
                        &query_scan.contiguous()?,
                        &key_scan.contiguous()?,
                        &exp_g_scan.contiguous()?,
                    )?
                } else {
                    let key_scan_t = key_scan.transpose(3, 2)?.contiguous()?;
                    backend.tensor_to_buffer(
                        query_scan
                            .matmul(&key_scan_t)?
                            .broadcast_mul(decay_scan.as_ref().ok_or_else(|| {
                                candle::Error::Msg(
                                    "prebatched-local attention requires decay scan".into(),
                                )
                            })?)?
                            .broadcast_mul(&lower_2d_scan)?,
                    )?
                }
            }),
            _ => None,
        };
        let needs_q_state_scan = !hip_full_scan_fast_path;
        let needs_hoisted_decays =
            !hip_full_scan_fast_path
                && matches!(
                    scan_mode,
                    DeltaNetScanMode::HoistedDecays | DeltaNetScanMode::PrebatchedLocal
                );
        let q_state_scan = if needs_q_state_scan {
            Some(query_scan.broadcast_mul(&exp_g_scan.unsqueeze(D::Minus1)?)?)
        } else {
            None
        };
        let (state_decay_scan, chunk_decay_scan) = if needs_hoisted_decays {
            let exp_g_last_scan = exp_g_scan.i((.., .., chunk_size - 1))?;
            (
                Some(exp_g_last_scan.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?),
                Some(
                    exp_g_last_scan
                        .unsqueeze(D::Minus1)?
                        .broadcast_div(&exp_g_scan)?
                        .unsqueeze(D::Minus1)?,
                ),
            )
        } else {
            (None, None)
        };
        let lower_2d = lower_2d.reshape((1, chunk_size, chunk_size))?;
        let mut last_recurrent_state = backend.zeros_state(
            query.device(),
            compute_dtype,
            &[batch_heads, k_head_dim, v_head_dim],
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        let full_scan = if use_full_scan_kernel {
            let full_pack_start = profile_start(device)?;
            let local_attn_scan = local_attn_scan
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg("delta-full-scan requires prebatched local attention".into())
                })?
                .contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let full_scan = if query.device().is_hip() {
                let packed_scan = backend.delta_full_scan_pack(
                    &query_scan.contiguous()?,
                    &key_scan.contiguous()?,
                    &exp_g_scan.contiguous()?,
                    &k_cumdecay_scan.contiguous()?,
                )?;
                let full_pack_elapsed = profile_elapsed(full_pack_start, device)?;
                profile.linear_full_kernel_pack_millis += full_pack_elapsed;
                profile.transfer_millis += full_pack_elapsed;
                let full_kernel_start = profile_start(device)?;
                let full_scan = backend.delta_full_scan_packed(
                    &last_recurrent_state,
                    &packed_scan,
                    &local_attn_scan,
                    &value_scan,
                )?;
                profile.linear_full_kernel_execute_millis +=
                    profile_elapsed(full_kernel_start, device)?;
                Some(full_scan)
            } else {
                let state_decay_scan = state_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted state decay".into())
                    })?
                    .squeeze(3)?
                    .squeeze(2)?
                    .contiguous()?;
                let weighted_key_scan = key_scan
                    .broadcast_mul(chunk_decay_scan.as_ref().ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted chunk decay".into())
                    })?)?
                    .contiguous()?;
                let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
                let q_state_scan = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted q-state".into())
                    })?
                    .contiguous()?;
                let full_pack_elapsed = profile_elapsed(full_pack_start, device)?;
                profile.linear_full_kernel_pack_millis += full_pack_elapsed;
                profile.transfer_millis += full_pack_elapsed;
                let full_kernel_start = profile_start(device)?;
                let full_scan = backend.delta_full_scan(
                    &last_recurrent_state,
                    &weighted_key_scan,
                    &k_cumdecay_scan,
                    &q_state_scan,
                    &local_attn_scan,
                    &state_decay_scan,
                    &value_scan,
                )?;
                profile.linear_full_kernel_execute_millis +=
                    profile_elapsed(full_kernel_start, device)?;
                Some(full_scan)
            };
            full_scan
        } else {
            None
        };
        let state_scan = if use_state_scan_kernel {
            let state_decay_scan = state_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted state decay".into())
            })?;
            let chunk_decay_scan = chunk_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted chunk decay".into())
            })?;
            let weighted_key_scan = key_scan.broadcast_mul(chunk_decay_scan)?;
            let state_decay_feature =
                state_decay_scan.broadcast_as((batch_heads, num_chunks, chunk_size, 1))?;
            let packed_scan = backend.pack_delta_state_scan(
                &weighted_key_scan,
                &k_cumdecay_scan,
                &state_decay_feature,
            )?;
            Some(backend.delta_state_scan(
                &last_recurrent_state,
                &packed_scan,
                &value_scan.contiguous()?,
            )?)
        } else {
            None
        };
        if let Some(full_scan) = &full_scan {
            let full_unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                full_scan,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let full_unpack_elapsed = profile_elapsed(full_unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += full_unpack_elapsed;
            profile.transfer_millis += full_unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        let scan_start = profile_start(device)?;
        for chunk_idx in 0..num_chunks {
            let index_start = profile_start(device)?;
            let q_i = query_scan.i((.., chunk_idx, .., ..))?;
            let k_i = key_scan.i((.., chunk_idx, .., ..))?;
            let v_i = value_scan.i((.., chunk_idx, .., ..))?;
            let g_i = g_scan.i((.., chunk_idx, ..))?;
            let prev_state_i = if let Some(state_scan) = &state_scan {
                backend.state_scan_chunk(state_scan, chunk_idx)?
            } else {
                last_recurrent_state.clone()
            };
            profile.linear_chunk_index_millis += profile_elapsed(index_start, device)?;

            let local_attn_start = profile_start(device)?;
            let attn = if let Some(local_attn_scan) = &local_attn_scan {
                local_attn_scan.tensor().i((.., chunk_idx, .., ..))?
            } else {
                let decay_i = decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta chunk fallback requires decay scan".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let k_i_t = k_i.transpose(2, 1)?.contiguous()?;
                q_i.matmul(&k_i_t)?
                    .broadcast_mul(&decay_i)?
                    .broadcast_mul(&lower_2d)?
            };
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let recurrent_read_start = profile_start(device)?;
            let (v_new, attn_inter, fused_next_state) = if use_chunk_fused_kernel
                && state_scan.is_none()
            {
                let weighted_key = chunk_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted chunk decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_mul(&k_i)?;
                let q_state = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted q-state".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let state_decay = state_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted state decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_as((batch_heads, chunk_size, 1))?;
                let packed_chunk = backend.pack_delta_chunk_fused(
                    &weighted_key,
                    &k_cumdecay_scan.i((.., chunk_idx, .., ..))?,
                    &q_state,
                    &state_decay,
                )?;
                let prev_state_i = prev_state_i.contiguous()?;
                let fused = backend.delta_chunk_fused(
                    &prev_state_i,
                    &packed_chunk,
                    &v_i.contiguous()?,
                )?;
                let (v_new, attn_inter, fused_next_state) = backend.unpack_chunk_fused(
                    &fused,
                    chunk_size,
                    k_head_dim,
                )?;
                (v_new, attn_inter, Some(fused_next_state))
            } else {
                let q_state_i = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk loop requires hoisted q-state".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let (v_new, attn_inter) = backend.delta_chunk_recurrent_read(
                    &prev_state_i,
                    &k_cumdecay_scan.i((.., chunk_idx, .., ..))?,
                    &q_state_i,
                    &v_i,
                )?;
                (v_new, attn_inter, None)
            };
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let local_mix_start = profile_start(device)?;
            outputs.push(
                backend
                    .mix_chunk_attention(&attn, &attn_inter, &v_new)?
                    .clone_tensor()
                    .unsqueeze(1)?,
            );
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_mix_start, device)?;

            let state_update_start = profile_start(device)?;
            let (state_decay, chunk_decay) =
                if let (Some(state_decay_scan), Some(chunk_decay_scan)) =
                    (&state_decay_scan, &chunk_decay_scan)
                {
                    (
                        state_decay_scan.i((.., chunk_idx, .., ..))?,
                        chunk_decay_scan.i((.., chunk_idx, .., ..))?,
                    )
                } else {
                    let g_last = g_i.i((.., chunk_size - 1))?;
                    (
                        g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?,
                        g_last
                            .unsqueeze(D::Minus1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )
                };
            if let Some(fused_next_state) = fused_next_state {
                last_recurrent_state = fused_next_state.contiguous()?;
            } else if let Some(state_scan) = &state_scan {
                last_recurrent_state = backend.state_scan_next_chunk(state_scan, chunk_idx + 1)?;
            } else {
                let prev_state_scaled = last_recurrent_state
                    .tensor()
                    .broadcast_mul(&state_decay)?
                    .contiguous()?;
                let weighted_key = k_i.broadcast_mul(&chunk_decay)?.contiguous()?;
                last_recurrent_state = backend.delta_state_update(
                    &prev_state_scaled,
                    &weighted_key,
                    &v_new,
                    use_state_kernel,
                )?;
            }
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state.clone_tensor(), profile))
    }

    fn forward_profiled_with_state(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let backend = backend_buffer_api::for_device(device);
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let mixed_qkv =
            backend.tensor_to_buffer(self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?)?;
        let z = backend.reshape_tensor_to_buffer(
            &self.in_proj_z.forward(&hidden_states)?,
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = backend.tensor_to_buffer(self.in_proj_b.forward(&hidden_states)?)?;
        let a = backend.tensor_to_buffer(self.in_proj_a.forward(&hidden_states)?)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;
        let (output, recurrent_state, linear_profile) = self.forward_profiled_with_state_projected(
            hidden_states.dtype(),
            batch_size,
            seq_len,
            &mixed_qkv,
            &z,
            &beta_raw,
            &a,
            compute_dtype,
        )?;
        profile.add_assign(&linear_profile);
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output.clone_tensor(), recurrent_state.clone_tensor(), profile))
    }

    fn forward_profiled_with_state_projected(
        &mut self,
        hidden_dtype: DType,
        batch_size: usize,
        seq_len: usize,
        mixed_qkv: &StateBuffer,
        z: &StateBuffer,
        beta_raw: &StateBuffer,
        a: &StateBuffer,
        compute_dtype: DType,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = mixed_qkv.device();
        let backend = backend_buffer_api::for_device(device);
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();

        if use_hip_combined_linear_decode(device, seq_len) {
            let (output, recurrent_state, decode_profile) = self.linear_decode_projected(
                hidden_dtype,
                batch_size,
                seq_len,
                mixed_qkv,
                z,
                beta_raw,
                a,
            )?;
            profile.add_assign(&decode_profile);
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, recurrent_state, profile));
        }

        let kv_append_start = profile_start(device)?;
        let (mixed_qkv, g) = if use_hip_combined_linear_prefill(device, seq_len) {
            let target_dtype = mixed_qkv.tensor().dtype();
            let a_tensor = if a.tensor().dtype() == target_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(target_dtype)?
            };
            let a = backend.tensor_to_buffer(a_tensor)?;
            let (dt_bias, a_log_exp) = self.value_cache(device, target_dtype)?;
            let weights = self.conv1d_weight_squeezed()?.contiguous()?;
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(target_dtype)?,
                None => backend
                    .zeros_tensor(
                        mixed_qkv.device(),
                        target_dtype,
                        &[mixed_qkv.tensor().dim(0)?, mixed_qkv.tensor().dim(1)?, state_len],
                    )?,
            };
            let fused = backend.linear_stateful_conv_value_decay_with_state(
                &mixed_qkv.contiguous()?,
                &prev_state,
                &weights,
                &a,
                &dt_bias,
                &a_log_exp,
                self.conv_kernel_size,
            )?;
            let conv_dim = self.conv_dim();
            let (mixed_qkv, g, conv_state) = backend.unpack_linear_prefill_output(
                &fused,
                batch_size,
                seq_len,
                conv_dim,
                self.num_v_heads,
                state_len,
            )?;
            self.conv_state = Some(conv_state);
            let g = if g.dtype() == compute_dtype {
                g
            } else {
                g.to_dtype(compute_dtype)?
            };
            (mixed_qkv, g)
        } else {
            let mixed_qkv = if seq_len == 1 {
                self.run_depthwise_conv_update(mixed_qkv.tensor())?
                    .transpose(1, 2)?
            } else if use_linear_prefill_packed_kernel(device, seq_len) {
                self.run_depthwise_conv_packed_prefill(mixed_qkv.tensor())?
            } else {
                self.run_depthwise_conv(mixed_qkv.tensor())?.transpose(1, 2)?
            };
            let a = if a.tensor().dtype() == compute_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(compute_dtype)?
            };
            let (dt_bias, a_log_exp) = self.value_cache(device, compute_dtype)?;
            let g = backend
                .value_decay(&backend.tensor_to_buffer(a.clone())?, &dt_bias, &a_log_exp)?
                .clone_tensor();
            (mixed_qkv, g)
        };
        let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
        profile.linear_conv_millis += kv_append_elapsed;
        profile.kv_append_write_millis += kv_append_elapsed;

        let layout_start = profile_start(device)?;
        let use_short_recurrent_prefill = use_hip_short_linear_prefill_recurrent(device, seq_len);
        let (query, key, value, beta, g) = backend.prepare_linear_attention_inputs(
            &mixed_qkv,
            beta_raw,
            &g,
            batch_size,
            seq_len,
            self.key_dim,
            self.value_dim,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            compute_dtype,
            seq_len == 1 || use_short_recurrent_prefill,
        )?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let (core_attn_out, recurrent_state, linear_profile) =
            if seq_len == 1 && self.recurrent_state.is_some() {
                self.recurrent_gated_delta_rule(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    self.recurrent_state.as_ref().map(StateBuffer::tensor),
                )?
            } else if seq_len == 1 {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else if use_short_recurrent_prefill {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else {
                self.chunk_gated_delta_rule(&query, &key, &value, &g, &beta, seq_len)?
            };
        profile.add_assign(&linear_profile);

        let output_start = profile_start(device)?;
        let output = self.finalize_linear_output_buffer(
            hidden_dtype,
            batch_size,
            seq_len,
            z,
            &core_attn_out,
        )?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output, backend.tensor_to_buffer(recurrent_state)?, profile))
    }

    fn linear_decode_projected(
        &mut self,
        hidden_dtype: DType,
        batch_size: usize,
        seq_len: usize,
        mixed_qkv: &StateBuffer,
        z: &StateBuffer,
        beta_raw: &StateBuffer,
        a: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = mixed_qkv.device();
        let backend = backend_buffer_api::for_device(device);
        let mut profile = RuntimeProfile::default();
        let kv_append_start = profile_start(device)?;
        let target_dtype = mixed_qkv.dtype();
        let weights = self.conv1d_weight_squeezed()?.contiguous()?;
        let state_len = self.conv_kernel_size.saturating_sub(1);
        let (batch_size_qkv, conv_dim, _) = mixed_qkv.dims3()?;
        let prev_conv_state = match &self.conv_state {
            Some(prev_state) => prev_state.clone_tensor_as(target_dtype)?,
            None => backend.zeros_tensor(
                mixed_qkv.device(),
                target_dtype,
                &[batch_size_qkv, conv_dim, state_len],
            )?,
        };
        let a = if a.dtype() == target_dtype {
            a.clone()
        } else {
            StateBuffer::from_tensor(a.tensor().to_dtype(target_dtype)?)?
        };
        let beta_raw = if beta_raw.dtype() == target_dtype {
            beta_raw.clone()
        } else {
            StateBuffer::from_tensor(beta_raw.tensor().to_dtype(target_dtype)?)?
        };
        let a_beta_raw = backend.concat_last_dim(&a, &beta_raw)?;
        let (dt_bias, a_log_exp) = self.value_cache(device, target_dtype)?;
        let initial_state = match &self.recurrent_state {
            Some(state) => {
                let state = state.clone_tensor();
                let state = if state.rank() == 3 {
                    state.reshape((batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim))?
                } else {
                    state
                };
                if state.dtype() == DType::F32 {
                    state
                } else {
                    state.to_dtype(DType::F32)?
                }
            }
            None => backend.zeros_tensor(
                device,
                DType::F32,
                &[batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim],
            )?,
        };
        let head_repeat = self.num_v_heads / self.num_k_heads;
        let mixed_qkv = mixed_qkv.contiguous()?;
        let fused = backend.linear_decode_step(
            &mixed_qkv,
            &prev_conv_state,
            &weights,
            &a_beta_raw,
            &dt_bias,
            &a_log_exp,
            &initial_state,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            head_repeat,
        )?;
        self.update_depthwise_conv_state_from_raw(mixed_qkv.tensor())?;
        let (core_attn_out, recurrent_state) = backend.unpack_linear_decode_output(
            &fused,
            batch_size,
            seq_len,
            self.value_dim,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )?;
        let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
        profile.linear_conv_millis += kv_append_elapsed;
        profile.kv_append_write_millis += kv_append_elapsed;
        profile.linear_recurrent_loop_millis += kv_append_elapsed;

        let output_start = profile_start(device)?;
        let output = self.finalize_linear_output_buffer(
            hidden_dtype,
            batch_size,
            seq_len,
            z,
            &core_attn_out,
        )?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((output, recurrent_state, profile))
    }

    pub(super) fn project_direct_decode_inputs(
        &self,
        hidden_states: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let backend = backend_buffer_api::for_device(device);
        let mut profile = RuntimeProfile::default();
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let qkv_start = profile_start(device)?;
        let mixed_qkv = backend.tensor_to_buffer(
            self.in_proj_qkv
                .forward_buffer(hidden_states)?
                .tensor()
                .transpose(1, 2)?,
        )?;
        let z = backend.reshape_tensor_to_buffer(
            self.in_proj_z.forward_buffer(hidden_states)?.tensor(),
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = self.in_proj_b.forward_buffer(hidden_states)?;
        let a = self.in_proj_a.forward_buffer(hidden_states)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;
        Ok((mixed_qkv, z, beta_raw, a, profile))
    }

    pub(super) fn trace_projection_components_buffer(
        &self,
        hidden_states: &StateBuffer,
    ) -> Result<LinearAttentionProjectionTrace> {
        let backend = backend_buffer_api::for_device(hidden_states.device());
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let qkv_output = self.in_proj_qkv.forward_buffer(hidden_states)?;
        let z_output = backend.reshape_tensor_to_buffer(
            self.in_proj_z.forward_buffer(hidden_states)?.tensor(),
            &[batch_size, seq_len, self.value_dim],
        )?;
        let b_output = self.in_proj_b.forward_buffer(hidden_states)?;
        let a_output = self.in_proj_a.forward_buffer(hidden_states)?;
        Ok(LinearAttentionProjectionTrace {
            qkv_output,
            z_output,
            b_output,
            a_output,
        })
    }

    pub(super) fn trace_core_components_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(LinearAttentionCoreTrace, StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let hidden_dtype = hidden_states.tensor().dtype();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.tensor().dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = backend_buffer_api::for_device(device).tensor_to_buffer(
            self.apply_mask_to_padding_states(hidden_states.tensor(), attention_mask)?,
        )?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let backend = backend_buffer_api::for_device(device);
        let mixed_qkv = backend.tensor_to_buffer(
            self.in_proj_qkv
                .forward_buffer(&hidden_states)?
                .tensor()
                .transpose(1, 2)?,
        )?;
        let z = backend.reshape_tensor_to_buffer(
            self.in_proj_z.forward_buffer(&hidden_states)?.tensor(),
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = self.in_proj_b.forward_buffer(&hidden_states)?;
        let a = self.in_proj_a.forward_buffer(&hidden_states)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let kv_append_start = profile_start(device)?;
        let (mixed_qkv, g) = if use_hip_combined_linear_prefill(device, seq_len) {
            let target_dtype = mixed_qkv.tensor().dtype();
            let a_tensor = if a.tensor().dtype() == target_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(target_dtype)?
            };
            let a = backend.tensor_to_buffer(a_tensor)?;
            let (dt_bias, a_log_exp) = self.value_cache(device, target_dtype)?;
            let weights = self.conv1d_weight_squeezed()?.contiguous()?;
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(target_dtype)?,
                None => backend.zeros_tensor(
                    mixed_qkv.device(),
                    target_dtype,
                    &[mixed_qkv.tensor().dim(0)?, mixed_qkv.tensor().dim(1)?, state_len],
                )?,
            };
            let fused = backend.linear_stateful_conv_value_decay_with_state(
                &mixed_qkv.contiguous()?,
                &prev_state,
                &weights,
                &a,
                &dt_bias,
                &a_log_exp,
                self.conv_kernel_size,
            )?;
            let conv_dim = self.conv_dim();
            let (mixed_qkv, g, conv_state) = backend.unpack_linear_prefill_output(
                &fused,
                batch_size,
                seq_len,
                conv_dim,
                self.num_v_heads,
                state_len,
            )?;
            self.conv_state = Some(conv_state);
            let g = if g.dtype() == compute_dtype {
                g
            } else {
                g.to_dtype(compute_dtype)?
            };
            (mixed_qkv, g)
        } else {
            let mixed_qkv = if seq_len == 1 {
                self.run_depthwise_conv_update(mixed_qkv.tensor())?
                    .transpose(1, 2)?
            } else if use_linear_prefill_packed_kernel(device, seq_len) {
                self.run_depthwise_conv_packed_prefill(mixed_qkv.tensor())?
            } else {
                self.run_depthwise_conv(mixed_qkv.tensor())?.transpose(1, 2)?
            };
            let a = if a.tensor().dtype() == compute_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(compute_dtype)?
            };
            let (dt_bias, a_log_exp) = self.value_cache(device, compute_dtype)?;
            let g = backend
                .value_decay(&backend.tensor_to_buffer(a.clone())?, &dt_bias, &a_log_exp)?
                .clone_tensor();
            (mixed_qkv, g)
        };
        let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
        profile.linear_conv_millis += kv_append_elapsed;
        profile.kv_append_write_millis += kv_append_elapsed;
        let post_conv_mixed_qkv = backend.tensor_to_buffer(mixed_qkv.clone())?;

        let layout_start = profile_start(device)?;
        let use_short_recurrent_prefill = use_hip_short_linear_prefill_recurrent(device, seq_len);
        let (query, key, value, beta, g) = backend.prepare_linear_attention_inputs(
            &mixed_qkv,
            &beta_raw,
            &g,
            batch_size,
            seq_len,
            self.key_dim,
            self.value_dim,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            compute_dtype,
            seq_len == 1 || use_short_recurrent_prefill,
        )?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let (core_attn_out, recurrent_state, linear_profile) =
            if seq_len == 1 && self.recurrent_state.is_some() {
                self.recurrent_gated_delta_rule(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    self.recurrent_state.as_ref().map(StateBuffer::tensor),
                )?
            } else if seq_len == 1 {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else if use_short_recurrent_prefill {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else {
                self.chunk_gated_delta_rule(&query, &key, &value, &g, &beta, seq_len)?
            };
        profile.add_assign(&linear_profile);

        let pre_gated_norm_output = backend.reshape_tensor_to_buffer(
            &core_attn_out,
            &[batch_size, seq_len, self.value_dim],
        )?;
        let pre_gated_norm_heads = backend.reshape_tensor_to_buffer(
            &core_attn_out,
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let pre_gated_norm_mean_square = backend.tensor_to_buffer(
            (pre_gated_norm_heads
                .tensor()
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_keepdim(D::Minus1)?
                / self.head_v_dim as f64)?
            .reshape((batch_size, seq_len, self.num_v_heads))?,
        )?;
        let pre_gated_norm_rsqrt = backend.tensor_to_buffer(
            pre_gated_norm_mean_square
                .tensor()
                .broadcast_add(&Tensor::new(self.norm.eps() as f32, device)?)?
                .sqrt()?
                .recip()?
                .reshape((batch_size, seq_len, self.num_v_heads))?,
        )?;
        let gated_norm_gate_input = backend.reshape_tensor_to_buffer(
            z.tensor(),
            &[batch_size, seq_len, self.value_dim],
        )?;
        let gated_norm_weight = backend.tensor_to_buffer(self.norm.weight().clone())?;
        let norm_input = backend.reshape_tensor_to_buffer(
            &core_attn_out,
            &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
        )?;
        let gated_norm_weighted_hidden = backend.reshape_tensor_to_buffer(
            &super::backend_ops::rms_norm(
                norm_input.tensor(),
                self.norm.weight(),
                self.norm.eps(),
                false,
            )?,
            &[batch_size, seq_len, self.value_dim],
        )?;
        let gated_norm_weighted_hidden_fallback = {
            let xs_dtype = norm_input.tensor().dtype();
            let xs = norm_input.tensor().to_dtype(DType::F32)?;
            let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / self.head_v_dim as f64)?;
            let xs = xs.broadcast_div(&(variance + self.norm.eps())?.sqrt()?)?;
            let xs = xs.broadcast_mul(&self.norm.weight().to_dtype(DType::F32)?)?;
            backend.reshape_tensor_to_buffer(
                &xs.to_dtype(xs_dtype)?,
                &[batch_size, seq_len, self.value_dim],
            )?
        };
        let gated_norm_silu_gate = backend.reshape_tensor_to_buffer(
            &super::ops::silu(gated_norm_gate_input.tensor())?,
            &[batch_size, seq_len, self.value_dim],
        )?;
        let gated_norm = self
            .norm
            .forward_buffer(
                &backend.reshape_tensor_to_buffer(
                    &core_attn_out,
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
                &backend.reshape_tensor_to_buffer(
                    z.tensor(),
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
            )?;
        let gated_norm = if gated_norm.tensor().dtype() == hidden_dtype {
            gated_norm
        } else {
            backend.tensor_to_buffer(gated_norm.tensor().to_dtype(hidden_dtype)?)?
        };
        let post_gated_norm_output =
            backend.reshape_tensor_to_buffer(gated_norm.tensor(), &[batch_size, seq_len, self.value_dim])?;
        let output = self.out_proj.forward_buffer(&post_gated_norm_output)?;
        profile.output_projection_millis += profile_elapsed(total_start, device)?;

        Ok((
            LinearAttentionCoreTrace {
                post_conv_mixed_qkv,
                pre_gated_norm_output,
                pre_gated_norm_mean_square,
                pre_gated_norm_rsqrt,
                gated_norm_gate_input,
                gated_norm_weight,
                gated_norm_weighted_hidden,
                gated_norm_weighted_hidden_fallback,
                gated_norm_silu_gate,
                post_gated_norm_output,
            },
            output,
            backend.tensor_to_buffer(recurrent_state)?,
            profile,
        ))
    }

    pub(super) fn project_direct_decode_inputs_into_scratch(
        &self,
        hidden_states: &StateBuffer,
        mixed_qkv_scratch: &StateBuffer,
        z_scratch: &StateBuffer,
        beta_raw_scratch: &StateBuffer,
        a_scratch: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer, RuntimeProfile)> {
        let backend = backend_buffer_api::for_device(hidden_states.device());
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        profile.layout_prepare_millis += profile_elapsed(total_start, device)?;

        let qkv_start = profile_start(device)?;
        let mixed_qkv = self
            .in_proj_qkv
            .forward_buffer(hidden_states)?;
        let mixed_qkv = backend.transpose_tensor_to_buffer_into_scratch(
            mixed_qkv.tensor(),
            1,
            2,
            mixed_qkv_scratch,
        )?;
        let z = self.in_proj_z.forward_buffer(hidden_states)?;
        let z = backend.reshape_tensor_to_buffer_into_scratch(
            z.tensor(),
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
            z_scratch,
        )?;
        let beta_raw = self
            .in_proj_b
            .forward_buffer_into_scratch(hidden_states, beta_raw_scratch)?;
        let a = self.in_proj_a.forward_buffer_into_scratch(hidden_states, a_scratch)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;
        Ok((mixed_qkv, z, beta_raw, a, profile))
    }

    pub(super) fn run_direct_decode_core(
        &mut self,
        hidden_dtype: DType,
        hidden_states: &StateBuffer,
        mixed_qkv: &StateBuffer,
        z: &StateBuffer,
        beta_raw: &StateBuffer,
        a: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let (batch_size, _, _) = hidden_states.dims3()?;
        let seq_len = 1;
        let compute_dtype = linear_attention_compute_dtype(device, hidden_dtype);
        self.forward_profiled_with_state_projected(
            hidden_dtype,
            batch_size,
            seq_len,
            mixed_qkv,
            z,
            beta_raw,
            a,
            compute_dtype,
        )
    }

    pub(super) fn commit_direct_decode_recurrent_state(&mut self, recurrent_state: StateBuffer) {
        self.recurrent_state = Some(recurrent_state);
    }

    fn forward_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let (output, recurrent_state, profile) =
            self.forward_profiled_with_state(hidden_states, attention_mask)?;
        let backend = backend_buffer_api::for_device(hidden_states.device());
        self.recurrent_state = Some(backend.tensor_to_buffer(recurrent_state)?);
        Ok((output, profile))
    }

    pub(super) fn forward_profiled_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let (output, recurrent_state, profile) =
            self.forward_profiled_with_state_buffer(hidden_states, attention_mask)?;
        self.recurrent_state = Some(recurrent_state);
        Ok((output, profile))
    }

    pub(super) fn forward_profiled_direct_decode_v1(
        &mut self,
        hidden_states: &StateBuffer,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let (_, seq_len, _) = hidden_states.dims3()?;
        if seq_len != 1 {
            candle::bail!(
                "direct-hip-v1 linear decode expects single-token hidden state, got seq_len={seq_len}"
            );
        }
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (mixed_qkv, z, beta_raw, a, projection_profile) =
            self.project_direct_decode_inputs(hidden_states)?;
        profile.add_assign(&projection_profile);
        let (output, recurrent_state, linear_profile) = self.run_direct_decode_core(
            hidden_states.tensor().dtype(),
            hidden_states,
            &mixed_qkv,
            &z,
            &beta_raw,
            &a,
        )?;
        profile.add_assign(&linear_profile);
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        self.commit_direct_decode_recurrent_state(recurrent_state);
        Ok((output, profile))
    }

    fn trace_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        self.forward_profiled_with_state(hidden_states, attention_mask)
    }

    pub(super) fn trace_profiled_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        self.forward_profiled_with_state_buffer(hidden_states, attention_mask)
    }

    fn forward_profiled_with_state_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.tensor().dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = backend_buffer_api::for_device(device).tensor_to_buffer(
            self.apply_mask_to_padding_states(hidden_states.tensor(), attention_mask)?,
        )?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let backend = backend_buffer_api::for_device(device);
        let mixed_qkv = backend.tensor_to_buffer(
            self.in_proj_qkv
                .forward_buffer(&hidden_states)?
                .tensor()
                .transpose(1, 2)?,
        )?;
        let z = backend.reshape_tensor_to_buffer(
            self.in_proj_z.forward_buffer(&hidden_states)?.tensor(),
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = self.in_proj_b.forward_buffer(&hidden_states)?;
        let a = self.in_proj_a.forward_buffer(&hidden_states)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let (output, recurrent_state, linear_profile) =
            self.forward_profiled_with_state_projected(
                hidden_states.tensor().dtype(),
                batch_size,
                seq_len,
                &mixed_qkv,
                &z,
                &beta_raw,
                &a,
                compute_dtype,
            )?;
        profile.add_assign(&linear_profile);
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output, recurrent_state, profile))
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_profiled(hidden_states, attention_mask)
            .map(|(output, _)| output)
    }

    pub(super) fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}
