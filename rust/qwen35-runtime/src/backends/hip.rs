use crate::{Qwen35Backend, Qwen35BackendDescriptor};
use crate::qwen35_minimal_impl::model::{
    full_attention_decode_megakernel, full_attention_prefill_megakernel, hip_causal_mask,
    hip_cumsum_last_dim, hip_embedding_lookup, hip_immutable_embedding_lookup, hip_l2norm,
    hip_rms_norm, hip_rms_norm_gated, hip_swiglu_mul, hip_value_decay,
    immutable_output_projection,
    linear_decode_step_hip, linear_prefill_conv_pack, linear_stateful_conv_hip,
    linear_stateful_conv_value_decay_with_state_hip, ImmutableEmbedding, StateBuffer,
};
use candle_core::{DType, Device, Result, Tensor};
use dotcache_runtime_core::{BackendKind, TargetSpec};

#[derive(Debug, Clone)]
struct HipBuffer(Tensor);

impl HipBuffer {
    fn from_tensor(tensor: Tensor) -> Self {
        Self(tensor)
    }

    fn into_tensor(self) -> Tensor {
        self.0
    }

    fn contiguous(&self) -> Result<Self> {
        Ok(Self(self.0.contiguous()?))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.0.dtype() == dtype {
            Ok(self.clone())
        } else {
            Ok(Self(self.0.to_dtype(dtype)?))
        }
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        Ok(Self(self.0.transpose(dim1, dim2)?))
    }

    fn reshape<T: candle_core::shape::ShapeWithOneHole>(&self, shape: T) -> Result<Self> {
        Ok(Self(self.0.reshape(shape)?))
    }

    fn expand<S: Into<candle_core::Shape>>(&self, shape: S) -> Result<Self> {
        Ok(Self(self.0.expand(shape)?))
    }

    fn narrow(&self, dim: impl candle_core::shape::Dim, start: usize, len: usize) -> Result<Self> {
        Ok(Self(self.0.narrow(dim, start, len)?))
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.matmul(&rhs.0)?))
    }

    fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_add(&rhs.0)?))
    }

    fn exp(&self) -> Result<Self> {
        Ok(Self(self.0.exp()?))
    }

    fn max_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        Ok(Self(self.0.max_keepdim(dim)?))
    }

    fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_sub(&rhs.0)?))
    }

    fn sum_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        Ok(Self(self.0.sum_keepdim(dim)?))
    }

    fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_div(&rhs.0)?))
    }

    fn sigmoid(&self) -> Result<Self> {
        Ok(Self((self.0.neg()?.exp()? + 1.0)?.recip()?))
    }

    fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        Ok(Self(self.0.pad_with_zeros(dim, left, right)?))
    }

    fn dim(&self, dim: usize) -> Result<usize> {
        self.0.dim(dim)
    }

    fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.0.dims3()
    }

    fn cat(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        Ok(Self(Tensor::cat(tensors, dim)?))
    }

    fn into_state_buffer(self) -> Result<StateBuffer> {
        StateBuffer::from_tensor(self.0)
    }
}

fn repeat_heads_impl(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    Ok(HipBuffer::from_tensor(xs.clone())
        .reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))?
        .into_tensor())
}

fn repeat_kv_impl(xs: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(xs.clone());
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![xs; repeats];
    Ok(HipBuffer::cat(&repeated, 2)?
        .reshape((b_sz, kv_heads * repeats, seq_len, head_dim))?
        .into_tensor())
}

pub fn descriptor(target: TargetSpec) -> Qwen35BackendDescriptor {
    debug_assert!(matches!(target.backend, BackendKind::Hip));
    Qwen35BackendDescriptor {
        target,
        optimized: true,
    }
}

pub fn backend(target: TargetSpec) -> Qwen35Backend {
    Qwen35Backend {
        descriptor: descriptor(target),
    }
}

pub(crate) fn embedding_lookup(embeddings: &Tensor, indexes: &Tensor) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_embedding_lookup(embeddings, indexes)?).into_state_buffer()
}

pub(crate) fn tensor_to_buffer(xs: Tensor) -> Result<StateBuffer> {
    HipBuffer::from_tensor(xs).into_state_buffer()
}

pub(crate) fn zeros_state(device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
    HipBuffer::from_tensor(Tensor::zeros(dims.to_vec(), dtype, device)?).into_state_buffer()
}

pub(crate) fn zeros_tensor(device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
    Tensor::zeros(dims.to_vec(), dtype, device)
}

pub(crate) fn reshape_tensor_to_buffer(xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
    HipBuffer::from_tensor(xs.reshape(dims.to_vec())?).into_state_buffer()
}

pub(crate) fn narrow_tensor_to_buffer(
    xs: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(xs.narrow(dim, start, len)?).into_state_buffer()
}

pub(crate) fn prepare_depthwise_conv_input(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Option<StateBuffer>)> {
    let mixed_qkv = match prev_state {
        Some(conv_state) => {
            let conv_state =
                HipBuffer::from_tensor(conv_state.clone_tensor_as(mixed_qkv.dtype())?);
            HipBuffer::cat(&[&conv_state.into_tensor(), mixed_qkv], 2)?.into_tensor()
        }
        None => HipBuffer::from_tensor(mixed_qkv.clone())
            .pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?
            .into_tensor(),
    };
    let mixed_qkv_buf = HipBuffer::from_tensor(mixed_qkv);
    let total_len = mixed_qkv_buf.dim(2)?;
    let state_len = kernel_size.saturating_sub(1);
    let next_state = if state_len == 0 {
        None
    } else {
        Some(
            mixed_qkv_buf
                .narrow(2, total_len - state_len, state_len)?
                .contiguous()?
                .into_state_buffer()?,
        )
    };
    Ok((mixed_qkv_buf.into_tensor(), next_state))
}

pub(crate) fn update_depthwise_conv_state(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<Option<StateBuffer>> {
    let state_len = kernel_size.saturating_sub(1);
    if state_len == 0 {
        return Ok(None);
    }

    let mixed_qkv = HipBuffer::from_tensor(mixed_qkv.clone());
    let seq_len = mixed_qkv.dim(2)?;
    let state = if seq_len >= state_len {
        mixed_qkv.narrow(2, seq_len - state_len, state_len)?.contiguous()?
    } else {
        match prev_state {
            Some(prev_state) => {
                let prev_state = HipBuffer::from_tensor(prev_state.clone_tensor_as(mixed_qkv.0.dtype())?);
                let keep = state_len - seq_len;
                let prev_tail = prev_state.narrow(2, prev_state.dim(2)? - keep, keep)?;
                HipBuffer::cat(&[&prev_tail.into_tensor(), &mixed_qkv.0], 2)?.contiguous()?
            }
            None => {
                let zeros = Tensor::zeros(
                    vec![mixed_qkv.dim(0)?, mixed_qkv.dim(1)?, state_len - seq_len],
                    mixed_qkv.0.dtype(),
                    mixed_qkv.0.device(),
                )?;
                HipBuffer::cat(&[&zeros, &mixed_qkv.0], 2)?.contiguous()?
            }
        }
    };
    Ok(Some(state.into_state_buffer()?))
}

pub(crate) fn concat_last_dim(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    HipBuffer::cat(&[lhs.tensor(), rhs.tensor()], lhs.tensor().dims().len() - 1)?
        .contiguous()?
        .into_state_buffer()
}

pub(crate) fn pack_delta_state_scan(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::cat(&[weighted_key_scan, k_cumdecay_scan, state_decay_feature], 3)?
        .contiguous()?
        .into_state_buffer()
}

pub(crate) fn pack_delta_chunk_fused(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::cat(&[weighted_key, k_cumdecay, q_state, state_decay], 2)?
        .contiguous()?
        .into_state_buffer()
}

pub(crate) fn unpack_linear_decode_output(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<(Tensor, StateBuffer)> {
    let fused = HipBuffer::from_tensor(fused.tensor().clone());
    let core_attn_out = fused
        .narrow(1, 0, value_dim)?
        .reshape((batch_size, seq_len, value_dim))?;
    let recurrent_state = fused
        .narrow(1, value_dim, num_v_heads * head_k_dim * head_v_dim)?
        .reshape((batch_size, num_v_heads, head_k_dim, head_v_dim))?
        .contiguous()?
        .into_state_buffer()?;
    Ok((core_attn_out.into_tensor(), recurrent_state))
}

pub(crate) fn unpack_linear_prefill_output(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    num_v_heads: usize,
    state_len: usize,
) -> Result<(Tensor, Tensor, StateBuffer)> {
    let out_width = conv_dim + num_v_heads;
    let fused = HipBuffer::from_tensor(fused.tensor().clone());
    let packed = fused
        .narrow(1, 0, seq_len * out_width)?
        .reshape((batch_size, seq_len, out_width))?;
    let mixed_qkv = packed.narrow(candle_core::D::Minus1, 0, conv_dim)?;
    let g = packed.narrow(candle_core::D::Minus1, conv_dim, num_v_heads)?;
    let conv_state = fused
        .narrow(1, seq_len * out_width, conv_dim * state_len)?
        .reshape((batch_size, conv_dim, state_len))?
        .contiguous()?
        .into_state_buffer()?;
    Ok((mixed_qkv.into_tensor(), g.into_tensor(), conv_state))
}

pub(crate) fn immutable_embedding_lookup(
    embedding: &ImmutableEmbedding,
    input_ids: &Tensor,
) -> Result<Tensor> {
    hip_immutable_embedding_lookup(embedding, input_ids)
}

pub(crate) fn output_projection_tensor(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Tensor> {
    immutable_output_projection(embedding, hidden_states)
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &StateBuffer,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(immutable_output_projection(embedding, hidden_states.tensor())?)
        .into_state_buffer()
}

pub(crate) fn linear_forward(
    x: &StateBuffer,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<StateBuffer> {
    let x = x.tensor();
    let projected = match *x.dims() {
        [b1, b2, m, k] => {
            if x.is_contiguous() {
                let w = weight.t()?;
                x.reshape((b1 * b2 * m, k))?
                    .matmul(&w)?
                    .reshape((b1, b2, m, ()))?
            } else {
                let w = weight.broadcast_left((b1, b2))?.t()?;
                x.matmul(&w)?
            }
        }
        [bsize, m, k] => {
            if x.is_contiguous() {
                let w = weight.t()?;
                x.reshape((bsize * m, k))?
                    .matmul(&w)?
                    .reshape((bsize, m, ()))?
            } else {
                let w = weight.broadcast_left(bsize)?.t()?;
                x.matmul(&w)?
            }
        }
        _ => {
            let w = weight.t()?;
            x.matmul(&w)?
        }
    };
    let projected = match bias {
        None => HipBuffer::from_tensor(projected),
        Some(bias) => HipBuffer::from_tensor(projected)
            .broadcast_add(&HipBuffer::from_tensor(bias.clone()))?,
    };
    StateBuffer::from_tensor(projected.into_tensor())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_full_attention_inputs(
    q_and_gate: &StateBuffer,
    k_proj: &StateBuffer,
    v_proj: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_norm_weight: &Tensor,
    q_norm_eps: f64,
    k_norm_weight: &Tensor,
    k_norm_eps: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let q_and_gate = HipBuffer::from_tensor(q_and_gate.tensor().reshape((
        b_sz,
        q_len,
        num_heads,
        head_dim * 2,
    ))?);
    let query_states = hip_rms_norm(
        &q_and_gate
            .narrow(candle_core::D::Minus1, 0, head_dim)?
            .into_tensor(),
        q_norm_weight,
        q_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let gate = q_and_gate
        .narrow(candle_core::D::Minus1, head_dim, head_dim)?
        .reshape((b_sz, q_len, num_heads * head_dim))?
        .into_tensor();
    let key_states = hip_rms_norm(
        &HipBuffer::from_tensor(k_proj.tensor().reshape((b_sz, q_len, num_kv_heads, head_dim))?)
            .into_tensor(),
        k_norm_weight,
        k_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let value_states = HipBuffer::from_tensor(
        v_proj.tensor().reshape((b_sz, q_len, num_kv_heads, head_dim))?,
    )
    .transpose(1, 2)?
    .into_tensor();
    Ok((query_states, gate, key_states, value_states))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_linear_attention_inputs(
    mixed_qkv: &Tensor,
    beta_raw: &StateBuffer,
    g: &Tensor,
    batch_size: usize,
    seq_len: usize,
    key_dim: usize,
    value_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    compute_dtype: DType,
    repeat_kv_heads: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let query = HipBuffer::from_tensor(
        mixed_qkv
            .narrow(candle_core::D::Minus1, 0, key_dim)?
            .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?,
    )
    .to_dtype(compute_dtype)?;
    let key = HipBuffer::from_tensor(
        mixed_qkv
            .narrow(candle_core::D::Minus1, key_dim, key_dim)?
            .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?,
    )
    .to_dtype(compute_dtype)?;
    let value = HipBuffer::from_tensor(
        mixed_qkv
            .narrow(candle_core::D::Minus1, key_dim * 2, value_dim)?
            .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?,
    )
    .to_dtype(compute_dtype)?;

    let query = hip_l2norm(&query.into_tensor(), 1e-6)?;
    let key = hip_l2norm(&key.into_tensor(), 1e-6)?;
    let head_repeat = num_v_heads / num_k_heads;
    let (query, key) = if repeat_kv_heads && head_repeat > 1 {
        (
            repeat_heads_impl(&query, head_repeat)?,
            repeat_heads_impl(&key, head_repeat)?,
        )
    } else {
        (query, key)
    };
    let beta = HipBuffer::from_tensor(beta_raw.tensor().clone())
        .sigmoid()?
        .to_dtype(compute_dtype)?
        .into_tensor();
    let g = HipBuffer::from_tensor(g.clone())
        .to_dtype(compute_dtype)?
        .into_tensor();
    Ok((query, key, value.into_tensor(), beta, g))
}

pub(crate) fn add(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    HipBuffer::from_tensor(lhs.tensor().clone())
        .broadcast_add(&HipBuffer::from_tensor(rhs.tensor().clone()))?
        .into_state_buffer()
}

pub(crate) fn slice_last_token(xs: &StateBuffer) -> Result<StateBuffer> {
    let xs = HipBuffer::from_tensor(xs.tensor().clone());
    let (_, seq_len, _) = xs.dims3()?;
    xs.narrow(1, seq_len - 1, 1)?.into_state_buffer()
}

pub(crate) fn causal_mask(
    device: &Device,
    dtype: DType,
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Tensor> {
    hip_causal_mask(device, dtype, b_size, tgt_len, seqlen_offset)
}

pub(crate) fn rms_norm(
    xs: &StateBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_rms_norm(xs.tensor(), weight, eps, add_unit_offset)?)
        .into_state_buffer()
}

pub(crate) fn rms_norm_gated(
    hidden_states: &StateBuffer,
    gate: &StateBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_rms_norm_gated(
        hidden_states.tensor(),
        gate.tensor(),
        weight,
        eps,
    )?)
    .into_state_buffer()
}

pub(crate) fn swiglu_mul(gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_swiglu_mul(gate.tensor(), up.tensor())?).into_state_buffer()
}

pub(crate) fn l2norm(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_l2norm(xs.tensor(), eps)?).into_state_buffer()
}

pub(crate) fn cumsum_last_dim(xs: &StateBuffer) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_cumsum_last_dim(xs.tensor())?).into_state_buffer()
}

pub(crate) fn value_decay(
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(hip_value_decay(a.tensor(), dt_bias, a_log_exp)?).into_state_buffer()
}

pub(crate) fn full_attention_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(full_attention_prefill_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?)
    .into_state_buffer()
}

pub(crate) fn full_attention_decode(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(full_attention_decode_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?)
    .into_state_buffer()
}

pub(crate) fn wrap_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    Ok((
        HipBuffer::from_tensor(key_states).into_state_buffer()?,
        HipBuffer::from_tensor(value_states).into_state_buffer()?,
    ))
}

pub(crate) fn prepare_full_attention_output(
    attn_output: &Tensor,
    gate: &Tensor,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    let attn_output = HipBuffer::from_tensor(attn_output.clone())
        .transpose(1, 2)?
        .reshape((b_sz, q_len, attention_size))?
        .to_dtype(hidden_dtype)?;
    let gate = HipBuffer::from_tensor(gate.clone()).sigmoid()?;
    HipBuffer::from_tensor(attn_output.into_tensor().broadcast_mul(&gate.into_tensor())?)
        .into_state_buffer()
}

pub(crate) fn append_full_attention_kv(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match (prev_k, prev_v) {
        (Some(prev_k), Some(prev_v)) => {
            let prev_k = HipBuffer::from_tensor(prev_k.clone_tensor_as(key_states.dtype())?);
            let prev_v = HipBuffer::from_tensor(prev_v.clone_tensor_as(value_states.dtype())?);
            Ok((
                HipBuffer::cat(&[&prev_k.into_tensor(), key_states], 2)?.into_tensor(),
                HipBuffer::cat(&[&prev_v.into_tensor(), value_states], 2)?.into_tensor(),
            ))
        }
        _ => Ok((key_states.clone(), value_states.clone())),
    }
}

pub(crate) fn prepare_full_attention_kernel_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        HipBuffer::from_tensor(query_states.clone()).contiguous()?.into_tensor(),
        HipBuffer::from_tensor(key_states.clone()).contiguous()?.into_tensor(),
        HipBuffer::from_tensor(value_states.clone()).contiguous()?.into_tensor(),
    ))
}

pub(crate) fn materialize_full_attention_dense_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_kv_groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let key_states = HipBuffer::from_tensor(repeat_kv_impl(key_states, num_kv_groups)?)
        .contiguous()?
        .to_dtype(DType::F32)?
        .into_tensor();
    let value_states = HipBuffer::from_tensor(repeat_kv_impl(value_states, num_kv_groups)?)
        .contiguous()?
        .to_dtype(DType::F32)?
        .into_tensor();
    Ok((
        HipBuffer::from_tensor(query_states.clone())
            .to_dtype(DType::F32)?
            .into_tensor(),
        key_states,
        value_states,
    ))
}

pub(crate) fn dense_full_attention_fallback(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let key_states_t = HipBuffer::from_tensor(key_states_f.clone())
        .transpose(2, 3)?
        .contiguous()?;
    let mut attn_weights =
        HipBuffer::from_tensor((query_states_f.matmul(&key_states_t.into_tensor())? * scale)?);
    if let Some(mask) = attention_mask {
        let mask = HipBuffer::from_tensor(mask.to_dtype(DType::F32)?);
        attn_weights = attn_weights.broadcast_add(&mask)?;
    }
    let max = attn_weights.max_keepdim(candle_core::D::Minus1)?;
    let diff = attn_weights.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(candle_core::D::Minus1)?;
    let attn_weights = num.broadcast_div(&den)?;
    Ok(attn_weights
        .matmul(&HipBuffer::from_tensor(value_states_f.clone()))?
        .into_tensor())
}

pub(crate) fn linear_prefill_conv(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Tensor> {
    linear_prefill_conv_pack(mixed_qkv, weights, seq_len, kernel_size)
}

pub(crate) fn linear_stateful_conv(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    linear_stateful_conv_hip(mixed_qkv, prev_state, weights, kernel_size)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_decode_step(
    mixed_qkv: &StateBuffer,
    prev_conv_state: &Tensor,
    weights: &Tensor,
    a_beta_raw: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    initial_state: &Tensor,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel_size: usize,
    head_repeat: usize,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(linear_decode_step_hip(
        mixed_qkv.tensor(),
        prev_conv_state,
        weights,
        a_beta_raw.tensor(),
        dt_bias,
        a_log_exp,
        initial_state,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        kernel_size,
        head_repeat,
    )?)
    .into_state_buffer()
}

pub(crate) fn linear_stateful_conv_value_decay_with_state(
    mixed_qkv: &StateBuffer,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(linear_stateful_conv_value_decay_with_state_hip(
        mixed_qkv.tensor(),
        prev_state,
        weights,
        a.tensor(),
        dt_bias,
        a_log_exp,
        kernel_size,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_recurrent_prefill(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_recurrent_prefill(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_single_prefill(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_single_prefill(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_scan_raw(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_scan_raw(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn unpack_scan_fused_output_and_state(
    fused: &StateBuffer,
    total_sequence_length: usize,
    output_sequence_length: usize,
    batch_size: usize,
    num_heads: usize,
    v_head_dim: usize,
    k_head_dim: usize,
    output_dtype: DType,
) -> Result<(StateBuffer, StateBuffer)> {
    let fused = HipBuffer::from_tensor(fused.tensor().clone());
    let output_scan = fused
        .narrow(1, 0, total_sequence_length)?
        .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?;
    let output = output_scan
        .narrow(2, 0, output_sequence_length)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(output_dtype)?
        .into_tensor();
    let recurrent_state = fused
        .narrow(1, total_sequence_length, k_head_dim)?
        .reshape((batch_size * num_heads, k_head_dim, v_head_dim))?
        .contiguous()?
        .into_tensor();
    Ok((
        HipBuffer::from_tensor(output).into_state_buffer()?,
        HipBuffer::from_tensor(recurrent_state).into_state_buffer()?,
    ))
}

pub(crate) fn state_scan_chunk(state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
    use candle_core::IndexOp;
    HipBuffer::from_tensor(state_scan.tensor().i((.., chunk_idx, .., ..))?).into_state_buffer()
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    use candle_core::IndexOp;
    HipBuffer::from_tensor(state_scan.tensor().i((.., next_chunk_idx, .., ..))?)
        .contiguous()?
        .into_state_buffer()
}

pub(crate) fn unpack_chunk_fused(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    let fused = HipBuffer::from_tensor(fused.tensor().clone());
    Ok((
        fused.narrow(1, 0, chunk_size)?.into_state_buffer()?,
        fused.narrow(1, chunk_size, chunk_size)?.into_state_buffer()?,
        fused.narrow(1, 2 * chunk_size, k_head_dim)?.into_state_buffer()?,
    ))
}

pub(crate) fn delta_base_attn_scan(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_base_attn_scan(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_attn_solve_from_inputs(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_attn_solve_from_inputs(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_attn_solve_scan(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_attn_solve_scan(
        base_attn_scan.tensor(),
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_local_attn_scan(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_local_attn_scan(
        query_scan,
        key_scan,
        exp_g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_full_scan_pack(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan_pack(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_full_scan_packed(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    local_attn_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan_packed(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )?)
    .into_state_buffer()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_full_scan(
    initial_state: &StateBuffer,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &StateBuffer,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan(
        initial_state.tensor(),
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan.tensor(),
        state_decay_scan,
        value,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_state_scan(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_state_scan(
        initial_state.tensor(),
        packed_scan.tensor(),
        value,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_fused(
    prev_state: &StateBuffer,
    packed_chunk: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_fused(
        prev_state.tensor(),
        packed_chunk.tensor(),
        value,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_recurrent_read(
    prev_state: &StateBuffer,
    k_cumdecay_chunk: &Tensor,
    q_state_chunk: &Tensor,
    value_chunk: &Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    let prev_state = HipBuffer::from_tensor(prev_state.tensor().clone());
    let v_prime = HipBuffer::from_tensor(k_cumdecay_chunk.clone()).matmul(&prev_state)?;
    let v_new = HipBuffer::from_tensor(value_chunk.clone()).broadcast_sub(&v_prime)?;
    let attn_inter = HipBuffer::from_tensor(q_state_chunk.clone()).matmul(&prev_state)?;
    Ok((
        v_new.into_state_buffer()?,
        attn_inter.into_state_buffer()?,
    ))
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    let attn_value = HipBuffer::from_tensor(attn.clone())
        .matmul(&HipBuffer::from_tensor(value_chunk.tensor().clone()))?;
    let mixed = HipBuffer::from_tensor(attn_inter.tensor().clone()).broadcast_add(&attn_value)?;
    mixed.into_state_buffer()
}

pub(crate) fn delta_state_update(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    HipBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_state_update(
        prev_state_scaled,
        weighted_key,
        value.tensor(),
        use_kernel,
    )?)
    .into_state_buffer()
}
