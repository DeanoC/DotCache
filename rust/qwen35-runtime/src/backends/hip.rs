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

fn repeat_heads_impl(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}

fn repeat_kv_impl(xs: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(xs.clone());
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![xs; repeats];
    Tensor::cat(&repeated, 2)?.reshape((b_sz, kv_heads * repeats, seq_len, head_dim))
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
    StateBuffer::from_tensor(hip_embedding_lookup(embeddings, indexes)?)
}

pub(crate) fn tensor_to_buffer(xs: Tensor) -> Result<StateBuffer> {
    StateBuffer::from_tensor(xs)
}

pub(crate) fn zeros_state(device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
    StateBuffer::from_tensor(Tensor::zeros(dims.to_vec(), dtype, device)?)
}

pub(crate) fn zeros_tensor(device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
    Tensor::zeros(dims.to_vec(), dtype, device)
}

pub(crate) fn reshape_tensor_to_buffer(xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
    StateBuffer::from_tensor(xs.reshape(dims.to_vec())?)
}

pub(crate) fn narrow_tensor_to_buffer(
    xs: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(xs.narrow(dim, start, len)?)
}

pub(crate) fn prepare_depthwise_conv_input(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Option<StateBuffer>)> {
    let mixed_qkv = match prev_state {
        Some(conv_state) => {
            let conv_state = conv_state.clone_tensor_as(mixed_qkv.dtype())?;
            Tensor::cat(&[&conv_state, mixed_qkv], 2)?
        }
        None => mixed_qkv.pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
    };
    let total_len = mixed_qkv.dim(2)?;
    let state_len = kernel_size.saturating_sub(1);
    let next_state = if state_len == 0 {
        None
    } else {
        Some(StateBuffer::from_tensor(
            mixed_qkv
                .narrow(2, total_len - state_len, state_len)?
                .contiguous()?,
        )?)
    };
    Ok((mixed_qkv, next_state))
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

    let seq_len = mixed_qkv.dim(2)?;
    let state = if seq_len >= state_len {
        mixed_qkv.narrow(2, seq_len - state_len, state_len)?.contiguous()?
    } else {
        match prev_state {
            Some(prev_state) => {
                let prev_state = prev_state.clone_tensor_as(mixed_qkv.dtype())?;
                let keep = state_len - seq_len;
                let prev_tail = prev_state.narrow(2, prev_state.dim(2)? - keep, keep)?;
                Tensor::cat(&[&prev_tail, mixed_qkv], 2)?.contiguous()?
            }
            None => {
                let zeros = Tensor::zeros(
                    vec![mixed_qkv.dim(0)?, mixed_qkv.dim(1)?, state_len - seq_len],
                    mixed_qkv.dtype(),
                    mixed_qkv.device(),
                )?;
                Tensor::cat(&[&zeros, mixed_qkv], 2)?.contiguous()?
            }
        }
    };
    Ok(Some(StateBuffer::from_tensor(state)?))
}

pub(crate) fn concat_last_dim(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(Tensor::cat(&[lhs.tensor(), rhs.tensor()], candle_core::D::Minus1)?.contiguous()?)
}

pub(crate) fn pack_delta_state_scan(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(
        Tensor::cat(&[weighted_key_scan, k_cumdecay_scan, state_decay_feature], 3)?.contiguous()?,
    )
}

pub(crate) fn pack_delta_chunk_fused(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(
        Tensor::cat(&[weighted_key, k_cumdecay, q_state, state_decay], 2)?.contiguous()?,
    )
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
    let core_attn_out = fused
        .tensor()
        .narrow(1, 0, value_dim)?
        .reshape((batch_size, seq_len, value_dim))?;
    let recurrent_state = StateBuffer::from_tensor(
        fused
            .tensor()
            .narrow(1, value_dim, num_v_heads * head_k_dim * head_v_dim)?
            .reshape((batch_size, num_v_heads, head_k_dim, head_v_dim))?
            .contiguous()?,
    )?;
    Ok((core_attn_out, recurrent_state))
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
    let packed = fused
        .tensor()
        .narrow(1, 0, seq_len * out_width)?
        .reshape((batch_size, seq_len, out_width))?;
    let mixed_qkv = packed.narrow(candle_core::D::Minus1, 0, conv_dim)?;
    let g = packed.narrow(candle_core::D::Minus1, conv_dim, num_v_heads)?;
    let conv_state = StateBuffer::from_tensor(
        fused
            .tensor()
            .narrow(1, seq_len * out_width, conv_dim * state_len)?
            .reshape((batch_size, conv_dim, state_len))?
            .contiguous()?,
    )?;
    Ok((mixed_qkv, g, conv_state))
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
    StateBuffer::from_tensor(immutable_output_projection(embedding, hidden_states.tensor())?)
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
        None => projected,
        Some(bias) => projected.broadcast_add(bias)?,
    };
    StateBuffer::from_tensor(projected)
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
    let q_and_gate = q_and_gate
        .tensor()
        .reshape((b_sz, q_len, num_heads, head_dim * 2))?;
    let query_states = hip_rms_norm(
        &q_and_gate.narrow(candle_core::D::Minus1, 0, head_dim)?,
        q_norm_weight,
        q_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let gate = q_and_gate
        .narrow(candle_core::D::Minus1, head_dim, head_dim)?
        .reshape((b_sz, q_len, num_heads * head_dim))?;
    let key_states = hip_rms_norm(
        &k_proj
            .tensor()
            .reshape((b_sz, q_len, num_kv_heads, head_dim))?,
        k_norm_weight,
        k_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let value_states = v_proj
        .tensor()
        .reshape((b_sz, q_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
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
    let query = mixed_qkv.narrow(candle_core::D::Minus1, 0, key_dim)?.reshape((
        batch_size,
        seq_len,
        num_k_heads,
        head_k_dim,
    ))?;
    let key = mixed_qkv
        .narrow(candle_core::D::Minus1, key_dim, key_dim)?
        .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?;
    let value = mixed_qkv
        .narrow(candle_core::D::Minus1, key_dim * 2, value_dim)?
        .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?;

    let query = if query.dtype() == compute_dtype {
        query
    } else {
        query.to_dtype(compute_dtype)?
    };
    let key = if key.dtype() == compute_dtype {
        key
    } else {
        key.to_dtype(compute_dtype)?
    };
    let query = hip_l2norm(&query, 1e-6)?;
    let key = hip_l2norm(&key, 1e-6)?;
    let head_repeat = num_v_heads / num_k_heads;
    let (query, key) = if repeat_kv_heads && head_repeat > 1 {
        (
            repeat_heads_impl(&query, head_repeat)?,
            repeat_heads_impl(&key, head_repeat)?,
        )
    } else {
        (query, key)
    };
    let value = if value.dtype() == compute_dtype {
        value
    } else {
        value.to_dtype(compute_dtype)?
    };
    let beta = (beta_raw.tensor().neg()?.exp()? + 1.0)?.recip()?;
    let beta = if beta.dtype() == compute_dtype {
        beta
    } else {
        beta.to_dtype(compute_dtype)?
    };
    let g = if g.dtype() == compute_dtype {
        g.clone()
    } else {
        g.to_dtype(compute_dtype)?
    };
    Ok((query, key, value, beta, g))
}

pub(crate) fn add(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(lhs.tensor().broadcast_add(rhs.tensor())?)
}

pub(crate) fn slice_last_token(xs: &StateBuffer) -> Result<StateBuffer> {
    let (_, seq_len, _) = xs.dims3()?;
    xs.narrow(1, seq_len - 1, 1)
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
    StateBuffer::from_tensor(hip_rms_norm(xs.tensor(), weight, eps, add_unit_offset)?)
}

pub(crate) fn rms_norm_gated(
    hidden_states: &StateBuffer,
    gate: &StateBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(hip_rms_norm_gated(
        hidden_states.tensor(),
        gate.tensor(),
        weight,
        eps,
    )?)
}

pub(crate) fn swiglu_mul(gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(hip_swiglu_mul(gate.tensor(), up.tensor())?)
}

pub(crate) fn l2norm(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    StateBuffer::from_tensor(hip_l2norm(xs.tensor(), eps)?)
}

pub(crate) fn cumsum_last_dim(xs: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(hip_cumsum_last_dim(xs.tensor())?)
}

pub(crate) fn value_decay(
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(hip_value_decay(a.tensor(), dt_bias, a_log_exp)?)
}

pub(crate) fn full_attention_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(full_attention_prefill_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?)
}

pub(crate) fn full_attention_decode(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(full_attention_decode_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?)
}

pub(crate) fn wrap_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    Ok((StateBuffer::from_tensor(key_states)?, StateBuffer::from_tensor(value_states)?))
}

pub(crate) fn prepare_full_attention_output(
    attn_output: &Tensor,
    gate: &Tensor,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    let attn_output = attn_output
        .transpose(1, 2)?
        .reshape((b_sz, q_len, attention_size))?
        .to_dtype(hidden_dtype)?;
    let gate = (gate.neg()?.exp()? + 1.0)?.recip()?;
    StateBuffer::from_tensor(attn_output.broadcast_mul(&gate)?)
}

pub(crate) fn append_full_attention_kv(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match (prev_k, prev_v) {
        (Some(prev_k), Some(prev_v)) => {
            let prev_k = prev_k.clone_tensor_as(key_states.dtype())?;
            let prev_v = prev_v.clone_tensor_as(value_states.dtype())?;
            Ok((
                Tensor::cat(&[&prev_k, key_states], 2)?,
                Tensor::cat(&[&prev_v, value_states], 2)?,
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
        query_states.contiguous()?,
        key_states.contiguous()?,
        value_states.contiguous()?,
    ))
}

pub(crate) fn materialize_full_attention_dense_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_kv_groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let key_states = repeat_kv_impl(key_states, num_kv_groups)?.contiguous()?;
    let value_states = repeat_kv_impl(value_states, num_kv_groups)?.contiguous()?;
    Ok((
        query_states.to_dtype(DType::F32)?,
        key_states.to_dtype(DType::F32)?,
        value_states.to_dtype(DType::F32)?,
    ))
}

pub(crate) fn dense_full_attention_fallback(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
    let mut attn_weights = (query_states_f.matmul(&key_states_t)? * scale)?;
    if let Some(mask) = attention_mask {
        attn_weights = attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?;
    }
    let max = attn_weights.max_keepdim(candle_core::D::Minus1)?;
    let diff = attn_weights.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(candle_core::D::Minus1)?;
    let attn_weights = num.broadcast_div(&den)?;
    attn_weights.matmul(value_states_f)
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
    StateBuffer::from_tensor(linear_decode_step_hip(
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
    StateBuffer::from_tensor(linear_stateful_conv_value_decay_with_state_hip(
        mixed_qkv.tensor(),
        prev_state,
        weights,
        a.tensor(),
        dt_bias,
        a_log_exp,
        kernel_size,
    )?)
}

pub(crate) fn delta_recurrent_prefill(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_recurrent_prefill(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
}

pub(crate) fn delta_chunk_single_prefill(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_single_prefill(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
}

pub(crate) fn delta_chunk_scan_raw(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_scan_raw(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
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
    let output_scan = fused.tensor().narrow(1, 0, total_sequence_length)?.reshape((
        batch_size,
        num_heads,
        total_sequence_length,
        v_head_dim,
    ))?;
    let output = output_scan
        .narrow(2, 0, output_sequence_length)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(output_dtype)?;
    let recurrent_state = fused
        .tensor()
        .narrow(1, total_sequence_length, k_head_dim)?
        .reshape((batch_size * num_heads, k_head_dim, v_head_dim))?
        .contiguous()?;
    Ok((
        StateBuffer::from_tensor(output)?,
        StateBuffer::from_tensor(recurrent_state)?,
    ))
}

pub(crate) fn state_scan_chunk(state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
    use candle_core::IndexOp;
    StateBuffer::from_tensor(state_scan.tensor().i((.., chunk_idx, .., ..))?)
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    use candle_core::IndexOp;
    StateBuffer::from_tensor(state_scan.tensor().i((.., next_chunk_idx, .., ..))?.contiguous()?)
}

pub(crate) fn unpack_chunk_fused(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    Ok((
        StateBuffer::from_tensor(fused.tensor().narrow(1, 0, chunk_size)?)?,
        StateBuffer::from_tensor(fused.tensor().narrow(1, chunk_size, chunk_size)?)?,
        StateBuffer::from_tensor(fused.tensor().narrow(1, 2 * chunk_size, k_head_dim)?)?,
    ))
}

pub(crate) fn delta_base_attn_scan(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_base_attn_scan(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
}

pub(crate) fn delta_attn_solve_from_inputs(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_attn_solve_from_inputs(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
}

pub(crate) fn delta_attn_solve_scan(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_attn_solve_scan(
        base_attn_scan.tensor(),
    )?)
}

pub(crate) fn delta_local_attn_scan(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_local_attn_scan(
        query_scan,
        key_scan,
        exp_g_scan,
    )?)
}

pub(crate) fn delta_full_scan_pack(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan_pack(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )?)
}

pub(crate) fn delta_full_scan_packed(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    local_attn_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan_packed(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )?)
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
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_full_scan(
        initial_state.tensor(),
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan.tensor(),
        state_decay_scan,
        value,
    )?)
}

pub(crate) fn delta_state_scan(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_state_scan(
        initial_state.tensor(),
        packed_scan.tensor(),
        value,
    )?)
}

pub(crate) fn delta_chunk_fused(
    prev_state: &StateBuffer,
    packed_chunk: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_chunk_fused(
        prev_state.tensor(),
        packed_chunk.tensor(),
        value,
    )?)
}

pub(crate) fn delta_chunk_recurrent_read(
    prev_state: &StateBuffer,
    k_cumdecay_chunk: &Tensor,
    q_state_chunk: &Tensor,
    value_chunk: &Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    let v_prime = k_cumdecay_chunk.matmul(prev_state.tensor())?;
    let v_new = value_chunk.broadcast_sub(&v_prime)?;
    let attn_inter = q_state_chunk.matmul(prev_state.tensor())?;
    Ok((
        StateBuffer::from_tensor(v_new)?,
        StateBuffer::from_tensor(attn_inter)?,
    ))
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(
        attn_inter
            .tensor()
            .broadcast_add(&attn.matmul(value_chunk.tensor())?)?,
    )
}

pub(crate) fn delta_state_update(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(crate::qwen35_minimal_impl::model::delta_state_update(
        prev_state_scaled,
        weighted_key,
        value.tensor(),
        use_kernel,
    )?)
}
