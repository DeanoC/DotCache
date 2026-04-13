use candle::{DType, Device, Result, Tensor, D};
use candle_core as candle;

use super::model::{
    delta_attn_solve_from_inputs, delta_attn_solve_scan, delta_base_attn_scan, delta_chunk_fused,
    delta_chunk_scan_raw, delta_chunk_single_prefill, delta_full_scan, delta_full_scan_pack,
    delta_full_scan_packed, delta_local_attn_scan, delta_recurrent_prefill, delta_state_scan,
    delta_state_update,
    full_attention_decode_megakernel, full_attention_prefill_megakernel, hip_causal_mask,
    hip_cumsum_last_dim, hip_embedding_lookup, hip_immutable_embedding_lookup, hip_l2norm,
    hip_rms_norm, hip_rms_norm_gated, hip_swiglu_mul, hip_value_decay,
    immutable_output_projection, linear_decode_step_hip, linear_prefill_conv_pack,
    linear_stateful_conv_hip, linear_stateful_conv_value_decay_with_state_hip, softplus,
    ImmutableEmbedding, StateBuffer,
};

pub(crate) fn embedding_lookup(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Tensor> {
    if embeddings.device().is_hip() && indexes.device().is_hip() {
        return hip_embedding_lookup(embeddings, indexes);
    }
    let mut final_dims = indexes.dims().to_vec();
    final_dims.push(embeddings.dim(1)?);
    let indexes = indexes.flatten_all()?;
    let values = embeddings.index_select(&indexes, 0)?;
    values.reshape(final_dims)
}

pub(crate) fn immutable_embedding_lookup(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Tensor> {
    if embedding.device().is_hip() && indexes.device().is_hip() {
        return hip_immutable_embedding_lookup(embedding, indexes);
    }
    embedding.fallback_forward(indexes)
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Tensor> {
    immutable_output_projection(embedding, hidden_states)
}

pub(crate) fn output_projection_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &StateBuffer,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(immutable_output_projection(embedding, hidden_states.tensor())?)
}

pub(crate) fn linear_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let x = match *x.dims() {
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
    match bias {
        None => Ok(x),
        Some(bias) => x.broadcast_add(bias),
    }
}

pub(crate) fn rms_norm(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Tensor> {
    if xs.device().is_hip() && weight.device().is_hip() {
        return hip_rms_norm(xs, weight, eps, add_unit_offset);
    }
    let xs_dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / xs.dim(D::Minus1)? as f64)?;
    let xs = xs.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let weight = if add_unit_offset {
        (weight.to_dtype(DType::F32)? + 1.0)?
    } else {
        weight.to_dtype(DType::F32)?
    };
    xs.broadcast_mul(&weight)?.to_dtype(xs_dtype)
}

pub(crate) fn rms_norm_gated(
    hidden_states: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if hidden_states.device().is_hip() && gate.device().is_hip() && weight.device().is_hip() {
        return hip_rms_norm_gated(hidden_states, gate, weight, eps);
    }
    let out_dtype = hidden_states.dtype();
    let hidden_states = hidden_states.to_dtype(DType::F32)?;
    let variance =
        (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_states.dim(D::Minus1)? as f64)?;
    let hidden_states = hidden_states.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let hidden_states = hidden_states.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    let gated = hidden_states.broadcast_mul(&super::ops::silu(&gate.to_dtype(DType::F32)?)?)?;
    gated.to_dtype(out_dtype)
}

pub(crate) fn swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_hip() && up.device().is_hip() {
        return hip_swiglu_mul(gate, up);
    }
    gate.apply(&super::activation::Activation::Silu)? * up
}

pub(crate) fn l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    if xs.device().is_hip() {
        return hip_l2norm(xs, eps);
    }
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?;
    xs.broadcast_div(&(norm + eps)?.sqrt()?)
}

pub(crate) fn causal_mask(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Tensor> {
    if device.is_hip() {
        return hip_causal_mask(device, dtype, batch_size, tgt_len, seqlen_offset);
    }
    let lower = Tensor::tril2(tgt_len, DType::U8, device)?;
    let on_true = lower.zeros_like()?.to_dtype(dtype)?;
    let on_false = Tensor::full(f32::NEG_INFINITY, (tgt_len, tgt_len), device)?.to_dtype(dtype)?;
    let mask = lower.where_cond(&on_true, &on_false)?;
    let mask = if seqlen_offset > 0 {
        let prefix = Tensor::zeros((tgt_len, seqlen_offset), dtype, device)?;
        Tensor::cat(&[&prefix, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((batch_size, 1, tgt_len, tgt_len + seqlen_offset))
}

pub(crate) fn cumsum_last_dim(xs: &Tensor) -> Result<Tensor> {
    if xs.device().is_hip() {
        return hip_cumsum_last_dim(xs);
    }
    let dims = xs.dims();
    let cols = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("cumsum-last-dim requires non-empty shape".into()))?;
    let rows = xs.elem_count() / cols;
    let flat = xs.reshape((rows, cols))?;
    let mut parts = Vec::with_capacity(cols);
    let mut running: Option<Tensor> = None;
    for idx in 0..cols {
        let col = flat.narrow(1, idx, 1)?;
        running = Some(match running {
            Some(acc) => acc.broadcast_add(&col)?,
            None => col,
        });
        parts.push(running.as_ref().expect("running cumsum initialized").clone());
    }
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?.reshape(dims)
}

pub(crate) fn value_decay(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Tensor> {
    if a.device().is_hip() && dt_bias.device().is_hip() && a_log_exp.device().is_hip() {
        return hip_value_decay(a, dt_bias, a_log_exp);
    }
    softplus(&a.broadcast_add(dt_bias)?)?
        .broadcast_mul(a_log_exp)?
        .neg()
}

pub(crate) fn full_attention_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Tensor> {
    full_attention_prefill_megakernel(query, key, value, num_kv_groups, scale, seqlen_offset)
}

pub(crate) fn full_attention_prefill_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(full_attention_prefill(
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
) -> Result<Tensor> {
    full_attention_decode_megakernel(query, key, value, num_kv_groups, scale, seqlen_offset)
}

pub(crate) fn full_attention_decode_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(full_attention_decode(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?)
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
    mixed_qkv: &Tensor,
    prev_conv_state: &Tensor,
    weights: &Tensor,
    a_beta_raw: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    initial_state: &Tensor,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel_size: usize,
    head_repeat: usize,
) -> Result<Tensor> {
    linear_decode_step_hip(
        mixed_qkv,
        prev_conv_state,
        weights,
        a_beta_raw,
        dt_bias,
        a_log_exp,
        initial_state,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        kernel_size,
        head_repeat,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_decode_step_buffer(
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
    StateBuffer::from_tensor(linear_decode_step(
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
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    linear_stateful_conv_value_decay_with_state_hip(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )
}

pub(crate) fn linear_stateful_conv_value_decay_with_state_buffer(
    mixed_qkv: &StateBuffer,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(linear_stateful_conv_value_decay_with_state(
        mixed_qkv.tensor(),
        prev_state,
        weights,
        a.tensor(),
        dt_bias,
        a_log_exp,
        kernel_size,
    )?)
}

pub(crate) fn delta_recurrent_prefill_buffer(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_recurrent_prefill(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
}

pub(crate) fn delta_chunk_single_prefill_buffer(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_chunk_single_prefill(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
}

pub(crate) fn delta_chunk_scan_raw_buffer(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_chunk_scan_raw(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
}

pub(crate) fn delta_base_attn_scan_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_base_attn_scan(k_beta_scan, key_scan, exp_g_scan)?)
}

pub(crate) fn delta_attn_solve_from_inputs_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_attn_solve_from_inputs(k_beta_scan, key_scan, exp_g_scan)?)
}

pub(crate) fn delta_attn_solve_scan_buffer(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_attn_solve_scan(base_attn_scan.tensor())?)
}

pub(crate) fn delta_local_attn_scan_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_local_attn_scan(query_scan, key_scan, exp_g_scan)?)
}

pub(crate) fn delta_full_scan_pack_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_full_scan_pack(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )?)
}

pub(crate) fn delta_full_scan_packed_buffer(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    local_attn_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_full_scan_packed(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_full_scan_buffer(
    initial_state: &StateBuffer,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &StateBuffer,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_full_scan(
        initial_state.tensor(),
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan.tensor(),
        state_decay_scan,
        value,
    )?)
}

pub(crate) fn delta_state_scan_buffer(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_state_scan(
        initial_state.tensor(),
        packed_scan.tensor(),
        value,
    )?)
}

pub(crate) fn delta_chunk_fused_buffer(
    prev_state: &StateBuffer,
    packed_chunk: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_chunk_fused(
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

pub(crate) fn delta_state_update_buffer(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    StateBuffer::from_tensor(delta_state_update(
        prev_state_scaled,
        weighted_key,
        value.tensor(),
        use_kernel,
    )?)
}
