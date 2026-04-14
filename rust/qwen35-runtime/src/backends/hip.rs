use super::hip_transport as transport;
use crate::{Qwen35Backend, Qwen35BackendDescriptor};
use crate::qwen35_minimal_impl::model::{ImmutableEmbedding, StateBuffer};
use dotcache_runtime_core::{BackendKind, TargetSpec};
use transport::{DType, Device, Result, Tensor};

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
    transport::embedding_lookup_buffer(embeddings, indexes)
}

pub(crate) fn tensor_to_buffer(xs: Tensor) -> Result<StateBuffer> {
    transport::tensor_to_state(xs)
}

pub(crate) fn state_buffer_from_host_bytes(
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
    device: &Device,
) -> Result<StateBuffer> {
    transport::state_buffer_from_host_bytes(bytes, shape, dtype, device)
}

pub(crate) fn zeros_state(device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
    transport::zeros_state(dims.to_vec(), dtype, device)
}

pub(crate) fn copy_state_into_scratch(
    src: &StateBuffer,
    scratch: &StateBuffer,
) -> Result<StateBuffer> {
    transport::copy_state_into_scratch(src, scratch)
}

pub(crate) fn zeros_tensor(device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
    transport::zeros(dims.to_vec(), dtype, device).map(|t| t.into_tensor())
}

pub(crate) fn reshape_tensor_to_buffer(xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
    transport::reshape_tensor_to_state(xs, dims)
}

pub(crate) fn narrow_tensor_to_buffer(
    xs: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<StateBuffer> {
    transport::narrow_tensor_to_state(xs, dim, start, len)
}

pub(crate) fn prepare_depthwise_conv_input(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Option<StateBuffer>)> {
    transport::prepare_depthwise_conv_input(prev_state, mixed_qkv, kernel_size)
}

pub(crate) fn update_depthwise_conv_state(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<Option<StateBuffer>> {
    transport::update_depthwise_conv_state(prev_state, mixed_qkv, kernel_size)
}

pub(crate) fn concat_last_dim(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    transport::concat_last_dim(lhs, rhs)
}

pub(crate) fn pack_delta_state_scan(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<StateBuffer> {
    transport::pack_delta_state_scan(weighted_key_scan, k_cumdecay_scan, state_decay_feature)
}

pub(crate) fn pack_delta_chunk_fused(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<StateBuffer> {
    transport::pack_delta_chunk_fused(weighted_key, k_cumdecay, q_state, state_decay)
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
    transport::unpack_linear_decode_output(
        fused,
        batch_size,
        seq_len,
        value_dim,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
}

pub(crate) fn unpack_linear_prefill_output(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    num_v_heads: usize,
    state_len: usize,
) -> Result<(Tensor, Tensor, StateBuffer)> {
    transport::unpack_linear_prefill_output(
        fused,
        batch_size,
        seq_len,
        conv_dim,
        num_v_heads,
        state_len,
    )
}

pub(crate) fn immutable_embedding_lookup(
    embedding: &ImmutableEmbedding,
    input_ids: &Tensor,
) -> Result<Tensor> {
    transport::immutable_embedding_lookup(embedding, input_ids).map(|t| t.into_tensor())
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &StateBuffer,
) -> Result<StateBuffer> {
    transport::output_projection_buffer(embedding, hidden_states)
}

pub(crate) fn output_projection_into_scratch(
    embedding: &ImmutableEmbedding,
    hidden_states: &StateBuffer,
    scratch: &StateBuffer,
) -> Result<StateBuffer> {
    let output = output_projection(embedding, hidden_states)?;
    copy_state_into_scratch(&output, scratch)
}

pub(crate) fn linear_forward(
    x: &StateBuffer,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<StateBuffer> {
    transport::linear_forward(x, weight, bias)
}

pub(crate) fn linear_forward_into_scratch(
    x: &StateBuffer,
    weight: &Tensor,
    bias: Option<&Tensor>,
    scratch: &StateBuffer,
) -> Result<StateBuffer> {
    let output = linear_forward(x, weight, bias)?;
    copy_state_into_scratch(&output, scratch)
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
) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer)> {
    transport::prepare_full_attention_inputs(
        q_and_gate,
        k_proj,
        v_proj,
        b_sz,
        q_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_norm_weight,
        q_norm_eps,
        k_norm_weight,
        k_norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_full_attention_inputs_into_scratch(
    q_and_gate: &StateBuffer,
    k_proj: &StateBuffer,
    v_proj: &StateBuffer,
    gate_scratch: &StateBuffer,
    query_scratch: &StateBuffer,
    key_scratch: &StateBuffer,
    value_scratch: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_norm_weight: &Tensor,
    q_norm_eps: f64,
    k_norm_weight: &Tensor,
    k_norm_eps: f64,
) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer)> {
    transport::prepare_full_attention_inputs_into_scratch(
        q_and_gate,
        k_proj,
        v_proj,
        gate_scratch,
        query_scratch,
        key_scratch,
        value_scratch,
        b_sz,
        q_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_norm_weight,
        q_norm_eps,
        k_norm_weight,
        k_norm_eps,
    )
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
    transport::prepare_linear_attention_inputs(
        mixed_qkv,
        beta_raw,
        g,
        batch_size,
        seq_len,
        key_dim,
        value_dim,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        compute_dtype,
        repeat_kv_heads,
    )
}

pub(crate) fn add(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    transport::add(lhs, rhs)
}

pub(crate) fn slice_last_token(xs: &StateBuffer) -> Result<StateBuffer> {
    transport::slice_last_token(xs)
}

pub(crate) fn causal_mask(
    device: &Device,
    dtype: DType,
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Tensor> {
    transport::causal_mask(device, dtype, b_size, tgt_len, seqlen_offset).map(|t| t.into_tensor())
}

pub(crate) fn rms_norm(
    xs: &StateBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<StateBuffer> {
    transport::rms_norm_buffer(xs, weight, eps, add_unit_offset)
}

pub(crate) fn rms_norm_gated(
    hidden_states: &StateBuffer,
    gate: &StateBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<StateBuffer> {
    transport::rms_norm_gated_buffer(hidden_states, gate, weight, eps)
}

pub(crate) fn swiglu_mul(gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
    transport::swiglu_mul_buffer(gate, up)
}

pub(crate) fn l2norm(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    transport::l2norm_buffer(xs, eps)
}

pub(crate) fn cumsum_last_dim(xs: &StateBuffer) -> Result<StateBuffer> {
    transport::cumsum_last_dim_buffer(xs)
}

pub(crate) fn value_decay(
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<StateBuffer> {
    transport::value_decay_buffer(a, dt_bias, a_log_exp)
}

pub(crate) fn full_attention_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    transport::full_attention_prefill_buffer(query, key, value, num_kv_groups, scale, seqlen_offset)
}

pub(crate) fn full_attention_decode(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    transport::full_attention_decode_buffer(query, key, value, num_kv_groups, scale, seqlen_offset)
}

pub(crate) fn prepare_full_attention_output(
    attn_output: &Tensor,
    gate: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    transport::prepare_full_attention_output(
        attn_output,
        gate,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )
}

pub(crate) fn prepare_full_attention_output_buffer(
    attn_output: &StateBuffer,
    gate: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    transport::prepare_full_attention_output_buffer(
        attn_output,
        gate,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )
}

pub(crate) fn append_full_attention_kv(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor)> {
    transport::append_full_attention_kv(prev_k, prev_v, key_states, value_states)
}

pub(crate) fn append_full_attention_kv_buffers(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    transport::append_full_attention_kv_buffers(prev_k, prev_v, key_states, value_states)
}

pub(crate) fn prepare_full_attention_kernel_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    transport::prepare_full_attention_kernel_inputs(query_states, key_states, value_states)
}

pub(crate) fn prepare_full_attention_kernel_inputs_with_buffer_kv(
    query_states: &StateBuffer,
    key_states: &StateBuffer,
    value_states: &StateBuffer,
) -> Result<(Tensor, Tensor, Tensor)> {
    transport::prepare_full_attention_kernel_inputs_with_buffer_kv(
        query_states,
        key_states,
        value_states,
    )
}

pub(crate) fn prepare_full_attention_kernel_input_buffers_with_buffer_kv(
    query_states: &StateBuffer,
    key_states: &StateBuffer,
    value_states: &StateBuffer,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    transport::prepare_full_attention_kernel_input_buffers_with_buffer_kv(
        query_states,
        key_states,
        value_states,
    )
}

pub(crate) fn rope_buffer(xs: &StateBuffer, cos: &Tensor, sin: &Tensor) -> Result<StateBuffer> {
    transport::rope_buffer(xs, cos, sin)
}

pub(crate) fn materialize_full_attention_dense_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_kv_groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    transport::materialize_full_attention_dense_inputs(
        query_states,
        key_states,
        value_states,
        num_kv_groups,
    )
}

pub(crate) fn dense_full_attention_fallback(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    transport::dense_full_attention_fallback(
        query_states_f,
        key_states_f,
        value_states_f,
        attention_mask,
        scale,
    )
}

pub(crate) fn dense_full_attention_fallback_buffer(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
    gate: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    transport::dense_full_attention_fallback_buffer(
        query_states_f,
        key_states_f,
        value_states_f,
        attention_mask,
        scale,
        gate,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )
}

pub(crate) fn linear_prefill_conv(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Tensor> {
    transport::linear_prefill_conv(mixed_qkv, weights, seq_len, kernel_size).map(|t| t.into_tensor())
}

pub(crate) fn linear_stateful_conv(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    transport::linear_stateful_conv(mixed_qkv, prev_state, weights, kernel_size).map(|t| t.into_tensor())
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
    transport::linear_decode_step_buffer(
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

pub(crate) fn linear_stateful_conv_value_decay_with_state(
    mixed_qkv: &StateBuffer,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<StateBuffer> {
    transport::linear_stateful_conv_value_decay_with_state_buffer(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )
}

pub(crate) fn delta_recurrent_prefill(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_recurrent_prefill_buffer(
        initial_state,
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )
}

pub(crate) fn delta_chunk_single_prefill(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_chunk_single_prefill_buffer(initial_state, query, key, value, beta, g)
}

pub(crate) fn delta_chunk_scan_raw(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_chunk_scan_raw_buffer(
        initial_state,
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )
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
    transport::unpack_scan_fused_output_and_state(
        fused,
        total_sequence_length,
        output_sequence_length,
        batch_size,
        num_heads,
        v_head_dim,
        k_head_dim,
        output_dtype,
    )
}

pub(crate) fn state_scan_chunk(state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
    transport::state_scan_chunk(state_scan, chunk_idx)
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    transport::state_scan_next_chunk(state_scan, next_chunk_idx)
}

pub(crate) fn unpack_chunk_fused(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    transport::unpack_chunk_fused(fused, chunk_size, k_head_dim)
}

pub(crate) fn unpack_delta_chunk_step_output(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer)> {
    transport::unpack_delta_chunk_step_output(fused, chunk_size, k_head_dim)
}

pub(crate) fn delta_base_attn_scan(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_base_attn_scan_buffer(k_beta_scan, key_scan, exp_g_scan)
}

pub(crate) fn delta_attn_solve_from_inputs(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_attn_solve_from_inputs_buffer(k_beta_scan, key_scan, exp_g_scan)
}

pub(crate) fn delta_attn_solve_scan(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    transport::delta_attn_solve_scan_buffer(base_attn_scan)
}

pub(crate) fn delta_local_attn_scan(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_local_attn_scan_buffer(query_scan, key_scan, exp_g_scan)
}

pub(crate) fn delta_full_scan_pack(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_full_scan_pack_buffer(query_scan, key_scan, exp_g_scan, k_cumdecay_scan)
}

pub(crate) fn delta_full_scan_packed(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    local_attn_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_full_scan_packed_buffer(initial_state, packed_scan, local_attn_scan, value)
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
    transport::delta_full_scan_buffer(
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
    )
}

pub(crate) fn delta_state_scan(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_state_scan_buffer(initial_state, packed_scan, value)
}

pub(crate) fn delta_chunk_fused(
    prev_state: &StateBuffer,
    packed_chunk: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    transport::delta_chunk_fused_buffer(prev_state, packed_chunk, value)
}

pub(crate) fn delta_chunk_recurrent_read(
    prev_state: &StateBuffer,
    k_cumdecay_chunk: &Tensor,
    q_state_chunk: &Tensor,
    value_chunk: &Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    transport::delta_chunk_recurrent_read(prev_state, k_cumdecay_chunk, q_state_chunk, value_chunk)
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    transport::mix_chunk_attention(attn, attn_inter, value_chunk)
}

pub(crate) fn delta_state_update(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    transport::delta_state_update_buffer(prev_state_scaled, weighted_key, value, use_kernel)
}
