use super::cpu_transport as transport;
use crate::Qwen35Backend;
use crate::qwen35_minimal_impl::model::StateBuffer;
use dotcache_runtime_core::TargetSpec;
use transport::{DType, Device, Result, Tensor};

pub fn backend(target: TargetSpec) -> Qwen35Backend {
    transport::backend(target)
}

pub(crate) fn tensor_to_buffer(xs: Tensor) -> Result<StateBuffer> {
    transport::tensor_to_state(xs)
}

pub(crate) fn zeros_state(device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
    transport::zeros_state(dims.to_vec(), dtype, device)
}

pub(crate) fn zeros_tensor(device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
    transport::zeros(dims.to_vec(), dtype, device)
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
