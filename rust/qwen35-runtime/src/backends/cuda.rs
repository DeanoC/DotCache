use super::cuda_transport as transport;
use crate::{Qwen35Backend, Qwen35BackendDescriptor};
use crate::qwen35_minimal_impl::model::StateBuffer;
use dotcache_runtime_core::TargetSpec;
use transport::{DType, Device, Result, Tensor};

pub fn descriptor(target: TargetSpec) -> Qwen35BackendDescriptor {
    transport::descriptor(target)
}

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
