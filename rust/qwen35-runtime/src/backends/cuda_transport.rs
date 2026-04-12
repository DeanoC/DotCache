use crate::{Qwen35Backend, Qwen35BackendDescriptor};
use crate::qwen35_minimal_impl::model::StateBuffer;
pub(crate) use candle_core::{DType, Device, Result, Tensor};
use dotcache_runtime_core::{BackendKind, TargetSpec};

pub(crate) fn descriptor(target: TargetSpec) -> Qwen35BackendDescriptor {
    debug_assert!(matches!(target.backend, BackendKind::Cuda));
    Qwen35BackendDescriptor {
        target,
        optimized: true,
    }
}

pub(crate) fn backend(target: TargetSpec) -> Qwen35Backend {
    Qwen35Backend {
        descriptor: descriptor(target),
    }
}

pub(crate) fn tensor_to_state(xs: Tensor) -> Result<StateBuffer> {
    StateBuffer::from_tensor(xs)
}

pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::zeros(dims, dtype, device)
}

pub(crate) fn zeros_state(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<StateBuffer> {
    tensor_to_state(zeros(dims, dtype, device)?)
}

pub(crate) fn reshape_tensor_to_state(xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
    tensor_to_state(xs.reshape(dims.to_vec())?)
}

pub(crate) fn narrow_tensor_to_state(
    xs: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<StateBuffer> {
    tensor_to_state(xs.narrow(dim, start, len)?)
}
