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
        Some(tensor_to_state(
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
    Ok(Some(tensor_to_state(state)?))
}

pub(crate) fn concat_last_dim(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    tensor_to_state(Tensor::cat(&[lhs.tensor(), rhs.tensor()], candle_core::D::Minus1)?.contiguous()?)
}

pub(crate) fn pack_delta_state_scan(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<StateBuffer> {
    tensor_to_state(
        Tensor::cat(&[weighted_key_scan, k_cumdecay_scan, state_decay_feature], 3)?.contiguous()?,
    )
}

pub(crate) fn pack_delta_chunk_fused(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<StateBuffer> {
    tensor_to_state(Tensor::cat(&[weighted_key, k_cumdecay, q_state, state_decay], 2)?.contiguous()?)
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
    let recurrent_state = tensor_to_state(
        fused.tensor()
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
    let conv_state = tensor_to_state(
        fused.tensor()
            .narrow(1, seq_len * out_width, conv_dim * state_len)?
            .reshape((batch_size, conv_dim, state_len))?
            .contiguous()?,
    )?;
    Ok((mixed_qkv, g, conv_state))
}
