use crate::qwen35_minimal_impl::model::{
    delta_attn_solve_from_inputs, delta_attn_solve_scan, delta_base_attn_scan, delta_chunk_fused,
    delta_chunk_scan_raw, delta_chunk_single_prefill, delta_full_scan, delta_full_scan_pack,
    delta_full_scan_packed, delta_local_attn_scan, delta_recurrent_prefill, delta_state_scan,
    delta_state_update, full_attention_decode_megakernel, full_attention_prefill_megakernel,
    hip_causal_mask, hip_cumsum_last_dim, hip_embedding_lookup, hip_immutable_embedding_lookup,
    hip_rms_norm, hip_rms_norm_gated, hip_swiglu_mul, hip_value_decay,
    immutable_output_projection, linear_decode_step_hip, linear_prefill_conv_pack,
    linear_stateful_conv_hip, linear_stateful_conv_value_decay_with_state_hip,
    ImmutableEmbedding, StateBuffer,
};
use half::{bf16, f16};
use std::sync::Arc;
use candle_core::shape::Dim;
use candle_core::IndexOp;

pub(crate) use candle_core::{DType, Device, Result, Shape, Tensor};

#[derive(Debug, Clone)]
pub(crate) enum HipNativeExpr {
    DeviceBuffer(HipDeviceBuffer),
    HostBytes {
        bytes: Arc<[u8]>,
    },
    PadWithZeros {
        source: Arc<HipNativeBuffer>,
        dim: usize,
        left: usize,
        right: usize,
    },
    Narrow {
        source: Arc<HipNativeBuffer>,
        dim: usize,
        start: usize,
        len: usize,
    },
    Concat {
        sources: Vec<Arc<HipNativeBuffer>>,
        dim: usize,
    },
    Reshape {
        source: Arc<HipNativeBuffer>,
        shape: Vec<usize>,
    },
    Expand {
        source: Arc<HipNativeBuffer>,
        shape: Vec<usize>,
    },
    Transpose {
        source: Arc<HipNativeBuffer>,
        dim1: usize,
        dim2: usize,
    },
    Cast {
        source: Arc<HipNativeBuffer>,
        dtype: DType,
    },
    Exp {
        source: Arc<HipNativeBuffer>,
    },
    BroadcastAdd {
        lhs: Arc<HipNativeBuffer>,
        rhs: Arc<HipNativeBuffer>,
    },
    BroadcastMul {
        lhs: Arc<HipNativeBuffer>,
        rhs: Arc<HipNativeBuffer>,
    },
    BroadcastSub {
        lhs: Arc<HipNativeBuffer>,
        rhs: Arc<HipNativeBuffer>,
    },
    BroadcastDiv {
        lhs: Arc<HipNativeBuffer>,
        rhs: Arc<HipNativeBuffer>,
    },
    MaxKeepdim {
        source: Arc<HipNativeBuffer>,
        dim: usize,
    },
    SumKeepdim {
        source: Arc<HipNativeBuffer>,
        dim: usize,
    },
    Neg {
        source: Arc<HipNativeBuffer>,
    },
    AddScalar {
        source: Arc<HipNativeBuffer>,
        value: f64,
    },
    MulScalar {
        source: Arc<HipNativeBuffer>,
        value: f64,
    },
    Recip {
        source: Arc<HipNativeBuffer>,
    },
    L2Norm {
        source: Arc<HipNativeBuffer>,
        eps: f64,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct HipNativeBuffer {
    expr: HipNativeExpr,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

#[derive(Debug, Clone)]
pub(crate) struct HipDeviceBuffer {
    tensor: Tensor,
}

#[derive(Debug, Clone)]
pub(crate) struct HipHostBuffer {
    bytes: Arc<[u8]>,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl HipHostBuffer {
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[cfg(test)]
    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn upload_to_device_buffer(self) -> Result<HipDeviceBuffer> {
        Ok(HipDeviceBuffer {
            tensor: Tensor::from_raw_buffer(
                self.bytes.as_ref(),
                self.dtype,
                &self.shape,
                &self.device,
            )?,
        })
    }

    pub(crate) fn upload_to_tensor(self) -> Result<Tensor> {
        self.upload_to_device_buffer().map(HipDeviceBuffer::into_tensor)
    }
}

impl HipDeviceBuffer {
    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            tensor: Tensor::zeros(dims.as_slice(), dtype, device)?,
        })
    }

    pub(crate) fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.tensor.dims();
        if dim >= dims.len() {
            candle_core::bail!("narrow dim {dim} out of range for {:?}", dims);
        }
        if start == 0 && len == dims[dim] {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.narrow(dim, start, len)?,
        })
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.pad_with_zeros(dim, left, right)?,
        })
    }

    pub(crate) fn cat(buffers: &[&HipDeviceBuffer], dim: usize) -> Result<Self> {
        let tensors = buffers.iter().map(|b| &b.tensor).collect::<Vec<_>>();
        Ok(Self {
            tensor: Tensor::cat(&tensors, dim)?,
        })
    }

    pub(crate) fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        if self.tensor.dims() == shape.as_slice() {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.reshape(shape)?,
        })
    }

    pub(crate) fn expand(&self, shape: Vec<usize>) -> Result<Self> {
        if self.tensor.dims() == shape.as_slice() {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.expand(shape)?,
        })
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.transpose(dim1, dim2)?,
        })
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.tensor.dtype() == dtype {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.to_dtype(dtype)?,
        })
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.exp()?,
        })
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        Ok(Self {
            tensor: (self.tensor.neg()?.exp()? + 1.0)?.recip()?,
        })
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.broadcast_add(&rhs.tensor)?,
        })
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.broadcast_sub(&rhs.tensor)?,
        })
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.broadcast_div(&rhs.tensor)?,
        })
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.broadcast_mul(&rhs.tensor)?,
        })
    }

    pub(crate) fn max_keepdim(&self, dim: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.max_keepdim(dim)?,
        })
    }

    pub(crate) fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.sum_keepdim(dim)?,
        })
    }

    pub(crate) fn mul_scalar(&self, value: f64) -> Result<Self> {
        Ok(Self {
            tensor: (&self.tensor * value)?,
        })
    }

    pub(crate) fn recip(&self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.recip()?,
        })
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.matmul(&rhs.tensor)?,
        })
    }

    pub(crate) fn l2norm(&self, eps: f64) -> Result<Self> {
        let norm = (self.tensor.sqr()?.sum_keepdim(candle_core::D::Minus1)? + eps)?.sqrt()?;
        Ok(Self {
            tensor: self.tensor.broadcast_div(&norm)?,
        })
    }

    pub(crate) fn rms_norm(
        &self,
        weight: &Tensor,
        eps: f64,
        add_unit_offset: bool,
    ) -> Result<Self> {
        let inner = *self.tensor.dims().last().ok_or_else(|| {
            candle_core::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into())
        })?;
        let mean_sq = (&self.tensor.sqr()?.sum_keepdim(candle_core::D::Minus1)?
            * (1.0 / inner as f64))?;
        let normed = self.tensor.broadcast_div(&(mean_sq + eps)?.sqrt()?)?;
        let weight = if weight.dtype() == self.tensor.dtype() {
            weight.clone()
        } else {
            weight.to_dtype(self.tensor.dtype())?
        };
        let weight = if add_unit_offset {
            (&weight + 1.0)?
        } else {
            weight
        };
        Ok(Self {
            tensor: normed.broadcast_mul(&weight)?,
        })
    }

    pub(crate) fn rms_norm_gated(
        &self,
        gate: &Self,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Self> {
        let normed = self.rms_norm(weight, eps, false)?;
        let sig = (gate.tensor.neg()?.exp()? + 1.0)?.recip()?;
        let silu = gate.tensor.broadcast_mul(&sig)?;
        Ok(Self {
            tensor: normed.tensor.broadcast_mul(&silu)?,
        })
    }

    pub(crate) fn swiglu_mul(&self, up: &Self) -> Result<Self> {
        let sig = (self.tensor.neg()?.exp()? + 1.0)?.recip()?;
        let silu = self.tensor.broadcast_mul(&sig)?;
        Ok(Self {
            tensor: silu.broadcast_mul(&up.tensor)?,
        })
    }

    pub(crate) fn contiguous(&self) -> Result<Self> {
        if self.tensor.is_contiguous() {
            return Ok(self.clone());
        }
        Ok(Self {
            tensor: self.tensor.contiguous()?,
        })
    }

    pub(crate) fn prepare_depthwise_conv_input(
        prev_state: Option<&HipDeviceBuffer>,
        mixed_qkv: &HipDeviceBuffer,
        kernel_size: usize,
    ) -> Result<(Self, Option<Self>)> {
        let mixed_qkv = match prev_state {
            Some(conv_state) => Self::cat(&[conv_state, mixed_qkv], 2)?,
            None => mixed_qkv.pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
        };
        let total_len = mixed_qkv.tensor.dims()[2];
        let state_len = kernel_size.saturating_sub(1);
        let next_state = if state_len == 0 {
            None
        } else {
            Some(
                mixed_qkv
                    .narrow(2, total_len - state_len, state_len)?
                    .contiguous()?,
            )
        };
        Ok((mixed_qkv, next_state))
    }

    pub(crate) fn update_depthwise_conv_state(
        prev_state: Option<&HipDeviceBuffer>,
        mixed_qkv: &HipDeviceBuffer,
        kernel_size: usize,
    ) -> Result<Option<Self>> {
        let state_len = kernel_size.saturating_sub(1);
        if state_len == 0 {
            return Ok(None);
        }

        let seq_len = mixed_qkv.tensor.dims()[2];
        let state = if seq_len >= state_len {
            mixed_qkv
                .narrow(2, seq_len - state_len, state_len)?
                .contiguous()?
        } else {
            match prev_state {
                Some(prev_state) => {
                    let keep = state_len - seq_len;
                    let prev_tail =
                        prev_state.narrow(2, prev_state.tensor.dims()[2] - keep, keep)?;
                    Self::cat(&[&prev_tail, mixed_qkv], 2)?.contiguous()?
                }
                None => {
                    let zeros = Self::zeros(
                        vec![
                            mixed_qkv.tensor.dims()[0],
                            mixed_qkv.tensor.dims()[1],
                            state_len - seq_len,
                        ],
                        mixed_qkv.tensor.dtype(),
                        mixed_qkv.tensor.device(),
                    )?;
                    Self::cat(&[&zeros, mixed_qkv], 2)?.contiguous()?
                }
            }
        };
        Ok(Some(state))
    }

    pub(crate) fn concat_last_dim(lhs: &HipDeviceBuffer, rhs: &HipDeviceBuffer) -> Result<Self> {
        Self::cat(&[lhs, rhs], lhs.tensor.dims().len() - 1)?.contiguous()
    }

    pub(crate) fn pack_delta_state_scan(
        weighted_key_scan: &HipDeviceBuffer,
        k_cumdecay_scan: &HipDeviceBuffer,
        state_decay_feature: &HipDeviceBuffer,
    ) -> Result<Self> {
        Self::cat(
            &[weighted_key_scan, k_cumdecay_scan, state_decay_feature],
            3,
        )?
        .contiguous()
    }

    pub(crate) fn pack_delta_chunk_fused(
        weighted_key: &HipDeviceBuffer,
        k_cumdecay: &HipDeviceBuffer,
        q_state: &HipDeviceBuffer,
        state_decay: &HipDeviceBuffer,
    ) -> Result<Self> {
        Self::cat(&[weighted_key, k_cumdecay, q_state, state_decay], 2)?.contiguous()
    }

    pub(crate) fn unpack_linear_decode_output(
        &self,
        batch_size: usize,
        seq_len: usize,
        value_dim: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<(Self, Self)> {
        let core_attn_out = self
            .narrow(1, 0, value_dim)?
            .reshape(vec![batch_size, seq_len, value_dim])?;
        let recurrent_state = self
            .narrow(1, value_dim, num_v_heads * head_k_dim * head_v_dim)?
            .reshape(vec![batch_size, num_v_heads, head_k_dim, head_v_dim])?
            .contiguous()?;
        Ok((core_attn_out, recurrent_state))
    }

    pub(crate) fn unpack_linear_prefill_output(
        &self,
        batch_size: usize,
        seq_len: usize,
        conv_dim: usize,
        num_v_heads: usize,
        state_len: usize,
    ) -> Result<(Self, Self, Self)> {
        let out_width = conv_dim + num_v_heads;
        let packed = self
            .narrow(1, 0, seq_len * out_width)?
            .reshape(vec![batch_size, seq_len, out_width])?;
        let last_dim = packed.tensor.dims().len() - 1;
        let mixed_qkv = packed.narrow(last_dim, 0, conv_dim)?;
        let g = packed.narrow(last_dim, conv_dim, num_v_heads)?;
        let conv_state = self
            .narrow(1, seq_len * out_width, conv_dim * state_len)?
            .reshape(vec![batch_size, conv_dim, state_len])?
            .contiguous()?;
        Ok((mixed_qkv, g, conv_state))
    }

    pub(crate) fn unpack_scan_fused_output_and_state(
        &self,
        total_sequence_length: usize,
        output_sequence_length: usize,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
        k_head_dim: usize,
        output_dtype: DType,
    ) -> Result<(Self, Self)> {
        let output_scan = self
            .narrow(1, 0, total_sequence_length)?
            .reshape(vec![batch_size, num_heads, total_sequence_length, v_head_dim])?;
        let output = output_scan
            .narrow(2, 0, output_sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .into_tensor()
            .to_dtype(output_dtype)?;
        let recurrent_state = self
            .narrow(1, total_sequence_length, k_head_dim)?
            .reshape(vec![batch_size * num_heads, k_head_dim, v_head_dim])?
            .contiguous()?;
        Ok((Self { tensor: output }, recurrent_state))
    }

    pub(crate) fn state_scan_chunk(&self, chunk_idx: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.i((.., chunk_idx, .., ..))?,
        })
    }

    pub(crate) fn state_scan_next_chunk(&self, next_chunk_idx: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.i((.., next_chunk_idx, .., ..))?.contiguous()?,
        })
    }

    pub(crate) fn unpack_chunk_fused(
        &self,
        chunk_size: usize,
        k_head_dim: usize,
    ) -> Result<(Self, Self, Self)> {
        Ok((
            self.narrow(1, 0, chunk_size)?,
            self.narrow(1, chunk_size, chunk_size)?,
            self.narrow(1, 2 * chunk_size, k_head_dim)?,
        ))
    }

    pub(crate) fn repeat_heads(&self, n_rep: usize) -> Result<Self> {
        let [b_sz, seq_len, heads, head_dim] = <[usize; 4]>::try_from(self.tensor.dims())
            .map_err(|_| candle_core::Error::Msg("unexpected rank, expected 4".into()))?;
        if n_rep == 1 {
            return Ok(self.clone());
        }
        self.reshape(vec![b_sz, seq_len, heads, 1, head_dim])?
            .expand(vec![b_sz, seq_len, heads, n_rep, head_dim])?
            .reshape(vec![b_sz, seq_len, heads * n_rep, head_dim])
    }

    pub(crate) fn repeat_kv(&self, repeats: usize) -> Result<Self> {
        let [b_sz, kv_heads, seq_len, head_dim] = <[usize; 4]>::try_from(self.tensor.dims())
            .map_err(|_| candle_core::Error::Msg("unexpected rank, expected 4".into()))?;
        if repeats <= 1 {
            return Ok(self.clone());
        }
        self.reshape(vec![b_sz, kv_heads, 1, seq_len, head_dim])?
            .expand(vec![b_sz, kv_heads, repeats, seq_len, head_dim])?
            .reshape(vec![b_sz, kv_heads * repeats, seq_len, head_dim])
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

impl HipNativeBuffer {
    fn direct_device_buffer(&self) -> Option<&HipDeviceBuffer> {
        match &self.expr {
            HipNativeExpr::DeviceBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }

    fn is_host_graph(&self) -> bool {
        match &self.expr {
            HipNativeExpr::DeviceBuffer(_) => false,
            HipNativeExpr::HostBytes { .. } => true,
            HipNativeExpr::PadWithZeros { source, .. }
            | HipNativeExpr::Narrow { source, .. }
            | HipNativeExpr::Reshape { source, .. }
            | HipNativeExpr::Expand { source, .. }
            | HipNativeExpr::Transpose { source, .. }
            | HipNativeExpr::Cast { source, .. }
            | HipNativeExpr::Exp { source }
            | HipNativeExpr::MaxKeepdim { source, .. }
            | HipNativeExpr::SumKeepdim { source, .. }
            | HipNativeExpr::Neg { source }
            | HipNativeExpr::AddScalar { source, .. }
            | HipNativeExpr::MulScalar { source, .. }
            | HipNativeExpr::Recip { source }
            | HipNativeExpr::L2Norm { source, .. } => source.is_host_graph(),
            HipNativeExpr::Concat { sources, .. } => sources.iter().all(|s| s.is_host_graph()),
            HipNativeExpr::BroadcastAdd { lhs, rhs }
            | HipNativeExpr::BroadcastMul { lhs, rhs }
            | HipNativeExpr::BroadcastSub { lhs, rhs }
            | HipNativeExpr::BroadcastDiv { lhs, rhs } => lhs.is_host_graph() && rhs.is_host_graph(),
        }
    }

    fn elem_count(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    fn byte_len(shape: &[usize], dtype: DType) -> usize {
        Self::elem_count(shape).saturating_mul(dtype.size_in_bytes())
    }

    fn supports_host_float_ops(dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    fn read_host_float(bytes: &[u8], dtype: DType, elem_idx: usize) -> Result<f64> {
        let elem_bytes = dtype.size_in_bytes();
        let start = elem_idx.saturating_mul(elem_bytes);
        let slice = &bytes[start..start + elem_bytes];
        Ok(match dtype {
            DType::F16 => f16::from_le_bytes([slice[0], slice[1]]).to_f64(),
            DType::BF16 => bf16::from_le_bytes([slice[0], slice[1]]).to_f64(),
            DType::F32 => f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as f64,
            DType::F64 => f64::from_le_bytes([
                slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            ]),
            _ => candle_core::bail!("unsupported host float dtype {:?}", dtype),
        })
    }

    fn write_host_float(out: &mut [u8], dtype: DType, elem_idx: usize, value: f64) -> Result<()> {
        let elem_bytes = dtype.size_in_bytes();
        let start = elem_idx.saturating_mul(elem_bytes);
        match dtype {
            DType::F16 => {
                out[start..start + 2].copy_from_slice(&f16::from_f64(value).to_le_bytes());
            }
            DType::BF16 => {
                out[start..start + 2].copy_from_slice(&bf16::from_f64(value).to_le_bytes());
            }
            DType::F32 => {
                out[start..start + 4].copy_from_slice(&(value as f32).to_le_bytes());
            }
            DType::F64 => {
                out[start..start + 8].copy_from_slice(&value.to_le_bytes());
            }
            _ => candle_core::bail!("unsupported host float dtype {:?}", dtype),
        }
        Ok(())
    }

    fn host_bytes_reshape(&self, source: &Arc<HipNativeBuffer>) -> Result<Option<Arc<[u8]>>> {
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        if bytes.len() != Self::byte_len(&self.shape, self.dtype) {
            candle_core::bail!(
                "invalid host reshape: {} bytes into shape {:?} {:?}",
                bytes.len(),
                self.shape,
                self.dtype
            )
        }
        Ok(Some(bytes))
    }

    fn host_bytes_narrow(
        &self,
        source: &Arc<HipNativeBuffer>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<Option<Arc<[u8]>>> {
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let src_shape = source.shape();
        if dim >= src_shape.len() {
            candle_core::bail!("invalid narrow dim {} for shape {:?}", dim, src_shape)
        }
        if start.saturating_add(len) > src_shape[dim] {
            candle_core::bail!(
                "invalid narrow start={} len={} for dim {} size {}",
                start,
                len,
                dim,
                src_shape[dim]
            )
        }
        let elem_bytes = self.dtype.size_in_bytes();
        let inner = Self::elem_count(&src_shape[dim + 1..]);
        let outer = Self::elem_count(&src_shape[..dim]);
        let chunk_bytes = len.saturating_mul(inner).saturating_mul(elem_bytes);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer {
            let src_off = ((outer_idx * src_shape[dim] + start) * inner) * elem_bytes;
            let dst_off = outer_idx * chunk_bytes;
            out[dst_off..dst_off + chunk_bytes]
                .copy_from_slice(&bytes[src_off..src_off + chunk_bytes]);
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_pad_with_zeros(
        &self,
        source: &Arc<HipNativeBuffer>,
        dim: usize,
        left: usize,
    ) -> Result<Option<Arc<[u8]>>> {
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let src_shape = source.shape();
        if dim >= src_shape.len() {
            candle_core::bail!("invalid pad dim {} for shape {:?}", dim, src_shape)
        }
        let elem_bytes = self.dtype.size_in_bytes();
        let inner = Self::elem_count(&src_shape[dim + 1..]);
        let outer = Self::elem_count(&src_shape[..dim]);
        let src_dim = src_shape[dim];
        let dst_dim = self.shape[dim];
        let src_chunk_bytes = src_dim.saturating_mul(inner).saturating_mul(elem_bytes);
        let dst_chunk_bytes = dst_dim.saturating_mul(inner).saturating_mul(elem_bytes);
        let left_bytes = left.saturating_mul(inner).saturating_mul(elem_bytes);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer {
            let src_off = outer_idx * src_chunk_bytes;
            let dst_off = outer_idx * dst_chunk_bytes + left_bytes;
            out[dst_off..dst_off + src_chunk_bytes]
                .copy_from_slice(&bytes[src_off..src_off + src_chunk_bytes]);
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_concat(
        &self,
        sources: &[Arc<HipNativeBuffer>],
        dim: usize,
    ) -> Result<Option<Arc<[u8]>>> {
        let mut source_bytes = Vec::with_capacity(sources.len());
        for source in sources {
            let Some(bytes) = source.try_materialize_host_bytes()? else {
                return Ok(None);
            };
            source_bytes.push((source.shape(), bytes));
        }
        let elem_bytes = self.dtype.size_in_bytes();
        let inner = Self::elem_count(&self.shape[dim + 1..]);
        let outer = Self::elem_count(&self.shape[..dim]);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer {
            let mut dst_off = outer_idx * self.shape[dim] * inner * elem_bytes;
            for (shape, bytes) in &source_bytes {
                let src_dim = shape[dim];
                let chunk_bytes = src_dim.saturating_mul(inner).saturating_mul(elem_bytes);
                let src_off = outer_idx * chunk_bytes;
                out[dst_off..dst_off + chunk_bytes]
                    .copy_from_slice(&bytes[src_off..src_off + chunk_bytes]);
                dst_off += chunk_bytes;
            }
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_reduce_keepdim(
        &self,
        source: &Arc<HipNativeBuffer>,
        dim: usize,
        sum: bool,
    ) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(self.dtype) {
            return Ok(None);
        }
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let src_shape = source.shape();
        if dim >= src_shape.len() {
            candle_core::bail!("invalid reduction dim {} for shape {:?}", dim, src_shape)
        }
        let inner = Self::elem_count(&src_shape[dim + 1..]);
        let outer = Self::elem_count(&src_shape[..dim]);
        let reduce = src_shape[dim];
        let out_elems = Self::elem_count(&self.shape);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let out_idx = outer_idx * inner + inner_idx;
                debug_assert!(out_idx < out_elems);
                let mut acc = if sum {
                    0.0
                } else {
                    Self::read_host_float(&bytes, self.dtype, (outer_idx * reduce) * inner + inner_idx)?
                };
                let start_r = if sum { 0 } else { 1 };
                for r in start_r..reduce {
                    let src_idx = ((outer_idx * reduce + r) * inner) + inner_idx;
                    let value = Self::read_host_float(&bytes, self.dtype, src_idx)?;
                    if sum {
                        acc += value;
                    } else if value > acc {
                        acc = value;
                    }
                }
                Self::write_host_float(&mut out, self.dtype, out_idx, acc)?;
            }
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_cast(&self, source: &Arc<HipNativeBuffer>, dtype: DType) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(source.dtype()) || !Self::supports_host_float_ops(dtype) {
            return Ok(None);
        }
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let elem_count = Self::elem_count(&self.shape);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for elem_idx in 0..elem_count {
            let value = Self::read_host_float(&bytes, source.dtype(), elem_idx)?;
            Self::write_host_float(&mut out, self.dtype, elem_idx, value)?;
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_map_float(
        &self,
        source: &Arc<HipNativeBuffer>,
        f: impl Fn(f64) -> f64,
    ) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(self.dtype) {
            return Ok(None);
        }
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let elem_count = Self::elem_count(&self.shape);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for elem_idx in 0..elem_count {
            let value = Self::read_host_float(&bytes, self.dtype, elem_idx)?;
            Self::write_host_float(&mut out, self.dtype, elem_idx, f(value))?;
        }
        Ok(Some(out.into()))
    }

    fn broadcast_elem_index(out_idx: usize, out_shape: &[usize], src_shape: &[usize]) -> usize {
        let rank_out = out_shape.len();
        let rank_src = src_shape.len();
        let lead = rank_out.saturating_sub(rank_src);
        let mut rem = out_idx;
        let mut src_idx = 0usize;
        let mut src_stride = 1usize;
        for out_dim in (0..rank_out).rev() {
            let coord = rem % out_shape[out_dim];
            rem /= out_shape[out_dim];
            if out_dim >= lead {
                let src_dim = out_dim - lead;
                let src_coord = if src_shape[src_dim] == 1 { 0 } else { coord };
                src_idx += src_coord * src_stride;
                src_stride = src_stride.saturating_mul(src_shape[src_dim]);
            }
        }
        src_idx
    }

    fn host_bytes_binary_float(
        &self,
        lhs: &Arc<HipNativeBuffer>,
        rhs: &Arc<HipNativeBuffer>,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(self.dtype) {
            return Ok(None);
        }
        let lhs_bytes = match lhs.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let rhs_bytes = match rhs.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let elem_count = Self::elem_count(&self.shape);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for out_idx in 0..elem_count {
            let lhs_idx = Self::broadcast_elem_index(out_idx, &self.shape, lhs.shape());
            let rhs_idx = Self::broadcast_elem_index(out_idx, &self.shape, rhs.shape());
            let lhs_val = Self::read_host_float(&lhs_bytes, self.dtype, lhs_idx)?;
            let rhs_val = Self::read_host_float(&rhs_bytes, self.dtype, rhs_idx)?;
            Self::write_host_float(&mut out, self.dtype, out_idx, f(lhs_val, rhs_val))?;
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_l2norm(
        &self,
        source: &Arc<HipNativeBuffer>,
        eps: f64,
    ) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(self.dtype) {
            return Ok(None);
        }
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let shape = source.shape();
        if shape.is_empty() {
            return Ok(None);
        }
        let inner = *shape.last().unwrap();
        let outer = Self::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let mut sum_sq = 0.0f64;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = Self::read_host_float(&bytes, self.dtype, idx)?;
                sum_sq += value * value;
            }
            let denom = (sum_sq + eps).sqrt();
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = Self::read_host_float(&bytes, self.dtype, idx)?;
                Self::write_host_float(&mut out, self.dtype, idx, value / denom)?;
            }
        }
        Ok(Some(out.into()))
    }

    fn host_bytes_matmul(lhs: &Arc<HipNativeBuffer>, rhs: &Arc<HipNativeBuffer>) -> Result<Option<Self>> {
        if lhs.dtype() != rhs.dtype() || !Self::supports_host_float_ops(lhs.dtype()) {
            return Ok(None);
        }
        let lhs_bytes = match lhs.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let rhs_bytes = match rhs.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            return Ok(None);
        }

        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let rhs_k = rhs_shape[rhs_shape.len() - 2];
        let n = rhs_shape[rhs_shape.len() - 1];
        if k != rhs_k {
            candle_core::bail!(
                "incompatible matmul shapes {:?} and {:?}",
                lhs_shape,
                rhs_shape
            )
        }

        let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
        let batch_shape =
            Self::broadcast_shape(lhs_batch, rhs_batch, "hip-native-host-matmul")?;
        let mut out_shape = batch_shape.clone();
        out_shape.push(m);
        out_shape.push(n);
        let out_elems = Self::elem_count(&out_shape);
        let mut out = vec![0u8; out_elems.saturating_mul(lhs.dtype().size_in_bytes())];

        let batch_count = Self::elem_count(&batch_shape);
        for batch_idx in 0..batch_count.max(1) {
            let lhs_batch_idx = Self::broadcast_elem_index(batch_idx, &batch_shape, lhs_batch);
            let rhs_batch_idx = Self::broadcast_elem_index(batch_idx, &batch_shape, rhs_batch);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        let lhs_idx = ((lhs_batch_idx * m + i) * k) + kk;
                        let rhs_idx = ((rhs_batch_idx * k + kk) * n) + j;
                        acc += Self::read_host_float(&lhs_bytes, lhs.dtype(), lhs_idx)?
                            * Self::read_host_float(&rhs_bytes, rhs.dtype(), rhs_idx)?;
                    }
                    let out_idx = ((batch_idx * m + i) * n) + j;
                    Self::write_host_float(&mut out, lhs.dtype(), out_idx, acc)?;
                }
            }
        }

        Ok(Some(Self {
            expr: HipNativeExpr::HostBytes { bytes: out.into() },
            shape: out_shape,
            dtype: lhs.dtype(),
            device: lhs.device().clone(),
        }))
    }

    fn try_materialize_host_bytes(&self) -> Result<Option<Arc<[u8]>>> {
        match &self.expr {
            HipNativeExpr::HostBytes { bytes } => Ok(Some(bytes.clone())),
            HipNativeExpr::DeviceBuffer(buffer) => {
                Self::tensor_to_host_float_bytes(&buffer.tensor, self.dtype)
            }
            HipNativeExpr::Reshape { source, .. } => self.host_bytes_reshape(source),
            HipNativeExpr::Narrow {
                source,
                dim,
                start,
                len,
            } => self.host_bytes_narrow(source, *dim, *start, *len),
            HipNativeExpr::PadWithZeros {
                source,
                dim,
                left,
                ..
            } => self.host_bytes_pad_with_zeros(source, *dim, *left),
            HipNativeExpr::Concat { sources, dim } => self.host_bytes_concat(sources, *dim),
            HipNativeExpr::Cast { source, dtype } => self.host_bytes_cast(source, *dtype),
            HipNativeExpr::MaxKeepdim { source, dim } => {
                self.host_bytes_reduce_keepdim(source, *dim, false)
            }
            HipNativeExpr::SumKeepdim { source, dim } => {
                self.host_bytes_reduce_keepdim(source, *dim, true)
            }
            HipNativeExpr::Exp { source } => self.host_bytes_map_float(source, |v| v.exp()),
            HipNativeExpr::BroadcastAdd { lhs, rhs } => {
                self.host_bytes_binary_float(lhs, rhs, |l, r| l + r)
            }
            HipNativeExpr::BroadcastMul { lhs, rhs } => {
                self.host_bytes_binary_float(lhs, rhs, |l, r| l * r)
            }
            HipNativeExpr::BroadcastSub { lhs, rhs } => {
                self.host_bytes_binary_float(lhs, rhs, |l, r| l - r)
            }
            HipNativeExpr::BroadcastDiv { lhs, rhs } => {
                self.host_bytes_binary_float(lhs, rhs, |l, r| l / r)
            }
            HipNativeExpr::Neg { source } => self.host_bytes_map_float(source, |v| -v),
            HipNativeExpr::AddScalar { source, value } => {
                self.host_bytes_map_float(source, |v| v + *value)
            }
            HipNativeExpr::MulScalar { source, value } => {
                self.host_bytes_map_float(source, |v| v * *value)
            }
            HipNativeExpr::Recip { source } => self.host_bytes_map_float(source, |v| v.recip()),
            HipNativeExpr::L2Norm { source, eps } => self.host_bytes_l2norm(source, *eps),
            _ => Ok(None),
        }
    }

    fn tensor_to_host_float_bytes(tensor: &Tensor, dtype: DType) -> Result<Option<Arc<[u8]>>> {
        if !Self::supports_host_float_ops(dtype) {
            return Ok(None);
        }
        let flat = if tensor.dtype() == dtype {
            tensor.flatten_all()?
        } else {
            tensor.to_dtype(dtype)?.flatten_all()?
        };
        let bytes = match dtype {
            DType::F16 => flat
                .to_vec1::<f16>()?
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            DType::BF16 => flat
                .to_vec1::<bf16>()?
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            DType::F32 => flat
                .to_vec1::<f32>()?
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            DType::F64 => flat
                .to_vec1::<f64>()?
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            _ => return Ok(None),
        };
        Ok(Some(bytes.into()))
    }

    pub(crate) fn imported_tensor(tensor: Tensor) -> Self {
        Self::device_buffer(HipDeviceBuffer { tensor })
    }

    pub(crate) fn device_buffer(buffer: HipDeviceBuffer) -> Self {
        Self {
            shape: buffer.tensor.dims().to_vec(),
            dtype: buffer.tensor.dtype(),
            device: buffer.tensor.device().clone(),
            expr: HipNativeExpr::DeviceBuffer(buffer),
        }
    }

    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        if device.is_hip() {
            return Ok(Self::device_buffer(HipDeviceBuffer::zeros(
                dims, dtype, device,
            )?));
        }
        let elem_count: usize = dims.iter().product();
        let byte_len = elem_count.saturating_mul(dtype.size_in_bytes());
        let bytes: Arc<[u8]> = vec![0u8; byte_len].into();
        Ok(Self {
            expr: HipNativeExpr::HostBytes { bytes },
            shape: dims,
            dtype,
            device: device.clone(),
        })
    }

    pub(crate) fn pad_with_zeros(
        source: Arc<HipNativeBuffer>,
        dim: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let mut shape = source.shape.clone();
        shape[dim] += left + right;
        Self {
            expr: HipNativeExpr::PadWithZeros {
                source: source.clone(),
                dim,
                left,
                right,
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn narrow(
        source: Arc<HipNativeBuffer>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Self {
        let mut shape = source.shape.clone();
        shape[dim] = len;
        Self {
            expr: HipNativeExpr::Narrow {
                source: source.clone(),
                dim,
                start,
                len,
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn concat(sources: Vec<Arc<HipNativeBuffer>>, dim: usize) -> Self {
        let mut shape = sources[0].shape.clone();
        shape[dim] = sources.iter().map(|s| s.shape[dim]).sum();
        Self {
            expr: HipNativeExpr::Concat {
                sources: sources.clone(),
                dim,
            },
            shape,
            dtype: sources[0].dtype,
            device: sources[0].device.clone(),
        }
    }

    pub(crate) fn reshape(source: Arc<HipNativeBuffer>, shape: Vec<usize>) -> Self {
        Self {
            expr: HipNativeExpr::Reshape {
                source: source.clone(),
                shape: shape.clone(),
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn expand(source: Arc<HipNativeBuffer>, shape: Vec<usize>) -> Self {
        Self {
            expr: HipNativeExpr::Expand {
                source: source.clone(),
                shape: shape.clone(),
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn transpose(source: Arc<HipNativeBuffer>, dim1: usize, dim2: usize) -> Self {
        let mut shape = source.shape.clone();
        shape.swap(dim1, dim2);
        Self {
            expr: HipNativeExpr::Transpose {
                source: source.clone(),
                dim1,
                dim2,
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn cast(source: Arc<HipNativeBuffer>, dtype: DType) -> Self {
        Self {
            expr: HipNativeExpr::Cast {
                source: source.clone(),
                dtype,
            },
            shape: source.shape.clone(),
            dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn exp(source: Arc<HipNativeBuffer>) -> Self {
        Self {
            expr: HipNativeExpr::Exp {
                source: source.clone(),
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    fn broadcast_shape(lhs: &[usize], rhs: &[usize], op: &'static str) -> Result<Vec<usize>> {
        Ok(Shape::from(lhs.to_vec())
            .broadcast_shape_binary_op(&Shape::from(rhs.to_vec()), op)?
            .into_dims())
    }

    pub(crate) fn broadcast_add(lhs: Arc<HipNativeBuffer>, rhs: Arc<HipNativeBuffer>) -> Result<Self> {
        Ok(Self {
            expr: HipNativeExpr::BroadcastAdd {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
            shape: Self::broadcast_shape(&lhs.shape, &rhs.shape, "hip-native-broadcast-add")?,
            dtype: lhs.dtype,
            device: lhs.device.clone(),
        })
    }

    pub(crate) fn broadcast_mul(lhs: Arc<HipNativeBuffer>, rhs: Arc<HipNativeBuffer>) -> Result<Self> {
        Ok(Self {
            expr: HipNativeExpr::BroadcastMul {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
            shape: Self::broadcast_shape(&lhs.shape, &rhs.shape, "hip-native-broadcast-mul")?,
            dtype: lhs.dtype,
            device: lhs.device.clone(),
        })
    }

    pub(crate) fn broadcast_sub(lhs: Arc<HipNativeBuffer>, rhs: Arc<HipNativeBuffer>) -> Result<Self> {
        Ok(Self {
            expr: HipNativeExpr::BroadcastSub {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
            shape: Self::broadcast_shape(&lhs.shape, &rhs.shape, "hip-native-broadcast-sub")?,
            dtype: lhs.dtype,
            device: lhs.device.clone(),
        })
    }

    pub(crate) fn broadcast_div(lhs: Arc<HipNativeBuffer>, rhs: Arc<HipNativeBuffer>) -> Result<Self> {
        Ok(Self {
            expr: HipNativeExpr::BroadcastDiv {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
            shape: Self::broadcast_shape(&lhs.shape, &rhs.shape, "hip-native-broadcast-div")?,
            dtype: lhs.dtype,
            device: lhs.device.clone(),
        })
    }

    pub(crate) fn max_keepdim(source: Arc<HipNativeBuffer>, dim: usize) -> Self {
        let mut shape = source.shape.clone();
        shape[dim] = 1;
        Self {
            expr: HipNativeExpr::MaxKeepdim {
                source: source.clone(),
                dim,
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn sum_keepdim(source: Arc<HipNativeBuffer>, dim: usize) -> Self {
        let mut shape = source.shape.clone();
        shape[dim] = 1;
        Self {
            expr: HipNativeExpr::SumKeepdim {
                source: source.clone(),
                dim,
            },
            shape,
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn neg(source: Arc<HipNativeBuffer>) -> Self {
        Self {
            expr: HipNativeExpr::Neg {
                source: source.clone(),
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn add_scalar(source: Arc<HipNativeBuffer>, value: f64) -> Self {
        Self {
            expr: HipNativeExpr::AddScalar {
                source: source.clone(),
                value,
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn mul_scalar(source: Arc<HipNativeBuffer>, value: f64) -> Self {
        Self {
            expr: HipNativeExpr::MulScalar {
                source: source.clone(),
                value,
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn recip(source: Arc<HipNativeBuffer>) -> Self {
        Self {
            expr: HipNativeExpr::Recip {
                source: source.clone(),
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn l2norm(source: Arc<HipNativeBuffer>, eps: f64) -> Self {
        Self {
            expr: HipNativeExpr::L2Norm {
                source: source.clone(),
                eps,
            },
            shape: source.shape.clone(),
            dtype: source.dtype,
            device: source.device.clone(),
        }
    }

    pub(crate) fn materialize_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        let Some(bytes) = self.try_materialize_host_bytes()? else {
            return Ok(None);
        };
        Ok(Some(HipHostBuffer {
            bytes,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        }))
    }

    pub(crate) fn materialize(&self) -> Result<Tensor> {
        if let Some(buffer) = self.materialize_host_buffer()? {
            return buffer.upload_to_tensor();
        }
        match &self.expr {
            HipNativeExpr::DeviceBuffer(buffer) => Ok(buffer.tensor.clone()),
            HipNativeExpr::HostBytes { bytes } => {
                Tensor::from_raw_buffer(bytes.as_ref(), self.dtype, &self.shape, &self.device)
            }
            HipNativeExpr::PadWithZeros {
                source,
                dim,
                left,
                right,
            } => source.materialize()?.pad_with_zeros(*dim, *left, *right),
            HipNativeExpr::Narrow {
                source,
                dim,
                start,
                len,
            } => source.materialize()?.narrow(*dim, *start, *len),
            HipNativeExpr::Concat { sources, dim } => {
                let tensors = sources
                    .iter()
                    .map(|s| s.materialize())
                    .collect::<Result<Vec<_>>>()?;
                let refs = tensors.iter().collect::<Vec<_>>();
                Tensor::cat(&refs, *dim)
            }
            HipNativeExpr::Reshape { source, shape } => {
                source.materialize()?.reshape(shape.clone())
            }
            HipNativeExpr::Expand { source, shape } => {
                source.materialize()?.expand(shape.clone())
            }
            HipNativeExpr::Transpose { source, dim1, dim2 } => {
                source.materialize()?.transpose(*dim1, *dim2)
            }
            HipNativeExpr::Cast { source, dtype } => {
                source.materialize()?.to_dtype(*dtype)
            }
            HipNativeExpr::Exp { source } => source.materialize()?.exp(),
            HipNativeExpr::BroadcastAdd { lhs, rhs } => {
                lhs.materialize()?.broadcast_add(&rhs.materialize()?)
            }
            HipNativeExpr::BroadcastMul { lhs, rhs } => {
                lhs.materialize()?.broadcast_mul(&rhs.materialize()?)
            }
            HipNativeExpr::BroadcastSub { lhs, rhs } => {
                lhs.materialize()?.broadcast_sub(&rhs.materialize()?)
            }
            HipNativeExpr::BroadcastDiv { lhs, rhs } => {
                lhs.materialize()?.broadcast_div(&rhs.materialize()?)
            }
            HipNativeExpr::MaxKeepdim { source, dim } => {
                source.materialize()?.max_keepdim(*dim)
            }
            HipNativeExpr::SumKeepdim { source, dim } => {
                source.materialize()?.sum_keepdim(*dim)
            }
            HipNativeExpr::Neg { source } => source.materialize()?.neg(),
            HipNativeExpr::AddScalar { source, value } => Ok((source.materialize()? + *value)?),
            HipNativeExpr::MulScalar { source, value } => Ok((source.materialize()? * *value)?),
            HipNativeExpr::Recip { source } => source.materialize()?.recip(),
            HipNativeExpr::L2Norm { source, eps } => {
                let source = source.materialize()?;
                let norm = source.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                source.broadcast_div(&(norm + *eps)?.sqrt()?)
            }
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

}

#[derive(Debug, Clone)]
pub(crate) struct HipStorage(pub(crate) HipNativeBuffer);

impl HipStorage {
    pub(crate) fn from_tensor(tensor: Tensor) -> Self {
        Self(HipNativeBuffer::imported_tensor(tensor))
    }

    pub(crate) fn imported_tensor(tensor: Tensor) -> Self {
        Self(HipNativeBuffer::imported_tensor(tensor))
    }

    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self(HipNativeBuffer::zeros(dims, dtype, device)?))
    }

    pub(crate) fn from_native_buffer(buffer: HipNativeBuffer) -> Self {
        Self(buffer)
    }

    pub(crate) fn from_device_buffer(buffer: HipDeviceBuffer) -> Self {
        Self(HipNativeBuffer::device_buffer(buffer))
    }

    pub(crate) fn materialize(&self) -> Result<Tensor> {
        self.0.materialize()
    }

    pub(crate) fn try_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        self.0.materialize_host_buffer()
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        self.0.materialize().expect("materialize native scaffold")
    }

    pub(crate) fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    pub(crate) fn dtype(&self) -> DType {
        self.0.dtype()
    }

    pub(crate) fn device(&self) -> Device {
        self.0.device().clone()
    }

    pub(crate) fn contiguous(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.contiguous()?));
        }
        if self.0.is_host_graph() {
            return Ok(self.clone());
        }
        Ok(Self::from_tensor(self.materialize()?.contiguous()?))
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            Ok(Self::from_native_buffer(HipNativeBuffer::cast(
                Arc::new(self.0.clone()),
                dtype,
            )))
        }
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.transpose(dim1, dim2)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::transpose(
            Arc::new(self.0.clone()),
            dim1,
            dim2,
        )))
    }

    pub(crate) fn reshape<T: candle_core::shape::ShapeWithOneHole>(&self, shape: T) -> Result<Self> {
        let shape = shape.into_shape(self.shape().iter().product())?;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.reshape(shape.into_dims())?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::reshape(
            Arc::new(self.0.clone()),
            shape.into_dims(),
        )))
    }

    pub(crate) fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let src = self.shape();
        let dst = shape.dims().to_vec();
        if dst.len() < src.len() {
            candle_core::bail!(
                "cannot expand rank {} shape {:?} to lower-rank {:?}",
                src.len(),
                src,
                dst
            )
        }
        for (src_dim, dst_dim) in src.iter().rev().zip(dst.iter().rev()) {
            if *src_dim != 1 && *src_dim != *dst_dim {
                candle_core::bail!("cannot expand shape {:?} to {:?}", src, dst)
            }
        }
        let _ = shape;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.expand(dst)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::expand(
            Arc::new(self.0.clone()),
            dst,
        )))
    }

    pub(crate) fn narrow(
        &self,
        dim: impl candle_core::shape::Dim,
        start: usize,
        len: usize,
    ) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-narrow")?;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.narrow(dim_index, start, len)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::narrow(
            Arc::new(self.0.clone()),
            dim_index,
            start,
            len,
        )))
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.0.direct_device_buffer(), rhs.0.direct_device_buffer()) {
            return Ok(Self::from_device_buffer(lhs.matmul(rhs)?));
        }
        if let Some(native) =
            HipNativeBuffer::host_bytes_matmul(&Arc::new(self.0.clone()), &Arc::new(rhs.0.clone()))?
        {
            return Ok(Self::from_native_buffer(native));
        }
        let lhs = self.materialize()?;
        let rhs = rhs.materialize()?;
        Ok(Self::from_tensor(lhs.matmul(&rhs)?))
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.0.direct_device_buffer(), rhs.0.direct_device_buffer()) {
            return Ok(Self::from_device_buffer(lhs.broadcast_add(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_add(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.0.direct_device_buffer(), rhs.0.direct_device_buffer()) {
            return Ok(Self::from_device_buffer(lhs.broadcast_mul(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_mul(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.exp()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::exp(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn max_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-max-keepdim")?;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.max_keepdim(dim_index)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::max_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.0.direct_device_buffer(), rhs.0.direct_device_buffer()) {
            return Ok(Self::from_device_buffer(lhs.broadcast_sub(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_sub(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn sum_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-sum-keepdim")?;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.sum_keepdim(dim_index)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::sum_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.0.direct_device_buffer(), rhs.0.direct_device_buffer()) {
            return Ok(Self::from_device_buffer(lhs.broadcast_div(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_div(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn recip(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.recip()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::recip(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn l2norm(&self, eps: f64) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.l2norm(eps)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::l2norm(
            Arc::new(self.0.clone()),
            eps,
        )))
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.sigmoid()?));
        }
        Self::from_native_buffer(HipNativeBuffer::add_scalar(
            Arc::new(HipNativeBuffer::exp(Arc::new(HipNativeBuffer::neg(
                Arc::new(self.0.clone()),
            )))),
            1.0,
        ))
        .recip()
    }

    pub(crate) fn mul_scalar(&self, value: f64) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.mul_scalar(value)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::mul_scalar(
            Arc::new(self.0.clone()),
            value,
        )))
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(
                buffer.pad_with_zeros(dim, left, right)?,
            ));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::pad_with_zeros(
            Arc::new(self.0.clone()),
            dim,
            left,
            right,
        )))
    }

    pub(crate) fn dim(&self, dim: usize) -> Result<usize> {
        Ok(self.shape()[dim])
    }

    pub(crate) fn dims3(&self) -> Result<(usize, usize, usize)> {
        let dims = self.shape();
        match dims.as_slice() {
            [d0, d1, d2] => Ok((*d0, *d1, *d2)),
            _ => candle_core::bail!("unexpected rank {}, expected 3", dims.len()),
        }
    }

    pub(crate) fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let dims = self.shape();
        match dims.as_slice() {
            [d0, d1, d2, d3] => Ok((*d0, *d1, *d2, *d3)),
            _ => candle_core::bail!("unexpected rank {}, expected 4", dims.len()),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HipTensor(pub(crate) HipStorage);

impl HipTensor {
    #[cfg(test)]
    pub(crate) fn from_host_buffer(buffer: HipHostBuffer) -> Self {
        Self(HipStorage::from_native_buffer(HipNativeBuffer {
            expr: HipNativeExpr::HostBytes { bytes: buffer.bytes },
            shape: buffer.shape,
            dtype: buffer.dtype,
            device: buffer.device,
        }))
    }

    pub(crate) fn from_device_buffer(buffer: HipDeviceBuffer) -> Self {
        Self(HipStorage::from_device_buffer(buffer))
    }

    pub(crate) fn from_scaffold_tensor(tensor: Tensor) -> Self {
        Self(HipStorage::imported_tensor(tensor))
    }

    pub(crate) fn from_state_buffer(state: &StateBuffer) -> Self {
        Self::from_scaffold_tensor(state.tensor().clone())
    }

    pub(crate) fn from_state_buffer_as(state: &StateBuffer, dtype: DType) -> Result<Self> {
        if state.tensor().dtype() == dtype {
            Ok(Self::from_state_buffer(state))
        } else {
            Ok(Self::from_scaffold_tensor(state.clone_tensor_as(dtype)?))
        }
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        if self.0 .0.is_host_graph() {
            if let Some(buffer) = self
                .try_host_buffer()
                .expect("materialize native scaffold host buffer")
            {
                return HipTensor::from_device_buffer(
                    buffer
                        .upload_to_device_buffer()
                        .expect("upload host buffer to device buffer"),
                )
                .0
                .into_tensor();
            }
        }
        self.0.into_tensor()
    }

    pub(crate) fn try_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        self.0.try_host_buffer()
    }

    pub(crate) fn contiguous(&self) -> Result<Self> {
        Ok(Self(self.0.contiguous()?))
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self(self.0.to_dtype(dtype)?))
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        Ok(Self(self.0.transpose(dim1, dim2)?))
    }

    pub(crate) fn reshape<T: candle_core::shape::ShapeWithOneHole>(&self, shape: T) -> Result<Self> {
        Ok(Self(self.0.reshape(shape)?))
    }

    pub(crate) fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        Ok(Self(self.0.expand(shape)?))
    }

    pub(crate) fn narrow(
        &self,
        dim: impl candle_core::shape::Dim,
        start: usize,
        len: usize,
    ) -> Result<Self> {
        Ok(Self(self.0.narrow(dim, start, len)?))
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.matmul(&rhs.0)?))
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_add(&rhs.0)?))
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_mul(&rhs.0)?))
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        Ok(Self(self.0.exp()?))
    }

    pub(crate) fn max_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        Ok(Self(self.0.max_keepdim(dim)?))
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_sub(&rhs.0)?))
    }

    pub(crate) fn sum_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        Ok(Self(self.0.sum_keepdim(dim)?))
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        Ok(Self(self.0.broadcast_div(&rhs.0)?))
    }

    #[cfg(test)]
    pub(crate) fn recip(&self) -> Result<Self> {
        Ok(Self(HipStorage::from_native_buffer(HipNativeBuffer::recip(
            Arc::new(self.0.0.clone()),
        ))))
    }

    pub(crate) fn l2norm(&self, eps: f64) -> Result<Self> {
        Ok(Self(self.0.l2norm(eps)?))
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        Ok(Self(self.0.sigmoid()?))
    }

    pub(crate) fn mul_scalar(&self, value: f64) -> Result<Self> {
        Ok(Self(self.0.mul_scalar(value)?))
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        Ok(Self(self.0.pad_with_zeros(dim, left, right)?))
    }

    pub(crate) fn dim(&self, dim: usize) -> Result<usize> {
        self.0.dim(dim)
    }

    pub(crate) fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.0.dims3()
    }

    pub(crate) fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        self.0.dims4()
    }

    pub(crate) fn cat(tensors: &[&HipTensor], dim: usize) -> Result<Self> {
        let device_buffers = tensors
            .iter()
            .map(|t| t.0 .0.direct_device_buffer())
            .collect::<Option<Vec<_>>>();
        if let Some(buffers) = device_buffers {
            return Ok(Self(HipStorage::from_device_buffer(HipDeviceBuffer::cat(
                &buffers, dim,
            )?)));
        }
        let sources = tensors
            .iter()
            .map(|t| Ok(Arc::new(t.0 .0.clone())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(HipStorage::from_native_buffer(HipNativeBuffer::concat(
            sources, dim,
        ))))
    }

    pub(crate) fn into_state_buffer(self) -> Result<StateBuffer> {
        if self.0 .0.is_host_graph() {
            if let Some(buffer) = self.try_host_buffer()? {
                return StateBuffer::from_tensor(buffer.upload_to_device_buffer()?.into_tensor());
            }
        }
        StateBuffer::from_tensor(self.0.into_tensor())
    }
}

fn from_kernel_tensor(tensor: Tensor) -> HipTensor {
    HipTensor::from_device_buffer(HipDeviceBuffer { tensor })
}

fn from_device_tensor(tensor: Tensor) -> HipTensor {
    HipTensor::from_device_buffer(HipDeviceBuffer { tensor })
}

pub(crate) fn to_state_buffer(tensor: Tensor) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(tensor).into_state_buffer()
}

pub(crate) fn tensor_to_state(tensor: Tensor) -> Result<StateBuffer> {
    to_state_buffer(tensor)
}

pub(crate) fn reshape_tensor_to_state(xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(xs.clone())
        .reshape(dims.to_vec())?
        .into_state_buffer()
}

pub(crate) fn narrow_tensor_to_state(
    xs: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(xs.clone())
        .narrow(dim, start, len)?
        .into_state_buffer()
}

fn prepare_depthwise_conv_input_hip(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<(HipTensor, Option<HipTensor>)> {
    let mixed_qkv = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let prev_state = match prev_state {
        Some(conv_state) => Some(HipTensor::from_state_buffer_as(conv_state, mixed_qkv.0.dtype())?),
        None => None,
    };
    if let Some(mixed_device) = mixed_qkv.0 .0.direct_device_buffer() {
        let prev_device = prev_state
            .as_ref()
            .and_then(|state| state.0 .0.direct_device_buffer());
        let (prepared, next_state) =
            HipDeviceBuffer::prepare_depthwise_conv_input(prev_device, mixed_device, kernel_size)?;
        return Ok((
            HipTensor::from_device_buffer(prepared),
            next_state.map(HipTensor::from_device_buffer),
        ));
    }

    let mixed_qkv = match prev_state {
        Some(conv_state) => HipTensor::cat(&[&conv_state, &mixed_qkv], 2)?,
        None => mixed_qkv.pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
    };
    let total_len = mixed_qkv.dim(2)?;
    let state_len = kernel_size.saturating_sub(1);
    let next_state = if state_len == 0 {
        None
    } else {
        Some(
            mixed_qkv
                .narrow(2, total_len - state_len, state_len)?
                .contiguous()?,
        )
    };
    Ok((mixed_qkv, next_state))
}

pub(crate) fn prepare_depthwise_conv_input(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Option<StateBuffer>)> {
    let (mixed_qkv, next_state) =
        prepare_depthwise_conv_input_hip(prev_state, mixed_qkv, kernel_size)?;
    Ok((
        mixed_qkv.into_tensor(),
        next_state.map(|t| t.into_state_buffer()).transpose()?,
    ))
}

fn update_depthwise_conv_state_hip(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let mixed_qkv = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let prev_state = match prev_state {
        Some(prev_state) => Some(HipTensor::from_state_buffer_as(prev_state, mixed_qkv.0.dtype())?),
        None => None,
    };
    if let Some(mixed_device) = mixed_qkv.0 .0.direct_device_buffer() {
        let prev_device = prev_state
            .as_ref()
            .and_then(|state| state.0 .0.direct_device_buffer());
        return HipDeviceBuffer::update_depthwise_conv_state(prev_device, mixed_device, kernel_size)
            .map(|state| state.map(HipTensor::from_device_buffer));
    }

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
                let keep = state_len - seq_len;
                let prev_tail = prev_state.narrow(2, prev_state.dim(2)? - keep, keep)?;
                HipTensor::cat(&[&prev_tail, &mixed_qkv], 2)?
                    .contiguous()?
            }
            None => {
                let zeros = zeros(
                    vec![mixed_qkv.dim(0)?, mixed_qkv.dim(1)?, state_len - seq_len],
                    mixed_qkv.0.dtype(),
                    &mixed_qkv.0.device(),
                )?;
                HipTensor::cat(&[&zeros, &mixed_qkv], 2)?.contiguous()?
            }
        }
    };
    Ok(Some(state))
}

pub(crate) fn update_depthwise_conv_state(
    prev_state: Option<&StateBuffer>,
    mixed_qkv: &Tensor,
    kernel_size: usize,
) -> Result<Option<StateBuffer>> {
    update_depthwise_conv_state_hip(prev_state, mixed_qkv, kernel_size)?
        .map(|t| t.into_state_buffer())
        .transpose()
}

fn concat_last_dim_hip(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<HipTensor> {
    let lhs = HipTensor::from_state_buffer(lhs);
    let rhs = HipTensor::from_state_buffer(rhs);
    if let (Some(lhs), Some(rhs)) = (lhs.0 .0.direct_device_buffer(), rhs.0 .0.direct_device_buffer()) {
        return Ok(HipTensor::from_device_buffer(HipDeviceBuffer::concat_last_dim(
            lhs, rhs,
        )?));
    }
    HipTensor::cat(&[&lhs, &rhs], lhs.0.shape().len() - 1)?
        .contiguous()
}

pub(crate) fn concat_last_dim(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    concat_last_dim_hip(lhs, rhs)?.into_state_buffer()
}

fn pack_delta_state_scan_hip(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<HipTensor> {
    let weighted_key_scan = HipTensor::from_scaffold_tensor(weighted_key_scan.clone());
    let k_cumdecay_scan = HipTensor::from_scaffold_tensor(k_cumdecay_scan.clone());
    let state_decay_feature = HipTensor::from_scaffold_tensor(state_decay_feature.clone());
    if let (Some(weighted_key_scan), Some(k_cumdecay_scan), Some(state_decay_feature)) = (
        weighted_key_scan.0 .0.direct_device_buffer(),
        k_cumdecay_scan.0 .0.direct_device_buffer(),
        state_decay_feature.0 .0.direct_device_buffer(),
    ) {
        return Ok(HipTensor::from_device_buffer(HipDeviceBuffer::pack_delta_state_scan(
            weighted_key_scan,
            k_cumdecay_scan,
            state_decay_feature,
        )?));
    }
    HipTensor::cat(&[&weighted_key_scan, &k_cumdecay_scan, &state_decay_feature], 3)?
        .contiguous()
}

pub(crate) fn pack_delta_state_scan(
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    state_decay_feature: &Tensor,
) -> Result<StateBuffer> {
    pack_delta_state_scan_hip(weighted_key_scan, k_cumdecay_scan, state_decay_feature)?
        .into_state_buffer()
}

fn pack_delta_chunk_fused_hip(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<HipTensor> {
    let weighted_key = HipTensor::from_scaffold_tensor(weighted_key.clone());
    let k_cumdecay = HipTensor::from_scaffold_tensor(k_cumdecay.clone());
    let q_state = HipTensor::from_scaffold_tensor(q_state.clone());
    let state_decay = HipTensor::from_scaffold_tensor(state_decay.clone());
    if let (Some(weighted_key), Some(k_cumdecay), Some(q_state), Some(state_decay)) = (
        weighted_key.0 .0.direct_device_buffer(),
        k_cumdecay.0 .0.direct_device_buffer(),
        q_state.0 .0.direct_device_buffer(),
        state_decay.0 .0.direct_device_buffer(),
    ) {
        return Ok(HipTensor::from_device_buffer(HipDeviceBuffer::pack_delta_chunk_fused(
            weighted_key,
            k_cumdecay,
            q_state,
            state_decay,
        )?));
    }
    HipTensor::cat(&[&weighted_key, &k_cumdecay, &q_state, &state_decay], 2)?
        .contiguous()
}

pub(crate) fn pack_delta_chunk_fused(
    weighted_key: &Tensor,
    k_cumdecay: &Tensor,
    q_state: &Tensor,
    state_decay: &Tensor,
) -> Result<StateBuffer> {
    pack_delta_chunk_fused_hip(weighted_key, k_cumdecay, q_state, state_decay)?
        .into_state_buffer()
}

fn unpack_linear_decode_output_hip(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<(HipTensor, HipTensor)> {
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_device_buffer() {
        let (core_attn_out, recurrent_state) = fused.unpack_linear_decode_output(
            batch_size,
            seq_len,
            value_dim,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )?;
        return Ok((
            HipTensor::from_device_buffer(core_attn_out),
            HipTensor::from_device_buffer(recurrent_state),
        ));
    }
    let core_attn_out = fused
        .narrow(1, 0, value_dim)?
        .reshape((batch_size, seq_len, value_dim))?;
    let recurrent_state = fused
        .narrow(1, value_dim, num_v_heads * head_k_dim * head_v_dim)?
        .reshape((batch_size, num_v_heads, head_k_dim, head_v_dim))?
        .contiguous()?;
    Ok((core_attn_out, recurrent_state))
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
    let (core_attn_out, recurrent_state) = unpack_linear_decode_output_hip(
        fused,
        batch_size,
        seq_len,
        value_dim,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )?;
    Ok((core_attn_out.into_tensor(), recurrent_state.into_state_buffer()?))
}

fn unpack_linear_prefill_output_hip(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    num_v_heads: usize,
    state_len: usize,
) -> Result<(HipTensor, HipTensor, HipTensor)> {
    let out_width = conv_dim + num_v_heads;
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_device_buffer() {
        let (mixed_qkv, g, conv_state) = fused.unpack_linear_prefill_output(
            batch_size,
            seq_len,
            conv_dim,
            num_v_heads,
            state_len,
        )?;
        return Ok((
            HipTensor::from_device_buffer(mixed_qkv),
            HipTensor::from_device_buffer(g),
            HipTensor::from_device_buffer(conv_state),
        ));
    }
    let packed = fused
        .narrow(1, 0, seq_len * out_width)?
        .reshape((batch_size, seq_len, out_width))?;
    let mixed_qkv = packed.narrow(candle_core::D::Minus1, 0, conv_dim)?;
    let g = packed.narrow(candle_core::D::Minus1, conv_dim, num_v_heads)?;
    let conv_state = fused
        .narrow(1, seq_len * out_width, conv_dim * state_len)?
        .reshape((batch_size, conv_dim, state_len))?
        .contiguous()?;
    Ok((mixed_qkv, g, conv_state))
}

pub(crate) fn unpack_linear_prefill_output(
    fused: &StateBuffer,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    num_v_heads: usize,
    state_len: usize,
) -> Result<(Tensor, Tensor, StateBuffer)> {
    let (mixed_qkv, g, conv_state) = unpack_linear_prefill_output_hip(
        fused,
        batch_size,
        seq_len,
        conv_dim,
        num_v_heads,
        state_len,
    )?;
    Ok((
        mixed_qkv.into_tensor(),
        g.into_tensor(),
        conv_state.into_state_buffer()?,
    ))
}

fn unpack_scan_fused_output_and_state_hip(
    fused: &StateBuffer,
    total_sequence_length: usize,
    output_sequence_length: usize,
    batch_size: usize,
    num_heads: usize,
    v_head_dim: usize,
    k_head_dim: usize,
    output_dtype: DType,
) -> Result<(HipTensor, HipTensor)> {
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_device_buffer() {
        let (output, recurrent_state) = fused.unpack_scan_fused_output_and_state(
            total_sequence_length,
            output_sequence_length,
            batch_size,
            num_heads,
            v_head_dim,
            k_head_dim,
            output_dtype,
        )?;
        return Ok((
            HipTensor::from_device_buffer(output),
            HipTensor::from_device_buffer(recurrent_state),
        ));
    }
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
        .contiguous()?;
    Ok((HipTensor::from_scaffold_tensor(output), recurrent_state))
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
    let (output, recurrent_state) = unpack_scan_fused_output_and_state_hip(
        fused,
        total_sequence_length,
        output_sequence_length,
        batch_size,
        num_heads,
        v_head_dim,
        k_head_dim,
        output_dtype,
    )?;
    Ok((output.into_state_buffer()?, recurrent_state.into_state_buffer()?))
}

pub(crate) fn state_scan_chunk(state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
    let state_scan_hip = HipTensor::from_state_buffer(state_scan);
    if let Some(state_scan) = state_scan_hip.0 .0.direct_device_buffer() {
        return HipTensor::from_device_buffer(state_scan.state_scan_chunk(chunk_idx)?)
            .into_state_buffer();
    }
    HipTensor::from_scaffold_tensor(state_scan.tensor().i((.., chunk_idx, .., ..))?).into_state_buffer()
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    let state_scan_hip = HipTensor::from_state_buffer(state_scan);
    if let Some(state_scan) = state_scan_hip.0 .0.direct_device_buffer() {
        return HipTensor::from_device_buffer(state_scan.state_scan_next_chunk(next_chunk_idx)?)
            .into_state_buffer();
    }
    HipTensor::from_scaffold_tensor(state_scan.tensor().i((.., next_chunk_idx, .., ..))?)
        .contiguous()?
        .into_state_buffer()
}

fn unpack_chunk_fused_hip(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(HipTensor, HipTensor, HipTensor)> {
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_device_buffer() {
        let (attn, local, q_state) = fused.unpack_chunk_fused(chunk_size, k_head_dim)?;
        return Ok((
            HipTensor::from_device_buffer(attn),
            HipTensor::from_device_buffer(local),
            HipTensor::from_device_buffer(q_state),
        ));
    }
    Ok((
        fused.narrow(1, 0, chunk_size)?,
        fused.narrow(1, chunk_size, chunk_size)?,
        fused.narrow(1, 2 * chunk_size, k_head_dim)?,
    ))
}

pub(crate) fn unpack_chunk_fused(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    let (attn, local, q_state) = unpack_chunk_fused_hip(fused, chunk_size, k_head_dim)?;
    Ok((
        attn.into_state_buffer()?,
        local.into_state_buffer()?,
        q_state.into_state_buffer()?,
    ))
}

fn linear_forward_matmul(
    x: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
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
    Ok(projected)
}

fn linear_forward_hip(
    x: &StateBuffer,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<HipTensor> {
    let projected = linear_forward_matmul(x.tensor(), weight)?;
    let projected = match bias {
        None => HipTensor::from_scaffold_tensor(projected),
        Some(bias) => HipTensor::from_scaffold_tensor(projected)
            .broadcast_add(&HipTensor::from_scaffold_tensor(bias.clone()))?,
    };
    Ok(projected)
}

pub(crate) fn linear_forward(
    x: &StateBuffer,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<StateBuffer> {
    linear_forward_hip(x, weight, bias)?.into_state_buffer()
}

pub(crate) fn add(lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
    HipTensor::from_state_buffer(lhs)
        .broadcast_add(&HipTensor::from_state_buffer(rhs))?
        .into_state_buffer()
}

pub(crate) fn slice_last_token(xs: &StateBuffer) -> Result<StateBuffer> {
    let xs = HipTensor::from_state_buffer(xs);
    let (_, seq_len, _) = xs.dims3()?;
    xs.narrow(1, seq_len - 1, 1)?.into_state_buffer()
}

fn repeat_heads_hip(xs: &HipTensor, n_rep: usize) -> Result<HipTensor> {
    if let Some(xs) = xs.0 .0.direct_device_buffer() {
        return Ok(HipTensor::from_device_buffer(xs.repeat_heads(n_rep)?));
    }
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}

fn repeat_kv_hip(xs: &HipTensor, repeats: usize) -> Result<HipTensor> {
    if let Some(xs) = xs.0 .0.direct_device_buffer() {
        return Ok(HipTensor::from_device_buffer(xs.repeat_kv(repeats)?));
    }
    if repeats <= 1 {
        return Ok(xs.clone());
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    xs.reshape((b_sz, kv_heads, 1, seq_len, head_dim))?
        .expand((b_sz, kv_heads, repeats, seq_len, head_dim))?
        .reshape((b_sz, kv_heads * repeats, seq_len, head_dim))
}

fn rms_norm_hip(
    xs: &HipTensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<HipTensor> {
    if let Some(xs) = xs.0 .0.direct_device_buffer() {
        if xs.tensor.device().is_hip() {
            return Ok(from_kernel_tensor(hip_rms_norm(
                &xs.tensor,
                weight,
                eps,
                add_unit_offset,
            )?));
        }
        return Ok(HipTensor::from_device_buffer(
            xs.rms_norm(weight, eps, add_unit_offset)?,
        ));
    }
    if let Some(host) = rms_norm_host(xs, weight, eps, add_unit_offset)? {
        return Ok(host);
    }
    rms_norm(&xs.clone().into_tensor(), weight, eps, add_unit_offset)
}

fn l2norm_hip(xs: &HipTensor, eps: f64) -> Result<HipTensor> {
    if let Some(xs) = xs.0 .0.direct_device_buffer() {
        return Ok(HipTensor::from_device_buffer(xs.l2norm(eps)?));
    }
    xs.l2norm(eps)
}

fn rms_norm_host(
    xs: &HipTensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    let source = &xs.0 .0;
    let Some(xs_bytes) = source.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    if !HipNativeBuffer::supports_host_float_ops(source.dtype()) {
        return Ok(None);
    }
    let Some(weight_bytes) = HipNativeBuffer::tensor_to_host_float_bytes(weight, DType::F32)? else {
        return Ok(None);
    };
    let shape = source.shape();
    if shape.is_empty() || weight.dim(0)? != *shape.last().unwrap() {
        return Ok(None);
    }
    let inner = *shape.last().unwrap();
    let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, source.dtype())];
    for outer_idx in 0..outer.max(1) {
        let mut sum_sq = 0.0f64;
        for inner_idx in 0..inner {
            let idx = outer_idx * inner + inner_idx;
            let value = HipNativeBuffer::read_host_float(&xs_bytes, source.dtype(), idx)?;
            sum_sq += value * value;
        }
        let denom = ((sum_sq / inner as f64) + eps).sqrt();
        for inner_idx in 0..inner {
            let idx = outer_idx * inner + inner_idx;
            let value = HipNativeBuffer::read_host_float(&xs_bytes, source.dtype(), idx)?;
            let mut w = HipNativeBuffer::read_host_float(&weight_bytes, DType::F32, inner_idx)?;
            if add_unit_offset {
                w += 1.0;
            }
            HipNativeBuffer::write_host_float(&mut out, source.dtype(), idx, (value / denom) * w)?;
        }
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape: shape.to_vec(),
        dtype: source.dtype(),
        device: source.device().clone(),
    }))))
}

fn rms_norm_gated_host(
    hidden_states: &HipTensor,
    gate: &HipTensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let hidden = &hidden_states.0 .0;
    let gate_src = &gate.0 .0;
    let Some(hidden_bytes) = hidden.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    let Some(gate_bytes) = gate_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    if hidden.shape() != gate_src.shape() || hidden.dtype() != gate_src.dtype() {
        return Ok(None);
    }
    if !HipNativeBuffer::supports_host_float_ops(hidden.dtype()) {
        return Ok(None);
    }
    let Some(weight_bytes) = HipNativeBuffer::tensor_to_host_float_bytes(weight, DType::F32)? else {
        return Ok(None);
    };
    let shape = hidden.shape();
    if shape.is_empty() || weight.dim(0)? != *shape.last().unwrap() {
        return Ok(None);
    }
    let inner = *shape.last().unwrap();
    let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, hidden.dtype())];
    for outer_idx in 0..outer.max(1) {
        let mut sum_sq = 0.0f64;
        for inner_idx in 0..inner {
            let idx = outer_idx * inner + inner_idx;
            let value = HipNativeBuffer::read_host_float(&hidden_bytes, hidden.dtype(), idx)?;
            sum_sq += value * value;
        }
        let denom = ((sum_sq / inner as f64) + eps).sqrt();
        for inner_idx in 0..inner {
            let idx = outer_idx * inner + inner_idx;
            let x = HipNativeBuffer::read_host_float(&hidden_bytes, hidden.dtype(), idx)?;
            let g = HipNativeBuffer::read_host_float(&gate_bytes, gate_src.dtype(), idx)?;
            let w = HipNativeBuffer::read_host_float(&weight_bytes, DType::F32, inner_idx)?;
            let silu = g / (1.0 + (-g).exp());
            HipNativeBuffer::write_host_float(
                &mut out,
                hidden.dtype(),
                idx,
                ((x / denom) * w) * silu,
            )?;
        }
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape: shape.to_vec(),
        dtype: hidden.dtype(),
        device: hidden.device().clone(),
    }))))
}

fn swiglu_mul_host(gate: &HipTensor, up: &HipTensor) -> Result<Option<HipTensor>> {
    let gate_src = &gate.0 .0;
    let up_src = &up.0 .0;
    let Some(gate_bytes) = gate_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    let Some(up_bytes) = up_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    if gate_src.shape() != up_src.shape() || gate_src.dtype() != up_src.dtype() {
        return Ok(None);
    }
    if !HipNativeBuffer::supports_host_float_ops(gate_src.dtype()) {
        return Ok(None);
    }
    let elem_count = HipNativeBuffer::elem_count(gate_src.shape());
    let mut out = vec![0u8; HipNativeBuffer::byte_len(gate_src.shape(), gate_src.dtype())];
    for idx in 0..elem_count {
        let gate_x = HipNativeBuffer::read_host_float(&gate_bytes, gate_src.dtype(), idx)?;
        let up_x = HipNativeBuffer::read_host_float(&up_bytes, up_src.dtype(), idx)?;
        let silu = gate_x / (1.0 + (-gate_x).exp());
        HipNativeBuffer::write_host_float(&mut out, gate_src.dtype(), idx, silu * up_x)?;
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape: gate_src.shape().to_vec(),
        dtype: gate_src.dtype(),
        device: gate_src.device().clone(),
    }))))
}

fn value_decay_host(a: &HipTensor, dt_bias: &HipTensor, a_log_exp: &HipTensor) -> Result<Option<HipTensor>> {
    let a_src = &a.0 .0;
    let dt_bias_src = &dt_bias.0 .0;
    let a_log_exp_src = &a_log_exp.0 .0;
    let Some(a_bytes) = a_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    let Some(dt_bias_bytes) = dt_bias_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    let Some(a_log_exp_bytes) = a_log_exp_src.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    if a_src.dtype() != dt_bias_src.dtype() || a_src.dtype() != a_log_exp_src.dtype() {
        return Ok(None);
    }
    if !HipNativeBuffer::supports_host_float_ops(a_src.dtype()) {
        return Ok(None);
    }
    let add_shape =
        HipNativeBuffer::broadcast_shape(a_src.shape(), dt_bias_src.shape(), "hip-native-value-decay-add")?;
    let out_shape =
        HipNativeBuffer::broadcast_shape(&add_shape, a_log_exp_src.shape(), "hip-native-value-decay-mul")?;
    let elem_count = HipNativeBuffer::elem_count(&out_shape);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&out_shape, a_src.dtype())];
    for out_idx in 0..elem_count {
        let a_idx = HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, a_src.shape());
        let dt_bias_idx =
            HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, dt_bias_src.shape());
        let a_log_exp_idx =
            HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, a_log_exp_src.shape());
        let a_val = HipNativeBuffer::read_host_float(&a_bytes, a_src.dtype(), a_idx)?;
        let dt_bias_val =
            HipNativeBuffer::read_host_float(&dt_bias_bytes, dt_bias_src.dtype(), dt_bias_idx)?;
        let a_log_exp_val = HipNativeBuffer::read_host_float(
            &a_log_exp_bytes,
            a_log_exp_src.dtype(),
            a_log_exp_idx,
        )?;
        let x = a_val + dt_bias_val;
        let softplus = if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0 + x.exp()).ln()
        };
        HipNativeBuffer::write_host_float(
            &mut out,
            a_src.dtype(),
            out_idx,
            -(softplus * a_log_exp_val),
        )?;
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape: out_shape,
        dtype: a_src.dtype(),
        device: a_src.device().clone(),
    }))))
}

fn cumsum_last_dim_host(xs: &HipTensor) -> Result<Option<HipTensor>> {
    let source = &xs.0 .0;
    let Some(bytes) = source.try_materialize_host_bytes()? else {
        return Ok(None);
    };
    if !HipNativeBuffer::supports_host_float_ops(source.dtype()) {
        return Ok(None);
    }
    let shape = source.shape();
    let Some(&inner) = shape.last() else {
        return Ok(None);
    };
    let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, source.dtype())];
    for outer_idx in 0..outer.max(1) {
        let mut running = 0.0f64;
        for inner_idx in 0..inner {
            let idx = outer_idx * inner + inner_idx;
            running += HipNativeBuffer::read_host_float(&bytes, source.dtype(), idx)?;
            HipNativeBuffer::write_host_float(&mut out, source.dtype(), idx, running)?;
        }
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape: shape.to_vec(),
        dtype: source.dtype(),
        device: source.device().clone(),
    }))))
}

fn causal_mask_host(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    if !HipNativeBuffer::supports_host_float_ops(dtype) {
        return Ok(None);
    }
    let kv_len = tgt_len + seqlen_offset;
    let shape = vec![batch_size, 1, tgt_len, kv_len];
    let elem_count = HipNativeBuffer::elem_count(&shape);
    let mut out = vec![0u8; elem_count.saturating_mul(dtype.size_in_bytes())];
    for b in 0..batch_size {
        for q in 0..tgt_len {
            for k in 0..kv_len {
                let allowed = k <= q + seqlen_offset;
                let value = if allowed { 0.0 } else { f64::NEG_INFINITY };
                let idx = ((b * tgt_len + q) * kv_len) + k;
                HipNativeBuffer::write_host_float(&mut out, dtype, idx, value)?;
            }
        }
    }
    Ok(Some(HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
        expr: HipNativeExpr::HostBytes { bytes: out.into() },
        shape,
        dtype,
        device: device.clone(),
    }))))
}

#[allow(clippy::too_many_arguments)]
fn prepare_full_attention_inputs_hip(
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
) -> Result<(HipTensor, HipTensor, HipTensor, HipTensor)> {
    let q_and_gate = HipTensor::from_state_buffer(q_and_gate);
    let k_proj = HipTensor::from_state_buffer(k_proj);
    let v_proj = HipTensor::from_state_buffer(v_proj);
    if let (Some(q_and_gate), Some(k_proj), Some(v_proj)) = (
        q_and_gate.0 .0.direct_device_buffer(),
        k_proj.0 .0.direct_device_buffer(),
        v_proj.0 .0.direct_device_buffer(),
    ) {
        let q_and_gate = q_and_gate.reshape(vec![b_sz, q_len, num_heads, head_dim * 2])?;
        let last_dim = q_and_gate.tensor.dims().len() - 1;
        let query_states = rms_norm_hip(
            &HipTensor::from_device_buffer(q_and_gate.narrow(last_dim, 0, head_dim)?),
            q_norm_weight,
            q_norm_eps,
            true,
        )?
        .transpose(1, 2)?;
        let gate = HipTensor::from_device_buffer(
            q_and_gate
                .narrow(last_dim, head_dim, head_dim)?
                .reshape(vec![b_sz, q_len, num_heads * head_dim])?,
        );
        let key_states = rms_norm_hip(
            &HipTensor::from_device_buffer(k_proj.reshape(vec![b_sz, q_len, num_kv_heads, head_dim])?),
            k_norm_weight,
            k_norm_eps,
            true,
        )?
        .transpose(1, 2)?;
        let value_states = HipTensor::from_device_buffer(
            v_proj
                .reshape(vec![b_sz, q_len, num_kv_heads, head_dim])?
                .transpose(1, 2)?,
        );
        return Ok((query_states, gate, key_states, value_states));
    }
    let q_and_gate = q_and_gate.reshape((
        b_sz,
        q_len,
        num_heads,
        head_dim * 2,
    ))?;
    let query_states = rms_norm_hip(
        &q_and_gate.narrow(candle_core::D::Minus1, 0, head_dim)?,
        q_norm_weight,
        q_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let gate = q_and_gate
        .narrow(candle_core::D::Minus1, head_dim, head_dim)?
        .reshape((b_sz, q_len, num_heads * head_dim))?;
    let key_states = rms_norm_hip(
        &k_proj.reshape((b_sz, q_len, num_kv_heads, head_dim))?,
        k_norm_weight,
        k_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let value_states = v_proj
        .reshape((b_sz, q_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
    Ok((query_states, gate, key_states, value_states))
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
    let (query_states, gate, key_states, value_states) = prepare_full_attention_inputs_hip(
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
    )?;
    Ok((
        query_states.into_tensor(),
        gate.into_tensor(),
        key_states.into_tensor(),
        value_states.into_tensor(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_linear_attention_inputs_hip(
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
) -> Result<(HipTensor, HipTensor, HipTensor, HipTensor, HipTensor)> {
    let mixed_qkv = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let beta_raw = HipTensor::from_state_buffer(beta_raw);
    let g = HipTensor::from_scaffold_tensor(g.clone());
    if let (Some(mixed_qkv), Some(beta_raw), Some(g)) = (
        mixed_qkv.0 .0.direct_device_buffer(),
        beta_raw.0 .0.direct_device_buffer(),
        g.0 .0.direct_device_buffer(),
    ) {
        let last_dim = mixed_qkv.tensor.dims().len() - 1;
        let query = mixed_qkv
            .narrow(last_dim, 0, key_dim)?
            .reshape(vec![batch_size, seq_len, num_k_heads, head_k_dim])?
            .to_dtype(compute_dtype)?
            .l2norm(1e-6)?;
        let key = mixed_qkv
            .narrow(last_dim, key_dim, key_dim)?
            .reshape(vec![batch_size, seq_len, num_k_heads, head_k_dim])?
            .to_dtype(compute_dtype)?
            .l2norm(1e-6)?;
        let value = mixed_qkv
            .narrow(last_dim, key_dim * 2, value_dim)?
            .reshape(vec![batch_size, seq_len, num_v_heads, head_v_dim])?
            .to_dtype(compute_dtype)?;
        let head_repeat = num_v_heads / num_k_heads;
        let (query, key) = if repeat_kv_heads && head_repeat > 1 {
            (query.repeat_heads(head_repeat)?, key.repeat_heads(head_repeat)?)
        } else {
            (query, key)
        };
        let beta = beta_raw.sigmoid()?.to_dtype(compute_dtype)?;
        let g = g.to_dtype(compute_dtype)?;
        return Ok((
            HipTensor::from_device_buffer(query),
            HipTensor::from_device_buffer(key),
            HipTensor::from_device_buffer(value),
            HipTensor::from_device_buffer(beta),
            HipTensor::from_device_buffer(g),
        ));
    }
    let query = mixed_qkv
        .narrow(candle_core::D::Minus1, 0, key_dim)?
        .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?
        .to_dtype(compute_dtype)?;
    let key = mixed_qkv
        .narrow(candle_core::D::Minus1, key_dim, key_dim)?
        .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?
        .to_dtype(compute_dtype)?;
    let value = mixed_qkv
        .narrow(candle_core::D::Minus1, key_dim * 2, value_dim)?
        .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?
        .to_dtype(compute_dtype)?;

    let query = l2norm_hip(&query, 1e-6)?;
    let key = l2norm_hip(&key, 1e-6)?;
    let head_repeat = num_v_heads / num_k_heads;
    let (query, key) = if repeat_kv_heads && head_repeat > 1 {
        (
            repeat_heads_hip(&query, head_repeat)?,
            repeat_heads_hip(&key, head_repeat)?,
        )
    } else {
        (query, key)
    };
    let beta = beta_raw
        .sigmoid()?
        .to_dtype(compute_dtype)?;
    let g = g
        .to_dtype(compute_dtype)?;
    Ok((query, key, value, beta, g))
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
    let (query, key, value, beta, g) = prepare_linear_attention_inputs_hip(
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
    )?;
    Ok((
        query.into_tensor(),
        key.into_tensor(),
        value.into_tensor(),
        beta.into_tensor(),
        g.into_tensor(),
    ))
}

fn wrap_kv_cache_hip(
    key_states: Tensor,
    value_states: Tensor,
) -> Result<(HipTensor, HipTensor)> {
    let key_states = HipTensor::from_scaffold_tensor(key_states);
    let value_states = HipTensor::from_scaffold_tensor(value_states);
    if let (Some(key_device), Some(value_device)) = (
        key_states.0 .0.direct_device_buffer(),
        value_states.0 .0.direct_device_buffer(),
    ) {
        return Ok((
            HipTensor::from_device_buffer(key_device.clone()),
            HipTensor::from_device_buffer(value_device.clone()),
        ));
    }
    Ok((
        key_states,
        value_states,
    ))
}

pub(crate) fn wrap_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    let (key_states, value_states) = wrap_kv_cache_hip(key_states, value_states)?;
    Ok((
        key_states.into_state_buffer()?,
        value_states.into_state_buffer()?,
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
    let attn_output_hip = HipTensor::from_scaffold_tensor(attn_output.clone());
    let gate_hip = HipTensor::from_scaffold_tensor(gate.clone());
    if let (Some(attn_output), Some(gate)) = (
        attn_output_hip.0 .0.direct_device_buffer(),
        gate_hip.0 .0.direct_device_buffer(),
    ) {
        return HipTensor::from_device_buffer(
            attn_output
                .transpose(1, 2)?
                .reshape(vec![b_sz, q_len, attention_size])?
                .to_dtype(hidden_dtype)?
                .broadcast_mul(&gate.sigmoid()?)?,
        )
        .into_state_buffer();
    }
    let attn_output = attn_output_hip
        .transpose(1, 2)?
        .reshape((b_sz, q_len, attention_size))?
        .to_dtype(hidden_dtype)?;
    let gate = gate_hip.sigmoid()?;
    attn_output.broadcast_mul(&gate)?.into_state_buffer()
}

fn append_full_attention_kv_hip(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(HipTensor, HipTensor)> {
    let key_states = HipTensor::from_scaffold_tensor(key_states.clone());
    let value_states = HipTensor::from_scaffold_tensor(value_states.clone());
    match (prev_k, prev_v) {
        (Some(prev_k), Some(prev_v)) => {
            let prev_k = HipTensor::from_state_buffer_as(prev_k, key_states.0.dtype())?;
            let prev_v = HipTensor::from_state_buffer_as(prev_v, value_states.0.dtype())?;
            if let (Some(prev_k_device), Some(prev_v_device), Some(key_device), Some(value_device)) = (
                prev_k.0 .0.direct_device_buffer(),
                prev_v.0 .0.direct_device_buffer(),
                key_states.0 .0.direct_device_buffer(),
                value_states.0 .0.direct_device_buffer(),
            ) {
                return Ok((
                    HipTensor::from_device_buffer(HipDeviceBuffer::cat(
                        &[prev_k_device, key_device],
                        2,
                    )?),
                    HipTensor::from_device_buffer(HipDeviceBuffer::cat(
                        &[prev_v_device, value_device],
                        2,
                    )?),
                ));
            }
            Ok((
                HipTensor::cat(&[&prev_k, &key_states], 2)?,
                HipTensor::cat(&[&prev_v, &value_states], 2)?,
            ))
        }
        _ => Ok((key_states, value_states)),
    }
}

pub(crate) fn append_full_attention_kv(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (key_states, value_states) =
        append_full_attention_kv_hip(prev_k, prev_v, key_states, value_states)?;
    Ok((key_states.into_tensor(), value_states.into_tensor()))
}

fn prepare_full_attention_kernel_inputs_hip(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(HipTensor, HipTensor, HipTensor)> {
    let query_states = HipTensor::from_scaffold_tensor(query_states.clone());
    let key_states = HipTensor::from_scaffold_tensor(key_states.clone());
    let value_states = HipTensor::from_scaffold_tensor(value_states.clone());
    if let (Some(query_device), Some(key_device), Some(value_device)) = (
        query_states.0 .0.direct_device_buffer(),
        key_states.0 .0.direct_device_buffer(),
        value_states.0 .0.direct_device_buffer(),
    ) {
        return Ok((
            HipTensor::from_device_buffer(query_device.contiguous()?),
            HipTensor::from_device_buffer(key_device.contiguous()?),
            HipTensor::from_device_buffer(value_device.contiguous()?),
        ));
    }
    Ok((
        query_states.contiguous()?,
        key_states.contiguous()?,
        value_states.contiguous()?,
    ))
}

pub(crate) fn prepare_full_attention_kernel_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (query_states, key_states, value_states) =
        prepare_full_attention_kernel_inputs_hip(query_states, key_states, value_states)?;
    Ok((
        query_states.into_tensor(),
        key_states.into_tensor(),
        value_states.into_tensor(),
    ))
}

fn materialize_full_attention_dense_inputs_hip(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_kv_groups: usize,
) -> Result<(HipTensor, HipTensor, HipTensor)> {
    let query_states = HipTensor::from_scaffold_tensor(query_states.clone());
    let key_states = HipTensor::from_scaffold_tensor(key_states.clone());
    let value_states = HipTensor::from_scaffold_tensor(value_states.clone());
    if let (Some(query_states), Some(key_states), Some(value_states)) = (
        query_states.0 .0.direct_device_buffer(),
        key_states.0 .0.direct_device_buffer(),
        value_states.0 .0.direct_device_buffer(),
    ) {
        return Ok((
            HipTensor::from_device_buffer(query_states.to_dtype(DType::F32)?),
            HipTensor::from_device_buffer(
                key_states.repeat_kv(num_kv_groups)?.contiguous()?.to_dtype(DType::F32)?,
            ),
            HipTensor::from_device_buffer(
                value_states
                    .repeat_kv(num_kv_groups)?
                    .contiguous()?
                    .to_dtype(DType::F32)?,
            ),
        ));
    }
    let key_states = repeat_kv_hip(&key_states, num_kv_groups)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let value_states = repeat_kv_hip(&value_states, num_kv_groups)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    Ok((query_states.to_dtype(DType::F32)?, key_states, value_states))
}

pub(crate) fn materialize_full_attention_dense_inputs(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_kv_groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (query_states, key_states, value_states) = materialize_full_attention_dense_inputs_hip(
        query_states,
        key_states,
        value_states,
        num_kv_groups,
    )?;
    Ok((
        query_states.into_tensor(),
        key_states.into_tensor(),
        value_states.into_tensor(),
    ))
}

fn dense_full_attention_fallback_hip(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<HipTensor> {
    let query_states_hip = HipTensor::from_scaffold_tensor(query_states_f.clone());
    let key_states_hip = HipTensor::from_scaffold_tensor(key_states_f.clone());
    let value_states_hip = HipTensor::from_scaffold_tensor(value_states_f.clone());
    let mask_hip = match attention_mask {
        Some(mask) => Some(HipTensor::from_scaffold_tensor(mask.to_dtype(DType::F32)?)),
        None => None,
    };
    if let (Some(query_states_f), Some(key_states_f), Some(value_states_f), mask_device) = (
        query_states_hip.0 .0.direct_device_buffer(),
        key_states_hip.0 .0.direct_device_buffer(),
        value_states_hip.0 .0.direct_device_buffer(),
        mask_hip
            .as_ref()
            .and_then(|mask| mask.0 .0.direct_device_buffer()),
    ) {
        let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
        let mut attn_weights = query_states_f.matmul(&key_states_t)?.mul_scalar(scale)?;
        if let Some(mask) = mask_device {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_weights = softmax_last_dim_device_hip(&attn_weights)?;
        return Ok(HipTensor::from_device_buffer(
            attn_weights.matmul(value_states_f)?,
        ));
    }
    let key_states_t = key_states_hip.transpose(2, 3)?.contiguous()?;
    let mut attn_weights = query_states_hip.matmul(&key_states_t)?.mul_scalar(scale)?;
    if let Some(mask) = mask_hip {
        attn_weights = attn_weights.broadcast_add(&mask)?;
    }
    let attn_weights = softmax_last_dim_hip(&attn_weights)?;
    attn_weights.matmul(&value_states_hip)
}

fn softmax_last_dim_hip(xs: &HipTensor) -> Result<HipTensor> {
    let max = xs.max_keepdim(candle_core::D::Minus1)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(candle_core::D::Minus1)?;
    num.broadcast_div(&den)
}

fn softmax_last_dim_device_hip(xs: &HipDeviceBuffer) -> Result<HipDeviceBuffer> {
    let last_dim = xs.tensor.dims().len() - 1;
    let max = xs.max_keepdim(last_dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(last_dim)?;
    num.broadcast_div(&den)
}

pub(crate) fn dense_full_attention_fallback(
    query_states_f: &Tensor,
    key_states_f: &Tensor,
    value_states_f: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    dense_full_attention_fallback_hip(
        query_states_f,
        key_states_f,
        value_states_f,
        attention_mask,
        scale,
    )
    .map(|t| t.into_tensor())
}

pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<HipTensor> {
    Ok(HipTensor(HipStorage::zeros(dims, dtype, device)?))
}

pub(crate) fn zeros_state(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<StateBuffer> {
    zeros(dims, dtype, device)?.into_state_buffer()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_f32_tensor(shape: &[usize], values: &[f32]) -> HipTensor {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        HipTensor(HipStorage::from_native_buffer(HipNativeBuffer {
            expr: HipNativeExpr::HostBytes { bytes: bytes.into() },
            shape: shape.to_vec(),
            dtype: DType::F32,
            device: Device::Cpu,
        }))
    }

    fn values_f32(tensor: HipTensor) -> Result<Vec<f32>> {
        tensor
            .into_tensor()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn host_buffer_values_f32(buffer: &HipHostBuffer) -> Result<Vec<f32>> {
        assert_eq!(buffer.dtype(), DType::F32);
        let elems = HipNativeBuffer::elem_count(buffer.shape());
        (0..elems)
            .map(|idx| {
                HipNativeBuffer::read_host_float(buffer.bytes(), DType::F32, idx).map(|v| v as f32)
            })
            .collect()
    }

    #[test]
    fn host_bytes_reshape_reuses_raw_bytes() -> Result<()> {
        let tensor = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).reshape((4,))?;
        assert_eq!(tensor.0.shape(), vec![4]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_narrow_copies_expected_slice() -> Result<()> {
        let tensor = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).narrow(1, 1, 1)?;
        assert_eq!(tensor.0.shape(), vec![2, 1]);
        assert_eq!(values_f32(tensor)?, vec![2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_pad_with_zeros_inserts_zero_columns() -> Result<()> {
        let tensor =
            host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).pad_with_zeros(1, 1, 1)?;
        assert_eq!(tensor.0.shape(), vec![2, 4]);
        assert_eq!(values_f32(tensor)?, vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_concat_preserves_row_major_layout() -> Result<()> {
        let a = host_f32_tensor(&[1, 2], &[1.0, 2.0]);
        let b = host_f32_tensor(&[1, 2], &[3.0, 4.0]);
        let tensor = HipTensor::cat(&[&a, &b], 0)?;
        assert_eq!(tensor.0.shape(), vec![2, 2]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_materialize_to_host_buffer_without_tensor() -> Result<()> {
        let tensor = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
            .reshape((1, 4))?
            .pad_with_zeros(1, 1, 1)?;
        let buffer = tensor.try_host_buffer()?.expect("host-backed buffer expected");
        assert_eq!(buffer.shape(), &[1, 6]);
        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(host_buffer_values_f32(&buffer)?, vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
        Ok(())
    }

    #[test]
    fn host_buffer_roundtrip_preserves_values() -> Result<()> {
        let tensor = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).transpose(0, 1)?;
        let buffer = tensor.try_host_buffer()?.expect("host-backed buffer expected");
        let roundtrip = HipTensor::from_host_buffer(buffer);
        assert_eq!(values_f32(roundtrip)?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn host_buffer_uploads_into_device_leaf() -> Result<()> {
        let tensor = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).transpose(0, 1)?;
        let buffer = tensor.try_host_buffer()?.expect("host-backed buffer expected");
        let roundtrip = HipTensor::from_device_buffer(buffer.upload_to_device_buffer()?);
        assert!(matches!(roundtrip.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(roundtrip)?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_kv_append_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let prev_k = Tensor::from_vec(vec![1f32, 2.0], (1, 1, 2, 1), &device)?;
        let prev_v = Tensor::from_vec(vec![3f32, 4.0], (1, 1, 2, 1), &device)?;
        let next_k = Tensor::from_vec(vec![5f32], (1, 1, 1, 1), &device)?;
        let next_v = Tensor::from_vec(vec![6f32], (1, 1, 1, 1), &device)?;
        let prev_k = StateBuffer::from_tensor(prev_k)?;
        let prev_v = StateBuffer::from_tensor(prev_v)?;

        let (key, value) =
            append_full_attention_kv_hip(Some(&prev_k), Some(&prev_v), &next_k, &next_v)?;

        assert!(matches!(key.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(value.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(key)?, vec![1.0, 2.0, 5.0]);
        assert_eq!(values_f32(value)?, vec![3.0, 4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_contiguous_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &device)?.transpose(0, 1)?;
        let tensor = HipTensor::from_scaffold_tensor(tensor).contiguous()?;
        assert!(matches!(tensor.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(tensor)?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pack_delta_state_scan_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let weighted_key_scan = Tensor::from_vec(vec![1f32, 2.0], (1, 1, 1, 2), &device)?;
        let k_cumdecay_scan = Tensor::from_vec(vec![3f32], (1, 1, 1, 1), &device)?;
        let state_decay_feature = Tensor::from_vec(vec![4f32], (1, 1, 1, 1), &device)?;

        let packed =
            pack_delta_state_scan_hip(&weighted_key_scan, &k_cumdecay_scan, &state_decay_feature)?;

        assert!(matches!(packed.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(packed)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_unpack_linear_decode_output_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let fused = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &device)?;
        let fused = StateBuffer::from_tensor(fused)?;

        let (core_attn_out, recurrent_state) =
            unpack_linear_decode_output_hip(&fused, 1, 1, 2, 1, 1, 2)?;

        assert!(matches!(core_attn_out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(recurrent_state.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(core_attn_out)?, vec![1.0, 2.0]);
        assert_eq!(values_f32(recurrent_state)?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_materialize_full_attention_dense_inputs_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let query_states = Tensor::from_vec(vec![1f32, 2.0], (1, 1, 1, 2), &device)?;
        let key_states = Tensor::from_vec(vec![3f32, 4.0], (1, 1, 1, 2), &device)?;
        let value_states = Tensor::from_vec(vec![5f32, 6.0], (1, 1, 1, 2), &device)?;

        let (query_states, key_states, value_states) = materialize_full_attention_dense_inputs_hip(
            &query_states,
            &key_states,
            &value_states,
            2,
        )?;

        assert!(matches!(query_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(key_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(value_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(query_states)?, vec![1.0, 2.0]);
        assert_eq!(values_f32(key_states)?, vec![3.0, 4.0, 3.0, 4.0]);
        assert_eq!(values_f32(value_states)?, vec![5.0, 6.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_dense_full_attention_fallback_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let query_states = Tensor::from_vec(vec![1f32, 0.0], (1, 1, 1, 2), &device)?;
        let key_states = Tensor::from_vec(vec![1f32, 0.0], (1, 1, 1, 2), &device)?;
        let value_states = Tensor::from_vec(vec![7f32, 8.0], (1, 1, 1, 2), &device)?;

        let out = dense_full_attention_fallback_hip(
            &query_states,
            &key_states,
            &value_states,
            None,
            1.0,
        )?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_prepare_full_attention_output_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let attn_output = Tensor::from_vec(vec![2f32, 4.0], (1, 1, 1, 2), &device)?;
        let gate = Tensor::from_vec(vec![0f32, 0.0], (1, 1, 2), &device)?;

        let out = prepare_full_attention_output(&attn_output, &gate, 1, 1, 2, DType::F32)?;
        let out = HipTensor::from_state_buffer(&out);

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_prepare_full_attention_inputs_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let q_and_gate =
            StateBuffer::from_tensor(Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 4), &device)?)?;
        let k_proj =
            StateBuffer::from_tensor(Tensor::from_vec(vec![5f32, 6.0], (1, 1, 2), &device)?)?;
        let v_proj =
            StateBuffer::from_tensor(Tensor::from_vec(vec![7f32, 8.0], (1, 1, 2), &device)?)?;
        let q_norm_weight = Tensor::ones((2,), DType::F32, &device)?;
        let k_norm_weight = Tensor::ones((2,), DType::F32, &device)?;

        let (query_states, gate, key_states, value_states) = prepare_full_attention_inputs_hip(
            &q_and_gate,
            &k_proj,
            &v_proj,
            1,
            1,
            1,
            1,
            2,
            &q_norm_weight,
            1e-6,
            &k_norm_weight,
            1e-6,
        )?;

        assert!(matches!(query_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(gate.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(key_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(value_states.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(query_states.0.shape(), vec![1, 1, 1, 2]);
        assert_eq!(key_states.0.shape(), vec![1, 1, 1, 2]);
        Ok(())
    }

    #[test]
    fn device_leaf_delta_chunk_recurrent_read_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let prev_state = StateBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?)?;
        let k_cumdecay_chunk = Tensor::from_vec(vec![1f32, 0.0], (1, 2), &device)?;
        let q_state_chunk = Tensor::from_vec(vec![0f32, 1.0], (1, 2), &device)?;
        let value_chunk = Tensor::from_vec(vec![10f32, 20.0], (1, 2), &device)?;

        let (v_new, attn_inter) = delta_chunk_recurrent_read(
            &prev_state,
            &k_cumdecay_chunk,
            &q_state_chunk,
            &value_chunk,
        )?;
        let v_new = HipTensor::from_state_buffer(&v_new);
        let attn_inter = HipTensor::from_state_buffer(&attn_inter);

        assert!(matches!(v_new.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(attn_inter.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(v_new)?, vec![9.0, 18.0]);
        assert_eq!(values_f32(attn_inter)?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_mix_chunk_attention_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let attn = Tensor::from_vec(vec![2f32], (1, 1), &device)?;
        let attn_inter = StateBuffer::from_tensor(Tensor::from_vec(vec![1f32, 2.0], (1, 2), &device)?)?;
        let value_chunk = StateBuffer::from_tensor(Tensor::from_vec(vec![3f32, 4.0], (1, 2), &device)?)?;

        let mixed = mix_chunk_attention(&attn, &attn_inter, &value_chunk)?;
        let mixed = HipTensor::from_state_buffer(&mixed);

        assert!(matches!(mixed.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(mixed)?, vec![7.0, 10.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_prepare_linear_attention_inputs_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let mixed_qkv =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 1, 6), &device)?;
        let beta_raw =
            StateBuffer::from_tensor(Tensor::from_vec(vec![0f32], (1, 1, 1), &device)?)?;
        let g = Tensor::from_vec(vec![1f32], (1, 1, 1), &device)?;

        let (query, key, value, beta, g) = prepare_linear_attention_inputs_hip(
            &mixed_qkv,
            &beta_raw,
            &g,
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            DType::F32,
            false,
        )?;

        assert!(matches!(query.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(key.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(value.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(beta.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert!(matches!(g.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_generic_matmul_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);
        let rhs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![5f32, 6.0, 7.0, 8.0],
            (2, 2),
            &device,
        )?);

        let out = lhs.matmul(&rhs)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_sigmoid_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![0f32, 1.0, -1.0],
            (3,),
            &device,
        )?);

        let out = xs.sigmoid()?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!((vals[0] - 0.5).abs() < 1e-6);
        assert!((vals[1] - 0.7310586).abs() < 1e-5);
        assert!((vals[2] - 0.26894143).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_rope_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 1, 1, 4),
            &device,
        )?);
        let cos = Tensor::from_vec(vec![1f32, 1.0], (1, 2), &device)?;
        let sin = Tensor::from_vec(vec![0f32, 0.0], (1, 2), &device)?;

        let out = rope_hip(&xs, &cos, &sin)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_causal_mask_stays_device_backed() -> Result<()> {
        let out = causal_mask(&Device::Cpu, DType::F32, 1, 3, 2)?;
        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_cumsum_last_dim_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![1f32, 2.0, 3.0, -1.0, 4.0, 2.0], (2, 3), &device)?;

        let out = cumsum_last_dim(&xs)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_value_decay_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![0f32, 1.0, -1.0, 2.0], (2, 2), &device)?;
        let dt_bias = Tensor::from_vec(vec![0.5f32, -0.25], (2,), &device)?;
        let a_log_exp = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (2, 2), &device)?;

        let out = value_decay(&a, &dt_bias, &a_log_exp)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_rms_norm_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![3f32, 4.0, 5.0, 12.0], (2, 2), &device)?;
        let weight = Tensor::ones((2,), DType::F32, &device)?;

        let out = rms_norm(&xs, &weight, 1e-6, true)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_rms_norm_gated_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let hidden = Tensor::from_vec(vec![3f32, 4.0, 5.0, 12.0], (2, 2), &device)?;
        let gate = Tensor::from_vec(vec![1f32, -1.0, 0.5, -0.5], (2, 2), &device)?;
        let weight = Tensor::ones((2,), DType::F32, &device)?;

        let out = rms_norm_gated(&hidden, &gate, &weight, 1e-6)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_swiglu_mul_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![1f32, -1.0, 0.5, -0.5], (2, 2), &device)?;
        let up = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (2, 2), &device)?;

        let out = swiglu_mul(&gate, &up)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        Ok(())
    }

    #[test]
    fn host_bytes_max_keepdim_reduces_without_materializing_first() -> Result<()> {
        let tensor =
            host_f32_tensor(&[2, 2], &[1.0, 5.0, 3.0, 4.0]).max_keepdim(candle_core::D::Minus(1))?;
        assert_eq!(tensor.0.shape(), vec![2, 1]);
        assert_eq!(values_f32(tensor)?, vec![5.0, 4.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_sum_keepdim_reduces_without_materializing_first() -> Result<()> {
        let tensor =
            host_f32_tensor(&[2, 2], &[1.0, 5.0, 3.0, 4.0]).sum_keepdim(candle_core::D::Minus(1))?;
        assert_eq!(tensor.0.shape(), vec![2, 1]);
        assert_eq!(values_f32(tensor)?, vec![6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_exp_stays_host_backed() -> Result<()> {
        let tensor = host_f32_tensor(&[2], &[0.0, 1.0]).exp()?;
        let values = values_f32(tensor)?;
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - std::f32::consts::E).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_float_cast_roundtrip_stays_host_backed() -> Result<()> {
        let tensor = host_f32_tensor(&[2], &[1.5, -2.25]).to_dtype(DType::F16)?.to_dtype(DType::F32)?;
        let values = values_f32(tensor)?;
        assert!((values[0] - 1.5).abs() < 1e-3);
        assert!((values[1] + 2.25).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn host_bytes_broadcast_add_matches_expected_layout() -> Result<()> {
        let lhs = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rhs = host_f32_tensor(&[1, 2], &[10.0, 20.0]);
        let tensor = lhs.broadcast_add(&rhs)?;
        assert_eq!(tensor.0.shape(), vec![2, 2]);
        assert_eq!(values_f32(tensor)?, vec![11.0, 22.0, 13.0, 24.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_matmul_stays_host_backed() -> Result<()> {
        let lhs = host_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = host_f32_tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let tensor = lhs.matmul(&rhs)?;
        assert_eq!(tensor.0.shape(), vec![2, 2]);
        assert_eq!(values_f32(tensor)?, vec![58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_l2norm_stays_host_backed() -> Result<()> {
        let xs = host_f32_tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0]);
        let tensor = l2norm(&xs.clone().into_tensor(), 1e-6)?;
        let values = values_f32(tensor)?;
        assert!((values[0] - 0.6).abs() < 1e-5);
        assert!((values[1] - 0.8).abs() < 1e-5);
        assert!((values[2] - (5.0f32 / 13.0)).abs() < 1e-5);
        assert!((values[3] - (12.0f32 / 13.0)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_rms_norm_stays_host_backed() -> Result<()> {
        let xs = host_f32_tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0]);
        let weight = Tensor::from_vec(vec![0.5f32, 1.5f32], (2,), &Device::Cpu)?;
        let tensor = rms_norm(&xs.clone().into_tensor(), &weight, 1e-6, true)?;
        let values = values_f32(tensor)?;
        let d0 = ((25.0f32 / 2.0) + 1e-6).sqrt();
        let d1 = ((169.0f32 / 2.0) + 1e-6).sqrt();
        assert!((values[0] - ((3.0 / d0) * 1.5)).abs() < 1e-5);
        assert!((values[1] - ((4.0 / d0) * 2.5)).abs() < 1e-5);
        assert!((values[2] - ((5.0 / d1) * 1.5)).abs() < 1e-5);
        assert!((values[3] - ((12.0 / d1) * 2.5)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_rms_norm_gated_stays_host_backed() -> Result<()> {
        let xs = host_f32_tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0]);
        let gate = host_f32_tensor(&[2, 2], &[1.0, -1.0, 0.5, -0.5]);
        let weight = Tensor::from_vec(vec![0.5f32, 1.5f32], (2,), &Device::Cpu)?;
        let tensor = rms_norm_gated(&xs.clone().into_tensor(), &gate.clone().into_tensor(), &weight, 1e-6)?;
        let values = values_f32(tensor)?;
        let d0 = ((25.0f32 / 2.0) + 1e-6).sqrt();
        let d1 = ((169.0f32 / 2.0) + 1e-6).sqrt();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert!((values[0] - (((3.0 / d0) * 0.5) * silu(1.0))).abs() < 1e-5);
        assert!((values[1] - (((4.0 / d0) * 1.5) * silu(-1.0))).abs() < 1e-5);
        assert!((values[2] - (((5.0 / d1) * 0.5) * silu(0.5))).abs() < 1e-5);
        assert!((values[3] - (((12.0 / d1) * 1.5) * silu(-0.5))).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_swiglu_mul_stays_host_backed() -> Result<()> {
        let gate = host_f32_tensor(&[2, 2], &[1.0, -1.0, 0.5, -0.5]);
        let up = host_f32_tensor(&[2, 2], &[2.0, 3.0, 4.0, 5.0]);
        let tensor = swiglu_mul(&gate.clone().into_tensor(), &up.clone().into_tensor())?;
        let values = values_f32(tensor)?;
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert!((values[0] - (silu(1.0) * 2.0)).abs() < 1e-5);
        assert!((values[1] - (silu(-1.0) * 3.0)).abs() < 1e-5);
        assert!((values[2] - (silu(0.5) * 4.0)).abs() < 1e-5);
        assert!((values[3] - (silu(-0.5) * 5.0)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_value_decay_stays_host_backed() -> Result<()> {
        let a = host_f32_tensor(&[2, 2], &[0.0, 1.0, -1.0, 2.0]);
        let dt_bias = host_f32_tensor(&[2], &[0.5, -0.25]);
        let a_log_exp = host_f32_tensor(&[2, 2], &[2.0, 3.0, 4.0, 5.0]);
        let tensor = value_decay(
            &a.clone().into_tensor(),
            &dt_bias.clone().into_tensor(),
            &a_log_exp.clone().into_tensor(),
        )?;
        let values = values_f32(tensor)?;
        let softplus = |v: f32| {
            if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            }
        };
        assert!((values[0] - (-(softplus(0.5) * 2.0))).abs() < 1e-5);
        assert!((values[1] - (-(softplus(0.75) * 3.0))).abs() < 1e-5);
        assert!((values[2] - (-(softplus(-0.5) * 4.0))).abs() < 1e-5);
        assert!((values[3] - (-(softplus(1.75) * 5.0))).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_cumsum_last_dim_stays_host_backed() -> Result<()> {
        let xs = host_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, -1.0, 4.0, 2.0]);
        let tensor = cumsum_last_dim(&xs.clone().into_tensor())?;
        assert_eq!(tensor.0.shape(), vec![2, 3]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 3.0, 6.0, -1.0, 3.0, 5.0]);
        Ok(())
    }

    #[test]
    fn host_bytes_causal_mask_stays_host_backed() -> Result<()> {
        let tensor = causal_mask(&Device::Cpu, DType::F32, 1, 3, 2)?;
        assert_eq!(tensor.0.shape(), vec![1, 1, 3, 5]);
        assert_eq!(
            values_f32(tensor)?,
            vec![
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn host_bytes_softmax_last_dim_stays_host_backed() -> Result<()> {
        let xs = host_f32_tensor(&[2, 2], &[0.0, 1.0, 2.0, 2.0]);
        let tensor = softmax_last_dim_hip(&xs)?;
        let values = values_f32(tensor)?;
        let row0_denom = 1.0f32 + std::f32::consts::E;
        assert!((values[0] - (1.0 / row0_denom)).abs() < 1e-5);
        assert!((values[1] - (std::f32::consts::E / row0_denom)).abs() < 1e-5);
        assert!((values[2] - 0.5).abs() < 1e-5);
        assert!((values[3] - 0.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn host_bytes_scalar_math_stays_host_backed() -> Result<()> {
        let tensor = host_f32_tensor(&[2], &[2.0, 4.0]).mul_scalar(0.5)?.recip()?.sigmoid()?;
        let values = values_f32(tensor)?;
        let expected0 = 1.0 / (1.0 + (-1.0f32).exp());
        let expected1 = 1.0 / (1.0 + (-0.5f32).exp());
        assert!((values[0] - expected0).abs() < 1e-5);
        assert!((values[1] - expected1).abs() < 1e-5);
        Ok(())
    }
}

pub(crate) fn embedding_lookup(embeddings: &Tensor, indexes: &Tensor) -> Result<HipTensor> {
    Ok(from_kernel_tensor(hip_embedding_lookup(embeddings, indexes)?))
}

pub(crate) fn embedding_lookup_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<StateBuffer> {
    embedding_lookup(embeddings, indexes)?.into_state_buffer()
}

pub(crate) fn immutable_embedding_lookup(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(hip_immutable_embedding_lookup(
        embedding, indexes,
    )?))
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(immutable_output_projection(
        embedding,
        hidden_states,
    )?))
}

pub(crate) fn output_projection_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &StateBuffer,
) -> Result<StateBuffer> {
    output_projection(embedding, hidden_states.tensor())?.into_state_buffer()
}

pub(crate) fn rms_norm(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<HipTensor> {
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    if xs_hip.0 .0.direct_device_buffer().is_some() {
        return rms_norm_hip(&xs_hip, weight, eps, add_unit_offset);
    }
    if let Some(host) = rms_norm_host(&xs_hip, weight, eps, add_unit_offset)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_rms_norm(
        xs,
        weight,
        eps,
        add_unit_offset,
    )?))
}

pub(crate) fn rms_norm_buffer(
    xs: &StateBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<StateBuffer> {
    rms_norm(xs.tensor(), weight, eps, add_unit_offset)?.into_state_buffer()
}

pub(crate) fn rms_norm_gated(
    hidden_states: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<HipTensor> {
    let hidden_states_hip = HipTensor::from_scaffold_tensor(hidden_states.clone());
    let gate_hip = HipTensor::from_scaffold_tensor(gate.clone());
    if let (Some(hidden_states), Some(gate)) = (
        hidden_states_hip.0 .0.direct_device_buffer(),
        gate_hip.0 .0.direct_device_buffer(),
    ) {
        if hidden_states.tensor.device().is_hip() {
            return Ok(from_kernel_tensor(hip_rms_norm_gated(
                &hidden_states.tensor,
                &gate.tensor,
                weight,
                eps,
            )?));
        }
        return Ok(HipTensor::from_device_buffer(
            hidden_states.rms_norm_gated(gate, weight, eps)?,
        ));
    }
    if let Some(host) = rms_norm_gated_host(&hidden_states_hip, &gate_hip, weight, eps)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_rms_norm_gated(
        hidden_states,
        gate,
        weight,
        eps,
    )?))
}

pub(crate) fn rms_norm_gated_buffer(
    hidden_states: &StateBuffer,
    gate: &StateBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<StateBuffer> {
    rms_norm_gated(
        hidden_states.tensor(),
        gate.tensor(),
        weight,
        eps,
    )?
    .into_state_buffer()
}

pub(crate) fn swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<HipTensor> {
    let gate_hip = HipTensor::from_scaffold_tensor(gate.clone());
    let up_hip = HipTensor::from_scaffold_tensor(up.clone());
    if let (Some(gate), Some(up)) = (
        gate_hip.0 .0.direct_device_buffer(),
        up_hip.0 .0.direct_device_buffer(),
    ) {
        if gate.tensor.device().is_hip() {
            return Ok(from_kernel_tensor(hip_swiglu_mul(
                &gate.tensor,
                &up.tensor,
            )?));
        }
        return Ok(HipTensor::from_device_buffer(gate.swiglu_mul(up)?));
    }
    if let Some(host) = swiglu_mul_host(&gate_hip, &up_hip)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_swiglu_mul(gate, up)?))
}

pub(crate) fn swiglu_mul_buffer(gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
    swiglu_mul(gate.tensor(), up.tensor())?.into_state_buffer()
}

pub(crate) fn causal_mask(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<HipTensor> {
    if device.is_hip() {
        return Ok(from_device_tensor(hip_causal_mask(
            device,
            dtype,
            batch_size,
            tgt_len,
            seqlen_offset,
        )?));
    }
    if let Some(host) = causal_mask_host(device, dtype, batch_size, tgt_len, seqlen_offset)? {
        if let Some(buffer) = host.try_host_buffer()? {
            return Ok(HipTensor::from_device_buffer(buffer.upload_to_device_buffer()?));
        }
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_causal_mask(
        device,
        dtype,
        batch_size,
        tgt_len,
        seqlen_offset,
    )?))
}

pub(crate) fn cumsum_last_dim(xs: &Tensor) -> Result<HipTensor> {
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    if let Some(xs) = xs_hip.0 .0.direct_device_buffer() {
        if xs.tensor.device().is_hip() {
            return Ok(from_device_tensor(hip_cumsum_last_dim(&xs.tensor)?));
        }
        if let Some(host) = cumsum_last_dim_host(&xs_hip)? {
            if let Some(buffer) = host.try_host_buffer()? {
                return Ok(HipTensor::from_device_buffer(buffer.upload_to_device_buffer()?));
            }
            return Ok(host);
        }
    }
    if let Some(host) = cumsum_last_dim_host(&xs_hip)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_cumsum_last_dim(xs)?))
}

pub(crate) fn cumsum_last_dim_buffer(xs: &StateBuffer) -> Result<StateBuffer> {
    cumsum_last_dim(xs.tensor())?.into_state_buffer()
}

pub(crate) fn l2norm(xs: &Tensor, eps: f64) -> Result<HipTensor> {
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    Ok(l2norm_hip(&xs_hip, eps)?)
}

pub(crate) fn l2norm_buffer(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    l2norm(xs.tensor(), eps)?.into_state_buffer()
}

pub(crate) fn value_decay(a: &Tensor, dt_bias: &Tensor, a_log_exp: &Tensor) -> Result<HipTensor> {
    let a_hip = HipTensor::from_scaffold_tensor(a.clone());
    let dt_bias_hip = HipTensor::from_scaffold_tensor(dt_bias.clone());
    let a_log_exp_hip = HipTensor::from_scaffold_tensor(a_log_exp.clone());
    if let (Some(a), Some(dt_bias), Some(a_log_exp)) = (
        a_hip.0 .0.direct_device_buffer(),
        dt_bias_hip.0 .0.direct_device_buffer(),
        a_log_exp_hip.0 .0.direct_device_buffer(),
    ) {
        if a.tensor.device().is_hip() {
            return Ok(from_device_tensor(hip_value_decay(
                &a.tensor,
                &dt_bias.tensor,
                &a_log_exp.tensor,
            )?));
        }
        let softplus = ((a.tensor.broadcast_add(&dt_bias.tensor)?.exp()? + 1.0)?).log()?;
        let out = softplus.broadcast_mul(&a_log_exp.tensor)?.neg()?;
        return Ok(from_device_tensor(out));
    }
    if let Some(host) = value_decay_host(&a_hip, &dt_bias_hip, &a_log_exp_hip)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_value_decay(
        a, dt_bias, a_log_exp,
    )?))
}

pub(crate) fn value_decay_buffer(
    a: &StateBuffer,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<StateBuffer> {
    value_decay(a.tensor(), dt_bias, a_log_exp)?.into_state_buffer()
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle_core::bail!("inconsistent batch size in rope {b_sz} {cs:?}")
            }
            Ok((t, d))
        }
        _ => candle_core::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

fn rope_hip(xs: &HipTensor, cos: &Tensor, sin: &Tensor) -> Result<HipTensor> {
    let (b_sz, n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle_core::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            Shape::from(xs.0.shape()),
            cos.shape(),
            sin.shape()
        )
    }

    let cos = HipTensor::from_scaffold_tensor(cos.clone());
    let sin = HipTensor::from_scaffold_tensor(sin.clone());
    if let (Some(xs), Some(cos), Some(sin)) = (
        xs.0 .0.direct_device_buffer(),
        cos.0 .0.direct_device_buffer(),
        sin.0 .0.direct_device_buffer(),
    ) {
        let cos = cos
            .narrow(0, 0, seq_len)?
            .reshape(vec![seq_len, n_embd / 2, 1])?
            .expand(vec![b_sz, 1, seq_len, n_embd / 2, 1])?;
        let sin = sin
            .narrow(0, 0, seq_len)?
            .reshape(vec![seq_len, n_embd / 2, 1])?
            .expand(vec![b_sz, 1, seq_len, n_embd / 2, 1])?;
        let x = xs
            .contiguous()?
            .reshape(vec![b_sz, n_head, seq_len, n_embd / 2, 2])?;
        let last_dim = x.tensor.dims().len() - 1;
        let x0 = x.narrow(last_dim, 0, 1)?;
        let x1 = x.narrow(last_dim, 1, 1)?;
        let y0 = x0.broadcast_mul(&cos)?.broadcast_sub(&x1.broadcast_mul(&sin)?)?;
        let y1 = x0.broadcast_mul(&sin)?.broadcast_add(&x1.broadcast_mul(&cos)?)?;
        return Ok(HipTensor::from_device_buffer(
            HipDeviceBuffer::cat(&[&y0, &y1], y0.tensor.dims().len() - 1)?
                .reshape(vec![b_sz, n_head, seq_len, n_embd])?,
        ));
    }

    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?
        .expand((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?
        .expand((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = xs.contiguous()?.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(candle_core::D::Minus1, 0, 1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 1, 1)?;
    let y0 = x0.broadcast_mul(&cos)?.broadcast_sub(&x1.broadcast_mul(&sin)?)?;
    let y1 = x0.broadcast_mul(&sin)?.broadcast_add(&x1.broadcast_mul(&cos)?)?;
    HipTensor::cat(&[&y0, &y1], y0.0.shape().len() - 1)?
        .reshape((b_sz, n_head, seq_len, n_embd))
}

pub(crate) fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    rope_hip(&HipTensor::from_scaffold_tensor(xs.clone()), cos, sin).map(|t| t.into_tensor())
}

pub(crate) fn linear_prefill_conv(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(linear_prefill_conv_pack(
        mixed_qkv,
        weights,
        seq_len,
        kernel_size,
    )?))
}

pub(crate) fn linear_stateful_conv(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(linear_stateful_conv_hip(
        mixed_qkv,
        prev_state,
        weights,
        kernel_size,
    )?))
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
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(linear_decode_step_hip(
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
    )?))
}

pub(crate) fn linear_stateful_conv_value_decay_with_state(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(linear_stateful_conv_value_decay_with_state_hip(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )?))
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
    linear_decode_step(
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
    )?
    .into_state_buffer()
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
    linear_stateful_conv_value_decay_with_state(
        mixed_qkv.tensor(),
        prev_state,
        weights,
        a.tensor(),
        dt_bias,
        a_log_exp,
        kernel_size,
    )?
    .into_state_buffer()
}

pub(crate) fn full_attention_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(full_attention_prefill_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?))
}

pub(crate) fn full_attention_prefill_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    full_attention_prefill(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?
    .into_state_buffer()
}

pub(crate) fn full_attention_decode(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<HipTensor> {
    Ok(from_kernel_tensor(full_attention_decode_megakernel(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?))
}

pub(crate) fn full_attention_decode_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<StateBuffer> {
    full_attention_decode(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )?
    .into_state_buffer()
}

pub(crate) fn delta_recurrent_prefill_buffer(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_recurrent_prefill(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_single_prefill_buffer(
    initial_state: &StateBuffer,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_chunk_single_prefill(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_scan_raw_buffer(
    initial_state: &StateBuffer,
    query_scan: &Tensor,
    key_scan: &Tensor,
    value_scan: &Tensor,
    beta_scan: &Tensor,
    g_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_chunk_scan_raw(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_base_attn_scan_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_base_attn_scan(k_beta_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_from_inputs_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_attn_solve_from_inputs(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_scan_buffer(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    from_kernel_tensor(delta_attn_solve_scan(base_attn_scan.tensor())?)
        .into_state_buffer()
}

pub(crate) fn delta_local_attn_scan_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_local_attn_scan(query_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_full_scan_pack_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_full_scan_pack(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_full_scan_packed_buffer(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    local_attn_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_full_scan_packed(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )?)
    .into_state_buffer()
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
    from_kernel_tensor(delta_full_scan(
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

pub(crate) fn delta_state_scan_buffer(
    initial_state: &StateBuffer,
    packed_scan: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_state_scan(
        initial_state.tensor(),
        packed_scan.tensor(),
        value,
    )?)
    .into_state_buffer()
}

pub(crate) fn delta_chunk_fused_buffer(
    prev_state: &StateBuffer,
    packed_chunk: &StateBuffer,
    value: &Tensor,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_chunk_fused(
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
    let prev_state = HipTensor::from_state_buffer(prev_state);
    let k_cumdecay_chunk = HipTensor::from_scaffold_tensor(k_cumdecay_chunk.clone());
    let q_state_chunk = HipTensor::from_scaffold_tensor(q_state_chunk.clone());
    let value_chunk = HipTensor::from_scaffold_tensor(value_chunk.clone());
    if let (Some(prev_state), Some(k_cumdecay_chunk), Some(q_state_chunk), Some(value_chunk)) = (
        prev_state.0 .0.direct_device_buffer(),
        k_cumdecay_chunk.0 .0.direct_device_buffer(),
        q_state_chunk.0 .0.direct_device_buffer(),
        value_chunk.0 .0.direct_device_buffer(),
    ) {
        let v_prime = k_cumdecay_chunk.matmul(prev_state)?;
        let v_new = value_chunk.broadcast_sub(&v_prime)?;
        let attn_inter = q_state_chunk.matmul(prev_state)?;
        return Ok((
            HipTensor::from_device_buffer(v_new).into_state_buffer()?,
            HipTensor::from_device_buffer(attn_inter).into_state_buffer()?,
        ));
    }
    let v_prime = k_cumdecay_chunk.matmul(&prev_state)?;
    let v_new = value_chunk.broadcast_sub(&v_prime)?;
    let attn_inter = q_state_chunk.matmul(&prev_state)?;
    Ok((v_new.into_state_buffer()?, attn_inter.into_state_buffer()?))
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    let attn = HipTensor::from_scaffold_tensor(attn.clone());
    let attn_inter = HipTensor::from_state_buffer(attn_inter);
    let value_chunk = HipTensor::from_state_buffer(value_chunk);
    if let (Some(attn), Some(attn_inter), Some(value_chunk)) = (
        attn.0 .0.direct_device_buffer(),
        attn_inter.0 .0.direct_device_buffer(),
        value_chunk.0 .0.direct_device_buffer(),
    ) {
        let attn_value = attn.matmul(value_chunk)?;
        let mixed = attn_inter.broadcast_add(&attn_value)?;
        return HipTensor::from_device_buffer(mixed).into_state_buffer();
    }
    let attn_value = attn.matmul(&value_chunk)?;
    let mixed = attn_inter.broadcast_add(&attn_value)?;
    mixed.into_state_buffer()
}

pub(crate) fn delta_state_update_buffer(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    from_kernel_tensor(delta_state_update(
        prev_state_scaled,
        weighted_key,
        value.tensor(),
        use_kernel,
    )?)
    .into_state_buffer()
}
