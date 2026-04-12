use crate::qwen35_minimal_impl::model::{
    delta_attn_solve_from_inputs, delta_attn_solve_scan, delta_base_attn_scan, delta_chunk_fused,
    delta_chunk_scan_raw, delta_chunk_single_prefill, delta_full_scan, delta_full_scan_pack,
    delta_full_scan_packed, delta_local_attn_scan, delta_recurrent_prefill, delta_state_scan,
    delta_state_update, full_attention_decode_megakernel, full_attention_prefill_megakernel,
    hip_causal_mask, hip_cumsum_last_dim, hip_embedding_lookup, hip_immutable_embedding_lookup,
    hip_l2norm, hip_rms_norm, hip_rms_norm_gated, hip_swiglu_mul, hip_value_decay,
    immutable_output_projection, linear_decode_step_hip, linear_prefill_conv_pack,
    linear_stateful_conv_hip, linear_stateful_conv_value_decay_with_state_hip,
    ImmutableEmbedding, StateBuffer,
};
use half::{bf16, f16};
use std::sync::Arc;
use candle_core::shape::Dim;

pub(crate) use candle_core::{DType, Device, Result, Shape, Tensor};

#[derive(Debug, Clone)]
pub(crate) enum HipNativeExpr {
    ImportedTensor(Tensor),
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
}

#[derive(Debug, Clone)]
pub(crate) struct HipNativeBuffer {
    expr: HipNativeExpr,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl HipNativeBuffer {
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

    fn try_materialize_host_bytes(&self) -> Result<Option<Arc<[u8]>>> {
        match &self.expr {
            HipNativeExpr::HostBytes { bytes } => Ok(Some(bytes.clone())),
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
            _ => Ok(None),
        }
    }

    pub(crate) fn imported_tensor(tensor: Tensor) -> Self {
        Self {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: tensor.device().clone(),
            expr: HipNativeExpr::ImportedTensor(tensor),
        }
    }

    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Self {
        let elem_count: usize = dims.iter().product();
        let byte_len = elem_count.saturating_mul(dtype.size_in_bytes());
        let bytes: Arc<[u8]> = vec![0u8; byte_len].into();
        Self {
            expr: HipNativeExpr::HostBytes { bytes },
            shape: dims,
            dtype,
            device: device.clone(),
        }
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

    pub(crate) fn materialize(&self) -> Result<Tensor> {
        if let Some(bytes) = self.try_materialize_host_bytes()? {
            return Tensor::from_raw_buffer(bytes.as_ref(), self.dtype, &self.shape, &self.device);
        }
        match &self.expr {
            HipNativeExpr::ImportedTensor(tensor) => Ok(tensor.clone()),
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

    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Self {
        Self(HipNativeBuffer::zeros(dims, dtype, device))
    }

    pub(crate) fn from_native_buffer(buffer: HipNativeBuffer) -> Self {
        Self(buffer)
    }

    pub(crate) fn materialize(&self) -> Result<Tensor> {
        self.0.materialize()
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
        Ok(self.clone())
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
        Ok(Self::from_native_buffer(HipNativeBuffer::transpose(
            Arc::new(self.0.clone()),
            dim1,
            dim2,
        )))
    }

    pub(crate) fn reshape<T: candle_core::shape::ShapeWithOneHole>(&self, shape: T) -> Result<Self> {
        let shape = shape.into_shape(self.shape().iter().product())?;
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
        Ok(Self::from_native_buffer(HipNativeBuffer::narrow(
            Arc::new(self.0.clone()),
            dim_index,
            start,
            len,
        )))
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        let lhs = self.materialize()?;
        let rhs = rhs.materialize()?;
        Ok(Self::from_tensor(lhs.matmul(&rhs)?))
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_add(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_mul(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::exp(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn max_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-max-keepdim")?;
        Ok(Self::from_native_buffer(HipNativeBuffer::max_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_sub(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn sum_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-sum-keepdim")?;
        Ok(Self::from_native_buffer(HipNativeBuffer::sum_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_div(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn recip(&self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::recip(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::recip(Arc::new(
            HipNativeBuffer::add_scalar(
                Arc::new(HipNativeBuffer::exp(Arc::new(HipNativeBuffer::neg(
                    Arc::new(self.0.clone()),
                )))),
                1.0,
            ),
        ))))
    }

    pub(crate) fn mul_scalar(&self, value: f64) -> Result<Self> {
        Ok(Self::from_native_buffer(HipNativeBuffer::mul_scalar(
            Arc::new(self.0.clone()),
            value,
        )))
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
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
}

#[derive(Debug, Clone)]
pub(crate) struct HipTensor(pub(crate) HipStorage);

impl HipTensor {
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
        self.0.into_tensor()
    }

    pub(crate) fn materialize(&self) -> Result<Tensor> {
        self.0.materialize()
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

    pub(crate) fn recip(&self) -> Result<Self> {
        Ok(Self(self.0.recip()?))
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

    pub(crate) fn cat(tensors: &[&HipTensor], dim: usize) -> Result<Self> {
        let sources = tensors
            .iter()
            .map(|t| Ok(Arc::new(t.0 .0.clone())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(HipStorage::from_native_buffer(HipNativeBuffer::concat(
            sources, dim,
        ))))
    }

    pub(crate) fn cat_tensors(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        Ok(Self(HipStorage::from_tensor(cat(tensors, dim)?)))
    }

    pub(crate) fn into_state_buffer(self) -> Result<StateBuffer> {
        StateBuffer::from_tensor(self.0.into_tensor())
    }
}

pub(crate) fn to_state_buffer(tensor: Tensor) -> Result<StateBuffer> {
    StateBuffer::from_tensor(tensor)
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
    let mixed_qkv = match prev_state {
        Some(conv_state) => {
            let conv_state = HipTensor::from_state_buffer_as(conv_state, mixed_qkv.dtype())?;
            let mixed_qkv = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
            HipTensor::cat(&[&conv_state, &mixed_qkv], 2)?
        }
        None => HipTensor::from_scaffold_tensor(mixed_qkv.clone())
            .pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
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
    let state_len = kernel_size.saturating_sub(1);
    if state_len == 0 {
        return Ok(None);
    }

    let mixed_qkv = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let seq_len = mixed_qkv.dim(2)?;
    let state = if seq_len >= state_len {
        mixed_qkv.narrow(2, seq_len - state_len, state_len)?.contiguous()?
    } else {
        match prev_state {
            Some(prev_state) => {
                let prev_state = HipTensor::from_state_buffer_as(prev_state, mixed_qkv.0.dtype())?;
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
    use candle_core::IndexOp;
    HipTensor::from_scaffold_tensor(state_scan.tensor().i((.., chunk_idx, .., ..))?)
        .into_state_buffer()
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    use candle_core::IndexOp;
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

fn repeat_heads_impl(xs: &Tensor, n_rep: usize) -> Result<HipTensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(HipTensor::from_scaffold_tensor(xs.clone()));
    }
    Ok(HipTensor::from_scaffold_tensor(xs.clone())
        .reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))?)
}

fn repeat_kv_impl(xs: &Tensor, repeats: usize) -> Result<HipTensor> {
    if repeats <= 1 {
        return Ok(HipTensor::from_scaffold_tensor(xs.clone()));
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![xs; repeats];
    Ok(HipTensor::cat_tensors(&repeated, 2)?
        .reshape((b_sz, kv_heads * repeats, seq_len, head_dim))?)
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
    let q_and_gate = HipTensor::from_state_buffer(q_and_gate).reshape((
        b_sz,
        q_len,
        num_heads,
        head_dim * 2,
    ))?;
    let query_states = rms_norm(
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
        .reshape((b_sz, q_len, num_heads * head_dim))?;
    let key_states = rms_norm(
        &HipTensor::from_state_buffer(k_proj)
            .reshape((b_sz, q_len, num_kv_heads, head_dim))?
            .into_tensor(),
        k_norm_weight,
        k_norm_eps,
        true,
    )?
    .transpose(1, 2)?;
    let value_states = HipTensor::from_state_buffer(v_proj)
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

    let query = l2norm(&query.into_tensor(), 1e-6)?;
    let key = l2norm(&key.into_tensor(), 1e-6)?;
    let head_repeat = num_v_heads / num_k_heads;
    let (query, key) = if repeat_kv_heads && head_repeat > 1 {
        (
            repeat_heads_impl(&query.into_tensor(), head_repeat)?,
            repeat_heads_impl(&key.into_tensor(), head_repeat)?,
        )
    } else {
        (query, key)
    };
    let beta = HipTensor::from_state_buffer(beta_raw)
        .sigmoid()?
        .to_dtype(compute_dtype)?;
    let g = HipTensor::from_scaffold_tensor(g.clone())
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
    Ok((
        HipTensor::from_scaffold_tensor(key_states),
        HipTensor::from_scaffold_tensor(value_states),
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
    let attn_output = HipTensor::from_scaffold_tensor(attn_output.clone())
        .transpose(1, 2)?
        .reshape((b_sz, q_len, attention_size))?
        .to_dtype(hidden_dtype)?;
    let gate = HipTensor::from_scaffold_tensor(gate.clone()).sigmoid()?;
    attn_output.broadcast_mul(&gate)?
        .into_state_buffer()
}

fn append_full_attention_kv_hip(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(HipTensor, HipTensor)> {
    match (prev_k, prev_v) {
        (Some(prev_k), Some(prev_v)) => {
            let prev_k = HipTensor::from_state_buffer_as(prev_k, key_states.dtype())?;
            let prev_v = HipTensor::from_state_buffer_as(prev_v, value_states.dtype())?;
            let key_states = HipTensor::from_scaffold_tensor(key_states.clone());
            let value_states = HipTensor::from_scaffold_tensor(value_states.clone());
            Ok((
                HipTensor::cat(&[&prev_k, &key_states], 2)?,
                HipTensor::cat(&[&prev_v, &value_states], 2)?,
            ))
        }
        _ => Ok((
            HipTensor::from_scaffold_tensor(key_states.clone()),
            HipTensor::from_scaffold_tensor(value_states.clone()),
        )),
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
    Ok((
        HipTensor::from_scaffold_tensor(query_states.clone()).contiguous()?,
        HipTensor::from_scaffold_tensor(key_states.clone()).contiguous()?,
        HipTensor::from_scaffold_tensor(value_states.clone()).contiguous()?,
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
    let key_states = repeat_kv_impl(key_states, num_kv_groups)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let value_states = repeat_kv_impl(value_states, num_kv_groups)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    Ok((
        HipTensor::from_scaffold_tensor(query_states.clone()).to_dtype(DType::F32)?,
        key_states,
        value_states,
    ))
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
    let key_states_t = HipTensor::from_scaffold_tensor(key_states_f.clone())
        .transpose(2, 3)?
        .contiguous()?;
    let mut attn_weights = HipTensor::from_scaffold_tensor(query_states_f.clone())
        .matmul(&key_states_t)?
        .mul_scalar(scale)?;
    if let Some(mask) = attention_mask {
        let mask = HipTensor::from_scaffold_tensor(mask.to_dtype(DType::F32)?);
        attn_weights = attn_weights.broadcast_add(&mask)?;
    }
    let max = attn_weights.max_keepdim(candle_core::D::Minus1)?;
    let diff = attn_weights.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(candle_core::D::Minus1)?;
    let attn_weights = num.broadcast_div(&den)?;
    attn_weights.matmul(&HipTensor::from_scaffold_tensor(value_states_f.clone()))
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
    Ok(HipTensor(HipStorage::zeros(dims, dtype, device)))
}

pub(crate) fn zeros_state(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<StateBuffer> {
    zeros(dims, dtype, device)?.into_state_buffer()
}

pub(crate) fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    Tensor::cat(tensors, dim)
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
    Ok(HipTensor::from_scaffold_tensor(hip_embedding_lookup(
        embeddings, indexes,
    )?))
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
    Ok(HipTensor::from_scaffold_tensor(hip_immutable_embedding_lookup(
        embedding, indexes,
    )?))
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<HipTensor> {
    Ok(HipTensor::from_scaffold_tensor(immutable_output_projection(
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
    Ok(HipTensor::from_scaffold_tensor(hip_rms_norm(
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
    Ok(HipTensor::from_scaffold_tensor(hip_rms_norm_gated(
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
    Ok(HipTensor::from_scaffold_tensor(hip_swiglu_mul(gate, up)?))
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
    Ok(HipTensor::from_scaffold_tensor(hip_causal_mask(
        device,
        dtype,
        batch_size,
        tgt_len,
        seqlen_offset,
    )?))
}

pub(crate) fn cumsum_last_dim(xs: &Tensor) -> Result<HipTensor> {
    Ok(HipTensor::from_scaffold_tensor(hip_cumsum_last_dim(xs)?))
}

pub(crate) fn cumsum_last_dim_buffer(xs: &StateBuffer) -> Result<StateBuffer> {
    cumsum_last_dim(xs.tensor())?.into_state_buffer()
}

pub(crate) fn l2norm(xs: &Tensor, eps: f64) -> Result<HipTensor> {
    Ok(HipTensor::from_scaffold_tensor(hip_l2norm(xs, eps)?))
}

pub(crate) fn l2norm_buffer(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    l2norm(xs.tensor(), eps)?.into_state_buffer()
}

pub(crate) fn value_decay(a: &Tensor, dt_bias: &Tensor, a_log_exp: &Tensor) -> Result<HipTensor> {
    Ok(HipTensor::from_scaffold_tensor(hip_value_decay(
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

pub(crate) fn linear_prefill_conv(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<HipTensor> {
    Ok(HipTensor::from_scaffold_tensor(linear_prefill_conv_pack(
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
    Ok(HipTensor::from_scaffold_tensor(linear_stateful_conv_hip(
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
    Ok(HipTensor::from_scaffold_tensor(linear_decode_step_hip(
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
    Ok(HipTensor::from_scaffold_tensor(linear_stateful_conv_value_decay_with_state_hip(
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
    Ok(HipTensor::from_scaffold_tensor(full_attention_prefill_megakernel(
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
    Ok(HipTensor::from_scaffold_tensor(full_attention_decode_megakernel(
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
    HipTensor::from_scaffold_tensor(delta_recurrent_prefill(
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
    HipTensor::from_scaffold_tensor(delta_chunk_single_prefill(
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
    HipTensor::from_scaffold_tensor(delta_chunk_scan_raw(
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
    HipTensor::from_scaffold_tensor(delta_base_attn_scan(k_beta_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_from_inputs_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(delta_attn_solve_from_inputs(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_scan_buffer(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(delta_attn_solve_scan(base_attn_scan.tensor())?)
        .into_state_buffer()
}

pub(crate) fn delta_local_attn_scan_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(delta_local_attn_scan(query_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_full_scan_pack_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(delta_full_scan_pack(
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
    HipTensor::from_scaffold_tensor(delta_full_scan_packed(
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
    HipTensor::from_scaffold_tensor(delta_full_scan(
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
    HipTensor::from_scaffold_tensor(delta_state_scan(
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
    HipTensor::from_scaffold_tensor(delta_chunk_fused(
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
    let v_prime = HipTensor::from_scaffold_tensor(k_cumdecay_chunk.clone()).matmul(&prev_state)?;
    let v_new = HipTensor::from_scaffold_tensor(value_chunk.clone()).broadcast_sub(&v_prime)?;
    let attn_inter = HipTensor::from_scaffold_tensor(q_state_chunk.clone()).matmul(&prev_state)?;
    Ok((v_new.into_state_buffer()?, attn_inter.into_state_buffer()?))
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    let attn_value = HipTensor::from_scaffold_tensor(attn.clone())
        .matmul(&HipTensor::from_state_buffer(value_chunk))?;
    let mixed = HipTensor::from_state_buffer(attn_inter).broadcast_add(&attn_value)?;
    mixed.into_state_buffer()
}

pub(crate) fn delta_state_update_buffer(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    HipTensor::from_scaffold_tensor(delta_state_update(
        prev_state_scaled,
        weighted_key,
        value.tensor(),
        use_kernel,
    )?)
    .into_state_buffer()
}
