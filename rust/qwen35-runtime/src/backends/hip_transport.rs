use crate::qwen35_minimal_impl::model::{
    delta_attn_solve_from_inputs, delta_attn_solve_from_inputs_host_buffer, delta_attn_solve_scan,
    delta_attn_solve_scan_host_buffer, delta_base_attn_scan, delta_base_attn_scan_host_buffer,
    delta_chunk_fused, delta_chunk_fused_host_buffer, delta_chunk_scan_raw,
    delta_chunk_scan_raw_host_buffer,
    delta_chunk_single_prefill, delta_chunk_single_prefill_host_buffer, delta_full_scan,
    delta_full_scan_host_buffer, delta_full_scan_pack, delta_full_scan_pack_host_buffer,
    delta_full_scan_packed, delta_full_scan_packed_host_buffer, delta_local_attn_scan,
    delta_local_attn_scan_host_buffer, delta_recurrent_prefill, delta_recurrent_prefill_host_buffer,
    delta_state_scan, delta_state_scan_host_buffer, full_attention_decode_megakernel,
    full_attention_prefill_megakernel, full_attention_prefill_host_buffer,
    hip_causal_mask, hip_causal_mask_host_buffer, hip_cumsum_last_dim,
    hip_cumsum_last_dim_host_buffer, hip_embedding_lookup, hip_embedding_lookup_host_buffer,
    hip_broadcast_add_host_buffer, hip_broadcast_div_host_buffer, hip_broadcast_mul_host_buffer,
    hip_broadcast_sub_host_buffer, hip_cast_host_buffer, hip_exp_host_buffer,
    hip_immutable_embedding_lookup, hip_immutable_embedding_lookup_host_buffer,
    hip_add_scalar_host_buffer,
    hip_log_host_buffer,
    hip_l2norm_host_buffer, hip_matmul_host_buffer, hip_max_keepdim_host_buffer,
    hip_mul_scalar_host_buffer, hip_rms_norm, hip_rms_norm_gated, hip_rms_norm_gated_host_buffer,
    hip_sqrt_host_buffer, hip_sum_keepdim_host_buffer,
    hip_recip_host_buffer, hip_rms_norm_host_buffer, hip_sigmoid_host_buffer, hip_swiglu_mul, hip_swiglu_mul_host_buffer, hip_value_decay,
    hip_value_decay_host_buffer, immutable_output_projection,
    immutable_output_projection_host_buffer, linear_decode_step_hip, linear_prefill_conv_pack,
    linear_prefill_conv_pack_host_buffer, linear_stateful_conv_hip,
    linear_decode_step_host_buffer, linear_stateful_conv_host_buffer,
    linear_stateful_conv_value_decay_with_state_hip,
    linear_stateful_conv_value_decay_with_state_host_buffer,
    ImmutableEmbedding, StateBuffer,
};
use crate::qwen35_minimal_impl::hip;
use half::{bf16, f16};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::OnceLock;
use candle_core::shape::Dim;
use candle_core::DeviceLocation;

pub(crate) use candle_core::{DType, Device, Result, Shape, Tensor};

fn hip_trace_candle_fallback_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("DOTCACHE_HIP_TRACE_CANDLE_FALLBACK")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn trace_candle_fallback(op: &str, tensor: &Tensor) {
    if hip_trace_candle_fallback_enabled() {
        eprintln!(
            "hip-candle-fallback op={} dtype={:?} shape={:?} device={:?}",
            op,
            tensor.dtype(),
            tensor.dims(),
            tensor.device().location()
        );
    }
}

fn trace_candle_storage_birth(reason: &str, tensor: &Tensor) {
    if hip_trace_candle_fallback_enabled() {
        eprintln!(
            "hip-candle-storage-birth reason={} dtype={:?} shape={:?} device={:?}",
            reason,
            tensor.dtype(),
            tensor.dims(),
            tensor.device().location()
        );
    }
}

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
    Select {
        source: Arc<HipNativeBuffer>,
        dim: usize,
        index: usize,
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
    Log {
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
    Sqrt {
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
    storage: HipDeviceStorage,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
    view_ops: Vec<HipDeviceViewOp>,
}

#[derive(Debug, Clone)]
enum HipDeviceStorage {
    CandleTensor(Tensor),
    OwnedDeviceBuffer(HipOwnedDeviceBuffer),
    MappedHostBuffer(HipMappedHostBuffer),
    HostBuffer(HipHostBuffer),
    PendingHostUpload(HipHostBuffer),
}

impl HipDeviceStorage {
    fn from_tensor(tensor: Tensor) -> Self {
        if tensor.device().is_hip() {
            if let Some(storage) = import_hip_tensor_storage(&tensor).ok().flatten() {
                return storage;
            }
            trace_candle_storage_birth("hip_import_failed", &tensor);
        }
        if !tensor.device().is_hip() {
            if let Some(bytes) = import_non_hip_tensor_bytes(&tensor).ok().flatten() {
                return Self::HostBuffer(HipHostBuffer {
                    bytes,
                    shape: tensor.dims().to_vec(),
                    dtype: tensor.dtype(),
                    device: tensor.device().clone(),
                });
            }
            trace_candle_storage_birth("non_hip_import_failed", &tensor);
        }
        Self::CandleTensor(tensor)
    }

    fn from_pending_host_upload(buffer: HipHostBuffer) -> Self {
        Self::PendingHostUpload(buffer)
    }

    fn from_host_buffer(buffer: HipHostBuffer) -> Self {
        if let Some(mapped) = HipMappedHostBuffer::new(buffer.clone()).ok() {
            Self::MappedHostBuffer(mapped)
        } else {
            Self::HostBuffer(buffer)
        }
    }

    fn is_contiguous(&self) -> bool {
        match self {
            Self::CandleTensor(tensor) => tensor.is_contiguous(),
            Self::OwnedDeviceBuffer(_) => true,
            Self::MappedHostBuffer(_) => true,
            Self::HostBuffer(_) => true,
            Self::PendingHostUpload(_) => true,
        }
    }

    fn as_host_buffer(&self) -> Option<&HipHostBuffer> {
        match self {
            Self::MappedHostBuffer(buffer) => Some(&buffer.buffer),
            Self::HostBuffer(buffer) | Self::PendingHostUpload(buffer) => Some(buffer),
            Self::OwnedDeviceBuffer(_) => None,
            Self::CandleTensor(_) => None,
        }
    }

    fn is_materialized(&self) -> bool {
        match self {
            Self::CandleTensor(_) => true,
            Self::OwnedDeviceBuffer(_) => true,
            Self::MappedHostBuffer(_) => true,
            Self::HostBuffer(_) => true,
            Self::PendingHostUpload(_) => false,
        }
    }

    fn materialize_tensor(&self) -> Result<Tensor> {
        match self {
            Self::CandleTensor(tensor) => Ok(tensor.clone()),
            Self::OwnedDeviceBuffer(buffer) => buffer.download_to_host_buffer()?.upload_to_tensor(),
            Self::MappedHostBuffer(buffer) => buffer.clone().into_host_buffer().upload_to_tensor(),
            Self::HostBuffer(buffer) => buffer.clone().upload_to_tensor(),
            Self::PendingHostUpload(buffer) => buffer.clone().upload_to_tensor(),
        }
    }

    fn into_tensor(self) -> Result<Tensor> {
        match self {
            Self::CandleTensor(tensor) => Ok(tensor),
            Self::OwnedDeviceBuffer(buffer) => buffer.download_to_host_buffer()?.upload_to_tensor(),
            Self::MappedHostBuffer(buffer) => buffer.into_host_buffer().upload_to_tensor(),
            Self::HostBuffer(buffer) => buffer.upload_to_tensor(),
            Self::PendingHostUpload(buffer) => buffer.upload_to_tensor(),
        }
    }

    fn try_extract_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        let Self::CandleTensor(tensor) = self else {
            return Ok(None);
        };
        let try_extract = |tensor: &Tensor| -> Result<Option<HipHostBuffer>> {
            let (storage, layout) = tensor.storage_and_layout();
            if !layout.is_contiguous() {
                return Ok(None);
            }
            let bytes = match &*storage {
                candle_core::Storage::Cpu(storage) => {
                    HipNativeBuffer::cpu_storage_to_bytes(storage, tensor.dtype())
                }
                _ => HipNativeBuffer::tensor_to_host_bytes(tensor, tensor.dtype())?,
            };
            let Some(bytes) = bytes else {
                return Ok(None);
            };
            Ok(Some(HipHostBuffer {
                bytes,
                shape: tensor.dims().to_vec(),
                dtype: tensor.dtype(),
                device: tensor.device().clone(),
            }))
        };
        if let Some(buffer) = try_extract(tensor)? {
            return Ok(Some(buffer));
        }
        let contiguous = tensor.contiguous()?;
        try_extract(&contiguous)
    }
}

fn import_non_hip_tensor_bytes(tensor: &Tensor) -> Result<Option<Arc<[u8]>>> {
    if tensor.device().is_hip() {
        return Ok(None);
    }
    let try_extract = |tensor: &Tensor| -> Result<Option<Arc<[u8]>>> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            return Ok(None);
        }
        Ok(match &*storage {
            candle_core::Storage::Cpu(storage) => {
                HipNativeBuffer::cpu_storage_to_bytes(storage, tensor.dtype())
            }
            _ => HipNativeBuffer::tensor_to_host_bytes(tensor, tensor.dtype())?,
        })
    };
    if let Some(bytes) = try_extract(tensor)? {
        return Ok(Some(bytes));
    }
    let contiguous = tensor.contiguous()?;
    try_extract(&contiguous)
}

fn import_hip_tensor_storage(tensor: &Tensor) -> Result<Option<HipDeviceStorage>> {
    if !tensor.device().is_hip() {
        return Ok(None);
    }
    if let Some(buffer) = HipOwnedDeviceBuffer::from_device_tensor_copy(tensor).ok().flatten() {
        return Ok(Some(HipDeviceStorage::OwnedDeviceBuffer(buffer)));
    }
    if let Some(host) = import_contiguous_hip_tensor_as_host_storage(tensor)? {
        return Ok(Some(host));
    }
    if tensor.is_contiguous() {
        return Ok(None);
    }
    let contiguous = tensor.contiguous()?;
    if let Some(buffer) = HipOwnedDeviceBuffer::from_device_tensor_copy(&contiguous).ok().flatten() {
        return Ok(Some(HipDeviceStorage::OwnedDeviceBuffer(buffer)));
    }
    if let Some(host) = import_contiguous_hip_tensor_as_host_storage(&contiguous)? {
        return Ok(Some(host));
    }
    Ok(None)
}

#[derive(Debug)]
struct RegisteredHipDeviceAllocation {
    device_ordinal: usize,
    device_ptr: usize,
}

impl RegisteredHipDeviceAllocation {
    fn device_ptr(&self) -> *const c_void {
        self.device_ptr as *const c_void
    }
}

impl Drop for RegisteredHipDeviceAllocation {
    fn drop(&mut self) {
        hip::free_device_bytes(self.device_ordinal, self.device_ptr as *mut c_void);
    }
}

#[derive(Debug, Clone)]
struct HipOwnedDeviceBuffer {
    len_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
    allocation: Arc<RegisteredHipDeviceAllocation>,
}

impl HipOwnedDeviceBuffer {
    fn allocate(shape: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        if !device.is_hip() {
            candle_core::bail!("owned HIP device buffer requires HIP device");
        }
        let len_bytes = HipNativeBuffer::byte_len(&shape, dtype);
        let ptr = hip::alloc_device_bytes(device.as_hip_device()?.ordinal(), len_bytes)?;
        Ok(Self {
            len_bytes,
            shape,
            dtype,
            device: device.clone(),
            allocation: Arc::new(RegisteredHipDeviceAllocation {
                device_ordinal: device.as_hip_device()?.ordinal(),
                device_ptr: ptr as usize,
            }),
        })
    }

    fn from_host_buffer(buffer: HipHostBuffer) -> Result<Self> {
        let out = Self::allocate(buffer.shape.clone(), buffer.dtype, &buffer.device)?;
        hip::copy_host_to_device(
            out.device.as_hip_device()?.ordinal(),
            out.raw_device_ptr() as *mut c_void,
            buffer.bytes.as_ptr() as *const c_void,
            out.len_bytes,
        )?;
        Ok(out)
    }

    fn from_device_tensor_copy(tensor: &Tensor) -> Result<Option<Self>> {
        use candle_core::Storage;

        if !tensor.device().is_hip() {
            return Ok(None);
        }
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Hip(storage) = &*storage else {
            return Ok(None);
        };
        if !layout.is_contiguous() {
            return Ok(None);
        }
        let out = Self::allocate(
            layout.shape().dims().to_vec(),
            tensor.dtype(),
            tensor.device(),
        )?;
        hip::copy_device_to_device(
            tensor.device().as_hip_device()?.ordinal(),
            out.raw_device_ptr() as *mut c_void,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            out.len_bytes,
        )?;
        Ok(Some(out))
    }

    fn raw_device_ptr(&self) -> *const c_void {
        self.allocation.device_ptr()
    }

    fn download_to_host_buffer(&self) -> Result<HipHostBuffer> {
        let mut bytes = vec![0u8; self.len_bytes];
        hip::copy_device_to_host(
            self.device.as_hip_device()?.ordinal(),
            bytes.as_mut_ptr() as *mut c_void,
            self.raw_device_ptr(),
            self.len_bytes,
        )?;
        Ok(HipHostBuffer {
            bytes: bytes.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

#[derive(Debug)]
struct RegisteredHipHostMapping {
    host_ptr: usize,
    device_ptr: usize,
}

impl RegisteredHipHostMapping {
    fn device_ptr(&self) -> *const c_void {
        self.device_ptr as *const c_void
    }
}

impl Drop for RegisteredHipHostMapping {
    fn drop(&mut self) {
        hip::unregister_host_mapping(self.host_ptr as *const c_void);
    }
}

#[derive(Debug, Clone)]
struct HipMappedHostBuffer {
    buffer: HipHostBuffer,
    mapping: Arc<RegisteredHipHostMapping>,
}

impl HipMappedHostBuffer {
    fn new(buffer: HipHostBuffer) -> Result<Self> {
        if !buffer.device.is_hip() {
            candle_core::bail!("mapped HIP host buffer requires a HIP device");
        }
        let host_ptr = buffer.bytes.as_ptr() as *const c_void;
        if host_ptr.is_null() || buffer.bytes.is_empty() {
            candle_core::bail!("mapped HIP host buffer requires non-empty host bytes");
        }
        let device_ptr = hip::register_host_mapping_for_device(
            buffer.device.as_hip_device()?.ordinal(),
            host_ptr,
            buffer.bytes.len(),
        )?;
        Ok(Self {
            buffer,
            mapping: Arc::new(RegisteredHipHostMapping {
                host_ptr: host_ptr as usize,
                device_ptr: device_ptr as usize,
            }),
        })
    }

    fn into_host_buffer(self) -> HipHostBuffer {
        self.buffer
    }

    #[allow(dead_code)]
    fn raw_device_ptr(&self) -> *const c_void {
        self.mapping.device_ptr()
    }
}

#[derive(Debug, Clone)]
enum HipDeviceViewOp {
    Narrow { dim: usize, start: usize, len: usize },
    Select { dim: usize, index: usize },
    Reshape { shape: Vec<usize> },
    Expand { shape: Vec<usize> },
    Transpose { dim1: usize, dim2: usize },
    Contiguous,
}

#[derive(Debug, Clone)]
pub(crate) struct HipHostBuffer {
    bytes: Arc<[u8]>,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl HipHostBuffer {
    fn elem_size(&self) -> usize {
        self.dtype.size_in_bytes()
    }

    fn outer_inner_counts(&self, dim: usize) -> Result<(usize, usize)> {
        if dim >= self.shape.len() {
            candle_core::bail!("dim {dim} out of range for host buffer shape {:?}", self.shape);
        }
        let outer = HipNativeBuffer::elem_count(&self.shape[..dim]);
        let inner = HipNativeBuffer::elem_count(&self.shape[dim + 1..]);
        Ok((outer.max(1), inner.max(1)))
    }

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
        Ok(HipDeviceBuffer::from_pending_host_upload(self))
    }

    pub(crate) fn upload_to_tensor(self) -> Result<Tensor> {
        Tensor::from_raw_buffer(self.bytes.as_ref(), self.dtype, &self.shape, &self.device)
    }

    pub(crate) fn upload_to_state_buffer(self) -> Result<StateBuffer> {
        StateBuffer::from_tensor(self.upload_to_device_buffer()?.into_tensor())
    }

    fn zeros(shape: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            bytes: vec![0u8; HipNativeBuffer::byte_len(&shape, dtype)].into(),
            shape,
            dtype,
            device: device.clone(),
        })
    }

    fn map_float(&self, op_name: &'static str, f: impl Fn(f64) -> f64) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("{op_name} unsupported for dtype {:?}", self.dtype);
        }
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&self.shape, self.dtype)];
        let elem_count = HipNativeBuffer::elem_count(&self.shape);
        for idx in 0..elem_count {
            let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
            HipNativeBuffer::write_host_float(&mut out, self.dtype, idx, f(value))?;
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        if dim >= self.shape.len() {
            candle_core::bail!("pad dim {dim} out of range for host buffer shape {:?}", self.shape);
        }
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let mut shape = self.shape.clone();
        let src_dim = self.shape[dim];
        shape[dim] = src_dim + left + right;
        let (outer, inner) = self.outer_inner_counts(dim)?;
        let elem_size = self.elem_size();
        let src_chunk = src_dim.saturating_mul(inner).saturating_mul(elem_size);
        let dst_chunk = shape[dim].saturating_mul(inner).saturating_mul(elem_size);
        let left_bytes = left.saturating_mul(inner).saturating_mul(elem_size);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        let src = self.bytes.as_ref();
        for outer_idx in 0..outer {
            let src_offset = outer_idx.saturating_mul(src_chunk);
            let dst_offset = outer_idx
                .saturating_mul(dst_chunk)
                .saturating_add(left_bytes);
            out[dst_offset..dst_offset + src_chunk]
                .copy_from_slice(&src[src_offset..src_offset + src_chunk]);
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn narrow_copy(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= self.shape.len() {
            candle_core::bail!("narrow dim {dim} out of range for host buffer shape {:?}", self.shape);
        }
        if start.saturating_add(len) > self.shape[dim] {
            candle_core::bail!(
                "narrow range [{start}, {}) out of bounds for dim size {}",
                start + len,
                self.shape[dim]
            );
        }
        if start == 0 && len == self.shape[dim] {
            return Ok(self.clone());
        }
        let mut shape = self.shape.clone();
        shape[dim] = len;
        let (outer, inner) = self.outer_inner_counts(dim)?;
        let elem_size = self.elem_size();
        let src_chunk = self.shape[dim].saturating_mul(inner).saturating_mul(elem_size);
        let dst_chunk = len.saturating_mul(inner).saturating_mul(elem_size);
        let skip_bytes = start.saturating_mul(inner).saturating_mul(elem_size);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        let src = self.bytes.as_ref();
        for outer_idx in 0..outer {
            let src_offset = outer_idx.saturating_mul(src_chunk).saturating_add(skip_bytes);
            let dst_offset = outer_idx.saturating_mul(dst_chunk);
            out[dst_offset..dst_offset + dst_chunk]
                .copy_from_slice(&src[src_offset..src_offset + dst_chunk]);
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn reshape_copy(&self, shape: Vec<usize>) -> Result<Self> {
        if HipNativeBuffer::elem_count(&self.shape) != HipNativeBuffer::elem_count(&shape) {
            candle_core::bail!("reshape changes element count: {:?} -> {:?}", self.shape, shape);
        }
        Ok(Self {
            bytes: self.bytes.clone(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn expand_copy(&self, shape: Vec<usize>) -> Result<Self> {
        if shape.len() < self.shape.len() {
            candle_core::bail!("expand rank shrinks: {:?} -> {:?}", self.shape, shape);
        }
        let leading = shape.len() - self.shape.len();
        for (src, dst) in self.shape.iter().zip(shape[leading..].iter()) {
            if *src != 1 && *src != *dst {
                candle_core::bail!("expand incompatible shapes: {:?} -> {:?}", self.shape, shape);
            }
        }
        if self.shape == shape {
            return Ok(self.clone());
        }

        let elem_count = HipNativeBuffer::elem_count(&shape);
        let elem_size = self.elem_size();
        let mut src_strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1].saturating_mul(self.shape[i + 1]);
        }
        let mut dst_strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            dst_strides[i] = dst_strides[i + 1].saturating_mul(shape[i + 1]);
        }
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        for dst_idx in 0..elem_count {
            let mut rem = dst_idx;
            let mut src_idx = 0usize;
            for axis in 0..shape.len() {
                let stride = dst_strides[axis];
                let coord = if stride == 0 { 0 } else { rem / stride };
                rem %= stride.max(1);
                if axis >= leading {
                    let src_axis = axis - leading;
                    let src_coord = if self.shape[src_axis] == 1 { 0 } else { coord };
                    src_idx = src_idx.saturating_add(src_coord.saturating_mul(src_strides[src_axis]));
                }
            }
            let src_byte = src_idx.saturating_mul(elem_size);
            let dst_byte = dst_idx.saturating_mul(elem_size);
            out[dst_byte..dst_byte + elem_size]
                .copy_from_slice(&self.bytes.as_ref()[src_byte..src_byte + elem_size]);
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn transpose_copy(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 >= self.shape.len() || dim2 >= self.shape.len() {
            candle_core::bail!("transpose dims out of range for host buffer shape {:?}", self.shape);
        }
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let mut shape = self.shape.clone();
        shape.swap(dim1, dim2);
        let elem_count = HipNativeBuffer::elem_count(&shape);
        let elem_size = self.elem_size();
        let src_strides = {
            let mut strides = vec![1usize; self.shape.len()];
            for i in (0..self.shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1].saturating_mul(self.shape[i + 1]);
            }
            strides
        };
        let dst_strides = {
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1].saturating_mul(shape[i + 1]);
            }
            strides
        };
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        for dst_idx in 0..elem_count {
            let mut rem = dst_idx;
            let mut coords = vec![0usize; shape.len()];
            for axis in 0..shape.len() {
                let stride = dst_strides[axis];
                coords[axis] = if stride == 0 { 0 } else { rem / stride };
                rem %= stride.max(1);
            }
            coords.swap(dim1, dim2);
            let src_idx = coords
                .iter()
                .zip(src_strides.iter())
                .fold(0usize, |acc, (coord, stride)| acc.saturating_add(coord.saturating_mul(*stride)));
            let src_byte = src_idx.saturating_mul(elem_size);
            let dst_byte = dst_idx.saturating_mul(elem_size);
            out[dst_byte..dst_byte + elem_size]
                .copy_from_slice(&self.bytes.as_ref()[src_byte..src_byte + elem_size]);
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn select_copy(&self, dim: usize, index: usize) -> Result<Self> {
        if dim >= self.shape.len() {
            candle_core::bail!("select dim {dim} out of range for host buffer shape {:?}", self.shape);
        }
        if index >= self.shape[dim] {
            candle_core::bail!(
                "select index {index} out of range for dim size {}",
                self.shape[dim]
            );
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        let elem_size = self.elem_size();
        let inner = HipNativeBuffer::elem_count(&self.shape[dim + 1..]);
        let outer = HipNativeBuffer::elem_count(&self.shape[..dim]);
        let chunk_bytes = inner.saturating_mul(elem_size);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let src_off = ((outer_idx * self.shape[dim] + index) * inner) * elem_size;
            let dst_off = outer_idx * chunk_bytes;
            out[dst_off..dst_off + chunk_bytes]
                .copy_from_slice(&self.bytes.as_ref()[src_off..src_off + chunk_bytes]);
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn cat(buffers: &[&HipHostBuffer], dim: usize) -> Result<Self> {
        let Some(first) = buffers.first() else {
            candle_core::bail!("cannot concatenate an empty host buffer list");
        };
        if dim >= first.shape.len() {
            candle_core::bail!("concat dim {dim} out of range for host buffer shape {:?}", first.shape);
        }
        let dtype = first.dtype;
        let device = first.device.clone();
        let mut shape = first.shape.clone();
        let (outer, inner) = first.outer_inner_counts(dim)?;
        let elem_size = first.elem_size();
        let mut dim_sum = 0usize;
        for buffer in buffers {
            if buffer.dtype != dtype {
                candle_core::bail!("host buffer concat dtype mismatch: {:?} vs {:?}", dtype, buffer.dtype);
            }
            if format!("{:?}", buffer.device) != format!("{:?}", device) {
                candle_core::bail!("host buffer concat device mismatch: {:?} vs {:?}", device, buffer.device);
            }
            if buffer.shape.len() != shape.len() {
                candle_core::bail!("host buffer concat rank mismatch: {:?} vs {:?}", shape, buffer.shape);
            }
            for (axis, (&lhs, &rhs)) in shape.iter().zip(buffer.shape.iter()).enumerate() {
                if axis != dim && lhs != rhs {
                    candle_core::bail!(
                        "host buffer concat shape mismatch on dim {axis}: {:?} vs {:?}",
                        shape,
                        buffer.shape
                    );
                }
            }
            dim_sum = dim_sum.saturating_add(buffer.shape[dim]);
        }
        shape[dim] = dim_sum;
        let dst_chunk = shape[dim].saturating_mul(inner).saturating_mul(elem_size);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, dtype)];
        for outer_idx in 0..outer {
            let mut write_offset = outer_idx.saturating_mul(dst_chunk);
            for buffer in buffers {
                let src_chunk = buffer.shape[dim]
                    .saturating_mul(inner)
                    .saturating_mul(elem_size);
                let src_offset = outer_idx.saturating_mul(src_chunk);
                let src = buffer.bytes.as_ref();
                out[write_offset..write_offset + src_chunk]
                    .copy_from_slice(&src[src_offset..src_offset + src_chunk]);
                write_offset += src_chunk;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype,
            device,
        })
    }

    fn cast(&self, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        if !HipNativeBuffer::supports_host_float_ops(self.dtype)
            || !HipNativeBuffer::supports_host_float_ops(dtype)
        {
            candle_core::bail!("cast unsupported for dtypes {:?} -> {:?}", self.dtype, dtype);
        }
        let elem_count = HipNativeBuffer::elem_count(&self.shape);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&self.shape, dtype)];
        for idx in 0..elem_count {
            let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
            HipNativeBuffer::write_host_float(&mut out, dtype, idx, value)?;
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype,
            device: self.device.clone(),
        })
    }

    fn exp(&self) -> Result<Self> {
        self.map_float("exp", f64::exp)
    }

    fn log(&self) -> Result<Self> {
        self.map_float("log", f64::ln)
    }

    fn broadcast_binary(
        lhs: &HipHostBuffer,
        rhs: &HipHostBuffer,
        f: impl Fn(f64, f64) -> f64,
        op_name: &'static str,
    ) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(lhs.dtype)
            || !HipNativeBuffer::supports_host_float_ops(rhs.dtype)
        {
            candle_core::bail!("{op_name} unsupported for dtypes {:?} and {:?}", lhs.dtype, rhs.dtype);
        }
        if lhs.dtype != rhs.dtype {
            candle_core::bail!(
                "{op_name} dtype mismatch: {:?} vs {:?}",
                lhs.dtype,
                rhs.dtype
            );
        }
        let shape = HipNativeBuffer::broadcast_shape(
            lhs.shape.as_slice(),
            rhs.shape.as_slice(),
            op_name,
        )?;
        let elem_count = HipNativeBuffer::elem_count(&shape);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, lhs.dtype)];
        for out_idx in 0..elem_count {
            let lhs_idx = HipNativeBuffer::broadcast_elem_index(out_idx, &shape, lhs.shape.as_slice());
            let rhs_idx = HipNativeBuffer::broadcast_elem_index(out_idx, &shape, rhs.shape.as_slice());
            let lhs_val = HipNativeBuffer::read_host_float(lhs.bytes.as_ref(), lhs.dtype, lhs_idx)?;
            let rhs_val = HipNativeBuffer::read_host_float(rhs.bytes.as_ref(), rhs.dtype, rhs_idx)?;
            HipNativeBuffer::write_host_float(&mut out, lhs.dtype, out_idx, f(lhs_val, rhs_val))?;
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: lhs.dtype,
            device: lhs.device.clone(),
        })
    }

    fn broadcast_add(lhs: &HipHostBuffer, rhs: &HipHostBuffer) -> Result<Self> {
        Self::broadcast_binary(lhs, rhs, |a, b| a + b, "broadcast add")
    }

    fn broadcast_sub(lhs: &HipHostBuffer, rhs: &HipHostBuffer) -> Result<Self> {
        Self::broadcast_binary(lhs, rhs, |a, b| a - b, "broadcast sub")
    }

    fn broadcast_div(lhs: &HipHostBuffer, rhs: &HipHostBuffer) -> Result<Self> {
        Self::broadcast_binary(lhs, rhs, |a, b| a / b, "broadcast div")
    }

    fn broadcast_mul(lhs: &HipHostBuffer, rhs: &HipHostBuffer) -> Result<Self> {
        Self::broadcast_binary(lhs, rhs, |a, b| a * b, "broadcast mul")
    }

    fn recip(&self) -> Result<Self> {
        self.map_float("recip", |x| x.recip())
    }

    fn sigmoid(&self) -> Result<Self> {
        self.map_float("sigmoid", |x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sqrt(&self) -> Result<Self> {
        self.map_float("sqrt", f64::sqrt)
    }

    fn reduce_keepdim(&self, dim: usize, sum: bool) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("reduction unsupported for dtype {:?}", self.dtype);
        }
        if dim >= self.shape.len() {
            candle_core::bail!("reduction dim {dim} out of range for host buffer shape {:?}", self.shape);
        }
        let mut shape = self.shape.clone();
        let reduce = shape[dim];
        shape[dim] = 1;
        let inner = HipNativeBuffer::elem_count(&self.shape[dim + 1..]);
        let outer = HipNativeBuffer::elem_count(&self.shape[..dim]);
        let out_elems = HipNativeBuffer::elem_count(&shape);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            for inner_idx in 0..inner.max(1) {
                let out_idx = outer_idx * inner.max(1) + inner_idx;
                debug_assert!(out_idx < out_elems);
                let mut acc = if sum {
                    0.0
                } else {
                    HipNativeBuffer::read_host_float(
                        self.bytes.as_ref(),
                        self.dtype,
                        (outer_idx * reduce) * inner.max(1) + inner_idx,
                    )?
                };
                let start_r = if sum { 0 } else { 1 };
                for r in start_r..reduce {
                    let src_idx = ((outer_idx * reduce + r) * inner.max(1)) + inner_idx;
                    let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, src_idx)?;
                    if sum {
                        acc += value;
                    } else if value > acc {
                        acc = value;
                    }
                }
                HipNativeBuffer::write_host_float(&mut out, self.dtype, out_idx, acc)?;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn max_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_keepdim(dim, false)
    }

    fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_keepdim(dim, true)
    }

    fn mul_scalar(&self, value: f64) -> Result<Self> {
        self.map_float("mul_scalar", |x| x * value)
    }

    fn add_scalar(&self, value: f64) -> Result<Self> {
        self.map_float("add_scalar", |x| x + value)
    }

    fn l2norm(&self, eps: f64) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("l2norm unsupported for dtype {:?}", self.dtype);
        }
        let shape = self.shape.as_slice();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("l2norm requires non-empty shape");
        };
        let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let mut sum_sq = 0.0f64;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                sum_sq += value * value;
            }
            let denom = (sum_sq + eps).sqrt();
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                HipNativeBuffer::write_host_float(&mut out, self.dtype, idx, value / denom)?;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn rms_norm(&self, weight: &Tensor, eps: f64, add_unit_offset: bool) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("rms_norm unsupported for dtype {:?}", self.dtype);
        }
        let Some(weight_bytes) = HipNativeBuffer::tensor_to_host_float_bytes(weight, DType::F32)? else {
            candle_core::bail!("rms_norm weight unsupported for host materialization");
        };
        let shape = self.shape.as_slice();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("rms_norm requires non-empty shape");
        };
        if weight.dim(0)? != inner {
            candle_core::bail!(
                "rms_norm weight dim mismatch: expected {inner}, got {}",
                weight.dim(0)?
            );
        }
        let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let mut sum_sq = 0.0f64;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                sum_sq += value * value;
            }
            let denom = ((sum_sq / inner as f64) + eps).sqrt();
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                let mut w = HipNativeBuffer::read_host_float(&weight_bytes, DType::F32, inner_idx)?;
                if add_unit_offset {
                    w += 1.0;
                }
                HipNativeBuffer::write_host_float(&mut out, self.dtype, idx, (value / denom) * w)?;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn rms_norm_gated(
        &self,
        gate: &HipHostBuffer,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Self> {
        if self.shape != gate.shape || self.dtype != gate.dtype {
            candle_core::bail!(
                "gated rms_norm shape/dtype mismatch: {:?}/{:?} vs {:?}/{:?}",
                self.shape,
                self.dtype,
                gate.shape,
                gate.dtype
            );
        }
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("gated rms_norm unsupported for dtype {:?}", self.dtype);
        }
        let Some(weight_bytes) = HipNativeBuffer::tensor_to_host_float_bytes(weight, DType::F32)? else {
            candle_core::bail!("gated rms_norm weight unsupported for host materialization");
        };
        let shape = self.shape.as_slice();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("gated rms_norm requires non-empty shape");
        };
        if weight.dim(0)? != inner {
            candle_core::bail!(
                "gated rms_norm weight dim mismatch: expected {inner}, got {}",
                weight.dim(0)?
            );
        }
        let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let mut sum_sq = 0.0f64;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let value = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                sum_sq += value * value;
            }
            let denom = ((sum_sq / inner as f64) + eps).sqrt();
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                let x = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                let g = HipNativeBuffer::read_host_float(gate.bytes.as_ref(), gate.dtype, idx)?;
                let w = HipNativeBuffer::read_host_float(&weight_bytes, DType::F32, inner_idx)?;
                let silu = g / (1.0 + (-g).exp());
                HipNativeBuffer::write_host_float(
                    &mut out,
                    self.dtype,
                    idx,
                    ((x / denom) * w) * silu,
                )?;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn swiglu_mul(&self, up: &HipHostBuffer) -> Result<Self> {
        if self.shape != up.shape || self.dtype != up.dtype {
            candle_core::bail!(
                "swiglu_mul shape/dtype mismatch: {:?}/{:?} vs {:?}/{:?}",
                self.shape,
                self.dtype,
                up.shape,
                up.dtype
            );
        }
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("swiglu_mul unsupported for dtype {:?}", self.dtype);
        }
        let elem_count = HipNativeBuffer::elem_count(&self.shape);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&self.shape, self.dtype)];
        for idx in 0..elem_count {
            let gate_x = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
            let up_x = HipNativeBuffer::read_host_float(up.bytes.as_ref(), up.dtype, idx)?;
            let silu = gate_x / (1.0 + (-gate_x).exp());
            HipNativeBuffer::write_host_float(&mut out, self.dtype, idx, silu * up_x)?;
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn value_decay(&self, dt_bias: &HipHostBuffer, a_log_exp: &HipHostBuffer) -> Result<Self> {
        if self.dtype != dt_bias.dtype || self.dtype != a_log_exp.dtype {
            candle_core::bail!(
                "value_decay dtype mismatch: {:?} vs {:?} vs {:?}",
                self.dtype,
                dt_bias.dtype,
                a_log_exp.dtype
            );
        }
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("value_decay unsupported for dtype {:?}", self.dtype);
        }
        let add_shape =
            HipNativeBuffer::broadcast_shape(self.shape.as_slice(), dt_bias.shape.as_slice(), "host-value-decay-add")?;
        let out_shape =
            HipNativeBuffer::broadcast_shape(&add_shape, a_log_exp.shape.as_slice(), "host-value-decay-mul")?;
        let elem_count = HipNativeBuffer::elem_count(&out_shape);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&out_shape, self.dtype)];
        for out_idx in 0..elem_count {
            let a_idx =
                HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, self.shape.as_slice());
            let dt_bias_idx =
                HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, dt_bias.shape.as_slice());
            let a_log_exp_idx =
                HipNativeBuffer::broadcast_elem_index(out_idx, &out_shape, a_log_exp.shape.as_slice());
            let a_val = HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, a_idx)?;
            let dt_bias_val =
                HipNativeBuffer::read_host_float(dt_bias.bytes.as_ref(), dt_bias.dtype, dt_bias_idx)?;
            let a_log_exp_val = HipNativeBuffer::read_host_float(
                a_log_exp.bytes.as_ref(),
                a_log_exp.dtype,
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
                self.dtype,
                out_idx,
                -(softplus * a_log_exp_val),
            )?;
        }
        Ok(Self {
            bytes: out.into(),
            shape: out_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn cumsum_last_dim(&self) -> Result<Self> {
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("cumsum_last_dim unsupported for dtype {:?}", self.dtype);
        }
        let shape = self.shape.as_slice();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("cumsum_last_dim requires non-empty shape");
        };
        let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(shape, self.dtype)];
        for outer_idx in 0..outer.max(1) {
            let mut running = 0.0f64;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                running += HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, idx)?;
                HipNativeBuffer::write_host_float(&mut out, self.dtype, idx, running)?;
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn matmul(&self, rhs: &HipHostBuffer) -> Result<Self> {
        if self.dtype != rhs.dtype {
            candle_core::bail!("host matmul dtype mismatch: {:?} vs {:?}", self.dtype, rhs.dtype);
        }
        if !HipNativeBuffer::supports_host_float_ops(self.dtype) {
            candle_core::bail!("host matmul unsupported for dtype {:?}", self.dtype);
        }
        let lhs_shape = self.shape.as_slice();
        let rhs_shape = rhs.shape.as_slice();
        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            candle_core::bail!("host matmul requires rank >= 1");
        }
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();
        let lhs_k = lhs_shape[lhs_rank - 1];
        let rhs_k = rhs_shape[rhs_rank.saturating_sub(2)];
        if lhs_k != rhs_k {
            candle_core::bail!("host matmul K mismatch: {} vs {}", lhs_k, rhs_k);
        }
        let m = if lhs_rank >= 2 { lhs_shape[lhs_rank - 2] } else { 1 };
        let n = rhs_shape[rhs_rank - 1];
        let lhs_batch = &lhs_shape[..lhs_rank.saturating_sub(2)];
        let rhs_batch = &rhs_shape[..rhs_rank.saturating_sub(2)];
        let batch = HipNativeBuffer::broadcast_shape(lhs_batch, rhs_batch, "host matmul")?;
        let mut out_shape = batch.clone();
        if lhs_rank >= 2 {
            out_shape.push(m);
        }
        out_shape.push(n);
        let batch_elems = HipNativeBuffer::elem_count(&batch).max(1);
        let lhs_batch_elems = HipNativeBuffer::elem_count(lhs_batch).max(1);
        let rhs_batch_elems = HipNativeBuffer::elem_count(rhs_batch).max(1);
        let lhs_matrix = m * lhs_k;
        let rhs_matrix = lhs_k * n;
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&out_shape, self.dtype)];
        for batch_idx in 0..batch_elems {
            let lhs_batch_idx = HipNativeBuffer::broadcast_elem_index(batch_idx, &batch, lhs_batch);
            let rhs_batch_idx = HipNativeBuffer::broadcast_elem_index(batch_idx, &batch, rhs_batch);
            debug_assert!(lhs_batch_idx < lhs_batch_elems);
            debug_assert!(rhs_batch_idx < rhs_batch_elems);
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..lhs_k {
                        let lhs_idx = lhs_batch_idx * lhs_matrix + row * lhs_k + kk;
                        let rhs_idx = rhs_batch_idx * rhs_matrix + kk * n + col;
                        acc += HipNativeBuffer::read_host_float(self.bytes.as_ref(), self.dtype, lhs_idx)?
                            * HipNativeBuffer::read_host_float(rhs.bytes.as_ref(), rhs.dtype, rhs_idx)?;
                    }
                    let out_idx = batch_idx * (m * n) + row * n + col;
                    HipNativeBuffer::write_host_float(&mut out, self.dtype, out_idx, acc)?;
                }
            }
        }
        Ok(Self {
            bytes: out.into(),
            shape: out_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

impl HipDeviceBuffer {
    fn has_pending_views(&self) -> bool {
        !self.view_ops.is_empty()
    }

    fn from_pending_host_upload(buffer: HipHostBuffer) -> Self {
        Self {
            shape: buffer.shape.clone(),
            dtype: buffer.dtype,
            device: buffer.device.clone(),
            storage: HipDeviceStorage::from_pending_host_upload(buffer),
            view_ops: Vec::new(),
        }
    }

    fn from_materialized_host_buffer(buffer: HipHostBuffer) -> Self {
        Self {
            shape: buffer.shape.clone(),
            dtype: buffer.dtype,
            device: buffer.device.clone(),
            storage: HipDeviceStorage::from_host_buffer(buffer),
            view_ops: Vec::new(),
        }
    }

    fn from_owned_device_buffer(buffer: HipOwnedDeviceBuffer) -> Self {
        Self {
            shape: buffer.shape.clone(),
            dtype: buffer.dtype,
            device: buffer.device.clone(),
            storage: HipDeviceStorage::OwnedDeviceBuffer(buffer),
            view_ops: Vec::new(),
        }
    }

    fn preserves_pending_upload(&self) -> bool {
        matches!(self.storage, HipDeviceStorage::PendingHostUpload(_))
    }

    fn from_host_computed_buffer_like(&self, buffer: HipHostBuffer) -> Self {
        if self.preserves_pending_upload() {
            Self::from_pending_host_upload(buffer)
        } else if buffer.device.is_hip() {
            if let Ok(device) = HipOwnedDeviceBuffer::from_host_buffer(buffer.clone()) {
                Self::from_owned_device_buffer(device)
            } else {
                Self::from_materialized_host_buffer(buffer)
            }
        } else {
            Self::from_materialized_host_buffer(buffer)
        }
    }

    fn from_host_computed_buffer_like_either(
        lhs: &Self,
        rhs: &Self,
        buffer: HipHostBuffer,
    ) -> Self {
        if lhs.preserves_pending_upload() || rhs.preserves_pending_upload() {
            Self::from_pending_host_upload(buffer)
        } else if buffer.device.is_hip() {
            if let Ok(device) = HipOwnedDeviceBuffer::from_host_buffer(buffer.clone()) {
                Self::from_owned_device_buffer(device)
            } else {
                Self::from_materialized_host_buffer(buffer)
            }
        } else {
            Self::from_materialized_host_buffer(buffer)
        }
    }

    pub(crate) fn from_tensor(tensor: Tensor) -> Self {
        Self {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: tensor.device().clone(),
            storage: HipDeviceStorage::from_tensor(tensor),
            view_ops: Vec::new(),
        }
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.dims().len()
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.view_ops.is_empty() && self.storage.is_contiguous()
    }

    fn is_materialized(&self) -> bool {
        self.storage.is_materialized()
    }

    fn try_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        let buffer = match &self.storage {
            HipDeviceStorage::MappedHostBuffer(buffer) => &buffer.buffer,
            HipDeviceStorage::HostBuffer(buffer) | HipDeviceStorage::PendingHostUpload(buffer) => buffer,
            HipDeviceStorage::OwnedDeviceBuffer(_) => return Ok(None),
            HipDeviceStorage::CandleTensor(_) => {
                return self.storage.try_extract_host_buffer();
            }
        };
        let mut native = HipNativeBuffer {
            expr: HipNativeExpr::HostBytes {
                bytes: buffer.bytes.clone(),
            },
            shape: buffer.shape.clone(),
            dtype: buffer.dtype,
            device: buffer.device.clone(),
        };
        for op in &self.view_ops {
            native = match op {
                HipDeviceViewOp::Narrow { dim, start, len } => {
                    HipNativeBuffer::narrow(Arc::new(native), *dim, *start, *len)
                }
                HipDeviceViewOp::Select { dim, index } => {
                    HipNativeBuffer::select(Arc::new(native), *dim, *index)
                }
                HipDeviceViewOp::Reshape { shape } => {
                    HipNativeBuffer::reshape(Arc::new(native), shape.clone())
                }
                HipDeviceViewOp::Expand { shape } => {
                    HipNativeBuffer::expand(Arc::new(native), shape.clone())
                }
                HipDeviceViewOp::Transpose { dim1, dim2 } => {
                    HipNativeBuffer::transpose(Arc::new(native), *dim1, *dim2)
                }
                HipDeviceViewOp::Contiguous => native,
            };
        }
        native.materialize_host_buffer()
    }

    fn with_view_op(&self, op: HipDeviceViewOp, shape: Vec<usize>) -> Self {
        let mut view_ops = self.view_ops.clone();
        view_ops.push(op);
        Self {
            storage: self.storage.clone(),
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
            view_ops,
        }
    }

    fn can_expand_shape(source: &[usize], target: &[usize]) -> bool {
        if target.len() < source.len() {
            return false;
        }
        let leading = target.len() - source.len();
        for (src, dst) in source.iter().zip(target[leading..].iter()) {
            if *src != 1 && src != dst {
                return false;
            }
        }
        true
    }

    fn materialize_host_buffer_with_views(&self) -> Result<Option<HipHostBuffer>> {
        let mut buffer = match &self.storage {
            HipDeviceStorage::MappedHostBuffer(buffer) => buffer.buffer.clone(),
            HipDeviceStorage::HostBuffer(buffer) | HipDeviceStorage::PendingHostUpload(buffer) => {
                buffer.clone()
            }
            HipDeviceStorage::OwnedDeviceBuffer(buffer) => buffer.download_to_host_buffer()?,
            HipDeviceStorage::CandleTensor(_) => match self.storage.try_extract_host_buffer()? {
                Some(buffer) => buffer,
                None => return Ok(None),
            },
        };
        for op in &self.view_ops {
            buffer = match op {
                HipDeviceViewOp::Narrow { dim, start, len } => buffer.narrow_copy(*dim, *start, *len)?,
                HipDeviceViewOp::Select { dim, index } => buffer.select_copy(*dim, *index)?,
                HipDeviceViewOp::Reshape { shape } => buffer.reshape_copy(shape.clone())?,
                HipDeviceViewOp::Expand { shape } => buffer.expand_copy(shape.clone())?,
                HipDeviceViewOp::Transpose { dim1, dim2 } => buffer.transpose_copy(*dim1, *dim2)?,
                HipDeviceViewOp::Contiguous => buffer,
            };
        }
        Ok(Some(buffer))
    }

    fn standard_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut running = 1usize;
        for (i, dim) in shape.iter().enumerate().rev() {
            strides[i] = running;
            running = running.saturating_mul(*dim);
        }
        strides
    }

    fn is_standard_contiguous_i32(shape: &[usize], strides: &[i32]) -> bool {
        if shape.len() != strides.len() {
            return false;
        }
        let expected = Self::standard_contiguous_strides(shape);
        expected
            .iter()
            .zip(strides.iter())
            .all(|(expected, actual)| i32::try_from(*expected).ok() == Some(*actual))
    }

    fn candle_view_launch_spec(
        &self,
    ) -> Result<Option<(usize, DType, Vec<usize>, Vec<i32>, *const c_void)>> {
        use candle_core::Storage;

        let (ordinal, dtype, mut offset, mut shape, mut strides, base_ptr) = match &self.storage {
            HipDeviceStorage::CandleTensor(tensor) => {
                let ordinal = match tensor.device().location() {
                    DeviceLocation::Hip { gpu_id } => gpu_id,
                    _ => return Ok(None),
                };
                let (storage, layout) = tensor.storage_and_layout();
                let Storage::Hip(storage) = &*storage else {
                    return Ok(None);
                };
                if !layout.is_contiguous() {
                    return Ok(None);
                }
                (
                    ordinal,
                    self.dtype,
                    layout.start_offset(),
                    layout.shape().dims().to_vec(),
                    Self::standard_contiguous_strides(layout.shape().dims()),
                    storage.raw_device_ptr_with_offset(0)? as *const c_void,
                )
            }
            HipDeviceStorage::OwnedDeviceBuffer(buffer) => (
                buffer.device.as_hip_device()?.ordinal(),
                buffer.dtype,
                0usize,
                buffer.shape.clone(),
                Self::standard_contiguous_strides(&buffer.shape),
                buffer.raw_device_ptr(),
            ),
            _ => return Ok(None),
        };
        for op in &self.view_ops {
            match op {
                HipDeviceViewOp::Narrow { dim, start, len } => {
                    offset = offset.saturating_add(start.saturating_mul(strides[*dim]));
                    shape[*dim] = *len;
                }
                HipDeviceViewOp::Select { dim, index } => {
                    offset = offset.saturating_add(index.saturating_mul(strides[*dim]));
                    shape.remove(*dim);
                    strides.remove(*dim);
                }
                HipDeviceViewOp::Reshape { shape: new_shape } => {
                    if HipNativeBuffer::elem_count(&shape) != HipNativeBuffer::elem_count(new_shape) {
                        return Ok(None);
                    }
                    if strides != Self::standard_contiguous_strides(&shape) {
                        return Ok(None);
                    }
                    shape = new_shape.clone();
                    strides = Self::standard_contiguous_strides(&shape);
                }
                HipDeviceViewOp::Expand { shape: new_shape } => {
                    if !Self::can_expand_shape(&shape, new_shape) {
                        return Ok(None);
                    }
                    let leading = new_shape.len().saturating_sub(shape.len());
                    let mut new_strides = vec![0usize; new_shape.len()];
                    for i in 0..new_shape.len() {
                        if i < leading {
                            new_strides[i] = 0;
                            continue;
                        }
                        let src_i = i - leading;
                        let src_dim = shape[src_i];
                        let dst_dim = new_shape[i];
                        if src_dim == dst_dim {
                            new_strides[i] = strides[src_i];
                        } else if src_dim == 1 {
                            new_strides[i] = 0;
                        } else {
                            return Ok(None);
                        }
                    }
                    shape = new_shape.clone();
                    strides = new_strides;
                }
                HipDeviceViewOp::Transpose { dim1, dim2 } => {
                    if *dim1 >= shape.len() || *dim2 >= shape.len() {
                        return Ok(None);
                    }
                    shape.swap(*dim1, *dim2);
                    strides.swap(*dim1, *dim2);
                }
                HipDeviceViewOp::Contiguous => {
                    if strides != Self::standard_contiguous_strides(&shape) {
                        return Ok(None);
                    }
                }
            }
        }
        let strides = strides
            .into_iter()
            .map(|stride| i32::try_from(stride).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some((
            ordinal,
            dtype,
            shape,
            strides,
            (base_ptr as usize + offset.saturating_mul(dtype.size_in_bytes())) as *const c_void,
        )))
    }

    fn standard_contiguous_launch_spec(
        &self,
    ) -> Result<Option<(usize, DType, Vec<usize>, *const c_void)>> {
        let Some((ordinal, dtype, shape, strides, ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        if !Self::is_standard_contiguous_i32(&shape, &strides) {
            return Ok(None);
        }
        Ok(Some((ordinal, dtype, shape, ptr)))
    }

    fn from_raw_hip_host_output(
        bytes: Vec<u8>,
        shape: Vec<usize>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        host_result_device_buffer(HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype,
            device: device.clone(),
        })
    }

    fn from_raw_hip_device_output(
        shape: Vec<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self::from_owned_device_buffer(HipOwnedDeviceBuffer::allocate(
            shape, dtype, device,
        )?))
    }

    fn unary_candle_view_device_output(
        &self,
        output_dtype: DType,
        op: i32,
        scalar: f32,
    ) -> Result<Option<Self>> {
        let Some((ordinal, input_dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        let total_elems = HipNativeBuffer::elem_count(&shape);
        let out_dims = shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let out = Self::from_raw_hip_device_output(shape.clone(), output_dtype, &self.device)?;
        let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
            return Ok(None);
        };
        let dtype_code = hip::dtype_code(input_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_unary_view(
                op,
                dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                scalar,
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                buffer.raw_device_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(hip::hip_error("hip-unary-view", status));
        }
        Ok(Some(out))
    }

    fn cast_candle_view_device_output(&self, output_dtype: DType) -> Result<Option<Self>> {
        let Some((ordinal, input_dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        let total_elems = HipNativeBuffer::elem_count(&shape);
        let out_dims = shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let out = Self::from_raw_hip_device_output(shape.clone(), output_dtype, &self.device)?;
        let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
            return Ok(None);
        };
        let input_dtype_code = hip::dtype_code(input_dtype)?;
        let output_dtype_code = hip::dtype_code(output_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_cast_view(
                input_dtype_code,
                output_dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                buffer.raw_device_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(hip::hip_error("hip-cast-view", status));
        }
        Ok(Some(out))
    }

    fn unary_candle_view_host_output(
        &self,
        output_dtype: DType,
        op: i32,
        scalar: f32,
    ) -> Result<Option<Self>> {
        let Some((ordinal, input_dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        let total_elems = HipNativeBuffer::elem_count(&shape);
        let out_dims = shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0u8; total_elems.saturating_mul(output_dtype.size_in_bytes())];
        let host_ptr = out.as_mut_ptr() as *const c_void;
        let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
        let dtype_code = hip::dtype_code(input_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_unary_view(
                op,
                dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                scalar,
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error("hip-unary-view", status));
        }
        Ok(Some(Self::from_raw_hip_host_output(
            out,
            shape,
            output_dtype,
            &self.device,
        )))
    }

    fn cast_candle_view_host_output(&self, output_dtype: DType) -> Result<Option<Self>> {
        let Some((ordinal, input_dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        let total_elems = HipNativeBuffer::elem_count(&shape);
        let out_dims = shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0u8; total_elems.saturating_mul(output_dtype.size_in_bytes())];
        let host_ptr = out.as_mut_ptr() as *const c_void;
        let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
        let input_dtype_code = hip::dtype_code(input_dtype)?;
        let output_dtype_code = hip::dtype_code(output_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_cast_view(
                input_dtype_code,
                output_dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error("hip-cast-view", status));
        }
        Ok(Some(Self::from_raw_hip_host_output(
            out,
            shape,
            output_dtype,
            &self.device,
        )))
    }

    fn binary_candle_view_host_output(&self, rhs: &Self, op: i32) -> Result<Option<Self>> {
        let Some((lhs_ordinal, lhs_dtype, lhs_shape, lhs_strides, lhs_ptr)) =
            self.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        let Some((rhs_ordinal, rhs_dtype, rhs_shape, rhs_strides, rhs_ptr)) =
            rhs.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        if lhs_ordinal != rhs_ordinal || lhs_dtype != rhs_dtype {
            return Ok(None);
        }
        let rank = lhs_shape.len().max(rhs_shape.len());
        let lhs_pad = rank.saturating_sub(lhs_shape.len());
        let rhs_pad = rank.saturating_sub(rhs_shape.len());
        let mut out_shape = vec![0usize; rank];
        let mut lhs_broadcast_strides = vec![0i32; rank];
        let mut rhs_broadcast_strides = vec![0i32; rank];
        let mut total_elems = 1usize;
        for dim in 0..rank {
            let lhs_dim = if dim < lhs_pad { 1 } else { lhs_shape[dim - lhs_pad] };
            let rhs_dim = if dim < rhs_pad { 1 } else { rhs_shape[dim - rhs_pad] };
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Ok(None);
            }
            let out_dim = lhs_dim.max(rhs_dim);
            out_shape[dim] = out_dim;
            total_elems = total_elems.saturating_mul(out_dim);
            lhs_broadcast_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
                0
            } else {
                lhs_strides[dim - lhs_pad]
            };
            rhs_broadcast_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
                0
            } else {
                rhs_strides[dim - rhs_pad]
            };
        }
        let out_dims = out_shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0u8; total_elems.saturating_mul(lhs_dtype.size_in_bytes())];
        let host_ptr = out.as_mut_ptr() as *const c_void;
        let device_ptr = hip::register_host_mapping_for_device(lhs_ordinal, host_ptr, out.len())?;
        let dtype_code = hip::dtype_code(lhs_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_binary_broadcast(
                op,
                dtype_code,
                lhs_ordinal,
                i32::try_from(rank).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                lhs_ptr,
                rhs_ptr,
                lhs_broadcast_strides.as_ptr(),
                rhs_broadcast_strides.as_ptr(),
                out_dims.as_ptr(),
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error("hip-binary-view", status));
        }
        Ok(Some(Self::from_raw_hip_host_output(
            out,
            out_shape,
            lhs_dtype,
            &self.device,
        )))
    }

    fn binary_candle_view_device_output(&self, rhs: &Self, op: i32) -> Result<Option<Self>> {
        let Some((lhs_ordinal, lhs_dtype, lhs_shape, lhs_strides, lhs_ptr)) =
            self.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        let Some((rhs_ordinal, rhs_dtype, rhs_shape, rhs_strides, rhs_ptr)) =
            rhs.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        if lhs_ordinal != rhs_ordinal || lhs_dtype != rhs_dtype {
            return Ok(None);
        }
        let rank = lhs_shape.len().max(rhs_shape.len());
        let lhs_pad = rank.saturating_sub(lhs_shape.len());
        let rhs_pad = rank.saturating_sub(rhs_shape.len());
        let mut out_shape = vec![0usize; rank];
        let mut lhs_broadcast_strides = vec![0i32; rank];
        let mut rhs_broadcast_strides = vec![0i32; rank];
        let mut total_elems = 1usize;
        for dim in 0..rank {
            let lhs_dim = if dim < lhs_pad { 1 } else { lhs_shape[dim - lhs_pad] };
            let rhs_dim = if dim < rhs_pad { 1 } else { rhs_shape[dim - rhs_pad] };
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Ok(None);
            }
            let out_dim = lhs_dim.max(rhs_dim);
            out_shape[dim] = out_dim;
            total_elems = total_elems.saturating_mul(out_dim);
            lhs_broadcast_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
                0
            } else {
                lhs_strides[dim - lhs_pad]
            };
            rhs_broadcast_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
                0
            } else {
                rhs_strides[dim - rhs_pad]
            };
        }
        let out_dims = out_shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let out = Self::from_raw_hip_device_output(out_shape.clone(), lhs_dtype, &self.device)?;
        let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
            return Ok(None);
        };
        let dtype_code = hip::dtype_code(lhs_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_binary_broadcast(
                op,
                dtype_code,
                lhs_ordinal,
                i32::try_from(rank).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                total_elems,
                lhs_ptr,
                rhs_ptr,
                lhs_broadcast_strides.as_ptr(),
                rhs_broadcast_strides.as_ptr(),
                out_dims.as_ptr(),
                buffer.raw_device_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(hip::hip_error("hip-binary-view", status));
        }
        Ok(Some(out))
    }

    fn reduce_candle_view_host_output(&self, dim: usize, sum: bool) -> Result<Option<Self>> {
        let Some((ordinal, dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        if dim >= shape.len() {
            return Ok(None);
        }
        let mut out_shape = shape.clone();
        let reduce_len = out_shape[dim];
        out_shape[dim] = 1;
        let total_out_elems = HipNativeBuffer::elem_count(&out_shape);
        let out_dims = out_shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0u8; total_out_elems.saturating_mul(dtype.size_in_bytes())];
        let host_ptr = out.as_mut_ptr() as *const c_void;
        let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
        let dtype_code = hip::dtype_code(dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_reduce_keepdim_view(
                dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                i32::try_from(dim).map_err(|_| candle_core::Error::Msg("dim overflow".into()))?,
                reduce_len,
                total_out_elems,
                if sum { 1 } else { 0 },
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error("hip-reduce-keepdim-view", status));
        }
        Ok(Some(Self::from_raw_hip_host_output(
            out,
            out_shape,
            dtype,
            &self.device,
        )))
    }

    fn reduce_candle_view_device_output(&self, dim: usize, sum: bool) -> Result<Option<Self>> {
        let Some((ordinal, dtype, shape, in_strides, input_ptr)) = self.candle_view_launch_spec()? else {
            return Ok(None);
        };
        if dim >= shape.len() {
            return Ok(None);
        }
        let mut out_shape = shape.clone();
        let reduce_len = out_shape[dim];
        out_shape[dim] = 1;
        let total_out_elems = HipNativeBuffer::elem_count(&out_shape);
        let out_dims = out_shape
            .iter()
            .copied()
            .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let out = Self::from_raw_hip_device_output(out_shape.clone(), dtype, &self.device)?;
        let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
            return Ok(None);
        };
        let dtype_code = hip::dtype_code(dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_reduce_keepdim_view(
                dtype_code,
                ordinal,
                i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
                i32::try_from(dim).map_err(|_| candle_core::Error::Msg("dim overflow".into()))?,
                reduce_len,
                total_out_elems,
                if sum { 1 } else { 0 },
                input_ptr,
                in_strides.as_ptr(),
                out_dims.as_ptr(),
                buffer.raw_device_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(hip::hip_error("hip-reduce-keepdim-view", status));
        }
        Ok(Some(out))
    }

    fn matmul_candle_view_host_output(&self, rhs: &Self) -> Result<Option<Self>> {
        let Some((lhs_ordinal, lhs_dtype, lhs_shape, lhs_strides, lhs_ptr)) =
            self.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        let Some((rhs_ordinal, rhs_dtype, rhs_shape, rhs_strides, rhs_ptr)) =
            rhs.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        if lhs_ordinal != rhs_ordinal || lhs_dtype != rhs_dtype {
            return Ok(None);
        }
        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            return Ok(None);
        }
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();
        let lhs_k = lhs_shape[lhs_rank - 1];
        let rhs_k = rhs_shape[rhs_rank.saturating_sub(2)];
        if lhs_k != rhs_k {
            return Ok(None);
        }
        let m = if lhs_rank >= 2 { lhs_shape[lhs_rank - 2] } else { 1 };
        let n = rhs_shape[rhs_rank - 1];
        let lhs_batch = &lhs_shape[..lhs_rank.saturating_sub(2)];
        let rhs_batch = &rhs_shape[..rhs_rank.saturating_sub(2)];
        let batch_rank = lhs_batch.len().max(rhs_batch.len());
        if batch_rank > 8 {
            return Ok(None);
        }
        let lhs_matrix_rank = lhs_rank.min(2);
        let rhs_matrix_rank = rhs_rank.min(2);
        let lhs_row_stride = if lhs_matrix_rank == 2 {
            lhs_strides[lhs_rank - 2]
        } else {
            0
        };
        let lhs_k_stride = lhs_strides[lhs_rank - 1];
        let rhs_k_stride = if rhs_matrix_rank == 2 {
            rhs_strides[rhs_rank - 2]
        } else {
            0
        };
        let rhs_col_stride = rhs_strides[rhs_rank - 1];
        let lhs_pad = batch_rank.saturating_sub(lhs_batch.len());
        let rhs_pad = batch_rank.saturating_sub(rhs_batch.len());
        let mut out_batch_dims = vec![1i32; batch_rank];
        let mut lhs_batch_strides = vec![0i32; batch_rank];
        let mut rhs_batch_strides = vec![0i32; batch_rank];
        let mut batch_elems = 1usize;
        for dim in 0..batch_rank {
            let lhs_dim = if dim < lhs_pad { 1 } else { lhs_batch[dim - lhs_pad] };
            let rhs_dim = if dim < rhs_pad { 1 } else { rhs_batch[dim - rhs_pad] };
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Ok(None);
            }
            let out_dim = lhs_dim.max(rhs_dim);
            out_batch_dims[dim] = i32::try_from(out_dim)
                .map_err(|_| candle_core::Error::Msg("matmul batch dim overflow".into()))?;
            lhs_batch_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
                0
            } else {
                lhs_strides[dim - lhs_pad]
            };
            rhs_batch_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
                0
            } else {
                rhs_strides[dim - rhs_pad]
            };
            batch_elems = batch_elems.saturating_mul(out_dim);
        }
        let mut out_shape = out_batch_dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
        if lhs_rank >= 2 {
            out_shape.push(m);
        }
        out_shape.push(n);
        let total_elems = batch_elems.saturating_mul(m).saturating_mul(n);
        let mut out = vec![0u8; total_elems.saturating_mul(lhs_dtype.size_in_bytes())];
        let host_ptr = out.as_mut_ptr() as *const c_void;
        let device_ptr = hip::register_host_mapping_for_device(lhs_ordinal, host_ptr, out.len())?;
        let dtype_code = hip::dtype_code(lhs_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_batched_matmul_view(
                dtype_code,
                lhs_ordinal,
                i32::try_from(batch_rank).map_err(|_| candle_core::Error::Msg("batch rank overflow".into()))?,
                batch_elems,
                i32::try_from(m).map_err(|_| candle_core::Error::Msg("m overflow".into()))?,
                i32::try_from(n).map_err(|_| candle_core::Error::Msg("n overflow".into()))?,
                i32::try_from(lhs_k).map_err(|_| candle_core::Error::Msg("k overflow".into()))?,
                lhs_batch_strides.as_ptr(),
                rhs_batch_strides.as_ptr(),
                out_batch_dims.as_ptr(),
                lhs_row_stride,
                lhs_k_stride,
                rhs_k_stride,
                rhs_col_stride,
                lhs_ptr,
                rhs_ptr,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error("hip-batched-matmul-view", status));
        }
        Ok(Some(Self::from_raw_hip_host_output(
            out,
            out_shape,
            lhs_dtype,
            &self.device,
        )))
    }

    fn matmul_candle_view_device_output(&self, rhs: &Self) -> Result<Option<Self>> {
        let Some((lhs_ordinal, lhs_dtype, lhs_shape, lhs_strides, lhs_ptr)) =
            self.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        let Some((rhs_ordinal, rhs_dtype, rhs_shape, rhs_strides, rhs_ptr)) =
            rhs.candle_view_launch_spec()?
        else {
            return Ok(None);
        };
        if lhs_ordinal != rhs_ordinal || lhs_dtype != rhs_dtype {
            return Ok(None);
        }
        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            return Ok(None);
        }
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();
        let lhs_k = lhs_shape[lhs_rank - 1];
        let rhs_k = rhs_shape[rhs_rank.saturating_sub(2)];
        if lhs_k != rhs_k {
            return Ok(None);
        }
        let m = if lhs_rank >= 2 { lhs_shape[lhs_rank - 2] } else { 1 };
        let n = rhs_shape[rhs_rank - 1];
        let lhs_batch = &lhs_shape[..lhs_rank.saturating_sub(2)];
        let rhs_batch = &rhs_shape[..rhs_rank.saturating_sub(2)];
        let batch_rank = lhs_batch.len().max(rhs_batch.len());
        if batch_rank > 8 {
            return Ok(None);
        }
        let lhs_matrix_rank = lhs_rank.min(2);
        let rhs_matrix_rank = rhs_rank.min(2);
        let lhs_row_stride = if lhs_matrix_rank == 2 { lhs_strides[lhs_rank - 2] } else { 0 };
        let lhs_k_stride = lhs_strides[lhs_rank - 1];
        let rhs_k_stride = if rhs_matrix_rank == 2 { rhs_strides[rhs_rank - 2] } else { 0 };
        let rhs_col_stride = rhs_strides[rhs_rank - 1];
        let lhs_pad = batch_rank.saturating_sub(lhs_batch.len());
        let rhs_pad = batch_rank.saturating_sub(rhs_batch.len());
        let mut out_batch_dims = vec![1i32; batch_rank];
        let mut lhs_batch_strides = vec![0i32; batch_rank];
        let mut rhs_batch_strides = vec![0i32; batch_rank];
        let mut batch_elems = 1usize;
        for dim in 0..batch_rank {
            let lhs_dim = if dim < lhs_pad { 1 } else { lhs_batch[dim - lhs_pad] };
            let rhs_dim = if dim < rhs_pad { 1 } else { rhs_batch[dim - rhs_pad] };
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Ok(None);
            }
            let out_dim = lhs_dim.max(rhs_dim);
            out_batch_dims[dim] = i32::try_from(out_dim)
                .map_err(|_| candle_core::Error::Msg("matmul batch dim overflow".into()))?;
            lhs_batch_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
                0
            } else {
                lhs_strides[dim - lhs_pad]
            };
            rhs_batch_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
                0
            } else {
                rhs_strides[dim - rhs_pad]
            };
            batch_elems = batch_elems.saturating_mul(out_dim);
        }
        let mut out_shape = out_batch_dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
        if lhs_rank >= 2 {
            out_shape.push(m);
        }
        out_shape.push(n);
        let out = Self::from_raw_hip_device_output(out_shape, lhs_dtype, &self.device)?;
        let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
            return Ok(None);
        };
        let dtype_code = hip::dtype_code(lhs_dtype)?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_batched_matmul_view(
                dtype_code,
                lhs_ordinal,
                i32::try_from(batch_rank).map_err(|_| candle_core::Error::Msg("batch rank overflow".into()))?,
                batch_elems,
                i32::try_from(m).map_err(|_| candle_core::Error::Msg("m overflow".into()))?,
                i32::try_from(n).map_err(|_| candle_core::Error::Msg("n overflow".into()))?,
                i32::try_from(lhs_k).map_err(|_| candle_core::Error::Msg("k overflow".into()))?,
                lhs_batch_strides.as_ptr(),
                rhs_batch_strides.as_ptr(),
                out_batch_dims.as_ptr(),
                lhs_row_stride,
                lhs_k_stride,
                rhs_k_stride,
                rhs_col_stride,
                lhs_ptr,
                rhs_ptr,
                buffer.raw_device_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(hip::hip_error("hip-batched-matmul-view", status));
        }
        Ok(Some(out))
    }

    pub(crate) fn materialize_tensor(&self) -> Result<Tensor> {
        if let Some(buffer) = self.materialize_host_buffer_with_views()? {
            return buffer.upload_to_tensor();
        }
        let mut tensor = self.storage.materialize_tensor()?;
        for op in &self.view_ops {
            tensor = match op {
                HipDeviceViewOp::Narrow { dim, start, len } => tensor.narrow(*dim, *start, *len)?,
                HipDeviceViewOp::Select { dim, index } => {
                    tensor.narrow(*dim, *index, 1)?.squeeze(*dim)?
                }
                HipDeviceViewOp::Reshape { shape } => tensor.reshape(shape.clone())?,
                HipDeviceViewOp::Expand { shape } => tensor.expand(shape.clone())?,
                HipDeviceViewOp::Transpose { dim1, dim2 } => tensor.transpose(*dim1, *dim2)?,
                HipDeviceViewOp::Contiguous => {
                    if tensor.is_contiguous() {
                        tensor
                    } else {
                        tensor.contiguous()?
                    }
                }
            };
        }
        Ok(tensor)
    }

    pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<Self> {
        if device.is_hip() {
            let out = Self::from_raw_hip_device_output(dims, dtype, device)?;
            let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
                candle_core::bail!("expected owned HIP device buffer for zeros");
            };
            hip::memset_device_bytes(
                device.as_hip_device()?.ordinal(),
                buffer.raw_device_ptr() as *mut c_void,
                0,
                buffer.len_bytes,
            )?;
            return Ok(out);
        }
        Ok(Self::from_tensor(Tensor::zeros(dims.as_slice(), dtype, device)?))
    }

    pub(crate) fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            candle_core::bail!("narrow dim {dim} out of range for {:?}", dims);
        }
        if start == 0 && len == dims[dim] {
            return Ok(self.clone());
        }
        let mut shape = dims.to_vec();
        shape[dim] = len;
        Ok(self.with_view_op(HipDeviceViewOp::Narrow { dim, start, len }, shape))
    }

    pub(crate) fn select(&self, dim: usize, index: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            candle_core::bail!("select dim {dim} out of range for {:?}", dims);
        }
        if index >= dims[dim] {
            candle_core::bail!("select index {index} out of range for dim size {}", dims[dim]);
        }
        let mut shape = dims.to_vec();
        shape.remove(dim);
        Ok(self.with_view_op(HipDeviceViewOp::Select { dim, index }, shape))
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.pad_with_zeros(dim, left, right)?,
            ));
        }
        if let Some(out) = pad_with_zeros_hip_owned_device(self, dim, left, right)? {
            return Ok(out);
        }
        Ok(Self::from_tensor(
            self.materialize_tensor()?.pad_with_zeros(dim, left, right)?,
        ))
    }

    pub(crate) fn cat(buffers: &[&HipDeviceBuffer], dim: usize) -> Result<Self> {
        if buffers.is_empty() {
            candle_core::bail!("cannot concatenate an empty buffer list");
        }
        if buffers.len() == 1 {
            return Ok(buffers[0].clone());
        }
        let host_buffers = buffers
            .iter()
            .map(|buffer| buffer.try_host_buffer())
            .collect::<Result<Vec<_>>>()?;
        if host_buffers.iter().all(|buffer| buffer.is_some()) {
            let refs = host_buffers.iter().flatten().collect::<Vec<_>>();
            let host_cat = HipHostBuffer::cat(refs.as_slice(), dim)?;
            let pending = buffers.iter().any(|buffer| buffer.preserves_pending_upload());
            return Ok(if pending {
                Self::from_pending_host_upload(host_cat)
            } else {
                host_result_device_buffer(host_cat)
            });
        }
        if let Some(out) = cat_hip_owned_device(buffers, dim)? {
            return Ok(out);
        }
        let tensors = buffers
            .iter()
            .map(|b| b.materialize_tensor())
            .collect::<Result<Vec<_>>>()?;
        let tensors = tensors.iter().collect::<Vec<_>>();
        Ok(Self::from_tensor(Tensor::cat(&tensors, dim)?))
    }

    pub(crate) fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        if self.dims() == shape.as_slice() {
            return Ok(self.clone());
        }
        if HipNativeBuffer::elem_count(self.dims()) != HipNativeBuffer::elem_count(&shape) {
            candle_core::bail!("reshape changes element count: {:?} -> {:?}", self.dims(), shape);
        }
        Ok(self.with_view_op(HipDeviceViewOp::Reshape { shape: shape.clone() }, shape))
    }

    pub(crate) fn expand(&self, shape: Vec<usize>) -> Result<Self> {
        if self.dims() == shape.as_slice() {
            return Ok(self.clone());
        }
        if !Self::can_expand_shape(self.dims(), &shape) {
            candle_core::bail!("cannot expand {:?} to {:?}", self.dims(), shape);
        }
        Ok(self.with_view_op(HipDeviceViewOp::Expand { shape: shape.clone() }, shape))
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let mut shape = self.dims().to_vec();
        if dim1 >= shape.len() || dim2 >= shape.len() {
            candle_core::bail!("transpose dims out of range for {:?}", shape);
        }
        shape.swap(dim1, dim2);
        Ok(self.with_view_op(HipDeviceViewOp::Transpose { dim1, dim2 }, shape))
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        if let Some(out) = self.cast_candle_view_device_output(dtype)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.cast(dtype)?));
        }
        if let Some(out) = self.cast_candle_view_host_output(dtype)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = cast_hip_owned_device(&tensor, dtype)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from cast owned device".into())
            })?);
        }
        if let Some(out) = cast_hip_host_buffer(&tensor, dtype)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from cast host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.to_dtype.tensor", &tensor);
        Ok(Self::from_tensor(tensor.to_dtype(dtype)?))
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 0, 0.0)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.exp()?));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 0, 0.0)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = exp_hip_owned_device(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from exp owned device".into())
            })?);
        }
        if let Some(out) = exp_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from exp host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.exp.tensor", &tensor);
        Ok(Self::from_tensor(tensor.exp()?))
    }

    pub(crate) fn log(&self) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 3, 0.0)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.log()?));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 3, 0.0)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = log_hip_owned_device(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from log owned device".into())
            })?);
        }
        if let Some(out) = log_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from log host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.log.tensor", &tensor);
        Ok(Self::from_tensor(tensor.log()?))
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 2, 0.0)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.sigmoid()?));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 2, 0.0)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = sigmoid_hip_owned_device(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sigmoid owned device".into())
            })?);
        }
        if let Some(out) = sigmoid_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sigmoid host buffer".into())
            })?);
        }
        self.mul_scalar(-1.0)?.exp()?.add_scalar(1.0)?.recip()
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs_buffer), Some(rhs_buffer)) = (self.try_host_buffer()?, rhs.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(self, rhs, HipHostBuffer::broadcast_add(
                &lhs_buffer, &rhs_buffer,
            )?));
        }
        if let Some(out) = self.binary_candle_view_device_output(rhs, 0)? {
            return Ok(out);
        }
        if let Some(out) = self.binary_candle_view_host_output(rhs, 0)? {
            return Ok(out);
        }
        let lhs = self.materialize_tensor()?;
        let rhs = rhs.materialize_tensor()?;
        if let Some(out) = binary_broadcast_hip_owned_device(&lhs, &rhs, 0)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_add owned device".into())
            })?);
        }
        if let Some(out) = binary_broadcast_hip_host_buffer(&lhs, &rhs, hip_broadcast_add_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_add host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.broadcast_add.tensor_lhs", &lhs);
        Ok(Self::from_tensor(lhs.broadcast_add(&rhs)?))
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs_buffer), Some(rhs_buffer)) = (self.try_host_buffer()?, rhs.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(self, rhs, HipHostBuffer::broadcast_sub(
                &lhs_buffer, &rhs_buffer,
            )?));
        }
        if let Some(out) = self.binary_candle_view_device_output(rhs, 1)? {
            return Ok(out);
        }
        if let Some(out) = self.binary_candle_view_host_output(rhs, 1)? {
            return Ok(out);
        }
        let lhs = self.materialize_tensor()?;
        let rhs = rhs.materialize_tensor()?;
        if let Some(out) = binary_broadcast_hip_owned_device(&lhs, &rhs, 1)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_sub owned device".into())
            })?);
        }
        if let Some(out) = binary_broadcast_hip_host_buffer(&lhs, &rhs, hip_broadcast_sub_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_sub host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.broadcast_sub.tensor_lhs", &lhs);
        Ok(Self::from_tensor(lhs.broadcast_sub(&rhs)?))
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs_buffer), Some(rhs_buffer)) = (self.try_host_buffer()?, rhs.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(self, rhs, HipHostBuffer::broadcast_div(
                &lhs_buffer, &rhs_buffer,
            )?));
        }
        if let Some(out) = self.binary_candle_view_device_output(rhs, 3)? {
            return Ok(out);
        }
        if let Some(out) = self.binary_candle_view_host_output(rhs, 3)? {
            return Ok(out);
        }
        let lhs = self.materialize_tensor()?;
        let rhs = rhs.materialize_tensor()?;
        if let Some(out) = binary_broadcast_hip_owned_device(&lhs, &rhs, 3)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_div owned device".into())
            })?);
        }
        if let Some(out) = binary_broadcast_hip_host_buffer(&lhs, &rhs, hip_broadcast_div_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_div host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.broadcast_div.tensor_lhs", &lhs);
        Ok(Self::from_tensor(lhs.broadcast_div(&rhs)?))
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs_buffer), Some(rhs_buffer)) = (self.try_host_buffer()?, rhs.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(self, rhs, HipHostBuffer::broadcast_mul(
                &lhs_buffer, &rhs_buffer,
            )?));
        }
        if let Some(out) = self.binary_candle_view_device_output(rhs, 2)? {
            return Ok(out);
        }
        if let Some(out) = self.binary_candle_view_host_output(rhs, 2)? {
            return Ok(out);
        }
        let lhs = self.materialize_tensor()?;
        let rhs = rhs.materialize_tensor()?;
        if let Some(out) = binary_broadcast_hip_owned_device(&lhs, &rhs, 2)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_mul owned device".into())
            })?);
        }
        if let Some(out) = binary_broadcast_hip_host_buffer(&lhs, &rhs, hip_broadcast_mul_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from broadcast_mul host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.broadcast_mul.tensor_lhs", &lhs);
        Ok(Self::from_tensor(lhs.broadcast_mul(&rhs)?))
    }

    pub(crate) fn max_keepdim(&self, dim: usize) -> Result<Self> {
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.max_keepdim(dim)?,
            ));
        }
        if let Some(out) = self.reduce_candle_view_device_output(dim, false)? {
            return Ok(out);
        }
        if let Some(out) = self.reduce_candle_view_host_output(dim, false)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = reduce_keepdim_hip_owned_device(&tensor, dim, false)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from max_keepdim owned device".into())
            })?);
        }
        if let Some(out) = reduce_keepdim_hip_host_buffer(&tensor, dim, hip_max_keepdim_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from max_keepdim host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.max_keepdim.tensor", &tensor);
        Ok(Self::from_tensor(tensor.max_keepdim(dim)?))
    }

    pub(crate) fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.sum_keepdim(dim)?,
            ));
        }
        if let Some(out) = self.reduce_candle_view_device_output(dim, true)? {
            return Ok(out);
        }
        if let Some(out) = self.reduce_candle_view_host_output(dim, true)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = reduce_keepdim_hip_owned_device(&tensor, dim, true)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sum_keepdim owned device".into())
            })?);
        }
        if let Some(out) = reduce_keepdim_hip_host_buffer(&tensor, dim, hip_sum_keepdim_host_buffer)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sum_keepdim host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.sum_keepdim.tensor", &tensor);
        Ok(Self::from_tensor(tensor.sum_keepdim(dim)?))
    }

    pub(crate) fn mul_scalar(&self, value: f64) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 5, value as f32)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.mul_scalar(value)?,
            ));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 5, value as f32)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = mul_scalar_hip_owned_device(&tensor, value)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from mul_scalar owned device".into())
            })?);
        }
        if let Some(out) = mul_scalar_hip_host_buffer(&tensor, value)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from mul_scalar host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.mul_scalar.tensor", &tensor);
        Ok(Self::from_tensor((tensor * value)?))
    }

    pub(crate) fn add_scalar(&self, value: f64) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 6, value as f32)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.add_scalar(value)?,
            ));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 6, value as f32)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = add_scalar_hip_owned_device(&tensor, value)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from add_scalar owned device".into())
            })?);
        }
        if let Some(out) = add_scalar_hip_host_buffer(&tensor, value)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from add_scalar host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.add_scalar.tensor", &tensor);
        Ok(Self::from_tensor((tensor + value)?))
    }

    pub(crate) fn recip(&self) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 1, 0.0)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.recip()?));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 1, 0.0)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = recip_hip_owned_device(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from recip owned device".into())
            })?);
        }
        if let Some(out) = recip_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from recip host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.recip.tensor", &tensor);
        Ok(Self::from_tensor(tensor.recip()?))
    }

    pub(crate) fn sqrt(&self) -> Result<Self> {
        if let Some(out) = self.unary_candle_view_device_output(self.dtype, 4, 0.0)? {
            return Ok(out);
        }
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.sqrt()?));
        }
        if let Some(out) = self.unary_candle_view_host_output(self.dtype, 4, 0.0)? {
            return Ok(out);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = sqrt_hip_owned_device(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sqrt owned device".into())
            })?);
        }
        if let Some(out) = sqrt_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from sqrt host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.sqrt.tensor", &tensor);
        Ok(Self::from_tensor(tensor.sqrt()?))
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs_buffer), Some(rhs_buffer)) = (self.try_host_buffer()?, rhs.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(
                self,
                rhs,
                lhs_buffer.matmul(&rhs_buffer)?,
            ));
        }
        if let Some(out) = self.matmul_candle_view_device_output(rhs)? {
            return Ok(out);
        }
        if let Some(out) = self.matmul_candle_view_host_output(rhs)? {
            return Ok(out);
        }
        let lhs = self.materialize_tensor()?;
        let rhs = rhs.materialize_tensor()?;
        if let Some(out) = matmul_hip_owned_device(&lhs, &rhs)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from matmul owned device".into())
            })?);
        }
        if let Some(out) = matmul_hip_host_buffer(&lhs, &rhs)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from matmul host buffer".into())
            })?);
        }
        trace_candle_fallback("device_buffer.matmul.tensor_lhs", &lhs);
        Ok(Self::from_tensor(lhs.matmul(&rhs)?))
    }

    pub(crate) fn l2norm(&self, eps: f64) -> Result<Self> {
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(self, buffer.l2norm(eps)?));
        }
        if let Some(out) = l2norm_hip_owned_device_buffer(self, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from l2norm owned device".into())
            })?);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = l2norm_hip_owned_device(&tensor, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from l2norm owned device".into())
            })?);
        }
        if let Some(out) = l2norm_hip_host_buffer(&tensor, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from l2norm host buffer".into())
            })?);
        }
        let last_dim = self.dims().len().saturating_sub(1);
        let sq = self.broadcast_mul(self)?;
        let sum = sq.sum_keepdim(last_dim)?;
        let denom = sum.add_scalar(eps)?.sqrt()?;
        self.broadcast_div(&denom)
    }

    pub(crate) fn rms_norm(
        &self,
        weight: &Tensor,
        eps: f64,
        add_unit_offset: bool,
    ) -> Result<Self> {
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.rms_norm(weight, eps, add_unit_offset)?,
            ));
        }
        if let Some(out) = rms_norm_hip_owned_device_buffer(self, weight, eps, add_unit_offset)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from rms_norm owned device".into())
            })?);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = rms_norm_hip_owned_device(&tensor, weight, eps, add_unit_offset)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from rms_norm owned device".into())
            })?);
        }
        if let Some(out) = rms_norm_hip_host_buffer(&tensor, weight, eps, add_unit_offset)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from rms_norm host buffer".into())
            })?);
        }
        let inner = *self.dims().last().ok_or_else(|| {
            candle_core::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into())
        })?;
        let last_dim = self.dims().len().saturating_sub(1);
        let sq = self.broadcast_mul(self)?;
        let sum = sq.sum_keepdim(last_dim)?;
        let mean_sq = sum.mul_scalar(1.0 / inner as f64)?;
        let denom = mean_sq.add_scalar(eps)?.sqrt()?;
        let mut normed = self.broadcast_div(&denom)?;
        if normed.dtype() != self.dtype() {
            normed = normed.to_dtype(self.dtype())?;
        }
        let weight = if weight.dtype() == normed.dtype() {
            weight.clone()
        } else {
            weight.to_dtype(normed.dtype())?
        };
        let mut weight_buffer = Self::from_tensor(weight);
        if add_unit_offset {
            weight_buffer = weight_buffer.add_scalar(1.0)?;
        }
        normed.broadcast_mul(&weight_buffer)
    }

    pub(crate) fn rms_norm_gated(
        &self,
        gate: &Self,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Self> {
        if let (Some(hidden_buffer), Some(gate_buffer)) = (self.try_host_buffer()?, gate.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(
                self,
                gate,
                hidden_buffer.rms_norm_gated(&gate_buffer, weight, eps)?,
            ));
        }
        if let Some(out) = rms_norm_gated_hip_owned_device_buffer(self, gate, weight, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected direct device buffer from rms_norm_gated owned device".into(),
                )
            })?);
        }
        let hidden = self.materialize_tensor()?;
        let gate_tensor = gate.materialize_tensor()?;
        if let Some(out) = rms_norm_gated_hip_owned_device(&hidden, &gate_tensor, weight, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected direct device buffer from rms_norm_gated owned device".into(),
                )
            })?);
        }
        if let Some(out) = rms_norm_gated_hip_host_buffer(&hidden, &gate_tensor, weight, eps)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected direct device buffer from rms_norm_gated host buffer".into(),
                )
            })?);
        }
        let normed = self.rms_norm(weight, eps, false)?;
        let sig = gate.sigmoid()?;
        let mut silu = gate.broadcast_mul(&sig)?;
        if silu.dtype() != normed.dtype() {
            silu = silu.to_dtype(normed.dtype())?;
        }
        normed.broadcast_mul(&silu)
    }

    pub(crate) fn value_decay(&self, dt_bias: &Self, a_log_exp: &Self) -> Result<Self> {
        if let (Some(a_buffer), Some(dt_bias_buffer), Some(a_log_exp_buffer)) = (
            self.try_host_buffer()?,
            dt_bias.try_host_buffer()?,
            a_log_exp.try_host_buffer()?,
        ) {
            return Ok(Self::from_host_computed_buffer_like_either(
                self,
                a_log_exp,
                a_buffer.value_decay(&dt_bias_buffer, &a_log_exp_buffer)?,
            ));
        }
        if let Some(out) = value_decay_hip_owned_device_buffer(self, dt_bias, a_log_exp)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from value_decay owned device".into())
            })?);
        }
        let a_tensor = self.materialize_tensor()?;
        let dt_bias_tensor = dt_bias.materialize_tensor()?;
        let a_log_exp_tensor = a_log_exp.materialize_tensor()?;
        if let Some(out) = value_decay_hip_owned_device(&a_tensor, &dt_bias_tensor, &a_log_exp_tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from value_decay owned device".into())
            })?);
        }
        if let Some(out) = value_decay_hip_host_buffer(&a_tensor, &dt_bias_tensor, &a_log_exp_tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from value_decay host buffer".into())
            })?);
        }
        self.broadcast_add(dt_bias)?
            .exp()?
            .add_scalar(1.0)?
            .log()?
            .broadcast_mul(a_log_exp)?
            .mul_scalar(-1.0)
    }

    pub(crate) fn cumsum_last_dim(&self) -> Result<Self> {
        if let Some(buffer) = self.try_host_buffer()? {
            return Ok(Self::from_host_computed_buffer_like(
                self,
                buffer.cumsum_last_dim()?,
            ));
        }
        if let Some(out) = owned_cumsum_last_dim_hip_device_buffer(self)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected direct device buffer from cumsum_last_dim owned device".into(),
                )
            })?);
        }
        let tensor = self.materialize_tensor()?;
        if let Some(out) = cumsum_last_dim_hip_host_buffer(&tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected direct device buffer from cumsum_last_dim host buffer".into(),
                )
            })?);
        }
        let shape = tensor.dims().to_vec();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("dotcache-hip-cumsum-last-dim requires non-empty shape");
        };
        let outer = HipNativeBuffer::elem_count(&shape[..shape.len() - 1]);
        let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, self.dtype())];
        let flat = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for outer_idx in 0..outer.max(1) {
            let mut running = 0.0f32;
            for inner_idx in 0..inner {
                let idx = outer_idx * inner + inner_idx;
                running += flat[idx];
                HipNativeBuffer::write_host_float(&mut out, self.dtype(), idx, running as f64)?;
            }
        }
        Ok(Self::from_host_computed_buffer_like(
            self,
            HipHostBuffer {
                bytes: out.into(),
                shape,
                dtype: self.dtype(),
                device: self.device().clone(),
            },
        ))
    }

    pub(crate) fn swiglu_mul(&self, up: &Self) -> Result<Self> {
        if let (Some(gate_buffer), Some(up_buffer)) = (self.try_host_buffer()?, up.try_host_buffer()?) {
            return Ok(Self::from_host_computed_buffer_like_either(
                self,
                up,
                gate_buffer.swiglu_mul(&up_buffer)?,
            ));
        }
        if let Some(out) = swiglu_mul_hip_owned_device_buffer(self, up)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from swiglu_mul owned device".into())
            })?);
        }
        let tensor = self.materialize_tensor()?;
        let up_tensor = up.materialize_tensor()?;
        if let Some(out) = swiglu_mul_hip_owned_device(&tensor, &up_tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from swiglu_mul owned device".into())
            })?);
        }
        if let Some(out) = swiglu_mul_hip_host_buffer(&tensor, &up_tensor)? {
            return Ok(out.0 .0.direct_device_buffer().cloned().ok_or_else(|| {
                candle_core::Error::Msg("expected direct device buffer from swiglu_mul host buffer".into())
            })?);
        }
        let sig = self.sigmoid()?;
        let silu = self.broadcast_mul(&sig)?;
        silu.broadcast_mul(up)
    }

    pub(crate) fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        Ok(self.with_view_op(HipDeviceViewOp::Contiguous, self.shape.clone()))
    }

    pub(crate) fn prepare_depthwise_conv_input(
        prev_state: Option<&HipDeviceBuffer>,
        mixed_qkv: &HipDeviceBuffer,
        kernel_size: usize,
    ) -> Result<(Self, Option<Self>)> {
        if let Some(mixed_qkv_host) = mixed_qkv.try_host_buffer()? {
            let preserve_pending = prev_state.is_some_and(HipDeviceBuffer::preserves_pending_upload)
                || mixed_qkv.preserves_pending_upload();
            let wrap = |buffer| {
                if preserve_pending {
                    Self::from_pending_host_upload(buffer)
                } else {
                    host_result_device_buffer(buffer)
                }
            };
            let prepared_host = match prev_state {
                Some(conv_state) => {
                    let conv_state_host = conv_state
                        .try_host_buffer()?
                        .ok_or_else(|| candle_core::Error::msg("missing host buffer for conv state"))?;
                    HipHostBuffer::cat(&[&conv_state_host, &mixed_qkv_host], 2)?
                }
                None => mixed_qkv_host.pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
            };
            let state_len = kernel_size.saturating_sub(1);
            let next_state = if state_len == 0 {
                None
            } else {
                Some(wrap(prepared_host.narrow_copy(
                    2,
                    prepared_host.shape[2] - state_len,
                    state_len,
                )?))
            };
            return Ok((wrap(prepared_host), next_state));
        }
        let mixed_qkv = match prev_state {
            Some(conv_state) => Self::cat(&[conv_state, mixed_qkv], 2)?,
            None => mixed_qkv.pad_with_zeros(2, kernel_size.saturating_sub(1), 0)?,
        };
        let total_len = mixed_qkv.dims()[2];
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
        if let Some(mixed_qkv_host) = mixed_qkv.try_host_buffer()? {
            let state_len = kernel_size.saturating_sub(1);
            if state_len == 0 {
                return Ok(None);
            }
            let preserve_pending = prev_state.is_some_and(HipDeviceBuffer::preserves_pending_upload)
                || mixed_qkv.preserves_pending_upload();
            let wrap = |buffer| {
                if preserve_pending {
                    Self::from_pending_host_upload(buffer)
                } else {
                    host_result_device_buffer(buffer)
                }
            };
            let seq_len = mixed_qkv_host.shape[2];
            let state_host = if seq_len >= state_len {
                mixed_qkv_host.narrow_copy(2, seq_len - state_len, state_len)?
            } else {
                match prev_state {
                    Some(prev_state) => {
                        let prev_state_host = prev_state
                            .try_host_buffer()?
                            .ok_or_else(|| candle_core::Error::msg("missing host buffer for conv state"))?;
                        let keep = state_len - seq_len;
                        let prev_tail =
                            prev_state_host.narrow_copy(2, prev_state_host.shape[2] - keep, keep)?;
                        HipHostBuffer::cat(&[&prev_tail, &mixed_qkv_host], 2)?
                    }
                    None => {
                        let zeros = HipHostBuffer::zeros(
                            vec![
                                mixed_qkv_host.shape[0],
                                mixed_qkv_host.shape[1],
                                state_len - seq_len,
                            ],
                            mixed_qkv_host.dtype,
                            &mixed_qkv_host.device,
                        )?;
                        HipHostBuffer::cat(&[&zeros, &mixed_qkv_host], 2)?
                    }
                }
            };
            return Ok(Some(wrap(state_host)));
        }
        let state_len = kernel_size.saturating_sub(1);
        if state_len == 0 {
            return Ok(None);
        }

        let seq_len = mixed_qkv.dims()[2];
        let state = if seq_len >= state_len {
            mixed_qkv
                .narrow(2, seq_len - state_len, state_len)?
                .contiguous()?
        } else {
            match prev_state {
                Some(prev_state) => {
                    let keep = state_len - seq_len;
                    let prev_tail = prev_state.narrow(2, prev_state.dims()[2] - keep, keep)?;
                    Self::cat(&[&prev_tail, mixed_qkv], 2)?.contiguous()?
                }
                None => {
                    let zeros = Self::zeros(
                        vec![
                            mixed_qkv.dims()[0],
                            mixed_qkv.dims()[1],
                            state_len - seq_len,
                        ],
                        mixed_qkv.dtype(),
                        mixed_qkv.device(),
                    )?;
                    Self::cat(&[&zeros, mixed_qkv], 2)?.contiguous()?
                }
            }
        };
        Ok(Some(state))
    }

    pub(crate) fn concat_last_dim(lhs: &HipDeviceBuffer, rhs: &HipDeviceBuffer) -> Result<Self> {
        Self::cat(&[lhs, rhs], lhs.dims().len() - 1)?.contiguous()
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
        if let Some(fused) = self.storage.as_host_buffer() {
            let core_attn_out = fused
                .narrow_copy(1, 0, value_dim)?
                .reshape_copy(vec![batch_size, seq_len, value_dim])?;
            let recurrent_state = fused
                .narrow_copy(1, value_dim, num_v_heads * head_k_dim * head_v_dim)?
                .reshape_copy(vec![batch_size, num_v_heads, head_k_dim, head_v_dim])?;
            return Ok((
                self.from_host_computed_buffer_like(core_attn_out),
                self.from_host_computed_buffer_like(recurrent_state),
            ));
        }
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
        if let Some(fused) = self.storage.as_host_buffer() {
            let out_width = conv_dim + num_v_heads;
            let packed = fused
                .narrow_copy(1, 0, seq_len * out_width)?
                .reshape_copy(vec![batch_size, seq_len, out_width])?;
            let mixed_qkv = packed.narrow_copy(2, 0, conv_dim)?;
            let g = packed.narrow_copy(2, conv_dim, num_v_heads)?;
            let conv_state = fused
                .narrow_copy(1, seq_len * out_width, conv_dim * state_len)?
                .reshape_copy(vec![batch_size, conv_dim, state_len])?;
            return Ok((
                self.from_host_computed_buffer_like(mixed_qkv),
                self.from_host_computed_buffer_like(g),
                self.from_host_computed_buffer_like(conv_state),
            ));
        }
        let out_width = conv_dim + num_v_heads;
        let packed = self
            .narrow(1, 0, seq_len * out_width)?
            .reshape(vec![batch_size, seq_len, out_width])?;
        let last_dim = packed.dims().len() - 1;
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
        if let Some(fused) = self.storage.as_host_buffer() {
            let output_scan = fused
                .narrow_copy(1, 0, total_sequence_length)?
                .reshape_copy(vec![batch_size, num_heads, total_sequence_length, v_head_dim])?;
            let output = output_scan
                .narrow_copy(2, 0, output_sequence_length)?
                .transpose_copy(1, 2)?
                .cast(output_dtype)?;
            let recurrent_state = fused
                .narrow_copy(1, total_sequence_length, k_head_dim)?
                .reshape_copy(vec![batch_size * num_heads, k_head_dim, v_head_dim])?;
            return Ok((
                self.from_host_computed_buffer_like(output),
                self.from_host_computed_buffer_like(recurrent_state),
            ));
        }
        let output_scan = self
            .narrow(1, 0, total_sequence_length)?
            .reshape(vec![batch_size, num_heads, total_sequence_length, v_head_dim])?;
        let output = output_scan
            .narrow(2, 0, output_sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(output_dtype)?;
        let recurrent_state = self
            .narrow(1, total_sequence_length, k_head_dim)?
            .reshape(vec![batch_size * num_heads, k_head_dim, v_head_dim])?
            .contiguous()?;
        Ok((output, recurrent_state))
    }

    pub(crate) fn unpack_chunk_fused(
        &self,
        chunk_size: usize,
        k_head_dim: usize,
    ) -> Result<(Self, Self, Self)> {
        if let Some(fused) = self.storage.as_host_buffer() {
            return Ok((
                self.from_host_computed_buffer_like(fused.narrow_copy(1, 0, chunk_size)?),
                self.from_host_computed_buffer_like(fused.narrow_copy(1, chunk_size, chunk_size)?),
                self.from_host_computed_buffer_like(fused.narrow_copy(1, 2 * chunk_size, k_head_dim)?),
            ));
        }
        Ok((
            self.narrow(1, 0, chunk_size)?,
            self.narrow(1, chunk_size, chunk_size)?,
            self.narrow(1, 2 * chunk_size, k_head_dim)?,
        ))
    }

    pub(crate) fn repeat_heads(&self, n_rep: usize) -> Result<Self> {
        let [b_sz, seq_len, heads, head_dim] = <[usize; 4]>::try_from(self.dims())
            .map_err(|_| candle_core::Error::Msg("unexpected rank, expected 4".into()))?;
        if n_rep == 1 {
            return Ok(self.clone());
        }
        self.reshape(vec![b_sz, seq_len, heads, 1, head_dim])?
            .expand(vec![b_sz, seq_len, heads, n_rep, head_dim])?
            .reshape(vec![b_sz, seq_len, heads * n_rep, head_dim])
    }

    pub(crate) fn repeat_kv(&self, repeats: usize) -> Result<Self> {
        let [b_sz, kv_heads, seq_len, head_dim] = <[usize; 4]>::try_from(self.dims())
            .map_err(|_| candle_core::Error::Msg("unexpected rank, expected 4".into()))?;
        if repeats <= 1 {
            return Ok(self.clone());
        }
        self.reshape(vec![b_sz, kv_heads, 1, seq_len, head_dim])?
            .expand(vec![b_sz, kv_heads, repeats, seq_len, head_dim])?
            .reshape(vec![b_sz, kv_heads * repeats, seq_len, head_dim])
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        if self.view_ops.is_empty() {
            self.storage
                .into_tensor()
                .expect("valid HipDeviceBuffer storage should materialize")
        } else {
            self.materialize_tensor()
                .expect("valid HipDeviceBuffer views should materialize")
        }
    }
}

impl HipNativeBuffer {
    fn direct_device_buffer(&self) -> Option<&HipDeviceBuffer> {
        match &self.expr {
            HipNativeExpr::DeviceBuffer(buffer) => Some(buffer),
            _ => None,
        }
    }

    fn direct_materialized_device_buffer(&self) -> Option<&HipDeviceBuffer> {
        self.direct_device_buffer()
            .filter(|buffer| !buffer.has_pending_views() && buffer.is_materialized())
    }

    fn try_materialize_device_buffer(&self) -> Result<Option<HipDeviceBuffer>> {
        match &self.expr {
            HipNativeExpr::DeviceBuffer(buffer) => Ok(Some(buffer.clone())),
            HipNativeExpr::HostBytes { .. } => Ok(None),
            HipNativeExpr::PadWithZeros {
                source,
                dim,
                left,
                right,
            } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.pad_with_zeros(*dim, *left, *right))
                .transpose(),
            HipNativeExpr::Narrow {
                source,
                dim,
                start,
                len,
            } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.narrow(*dim, *start, *len))
                .transpose(),
            HipNativeExpr::Select { source, dim, index } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.select(*dim, *index))
                .transpose(),
            HipNativeExpr::Concat { sources, dim } => {
                let mut materialized = Vec::with_capacity(sources.len());
                for source in sources {
                    let Some(buffer) = source.try_materialize_device_buffer()? else {
                        return Ok(None);
                    };
                    materialized.push(buffer);
                }
                let refs = materialized.iter().collect::<Vec<_>>();
                Ok(Some(HipDeviceBuffer::cat(refs.as_slice(), *dim)?))
            }
            HipNativeExpr::Reshape { source, shape } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.reshape(shape.clone()))
                .transpose(),
            HipNativeExpr::Expand { source, shape } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.expand(shape.clone()))
                .transpose(),
            HipNativeExpr::Transpose { source, dim1, dim2 } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.transpose(*dim1, *dim2))
                .transpose(),
            HipNativeExpr::Cast { source, dtype } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.to_dtype(*dtype))
                .transpose(),
            HipNativeExpr::Exp { source } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.exp())
                .transpose(),
            HipNativeExpr::Log { source } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.log())
                .transpose(),
            HipNativeExpr::BroadcastAdd { lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    lhs.try_materialize_device_buffer()?,
                    rhs.try_materialize_device_buffer()?,
                ) else {
                    return Ok(None);
                };
                Ok(Some(lhs.broadcast_add(&rhs)?))
            }
            HipNativeExpr::BroadcastMul { lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    lhs.try_materialize_device_buffer()?,
                    rhs.try_materialize_device_buffer()?,
                ) else {
                    return Ok(None);
                };
                Ok(Some(lhs.broadcast_mul(&rhs)?))
            }
            HipNativeExpr::BroadcastSub { lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    lhs.try_materialize_device_buffer()?,
                    rhs.try_materialize_device_buffer()?,
                ) else {
                    return Ok(None);
                };
                Ok(Some(lhs.broadcast_sub(&rhs)?))
            }
            HipNativeExpr::BroadcastDiv { lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    lhs.try_materialize_device_buffer()?,
                    rhs.try_materialize_device_buffer()?,
                ) else {
                    return Ok(None);
                };
                Ok(Some(lhs.broadcast_div(&rhs)?))
            }
            HipNativeExpr::MaxKeepdim { source, dim } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.max_keepdim(*dim))
                .transpose(),
            HipNativeExpr::SumKeepdim { source, dim } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.sum_keepdim(*dim))
                .transpose(),
            HipNativeExpr::Neg { source } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.mul_scalar(-1.0))
                .transpose(),
            HipNativeExpr::AddScalar { source, value } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.add_scalar(*value))
                .transpose(),
            HipNativeExpr::MulScalar { source, value } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.mul_scalar(*value))
                .transpose(),
            HipNativeExpr::Recip { source } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.recip())
                .transpose(),
            HipNativeExpr::Sqrt { source } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.sqrt())
                .transpose(),
            HipNativeExpr::L2Norm { source, eps } => source
                .try_materialize_device_buffer()?
                .map(|buffer| buffer.l2norm(*eps))
                .transpose(),
        }
    }

    fn is_host_graph(&self) -> bool {
        match &self.expr {
            HipNativeExpr::DeviceBuffer(_) => false,
            HipNativeExpr::HostBytes { .. } => true,
            HipNativeExpr::PadWithZeros { source, .. }
            | HipNativeExpr::Narrow { source, .. }
            | HipNativeExpr::Select { source, .. }
            | HipNativeExpr::Reshape { source, .. }
            | HipNativeExpr::Expand { source, .. }
            | HipNativeExpr::Transpose { source, .. }
            | HipNativeExpr::Cast { source, .. }
            | HipNativeExpr::Exp { source }
            | HipNativeExpr::Log { source }
            | HipNativeExpr::MaxKeepdim { source, .. }
            | HipNativeExpr::SumKeepdim { source, .. }
            | HipNativeExpr::Neg { source }
            | HipNativeExpr::AddScalar { source, .. }
            | HipNativeExpr::MulScalar { source, .. }
            | HipNativeExpr::Recip { source }
            | HipNativeExpr::Sqrt { source }
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

    fn host_bytes_select(
        &self,
        source: &Arc<HipNativeBuffer>,
        dim: usize,
        index: usize,
    ) -> Result<Option<Arc<[u8]>>> {
        let bytes = match source.try_materialize_host_bytes()? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let src_shape = source.shape();
        if dim >= src_shape.len() {
            candle_core::bail!("invalid select dim {} for shape {:?}", dim, src_shape)
        }
        if index >= src_shape[dim] {
            candle_core::bail!(
                "invalid select index={} for dim {} size {}",
                index,
                dim,
                src_shape[dim]
            )
        }
        let elem_bytes = self.dtype.size_in_bytes();
        let inner = Self::elem_count(&src_shape[dim + 1..]);
        let outer = Self::elem_count(&src_shape[..dim]);
        let chunk_bytes = inner.saturating_mul(elem_bytes);
        let mut out = vec![0u8; Self::byte_len(&self.shape, self.dtype)];
        for outer_idx in 0..outer {
            let src_off = ((outer_idx * src_shape[dim] + index) * inner) * elem_bytes;
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
                if let Some(buffer) = buffer.try_host_buffer()? {
                    return Ok(Some(buffer.bytes));
                }
                if let Some(buffer) = buffer.materialize_host_buffer_with_views()? {
                    return Ok(Some(buffer.bytes));
                }
                Self::tensor_to_host_bytes(&buffer.materialize_tensor()?, self.dtype)
            }
            HipNativeExpr::Reshape { source, .. } => self.host_bytes_reshape(source),
            HipNativeExpr::Narrow {
                source,
                dim,
                start,
                len,
            } => self.host_bytes_narrow(source, *dim, *start, *len),
            HipNativeExpr::Select { source, dim, index } => {
                self.host_bytes_select(source, *dim, *index)
            }
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

    fn tensor_to_host_bytes(tensor: &Tensor, dtype: DType) -> Result<Option<Arc<[u8]>>> {
        let (storage, layout) = tensor.storage_and_layout();
        if layout.is_contiguous() {
            if let candle_core::Storage::Cpu(storage) = &*storage {
                if let Some(bytes) = Self::cpu_storage_to_bytes(storage, dtype) {
                    return Ok(Some(bytes));
                }
            }
        }
        Self::tensor_to_host_float_bytes(tensor, dtype)
    }

    fn cpu_storage_to_bytes(storage: &candle_core::CpuStorage, dtype: DType) -> Option<Arc<[u8]>> {
        let bytes = match (storage, dtype) {
            (candle_core::CpuStorage::U8(values), DType::U8) => values.clone(),
            (candle_core::CpuStorage::U32(values), DType::U32) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::I16(values), DType::I16) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::I32(values), DType::I32) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::I64(values), DType::I64) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::BF16(values), DType::BF16) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::F16(values), DType::F16) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::F32(values), DType::F32) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            (candle_core::CpuStorage::F64(values), DType::F64) => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            _ => return None,
        };
        Some(bytes.into())
    }

    pub(crate) fn imported_tensor(tensor: Tensor) -> Self {
        Self::device_buffer(HipDeviceBuffer::from_tensor(tensor))
    }

    pub(crate) fn device_buffer(buffer: HipDeviceBuffer) -> Self {
        Self {
            shape: buffer.dims().to_vec(),
            dtype: buffer.dtype(),
            device: buffer.device().clone(),
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

    pub(crate) fn select(source: Arc<HipNativeBuffer>, dim: usize, index: usize) -> Self {
        let mut shape = source.shape.clone();
        shape.remove(dim);
        Self {
            expr: HipNativeExpr::Select {
                source: source.clone(),
                dim,
                index,
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

    pub(crate) fn log(source: Arc<HipNativeBuffer>) -> Self {
        Self {
            expr: HipNativeExpr::Log {
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

    pub(crate) fn sqrt(source: Arc<HipNativeBuffer>) -> Self {
        Self {
            expr: HipNativeExpr::Sqrt {
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
            return Ok(buffer.upload_to_device_buffer()?.into_tensor());
        }
        if let Some(buffer) = self.try_materialize_device_buffer()? {
            return buffer.materialize_tensor();
        }
        match &self.expr {
            HipNativeExpr::DeviceBuffer(buffer) => buffer.materialize_tensor(),
            HipNativeExpr::HostBytes { bytes } => {
                Tensor::from_raw_buffer(bytes.as_ref(), self.dtype, &self.shape, &self.device)
            }
            HipNativeExpr::PadWithZeros {
                source,
                dim,
                left,
                right,
            } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.pad_with_zeros(*dim, *left, *right)?.into_tensor())
                } else {
                    source.materialize()?.pad_with_zeros(*dim, *left, *right)
                }
            }
            HipNativeExpr::Narrow {
                source,
                dim,
                start,
                len,
            } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.narrow(*dim, *start, *len)?.into_tensor())
                } else {
                    source.materialize()?.narrow(*dim, *start, *len)
                }
            }
            HipNativeExpr::Select { source, dim, index } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.select(*dim, *index)?.into_tensor())
                } else {
                    source.materialize()?.narrow(*dim, *index, 1)?.squeeze(*dim)
                }
            }
            HipNativeExpr::Concat { sources, dim } => {
                let device_buffers = sources
                    .iter()
                    .map(|s| match &s.expr {
                        HipNativeExpr::DeviceBuffer(buffer) => Some(buffer),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if device_buffers.iter().all(|buffer| buffer.is_some()) {
                    let refs = device_buffers.into_iter().flatten().collect::<Vec<_>>();
                    Ok(HipDeviceBuffer::cat(refs.as_slice(), *dim)?.into_tensor())
                } else {
                    let tensors = sources
                        .iter()
                        .map(|s| s.materialize())
                        .collect::<Result<Vec<_>>>()?;
                    let refs = tensors.iter().collect::<Vec<_>>();
                    Tensor::cat(&refs, *dim)
                }
            }
            HipNativeExpr::Reshape { source, shape } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.reshape(shape.clone())?.into_tensor())
                } else {
                    source.materialize()?.reshape(shape.clone())
                }
            }
            HipNativeExpr::Expand { source, shape } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.expand(shape.clone())?.into_tensor())
                } else {
                    source.materialize()?.expand(shape.clone())
                }
            }
            HipNativeExpr::Transpose { source, dim1, dim2 } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.transpose(*dim1, *dim2)?.into_tensor())
                } else {
                    source.materialize()?.transpose(*dim1, *dim2)
                }
            }
            HipNativeExpr::Cast { source, dtype } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.to_dtype(*dtype)?.into_tensor())
                } else {
                    source.materialize()?.to_dtype(*dtype)
                }
            }
            HipNativeExpr::Exp { source } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.exp()?.into_tensor())
                } else {
                    source.materialize()?.exp()
                }
            }
            HipNativeExpr::Log { source } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.log()?.into_tensor())
                } else {
                    source.materialize()?.log()
                }
            }
            HipNativeExpr::BroadcastAdd { lhs, rhs } => {
                if let (HipNativeExpr::DeviceBuffer(lhs), HipNativeExpr::DeviceBuffer(rhs)) =
                    (&lhs.expr, &rhs.expr)
                {
                    Ok(lhs.broadcast_add(rhs)?.into_tensor())
                } else {
                    lhs.materialize()?.broadcast_add(&rhs.materialize()?)
                }
            }
            HipNativeExpr::BroadcastMul { lhs, rhs } => {
                if let (HipNativeExpr::DeviceBuffer(lhs), HipNativeExpr::DeviceBuffer(rhs)) =
                    (&lhs.expr, &rhs.expr)
                {
                    Ok(lhs.broadcast_mul(rhs)?.into_tensor())
                } else {
                    lhs.materialize()?.broadcast_mul(&rhs.materialize()?)
                }
            }
            HipNativeExpr::BroadcastSub { lhs, rhs } => {
                if let (HipNativeExpr::DeviceBuffer(lhs), HipNativeExpr::DeviceBuffer(rhs)) =
                    (&lhs.expr, &rhs.expr)
                {
                    Ok(lhs.broadcast_sub(rhs)?.into_tensor())
                } else {
                    lhs.materialize()?.broadcast_sub(&rhs.materialize()?)
                }
            }
            HipNativeExpr::BroadcastDiv { lhs, rhs } => {
                if let (HipNativeExpr::DeviceBuffer(lhs), HipNativeExpr::DeviceBuffer(rhs)) =
                    (&lhs.expr, &rhs.expr)
                {
                    Ok(lhs.broadcast_div(rhs)?.into_tensor())
                } else {
                    lhs.materialize()?.broadcast_div(&rhs.materialize()?)
                }
            }
            HipNativeExpr::MaxKeepdim { source, dim } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.max_keepdim(*dim)?.into_tensor())
                } else {
                    source.materialize()?.max_keepdim(*dim)
                }
            }
            HipNativeExpr::SumKeepdim { source, dim } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.sum_keepdim(*dim)?.into_tensor())
                } else {
                    source.materialize()?.sum_keepdim(*dim)
                }
            }
            HipNativeExpr::Neg { source } => source.materialize()?.neg(),
            HipNativeExpr::AddScalar { source, value } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.add_scalar(*value)?.into_tensor())
                } else {
                    Ok((source.materialize()? + *value)?)
                }
            }
            HipNativeExpr::MulScalar { source, value } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.mul_scalar(*value)?.into_tensor())
                } else {
                    Ok((source.materialize()? * *value)?)
                }
            }
            HipNativeExpr::Recip { source } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.recip()?.into_tensor())
                } else {
                    source.materialize()?.recip()
                }
            }
            HipNativeExpr::Sqrt { source } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.sqrt()?.into_tensor())
                } else {
                    source.materialize()?.sqrt()
                }
            }
            HipNativeExpr::L2Norm { source, eps } => {
                if let HipNativeExpr::DeviceBuffer(buffer) = &source.expr {
                    Ok(buffer.l2norm(*eps)?.into_tensor())
                } else {
                    let source = source.materialize()?;
                    let norm = source.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                    source.broadcast_div(&(norm + *eps)?.sqrt()?)
                }
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
        if let Some(buffer) = self.0.try_materialize_device_buffer()? {
            return Ok(Self::from_device_buffer(buffer.contiguous()?));
        }
        Ok(Self::from_tensor(self.materialize()?.contiguous()?))
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            if let Some(buffer) = self.0.direct_materialized_device_buffer() {
                return Ok(Self::from_device_buffer(buffer.to_dtype(dtype)?));
            }
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

    pub(crate) fn select(&self, dim: impl candle_core::shape::Dim, index: usize) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-select")?;
        if let Some(buffer) = self.0.direct_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.select(dim_index, index)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::select(
            Arc::new(self.0.clone()),
            dim_index,
            index,
        )))
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (
            self.0.direct_materialized_device_buffer(),
            rhs.0.direct_materialized_device_buffer(),
        ) {
            return Ok(Self::from_device_buffer(lhs.matmul(rhs)?));
        }
        if let Some(native) =
            HipNativeBuffer::host_bytes_matmul(&Arc::new(self.0.clone()), &Arc::new(rhs.0.clone()))?
        {
            return Ok(Self::from_native_buffer(native));
        }
        if let (Some(lhs), Some(rhs)) = (
            self.0.try_materialize_device_buffer()?,
            rhs.0.try_materialize_device_buffer()?,
        ) {
            return Ok(Self::from_device_buffer(lhs.matmul(&rhs)?));
        }
        let lhs = self.materialize()?;
        let rhs = rhs.materialize()?;
        Ok(Self::from_tensor(lhs.matmul(&rhs)?))
    }

    pub(crate) fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (
            self.0.direct_materialized_device_buffer(),
            rhs.0.direct_materialized_device_buffer(),
        ) {
            return Ok(Self::from_device_buffer(lhs.broadcast_add(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_add(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (
            self.0.direct_materialized_device_buffer(),
            rhs.0.direct_materialized_device_buffer(),
        ) {
            return Ok(Self::from_device_buffer(lhs.broadcast_mul(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_mul(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn exp(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.exp()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::exp(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn log(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.log()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::log(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn max_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-max-keepdim")?;
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.max_keepdim(dim_index)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::max_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (
            self.0.direct_materialized_device_buffer(),
            rhs.0.direct_materialized_device_buffer(),
        ) {
            return Ok(Self::from_device_buffer(lhs.broadcast_sub(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_sub(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn sum_keepdim(&self, dim: candle_core::D) -> Result<Self> {
        let dim_index = dim.to_index(&Shape::from(self.shape()), "hip-native-sum-keepdim")?;
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.sum_keepdim(dim_index)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::sum_keepdim(
            Arc::new(self.0.clone()),
            dim_index,
        )))
    }

    pub(crate) fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (
            self.0.direct_materialized_device_buffer(),
            rhs.0.direct_materialized_device_buffer(),
        ) {
            return Ok(Self::from_device_buffer(lhs.broadcast_div(rhs)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::broadcast_div(
            Arc::new(self.0.clone()),
            Arc::new(rhs.0.clone()),
        )?))
    }

    pub(crate) fn recip(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.recip()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::recip(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn sqrt(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.sqrt()?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::sqrt(Arc::new(
            self.0.clone(),
        ))))
    }

    pub(crate) fn l2norm(&self, eps: f64) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.l2norm(eps)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::l2norm(
            Arc::new(self.0.clone()),
            eps,
        )))
    }

    pub(crate) fn sigmoid(&self) -> Result<Self> {
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
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
        if let Some(buffer) = self.0.direct_materialized_device_buffer() {
            return Ok(Self::from_device_buffer(buffer.mul_scalar(value)?));
        }
        Ok(Self::from_native_buffer(HipNativeBuffer::mul_scalar(
            Arc::new(self.0.clone()),
            value,
        )))
    }

    pub(crate) fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        if let Some(buffer) = self.0.direct_device_buffer() {
            if buffer.has_pending_views() {
                return Ok(Self::from_native_buffer(HipNativeBuffer::pad_with_zeros(
                    Arc::new(self.0.clone()),
                    dim,
                    left,
                    right,
                )));
            }
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

    pub(crate) fn rank(&self) -> usize {
        self.shape().len()
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
                return buffer
                    .upload_to_device_buffer()
                    .expect("upload host buffer to device buffer")
                    .into_tensor();
            }
        }
        self.0.into_tensor()
    }

    pub(crate) fn try_host_buffer(&self) -> Result<Option<HipHostBuffer>> {
        self.0.try_host_buffer()
    }

    pub(crate) fn try_materialized_device_buffer(&self) -> Result<Option<HipDeviceBuffer>> {
        if let Some(buffer) = self.0 .0.direct_materialized_device_buffer() {
            return Ok(Some(buffer.clone()));
        }
        self.0 .0.try_materialize_device_buffer()
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

    pub(crate) fn select(&self, dim: impl candle_core::shape::Dim, index: usize) -> Result<Self> {
        Ok(Self(self.0.select(dim, index)?))
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

    pub(crate) fn log(&self) -> Result<Self> {
        Ok(Self(self.0.log()?))
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
        Ok(Self(self.0.recip()?))
    }

    pub(crate) fn sqrt(&self) -> Result<Self> {
        Ok(Self(self.0.sqrt()?))
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

    pub(crate) fn rank(&self) -> usize {
        self.0.rank()
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
            if buffers.iter().any(|b| b.has_pending_views()) {
                let sources = tensors
                    .iter()
                    .map(|t| Ok(Arc::new(t.0 .0.clone())))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Self(HipStorage::from_native_buffer(HipNativeBuffer::concat(
                    sources, dim,
                ))));
            }
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
                return buffer.upload_to_state_buffer();
            }
        }
        StateBuffer::from_tensor(self.0.into_tensor())
    }
}

fn from_kernel_tensor(tensor: Tensor) -> HipTensor {
    if let Some(storage) = import_hip_tensor_storage(&tensor).ok().flatten() {
        return HipTensor::from_device_buffer(HipDeviceBuffer {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: tensor.device().clone(),
            storage,
            view_ops: Vec::new(),
        });
    }
    HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(tensor))
}

fn import_contiguous_hip_tensor_as_host_storage(tensor: &Tensor) -> Result<Option<HipDeviceStorage>> {
    if !tensor.device().is_hip() {
        return Ok(None);
    }
    let (storage, layout) = tensor.storage_and_layout();
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let bytes = match &*storage {
        candle_core::Storage::Hip(storage) => {
            HipNativeBuffer::cpu_storage_to_bytes(storage.cpu_storage(), tensor.dtype())
        }
        _ => HipNativeBuffer::tensor_to_host_float_bytes(tensor, tensor.dtype())?,
    };
    let Some(bytes) = bytes else { return Ok(None); };
    Ok(Some(HipDeviceStorage::from_host_buffer(HipHostBuffer {
        bytes,
        shape: tensor.dims().to_vec(),
        dtype: tensor.dtype(),
        device: tensor.device().clone(),
    })))
}

pub(crate) fn state_buffer_from_host_bytes(
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
    device: &Device,
) -> Result<StateBuffer> {
    HipTensor::from_device_buffer(host_result_device_buffer(HipHostBuffer {
        bytes: bytes.into(),
        shape,
        dtype,
        device: device.clone(),
    }))
    .into_state_buffer()
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
    if let Some(mixed_device) = mixed_qkv.0 .0.direct_materialized_device_buffer() {
        let prev_device = prev_state
            .as_ref()
            .and_then(|state| state.0 .0.direct_materialized_device_buffer());
        if prev_device.is_some() || prev_state.is_none() {
            let (prepared, next_state) = HipDeviceBuffer::prepare_depthwise_conv_input(
                prev_device,
                mixed_device,
                kernel_size,
            )?;
            return Ok((
                HipTensor::from_device_buffer(prepared),
                next_state.map(HipTensor::from_device_buffer),
            ));
        }
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
    if let Some(mixed_device) = mixed_qkv.0 .0.direct_materialized_device_buffer() {
        let prev_device = prev_state
            .as_ref()
            .and_then(|state| state.0 .0.direct_materialized_device_buffer());
        if prev_device.is_some() || prev_state.is_none() {
            return HipDeviceBuffer::update_depthwise_conv_state(
                prev_device,
                mixed_device,
                kernel_size,
            )
            .map(|state| state.map(HipTensor::from_device_buffer));
        }
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
    if let (Some(lhs_host), Some(rhs_host)) = (lhs.try_host_buffer()?, rhs.try_host_buffer()?) {
        return Ok(HipTensor::from_device_buffer(host_result_device_buffer(
            HipHostBuffer::cat(&[&lhs_host, &rhs_host], lhs.rank() - 1)?,
        )));
    }
    if let (Some(lhs), Some(rhs)) = (
        lhs.0 .0.direct_materialized_device_buffer(),
        rhs.0 .0.direct_materialized_device_buffer(),
    ) {
        return Ok(HipTensor::from_device_buffer(HipDeviceBuffer::concat_last_dim(
            lhs, rhs,
        )?));
    }
    HipTensor::cat(&[&lhs, &rhs], lhs.rank() - 1)?
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
        weighted_key_scan.try_host_buffer()?,
        k_cumdecay_scan.try_host_buffer()?,
        state_decay_feature.try_host_buffer()?,
    ) {
        return Ok(HipTensor::from_device_buffer(host_result_device_buffer(
            HipHostBuffer::cat(&[&weighted_key_scan, &k_cumdecay_scan, &state_decay_feature], 3)?,
        )));
    }
    if let (Some(weighted_key_scan), Some(k_cumdecay_scan), Some(state_decay_feature)) = (
        weighted_key_scan.0 .0.direct_materialized_device_buffer(),
        k_cumdecay_scan.0 .0.direct_materialized_device_buffer(),
        state_decay_feature.0 .0.direct_materialized_device_buffer(),
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
        weighted_key.try_host_buffer()?,
        k_cumdecay.try_host_buffer()?,
        q_state.try_host_buffer()?,
        state_decay.try_host_buffer()?,
    ) {
        return Ok(HipTensor::from_device_buffer(host_result_device_buffer(
            HipHostBuffer::cat(&[&weighted_key, &k_cumdecay, &q_state, &state_decay], 2)?,
        )));
    }
    if let (Some(weighted_key), Some(k_cumdecay), Some(q_state), Some(state_decay)) = (
        weighted_key.0 .0.direct_materialized_device_buffer(),
        k_cumdecay.0 .0.direct_materialized_device_buffer(),
        q_state.0 .0.direct_materialized_device_buffer(),
        state_decay.0 .0.direct_materialized_device_buffer(),
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
    if let Some(fused) = fused.0 .0.direct_materialized_device_buffer() {
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
    if let Some(fused) = fused.0 .0.direct_materialized_device_buffer() {
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
    if let Some(fused) = fused.0 .0.direct_materialized_device_buffer() {
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
    let state_scan = HipTensor::from_state_buffer(state_scan);
    if let Some(host) = state_scan.try_host_buffer()? {
        return HipTensor::from_device_buffer(host_result_device_buffer(
            host.select_copy(1, chunk_idx)?,
        ))
        .into_state_buffer();
    }
    state_scan.select(1, chunk_idx)?.into_state_buffer()
}

pub(crate) fn state_scan_next_chunk(
    state_scan: &StateBuffer,
    next_chunk_idx: usize,
) -> Result<StateBuffer> {
    let state_scan = HipTensor::from_state_buffer(state_scan);
    if let Some(host) = state_scan.try_host_buffer()? {
        return HipTensor::from_device_buffer(host_result_device_buffer(
            host.select_copy(1, next_chunk_idx)?,
        ))
        .into_state_buffer();
    }
    state_scan
        .select(1, next_chunk_idx)?
        .contiguous()?
        .into_state_buffer()
}

fn unpack_chunk_fused_hip(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(HipTensor, HipTensor, HipTensor)> {
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_materialized_device_buffer() {
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
    if let Some(xs) = xs.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(mapped) = &xs.storage {
            if let Some(out) = mapped_rms_norm_hip_host_buffer(mapped, weight, eps, add_unit_offset)? {
                return Ok(out);
            }
        }
        if xs.storage.as_host_buffer().is_some() {
            return Ok(HipTensor::from_device_buffer(
                xs.rms_norm(weight, eps, add_unit_offset)?,
            ));
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
    if let Some(xs) = xs.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(mapped) = &xs.storage {
            if let Some(out) = mapped_l2norm_hip_host_buffer(mapped, eps)? {
                return Ok(out);
            }
        }
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

fn causal_mask_hip_host_buffer(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        hip_causal_mask_host_buffer(device, dtype, batch_size, tgt_len, seqlen_offset)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype,
            device: device.clone(),
        },
    ))))
}

fn cumsum_last_dim_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_cumsum_last_dim_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn pad_with_zeros_hip_owned_device(
    xs: &HipDeviceBuffer,
    dim: usize,
    left: usize,
    right: usize,
) -> Result<Option<HipDeviceBuffer>> {
    let Some((ordinal, dtype, src_shape, src_ptr)) = xs.standard_contiguous_launch_spec()? else {
        return Ok(None);
    };
    if dim >= src_shape.len() {
        candle_core::bail!("invalid pad dim {} for shape {:?}", dim, src_shape);
    }
    let mut out_shape = src_shape.clone();
    out_shape[dim] = out_shape[dim].saturating_add(left).saturating_add(right);
    let out = HipDeviceBuffer::from_raw_hip_device_output(out_shape.clone(), dtype, xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    hip::memset_device_bytes(
        ordinal,
        buffer.raw_device_ptr() as *mut c_void,
        0,
        buffer.len_bytes,
    )?;
    let elem_bytes = dtype.size_in_bytes();
    let inner = HipNativeBuffer::elem_count(&src_shape[dim + 1..]);
    let outer = HipNativeBuffer::elem_count(&src_shape[..dim]);
    let src_chunk_bytes = src_shape[dim]
        .saturating_mul(inner)
        .saturating_mul(elem_bytes);
    let dst_chunk_bytes = out_shape[dim]
        .saturating_mul(inner)
        .saturating_mul(elem_bytes);
    let left_bytes = left.saturating_mul(inner).saturating_mul(elem_bytes);
    for outer_idx in 0..outer {
        let src_off = outer_idx.saturating_mul(src_chunk_bytes);
        let dst_off = outer_idx
            .saturating_mul(dst_chunk_bytes)
            .saturating_add(left_bytes);
        hip::copy_device_to_device(
            ordinal,
            (buffer.raw_device_ptr() as usize + dst_off) as *mut c_void,
            (src_ptr as usize + src_off) as *const c_void,
            src_chunk_bytes,
        )?;
    }
    Ok(Some(out))
}

fn cat_hip_owned_device(buffers: &[&HipDeviceBuffer], dim: usize) -> Result<Option<HipDeviceBuffer>> {
    let mut specs = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let Some(spec) = buffer.standard_contiguous_launch_spec()? else {
            return Ok(None);
        };
        specs.push(spec);
    }
    let (ordinal, dtype, first_shape, _) = &specs[0];
    if dim >= first_shape.len() {
        candle_core::bail!("invalid concat dim {} for shape {:?}", dim, first_shape);
    }
    let mut out_shape = first_shape.clone();
    out_shape[dim] = 0;
    for (src_ordinal, src_dtype, shape, _) in &specs {
        if src_ordinal != ordinal || *src_dtype != *dtype || shape.len() != first_shape.len() {
            return Ok(None);
        }
        for axis in 0..shape.len() {
            if axis == dim {
                continue;
            }
            if shape[axis] != first_shape[axis] {
                candle_core::bail!(
                    "incompatible cat shapes {:?} and {:?} on dim {}",
                    first_shape,
                    shape,
                    dim
                );
            }
        }
        out_shape[dim] = out_shape[dim].saturating_add(shape[dim]);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        out_shape.clone(),
        *dtype,
        buffers[0].device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let elem_bytes = dtype.size_in_bytes();
    let inner = HipNativeBuffer::elem_count(&out_shape[dim + 1..]);
    let outer = HipNativeBuffer::elem_count(&out_shape[..dim]);
    let out_row_bytes = out_shape[dim].saturating_mul(inner).saturating_mul(elem_bytes);
    for outer_idx in 0..outer {
        let mut dst_off = outer_idx.saturating_mul(out_row_bytes);
        for (_, _, shape, src_ptr) in &specs {
            let chunk_bytes = shape[dim].saturating_mul(inner).saturating_mul(elem_bytes);
            let src_off = outer_idx.saturating_mul(chunk_bytes);
            hip::copy_device_to_device(
                *ordinal,
                (buffer.raw_device_ptr() as usize + dst_off) as *mut c_void,
                (*src_ptr as usize + src_off) as *const c_void,
                chunk_bytes,
            )?;
            dst_off = dst_off.saturating_add(chunk_bytes);
        }
    }
    Ok(Some(out))
}

fn exp_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_exp_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn exp_hip_owned_device(xs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_exp(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-exp-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn recip_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_recip_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn recip_hip_owned_device(xs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_recip(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-recip-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn log_hip_owned_device(xs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_log(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-log-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn sigmoid_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_sigmoid_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn sigmoid_hip_owned_device(xs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sigmoid(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-sigmoid-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn cast_hip_host_buffer(xs: &Tensor, dtype: DType) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_cast_host_buffer(xs, dtype)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype,
            device: xs.device().clone(),
        },
    ))))
}

fn cast_hip_owned_device(xs: &Tensor, dtype: DType) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cast(
            hip::dtype_code(xs.dtype())?,
            hip::dtype_code(dtype)?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-cast-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn mul_scalar_hip_owned_device(xs: &Tensor, value: f64) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_mul_scalar(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            value as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-mul-scalar-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn add_scalar_hip_owned_device(xs: &Tensor, value: f64) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_add_scalar(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            value as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-add-scalar-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn sqrt_hip_owned_device(xs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sqrt(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            layout.shape().elem_count(),
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-sqrt-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn binary_broadcast_hip_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    helper: fn(&Tensor, &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>>,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = helper(lhs, rhs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: lhs.dtype(),
            device: lhs.device().clone(),
        },
    ))))
}

fn binary_broadcast_hip_owned_device(
    lhs: &Tensor,
    rhs: &Tensor,
    op: i32,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    if !lhs.device().is_hip() || !rhs.device().is_hip() {
        return Ok(None);
    }
    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let (Storage::Hip(lhs_storage), Storage::Hip(rhs_storage)) = (&*lhs_storage, &*rhs_storage) else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }
    let lhs_shape = lhs_layout.shape().dims().to_vec();
    let rhs_shape = rhs_layout.shape().dims().to_vec();
    let rank = lhs_shape.len().max(rhs_shape.len());
    let lhs_pad = rank.saturating_sub(lhs_shape.len());
    let rhs_pad = rank.saturating_sub(rhs_shape.len());
    let lhs_strides = HipDeviceBuffer::standard_contiguous_strides(&lhs_shape);
    let rhs_strides = HipDeviceBuffer::standard_contiguous_strides(&rhs_shape);
    let mut out_shape = vec![0usize; rank];
    let mut lhs_broadcast_strides = vec![0i32; rank];
    let mut rhs_broadcast_strides = vec![0i32; rank];
    let mut total_elems = 1usize;
    for dim in 0..rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_shape[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_shape[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_shape[dim] = out_dim;
        total_elems = total_elems.saturating_mul(out_dim);
        lhs_broadcast_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
            0
        } else {
            i32::try_from(lhs_strides[dim - lhs_pad])
                .map_err(|_| candle_core::Error::Msg("lhs stride overflow".into()))?
        };
        rhs_broadcast_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
            0
        } else {
            i32::try_from(rhs_strides[dim - rhs_pad])
                .map_err(|_| candle_core::Error::Msg("rhs stride overflow".into()))?
        };
    }
    let out_dims = out_shape
        .iter()
        .copied()
        .map(|dim| i32::try_from(dim).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
        .collect::<Result<Vec<_>>>()?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(out_shape, lhs.dtype(), lhs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = lhs.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_binary_broadcast(
            op,
            hip::dtype_code(lhs.dtype())?,
            ordinal,
            i32::try_from(rank).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
            total_elems,
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            lhs_broadcast_strides.as_ptr(),
            rhs_broadcast_strides.as_ptr(),
            out_dims.as_ptr(),
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-binary-broadcast-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn reduce_keepdim_hip_owned_device(
    xs: &Tensor,
    dim: usize,
    sum: bool,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    if !xs.device().is_hip() {
        return Ok(None);
    }
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() || dim >= layout.shape().rank() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let mut out_shape = shape.clone();
    let reduce_len = out_shape[dim];
    out_shape[dim] = 1;
    let total_out_elems = HipNativeBuffer::elem_count(&out_shape);
    let out_dims = out_shape
        .iter()
        .copied()
        .map(|d| i32::try_from(d).map_err(|_| candle_core::Error::Msg("shape overflow".into())))
        .collect::<Result<Vec<_>>>()?;
    let in_strides = HipDeviceBuffer::standard_contiguous_strides(&shape)
        .into_iter()
        .map(|s| i32::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<Vec<_>>>()?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(out_shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = xs.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_reduce_keepdim_view(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            i32::try_from(shape.len()).map_err(|_| candle_core::Error::Msg("rank overflow".into()))?,
            i32::try_from(dim).map_err(|_| candle_core::Error::Msg("dim overflow".into()))?,
            reduce_len,
            total_out_elems,
            if sum { 1 } else { 0 },
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            in_strides.as_ptr(),
            out_dims.as_ptr(),
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-reduce-keepdim-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn matmul_hip_owned_device(lhs: &Tensor, rhs: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    if !lhs.device().is_hip() || !rhs.device().is_hip() {
        return Ok(None);
    }
    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let (Storage::Hip(lhs_storage), Storage::Hip(rhs_storage)) = (&*lhs_storage, &*rhs_storage) else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }
    let lhs_shape = lhs_layout.shape().dims().to_vec();
    let rhs_shape = rhs_layout.shape().dims().to_vec();
    if lhs_shape.is_empty() || rhs_shape.is_empty() {
        return Ok(None);
    }
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    let lhs_k = lhs_shape[lhs_rank - 1];
    let rhs_k = rhs_shape[rhs_rank.saturating_sub(2)];
    if lhs_k != rhs_k {
        return Ok(None);
    }
    let m = if lhs_rank >= 2 { lhs_shape[lhs_rank - 2] } else { 1 };
    let n = rhs_shape[rhs_rank - 1];
    let lhs_batch = &lhs_shape[..lhs_rank.saturating_sub(2)];
    let rhs_batch = &rhs_shape[..rhs_rank.saturating_sub(2)];
    let batch_rank = lhs_batch.len().max(rhs_batch.len());
    if batch_rank > 8 {
        return Ok(None);
    }
    let lhs_strides = HipDeviceBuffer::standard_contiguous_strides(&lhs_shape);
    let rhs_strides = HipDeviceBuffer::standard_contiguous_strides(&rhs_shape);
    let lhs_matrix_rank = lhs_rank.min(2);
    let rhs_matrix_rank = rhs_rank.min(2);
    let lhs_row_stride = if lhs_matrix_rank == 2 {
        i32::try_from(lhs_strides[lhs_rank - 2]).map_err(|_| candle_core::Error::Msg("lhs row stride overflow".into()))?
    } else { 0 };
    let lhs_k_stride = i32::try_from(lhs_strides[lhs_rank - 1]).map_err(|_| candle_core::Error::Msg("lhs k stride overflow".into()))?;
    let rhs_k_stride = if rhs_matrix_rank == 2 {
        i32::try_from(rhs_strides[rhs_rank - 2]).map_err(|_| candle_core::Error::Msg("rhs k stride overflow".into()))?
    } else { 0 };
    let rhs_col_stride = i32::try_from(rhs_strides[rhs_rank - 1]).map_err(|_| candle_core::Error::Msg("rhs col stride overflow".into()))?;
    let lhs_pad = batch_rank.saturating_sub(lhs_batch.len());
    let rhs_pad = batch_rank.saturating_sub(rhs_batch.len());
    let mut out_batch_dims = vec![1i32; batch_rank];
    let mut lhs_batch_strides = vec![0i32; batch_rank];
    let mut rhs_batch_strides = vec![0i32; batch_rank];
    let mut batch_elems = 1usize;
    for dim in 0..batch_rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_batch[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_batch[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_batch_dims[dim] = i32::try_from(out_dim)
            .map_err(|_| candle_core::Error::Msg("matmul batch dim overflow".into()))?;
        lhs_batch_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
            0
        } else {
            i32::try_from(lhs_strides[dim - lhs_pad])
                .map_err(|_| candle_core::Error::Msg("lhs batch stride overflow".into()))?
        };
        rhs_batch_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
            0
        } else {
            i32::try_from(rhs_strides[dim - rhs_pad])
                .map_err(|_| candle_core::Error::Msg("rhs batch stride overflow".into()))?
        };
        batch_elems = batch_elems.saturating_mul(out_dim);
    }
    let mut out_shape = out_batch_dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
    if lhs_rank >= 2 {
        out_shape.push(m);
    }
    out_shape.push(n);
    let out = HipDeviceBuffer::from_raw_hip_device_output(out_shape, lhs.dtype(), lhs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = lhs.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_batched_matmul_view(
            hip::dtype_code(lhs.dtype())?,
            ordinal,
            i32::try_from(batch_rank).map_err(|_| candle_core::Error::Msg("batch rank overflow".into()))?,
            batch_elems,
            i32::try_from(m).map_err(|_| candle_core::Error::Msg("m overflow".into()))?,
            i32::try_from(n).map_err(|_| candle_core::Error::Msg("n overflow".into()))?,
            i32::try_from(lhs_k).map_err(|_| candle_core::Error::Msg("k overflow".into()))?,
            lhs_batch_strides.as_ptr(),
            rhs_batch_strides.as_ptr(),
            out_batch_dims.as_ptr(),
            lhs_row_stride,
            lhs_k_stride,
            rhs_k_stride,
            rhs_col_stride,
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-matmul-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

fn matmul_hip_host_buffer(lhs: &Tensor, rhs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_matmul_host_buffer(lhs, rhs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: lhs.dtype(),
            device: lhs.device().clone(),
        },
    ))))
}

fn mul_scalar_hip_host_buffer(xs: &Tensor, value: f64) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_mul_scalar_host_buffer(xs, value)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn reduce_keepdim_hip_host_buffer(
    xs: &Tensor,
    dim: usize,
    helper: fn(&Tensor, usize) -> Result<Option<(Vec<u8>, Vec<usize>)>>,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = helper(xs, dim)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn add_scalar_hip_host_buffer(xs: &Tensor, value: f64) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_add_scalar_host_buffer(xs, value)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn log_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_log_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn sqrt_hip_host_buffer(xs: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_sqrt_host_buffer(xs)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

fn l2norm_hip_host_buffer(xs: &Tensor, eps: f64) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_l2norm_host_buffer(xs, eps)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn l2norm_hip_owned_device_buffer(xs: &HipDeviceBuffer, eps: f64) -> Result<Option<HipTensor>> {
    let Some((ordinal, dtype, shape, _strides, ptr)) = xs.candle_view_launch_spec()? else {
        return Ok(None);
    };
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_l2norm(
            hip::dtype_code(dtype)?,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            ptr as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error(
            "dotcache-hip-l2norm-owned-device-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn l2norm_hip_owned_device_buffer(xs: &HipDeviceBuffer, eps: f64) -> Result<Option<HipTensor>> {
    let _ = (xs, eps);
    Ok(None)
}

fn l2norm_hip_owned_device(xs: &Tensor, eps: f64) -> Result<Option<HipTensor>> {
    use candle_core::Storage;
    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_l2norm(
            hip::dtype_code(xs.dtype())?,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-l2norm-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_cumsum_last_dim_hip_host_buffer(xs: &HipMappedHostBuffer) -> Result<Option<HipTensor>> {
    let ordinal = match xs.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.buffer.dtype) else {
        return Ok(None);
    };
    let shape = xs.buffer.shape.clone();
    let cols = *shape.last().ok_or_else(|| {
        candle_core::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = HipNativeBuffer::elem_count(&shape) / cols;
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, xs.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
            dtype_code,
            ordinal,
            rows,
            cols,
            xs.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-cumsum-last-dim-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: xs.buffer.dtype,
            device: xs.buffer.device.clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn owned_cumsum_last_dim_hip_device_buffer(xs: &HipDeviceBuffer) -> Result<Option<HipTensor>> {
    let Some((ordinal, dtype, shape, _strides, input_ptr)) = xs.candle_view_launch_spec()? else {
        return Ok(None);
    };
    let cols = *shape.last().ok_or_else(|| {
        candle_core::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = HipNativeBuffer::elem_count(&shape) / cols;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape.clone(), dtype, xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
            hip::dtype_code(dtype)?,
            ordinal,
            rows,
            cols,
            input_ptr,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-cumsum-last-dim-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn owned_cumsum_last_dim_hip_device_buffer(xs: &HipDeviceBuffer) -> Result<Option<HipTensor>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn causal_mask_hip_owned_device(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let ordinal = match device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let kv_len = tgt_len + seqlen_offset;
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, 1, tgt_len, kv_len],
        dtype,
        device,
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_causal_mask(
            hip::dtype_code(dtype)?,
            ordinal,
            batch_size,
            tgt_len,
            seqlen_offset,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("hip-causal-mask-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn causal_mask_hip_owned_device(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let _ = (device, dtype, batch_size, tgt_len, seqlen_offset);
    Ok(None)
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_cumsum_last_dim_hip_host_buffer(
    xs: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_l2norm_hip_host_buffer(xs: &HipMappedHostBuffer, eps: f64) -> Result<Option<HipTensor>> {
    let ordinal = match xs.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.buffer.dtype) else {
        return Ok(None);
    };
    let shape = xs.buffer.shape.clone();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, xs.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_l2norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            xs.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-l2norm-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: xs.buffer.dtype,
            device: xs.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_l2norm_hip_host_buffer(
    xs: &HipMappedHostBuffer,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let _ = (xs, eps);
    Ok(None)
}

fn rms_norm_hip_host_buffer(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_rms_norm_host_buffer(xs, weight, eps, add_unit_offset)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: xs.dtype(),
            device: xs.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn rms_norm_hip_owned_device_buffer(
    xs: &HipDeviceBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let Some((ordinal, dtype, shape, _strides, xs_ptr)) = xs.candle_view_launch_spec()? else {
        return Ok(None);
    };
    if !weight.device().same_device(xs.device()) {
        return Ok(None);
    }
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Hip(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    if !weight_layout.is_contiguous() || dtype != weight.dtype() {
        return Ok(None);
    }
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(dtype)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            if add_unit_offset { 1 } else { 0 },
            xs_ptr,
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-owned-device-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn rms_norm_hip_owned_device_buffer(
    xs: &HipDeviceBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    let _ = (xs, weight, eps, add_unit_offset);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn rms_norm_hip_owned_device(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    if !xs.device().is_hip() || !weight.device().same_device(xs.device()) {
        return Ok(None);
    }
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (Storage::Hip(xs_storage), Storage::Hip(weight_storage)) = (&*xs_storage, &*weight_storage) else {
        return Ok(None);
    };
    if !xs_layout.is_contiguous() || !weight_layout.is_contiguous() || xs.dtype() != weight.dtype() {
        return Ok(None);
    }
    let shape = xs_layout.shape().dims().to_vec();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(xs.dtype())?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, xs.dtype(), xs.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = xs.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            if add_unit_offset { 1 } else { 0 },
            xs_storage.raw_device_ptr_with_offset(xs_layout.start_offset())? as *const c_void,
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn rms_norm_hip_owned_device(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    let _ = (xs, weight, eps, add_unit_offset);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_rms_norm_hip_host_buffer(
    xs: &HipMappedHostBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match xs.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !weight.device().same_device(&xs.buffer.device) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(xs.buffer.dtype) else {
        return Ok(None);
    };
    if xs.buffer.dtype != weight.dtype() {
        return Ok(None);
    }
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Hip(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    if !weight_layout.is_contiguous() {
        return Ok(None);
    }
    let shape = xs.buffer.shape.clone();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into()))?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, xs.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            if add_unit_offset { 1 } else { 0 },
            xs.raw_device_ptr(),
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: xs.buffer.dtype,
            device: xs.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_rms_norm_hip_host_buffer(
    xs: &HipMappedHostBuffer,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<HipTensor>> {
    let _ = (xs, weight, eps, add_unit_offset);
    Ok(None)
}

fn rms_norm_gated_hip_host_buffer(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_rms_norm_gated_host_buffer(hidden, gate, weight, eps)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: hidden.dtype(),
            device: hidden.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn rms_norm_gated_hip_owned_device_buffer(
    hidden: &HipDeviceBuffer,
    gate: &HipDeviceBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let Some((ordinal, dtype, shape, _hidden_strides, hidden_ptr)) = hidden.candle_view_launch_spec()? else {
        return Ok(None);
    };
    let Some((_gate_ordinal, gate_dtype, gate_shape, _gate_strides, gate_ptr)) =
        gate.candle_view_launch_spec()?
    else {
        return Ok(None);
    };
    if !(gate.device().same_device(hidden.device()) && weight.device().same_device(hidden.device())) {
        return Ok(None);
    }
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Hip(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    if gate_dtype != dtype || gate_shape != shape || !weight_layout.is_contiguous() || weight.dtype() != dtype {
        return Ok(None);
    }
    let n_cols = *shape.last().ok_or_else(|| {
        candle_core::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
    })?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(dtype)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, hidden.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm_gated(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            hidden_ptr,
            gate_ptr,
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-gated-owned-device-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn rms_norm_gated_hip_owned_device_buffer(
    hidden: &HipDeviceBuffer,
    gate: &HipDeviceBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let _ = (hidden, gate, weight, eps);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn rms_norm_gated_hip_owned_device(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    if !(hidden.device().is_hip()
        && gate.device().same_device(hidden.device())
        && weight.device().same_device(hidden.device()))
    {
        return Ok(None);
    }
    let (hidden_storage, hidden_layout) = hidden.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (Storage::Hip(hidden_storage), Storage::Hip(gate_storage), Storage::Hip(weight_storage)) =
        (&*hidden_storage, &*gate_storage, &*weight_storage)
    else {
        return Ok(None);
    };
    if !hidden_layout.is_contiguous()
        || !gate_layout.is_contiguous()
        || !weight_layout.is_contiguous()
        || hidden.dtype() != gate.dtype()
        || hidden.dtype() != weight.dtype()
        || hidden_layout.shape() != gate_layout.shape()
    {
        return Ok(None);
    }
    let shape = hidden_layout.shape().dims().to_vec();
    let n_cols = *shape.last().ok_or_else(|| {
        candle_core::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
    })?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(hidden.dtype())?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, hidden.dtype(), hidden.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = hidden.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm_gated(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            hidden_storage.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
            gate_storage.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-gated-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn rms_norm_gated_hip_owned_device(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let _ = (hidden, gate, weight, eps);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_rms_norm_gated_hip_host_buffer(
    hidden: &HipMappedHostBuffer,
    gate: &HipMappedHostBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match hidden.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(gate.buffer.device.same_device(&hidden.buffer.device)
        && weight.device().same_device(&hidden.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(hidden.buffer.dtype) else {
        return Ok(None);
    };
    if hidden.buffer.dtype != gate.buffer.dtype || hidden.buffer.dtype != weight.dtype() {
        return Ok(None);
    }
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Hip(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    if !weight_layout.is_contiguous() {
        return Ok(None);
    }
    let shape = hidden.buffer.shape.clone();
    let n_cols = *shape.last().ok_or_else(|| {
        candle_core::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
    })?;
    let n_rows = HipNativeBuffer::elem_count(&shape) / n_cols;
    if gate.buffer.shape != shape || weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, hidden.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_rms_norm_gated(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            hidden.raw_device_ptr(),
            gate.raw_device_ptr(),
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-gated-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: hidden.buffer.dtype,
            device: hidden.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_rms_norm_gated_hip_host_buffer(
    hidden: &HipMappedHostBuffer,
    gate: &HipMappedHostBuffer,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<HipTensor>> {
    let _ = (hidden, gate, weight, eps);
    Ok(None)
}

fn swiglu_mul_hip_host_buffer(gate: &Tensor, up: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_swiglu_mul_host_buffer(gate, up)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: gate.dtype(),
            device: gate.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn swiglu_mul_hip_owned_device_buffer(
    gate: &HipDeviceBuffer,
    up: &HipDeviceBuffer,
) -> Result<Option<HipTensor>> {
    let Some((ordinal, dtype, shape, _gate_strides, gate_ptr)) = gate.candle_view_launch_spec()? else {
        return Ok(None);
    };
    let Some((_up_ordinal, up_dtype, up_shape, _up_strides, up_ptr)) = up.candle_view_launch_spec()? else {
        return Ok(None);
    };
    if !up.device().same_device(gate.device()) || up_dtype != dtype || up_shape != shape {
        return Ok(None);
    }
    let elem_count = HipNativeBuffer::elem_count(&shape);
    let dtype_code = hip::dtype_code(dtype)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, gate.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_swiglu_mul(
            dtype_code,
            ordinal,
            elem_count,
            gate_ptr,
            up_ptr,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-swiglu-mul-owned-device-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn swiglu_mul_hip_owned_device_buffer(
    gate: &HipDeviceBuffer,
    up: &HipDeviceBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (gate, up);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn swiglu_mul_hip_owned_device(gate: &Tensor, up: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    if !gate.device().is_hip() || !up.device().same_device(gate.device()) {
        return Ok(None);
    }
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (up_storage, up_layout) = up.storage_and_layout();
    let (Storage::Hip(gate_storage), Storage::Hip(up_storage)) = (&*gate_storage, &*up_storage) else {
        return Ok(None);
    };
    if !gate_layout.is_contiguous()
        || !up_layout.is_contiguous()
        || gate.dtype() != up.dtype()
        || gate_layout.shape() != up_layout.shape()
    {
        return Ok(None);
    }
    let shape = gate_layout.shape().dims().to_vec();
    let elem_count = HipNativeBuffer::elem_count(&shape);
    let dtype_code = hip::dtype_code(gate.dtype())?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, gate.dtype(), gate.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = gate.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_swiglu_mul(
            dtype_code,
            ordinal,
            elem_count,
            gate_storage.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
            up_storage.raw_device_ptr_with_offset(up_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-swiglu-mul-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn swiglu_mul_hip_owned_device(gate: &Tensor, up: &Tensor) -> Result<Option<HipTensor>> {
    let _ = (gate, up);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_swiglu_mul_hip_host_buffer(
    gate: &HipMappedHostBuffer,
    up: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match gate.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !up.buffer.device.same_device(&gate.buffer.device) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(gate.buffer.dtype) else {
        return Ok(None);
    };
    if gate.buffer.dtype != up.buffer.dtype || gate.buffer.shape != up.buffer.shape {
        return Ok(None);
    }
    let shape = gate.buffer.shape.clone();
    let elem_count = HipNativeBuffer::elem_count(&shape);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, gate.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_swiglu_mul(
            dtype_code,
            ordinal,
            elem_count,
            gate.raw_device_ptr(),
            up.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-swiglu-mul-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: gate.buffer.dtype,
            device: gate.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_swiglu_mul_hip_host_buffer(
    gate: &HipMappedHostBuffer,
    up: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (gate, up);
    Ok(None)
}

fn value_decay_hip_host_buffer(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_value_decay_host_buffer(a, dt_bias, a_log_exp)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: a.dtype(),
            device: a.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn value_decay_hip_owned_device_buffer(
    a: &HipDeviceBuffer,
    dt_bias: &HipDeviceBuffer,
    a_log_exp: &HipDeviceBuffer,
) -> Result<Option<HipTensor>> {
    let Some((ordinal, dtype, shape, _a_strides, a_ptr)) = a.candle_view_launch_spec()? else {
        return Ok(None);
    };
    let Some((_dt_ordinal, dt_dtype, dt_shape, _dt_strides, dt_ptr)) =
        dt_bias.candle_view_launch_spec()?
    else {
        return Ok(None);
    };
    let Some((_exp_ordinal, exp_dtype, exp_shape, _exp_strides, exp_ptr)) =
        a_log_exp.candle_view_launch_spec()?
    else {
        return Ok(None);
    };
    if !(dt_bias.device().same_device(a.device()) && a_log_exp.device().same_device(a.device())) {
        return Ok(None);
    }
    if dt_dtype != dtype || exp_dtype != dtype {
        return Ok(None);
    }
    let total_elems = HipNativeBuffer::elem_count(&shape);
    let num_heads = HipNativeBuffer::elem_count(&dt_shape);
    if HipNativeBuffer::elem_count(&exp_shape) != num_heads {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(dtype)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, dtype, a.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_value_decay(
            dtype_code,
            ordinal,
            total_elems,
            num_heads,
            a_ptr,
            dt_ptr,
            exp_ptr,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-value-decay-owned-device-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn value_decay_hip_owned_device_buffer(
    a: &HipDeviceBuffer,
    dt_bias: &HipDeviceBuffer,
    a_log_exp: &HipDeviceBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (a, dt_bias, a_log_exp);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn value_decay_hip_owned_device(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    if !(a.device().is_hip()
        && dt_bias.device().same_device(a.device())
        && a_log_exp.device().same_device(a.device()))
    {
        return Ok(None);
    }
    let (a_storage, a_layout) = a.storage_and_layout();
    let (dt_bias_storage, dt_bias_layout) = dt_bias.storage_and_layout();
    let (a_log_exp_storage, a_log_exp_layout) = a_log_exp.storage_and_layout();
    let (Storage::Hip(a_storage), Storage::Hip(dt_bias_storage), Storage::Hip(a_log_exp_storage)) =
        (&*a_storage, &*dt_bias_storage, &*a_log_exp_storage)
    else {
        return Ok(None);
    };
    if !a_layout.is_contiguous()
        || !dt_bias_layout.is_contiguous()
        || !a_log_exp_layout.is_contiguous()
        || a.dtype() != dt_bias.dtype()
        || a.dtype() != a_log_exp.dtype()
    {
        return Ok(None);
    }
    let shape = a_layout.shape().dims().to_vec();
    let total_elems = HipNativeBuffer::elem_count(&shape);
    let num_heads = dt_bias_layout.shape().elem_count();
    if a_log_exp_layout.shape().elem_count() != num_heads {
        return Ok(None);
    }
    let dtype_code = hip::dtype_code(a.dtype())?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, a.dtype(), a.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let ordinal = a.device().as_hip_device()?.ordinal();
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_value_decay(
            dtype_code,
            ordinal,
            total_elems,
            num_heads,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-value-decay-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn value_decay_hip_owned_device(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (a, dt_bias, a_log_exp);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_value_decay_hip_host_buffer(
    a: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match a.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(dt_bias.buffer.device.same_device(&a.buffer.device)
        && a_log_exp.buffer.device.same_device(&a.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(a.buffer.dtype) else {
        return Ok(None);
    };
    if a.buffer.dtype != dt_bias.buffer.dtype || a.buffer.dtype != a_log_exp.buffer.dtype {
        return Ok(None);
    }
    let total_elems = HipNativeBuffer::elem_count(&a.buffer.shape);
    let num_heads = HipNativeBuffer::elem_count(&dt_bias.buffer.shape);
    if HipNativeBuffer::elem_count(&a_log_exp.buffer.shape) != num_heads {
        return Ok(None);
    }
    let shape = a.buffer.shape.clone();
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, a.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_value_decay(
            dtype_code,
            ordinal,
            total_elems,
            num_heads,
            a.raw_device_ptr(),
            dt_bias.raw_device_ptr(),
            a_log_exp.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-value-decay-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: a.buffer.dtype,
            device: a.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_value_decay_hip_host_buffer(
    a: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (a, dt_bias, a_log_exp);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_linear_prefill_conv_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match mixed_qkv.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !weights.device().same_device(&mixed_qkv.buffer.device) {
        return Ok(None);
    }
    let weights = weights.contiguous()?;
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let Storage::Hip(weights_storage) = &*weights_storage else {
        return Ok(None);
    };
    if !weights_layout.is_contiguous() {
        return Ok(None);
    }
    let Ok(dtype_code) = candle_core::hip::qwen35_dtype_code(mixed_qkv.buffer.dtype) else {
        return Ok(None);
    };
    if weights.dtype() != mixed_qkv.buffer.dtype {
        return Ok(None);
    }
    let [batch_size, conv_dim, total_len] = *mixed_qkv.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    if weights_conv_dim != conv_dim || weights_kernel_size != kernel_size {
        return Ok(None);
    }
    if total_len < seq_len + kernel_size.saturating_sub(1) {
        return Ok(None);
    }
    let shape = vec![batch_size, seq_len, conv_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, mixed_qkv.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_prefill_conv_pack(
            dtype_code,
            ordinal,
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
            mixed_qkv.raw_device_ptr(),
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "dotcache-hip-linear-prefill-conv-pack-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: mixed_qkv.buffer.dtype,
            device: mixed_qkv.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_linear_prefill_conv_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, weights, seq_len, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn linear_prefill_conv_hip_owned_device(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weights = weights.contiguous()?;
    if !mixed_qkv.device().is_hip() || !weights.device().same_device(mixed_qkv.device()) {
        return Ok(None);
    }
    let (mixed_qkv_storage, mixed_qkv_layout) = mixed_qkv.storage_and_layout();
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let (Storage::Hip(mixed_qkv_storage), Storage::Hip(weights_storage)) =
        (&*mixed_qkv_storage, &*weights_storage)
    else {
        return Ok(None);
    };
    if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
        return Ok(None);
    }
    let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    if weights_conv_dim != conv_dim
        || weights_kernel_size != kernel_size
        || total_len < seq_len + kernel_size.saturating_sub(1)
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, seq_len, conv_dim],
        mixed_qkv.dtype(),
        mixed_qkv.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_prefill_conv_pack(
            hip::dtype_code(mixed_qkv.dtype())?,
            mixed_qkv.device().as_hip_device()?.ordinal(),
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-linear-prefill-conv-pack-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn linear_prefill_conv_hip_owned_device(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, weights, seq_len, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_linear_stateful_conv_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_state: &HipMappedHostBuffer,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match mixed_qkv.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_state.buffer.device.same_device(&mixed_qkv.buffer.device)
        && weights.device().same_device(&mixed_qkv.buffer.device))
    {
        return Ok(None);
    }
    let weights = weights.contiguous()?;
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let Storage::Hip(weights_storage) = &*weights_storage else {
        return Ok(None);
    };
    if !weights_layout.is_contiguous() {
        return Ok(None);
    }
    let Ok(dtype_code) = candle_core::hip::qwen35_dtype_code(mixed_qkv.buffer.dtype) else {
        return Ok(None);
    };
    if mixed_qkv.buffer.dtype != prev_state.buffer.dtype || mixed_qkv.buffer.dtype != weights.dtype() {
        return Ok(None);
    }
    let [batch_size, conv_dim, seq_len] = *mixed_qkv.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [state_batch, state_conv_dim, state_len] = *prev_state.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    if state_batch != batch_size
        || state_conv_dim != conv_dim
        || weights_conv_dim != conv_dim
        || weights_kernel_size != kernel_size
    {
        return Ok(None);
    }
    let shape = vec![batch_size, seq_len, conv_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, mixed_qkv.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_stateful_conv(
            dtype_code,
            ordinal,
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            mixed_qkv.raw_device_ptr(),
            prev_state.raw_device_ptr(),
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "dotcache-hip-linear-stateful-conv-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: mixed_qkv.buffer.dtype,
            device: mixed_qkv.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_linear_stateful_conv_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_state: &HipMappedHostBuffer,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, prev_state, weights, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn linear_stateful_conv_hip_owned_device(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    if !(mixed_qkv.device().is_hip()
        && prev_state.device().same_device(mixed_qkv.device())
        && weights.device().same_device(mixed_qkv.device()))
    {
        return Ok(None);
    }
    let (mixed_qkv_storage, mixed_qkv_layout) = mixed_qkv.storage_and_layout();
    let (prev_state_storage, prev_state_layout) = prev_state.storage_and_layout();
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let (
        Storage::Hip(mixed_qkv_storage),
        Storage::Hip(prev_state_storage),
        Storage::Hip(weights_storage),
    ) = (&*mixed_qkv_storage, &*prev_state_storage, &*weights_storage)
    else {
        return Ok(None);
    };
    if !(mixed_qkv_layout.is_contiguous()
        && prev_state_layout.is_contiguous()
        && weights_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_size, conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
    let (state_batch, state_conv_dim, state_len) = prev_state_layout.shape().dims3()?;
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    if state_batch != batch_size
        || state_conv_dim != conv_dim
        || weights_conv_dim != conv_dim
        || weights_kernel_size != kernel_size
        || mixed_qkv.dtype() != prev_state.dtype()
        || mixed_qkv.dtype() != weights.dtype()
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, seq_len, conv_dim],
        mixed_qkv.dtype(),
        mixed_qkv.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_stateful_conv(
            hip::dtype_code(mixed_qkv.dtype())?,
            mixed_qkv.device().as_hip_device()?.ordinal(),
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            prev_state_storage.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("linear-stateful-conv-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn linear_stateful_conv_hip_owned_device(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, prev_state, weights, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn mapped_linear_stateful_conv_value_decay_with_state_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_state: &HipMappedHostBuffer,
    weights: &Tensor,
    a: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match mixed_qkv.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_state.buffer.device.same_device(&mixed_qkv.buffer.device)
        && a.buffer.device.same_device(&mixed_qkv.buffer.device)
        && dt_bias.buffer.device.same_device(&mixed_qkv.buffer.device)
        && a_log_exp.buffer.device.same_device(&mixed_qkv.buffer.device)
        && weights.device().same_device(&mixed_qkv.buffer.device))
    {
        return Ok(None);
    }
    let weights = weights.contiguous()?;
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let Storage::Hip(weights_storage) = &*weights_storage else {
        return Ok(None);
    };
    if !weights_layout.is_contiguous() {
        return Ok(None);
    }
    let Ok(dtype_code) = candle_core::hip::qwen35_dtype_code(mixed_qkv.buffer.dtype) else {
        return Ok(None);
    };
    if mixed_qkv.buffer.dtype != prev_state.buffer.dtype
        || mixed_qkv.buffer.dtype != a.buffer.dtype
        || mixed_qkv.buffer.dtype != dt_bias.buffer.dtype
        || mixed_qkv.buffer.dtype != a_log_exp.buffer.dtype
        || mixed_qkv.buffer.dtype != weights.dtype()
    {
        return Ok(None);
    }
    let [batch_size, conv_dim, seq_len] = *mixed_qkv.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [state_batch, state_conv_dim, state_len] = *prev_state.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [a_batch, a_seq_len, num_heads] = *a.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    if state_batch != batch_size
        || state_conv_dim != conv_dim
        || weights_conv_dim != conv_dim
        || weights_kernel_size != kernel_size
        || a_batch != batch_size
        || a_seq_len != seq_len
        || HipNativeBuffer::elem_count(&dt_bias.buffer.shape) != num_heads
        || HipNativeBuffer::elem_count(&a_log_exp.buffer.shape) != num_heads
    {
        return Ok(None);
    }
    let flat_width = seq_len * (conv_dim + num_heads) + conv_dim * state_len;
    let shape = vec![batch_size, flat_width];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, mixed_qkv.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_stateful_conv_value_decay_with_state(
            dtype_code,
            ordinal,
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv.raw_device_ptr(),
            prev_state.raw_device_ptr(),
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            a.raw_device_ptr(),
            dt_bias.raw_device_ptr(),
            a_log_exp.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "dotcache-hip-linear-stateful-conv-value-decay-with-state-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: mixed_qkv.buffer.dtype,
            device: mixed_qkv.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn mapped_linear_stateful_conv_value_decay_with_state_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_state: &HipMappedHostBuffer,
    weights: &Tensor,
    a: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn linear_stateful_conv_value_decay_with_state_hip_owned_device(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a = a.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    if !(mixed_qkv.device().is_hip()
        && prev_state.device().same_device(mixed_qkv.device())
        && weights.device().same_device(mixed_qkv.device())
        && a.device().same_device(mixed_qkv.device())
        && dt_bias.device().same_device(mixed_qkv.device())
        && a_log_exp.device().same_device(mixed_qkv.device()))
    {
        return Ok(None);
    }
    let (mixed_qkv_storage, mixed_qkv_layout) = mixed_qkv.storage_and_layout();
    let (prev_state_storage, prev_state_layout) = prev_state.storage_and_layout();
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let (a_storage, a_layout) = a.storage_and_layout();
    let (dt_bias_storage, dt_bias_layout) = dt_bias.storage_and_layout();
    let (a_log_exp_storage, a_log_exp_layout) = a_log_exp.storage_and_layout();
    let (
        Storage::Hip(mixed_qkv_storage),
        Storage::Hip(prev_state_storage),
        Storage::Hip(weights_storage),
        Storage::Hip(a_storage),
        Storage::Hip(dt_bias_storage),
        Storage::Hip(a_log_exp_storage),
    ) = (
        &*mixed_qkv_storage,
        &*prev_state_storage,
        &*weights_storage,
        &*a_storage,
        &*dt_bias_storage,
        &*a_log_exp_storage,
    ) else {
        return Ok(None);
    };
    if !(mixed_qkv_layout.is_contiguous()
        && prev_state_layout.is_contiguous()
        && weights_layout.is_contiguous()
        && a_layout.is_contiguous()
        && dt_bias_layout.is_contiguous()
        && a_log_exp_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_size, conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
    let (state_batch, state_conv_dim, state_len) = prev_state_layout.shape().dims3()?;
    let (weights_conv_dim, weights_kernel_size) = weights_layout.shape().dims2()?;
    let (a_batch, a_seq_len, num_heads) = a_layout.shape().dims3()?;
    if state_batch != batch_size
        || state_conv_dim != conv_dim
        || weights_conv_dim != conv_dim
        || weights_kernel_size != kernel_size
        || a_batch != batch_size
        || a_seq_len != seq_len
        || dt_bias_layout.shape().elem_count() != num_heads
        || a_log_exp_layout.shape().elem_count() != num_heads
        || mixed_qkv.dtype() != prev_state.dtype()
        || mixed_qkv.dtype() != weights.dtype()
        || mixed_qkv.dtype() != a.dtype()
        || mixed_qkv.dtype() != dt_bias.dtype()
        || mixed_qkv.dtype() != a_log_exp.dtype()
    {
        return Ok(None);
    }
    let flat_width = seq_len * (conv_dim + num_heads) + conv_dim * state_len;
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, flat_width],
        mixed_qkv.dtype(),
        mixed_qkv.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_stateful_conv_value_decay_with_state(
            hip::dtype_code(mixed_qkv.dtype())?,
            mixed_qkv.device().as_hip_device()?.ordinal(),
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            prev_state_storage.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())?
                as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error(
            "linear-stateful-conv-value-decay-with-state-owned-device",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn linear_stateful_conv_value_decay_with_state_hip_owned_device(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let _ = (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, kernel_size);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn mapped_linear_decode_step_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_conv_state: &HipMappedHostBuffer,
    weights: &Tensor,
    a_beta_raw: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
    initial_state: &HipMappedHostBuffer,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel_size: usize,
    head_repeat: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let ordinal = match mixed_qkv.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_conv_state.buffer.device.same_device(&mixed_qkv.buffer.device)
        && a_beta_raw.buffer.device.same_device(&mixed_qkv.buffer.device)
        && dt_bias.buffer.device.same_device(&mixed_qkv.buffer.device)
        && a_log_exp.buffer.device.same_device(&mixed_qkv.buffer.device)
        && initial_state.buffer.device.same_device(&mixed_qkv.buffer.device)
        && weights.device().same_device(&mixed_qkv.buffer.device))
    {
        return Ok(None);
    }
    let weights = weights.contiguous()?;
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let Storage::Hip(weights_storage) = &*weights_storage else {
        return Ok(None);
    };
    if !weights_layout.is_contiguous() || initial_state.buffer.dtype != DType::F32 {
        return Ok(None);
    }
    let Ok(dtype_code) = candle_core::hip::qwen35_dtype_code(mixed_qkv.buffer.dtype) else {
        return Ok(None);
    };
    if mixed_qkv.buffer.dtype != prev_conv_state.buffer.dtype
        || mixed_qkv.buffer.dtype != a_beta_raw.buffer.dtype
        || mixed_qkv.buffer.dtype != dt_bias.buffer.dtype
        || mixed_qkv.buffer.dtype != a_log_exp.buffer.dtype
        || mixed_qkv.buffer.dtype != weights.dtype()
    {
        return Ok(None);
    }
    let [batch_size, _conv_dim, seq_len] = *mixed_qkv.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [_, _, state_len] = *prev_conv_state.buffer.shape.as_slice() else {
        return Ok(None);
    };
    if seq_len != 1 {
        return Ok(None);
    }
    let packed_width = 2 * head_k_dim + head_v_dim + 2;
    let packed_len = batch_size
        .saturating_mul(num_v_heads)
        .saturating_mul(packed_width);
    let mut packed = vec![0u8; packed_len.saturating_mul(DType::F32.size_in_bytes())];
    let packed_host_ptr = packed.as_mut_ptr() as *const c_void;
    let packed_device_ptr =
        hip::register_host_mapping_for_device(ordinal, packed_host_ptr, packed.len())?;
    let prepare_status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_decode_prepare(
            dtype_code,
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_len,
            kernel_size,
            head_repeat,
            mixed_qkv.raw_device_ptr(),
            prev_conv_state.raw_device_ptr(),
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            a_beta_raw.raw_device_ptr(),
            dt_bias.raw_device_ptr(),
            a_log_exp.raw_device_ptr(),
            packed_device_ptr as *mut c_void,
        )
    };
    if prepare_status != 0 {
        hip::unregister_host_mapping(packed_host_ptr);
        return Err(hip::hip_error(
            "linear-decode-prepare-mapped-host-buffer",
            prepare_status,
        ));
    }
    let output_width = num_v_heads * head_v_dim + num_v_heads * head_k_dim * head_v_dim;
    let output_shape = vec![batch_size, output_width];
    let mut out = vec![
        0u8;
        batch_size
            .saturating_mul(output_width)
            .saturating_mul(DType::F32.size_in_bytes())
    ];
    let out_host_ptr = out.as_mut_ptr() as *const c_void;
    let out_device_ptr = hip::register_host_mapping_for_device(ordinal, out_host_ptr, out.len())?;
    let apply_status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_decode_apply(
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            packed_device_ptr as *const c_void,
            initial_state.raw_device_ptr(),
            out_device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(out_host_ptr);
    hip::unregister_host_mapping(packed_host_ptr);
    if apply_status != 0 {
        return Err(hip::hip_error(
            "linear-decode-apply-mapped-host-buffer",
            apply_status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape: output_shape,
            dtype: DType::F32,
            device: mixed_qkv.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn mapped_linear_decode_step_hip_host_buffer(
    mixed_qkv: &HipMappedHostBuffer,
    prev_conv_state: &HipMappedHostBuffer,
    weights: &Tensor,
    a_beta_raw: &HipMappedHostBuffer,
    dt_bias: &HipMappedHostBuffer,
    a_log_exp: &HipMappedHostBuffer,
    initial_state: &HipMappedHostBuffer,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel_size: usize,
    head_repeat: usize,
) -> Result<Option<HipTensor>> {
    let _ = (
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
    );
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn linear_decode_step_hip_owned_device(
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
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_conv_state = prev_conv_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a_beta_raw = a_beta_raw.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let initial_state = initial_state.contiguous()?;
    if !(mixed_qkv.device().is_hip()
        && prev_conv_state.device().same_device(mixed_qkv.device())
        && weights.device().same_device(mixed_qkv.device())
        && a_beta_raw.device().same_device(mixed_qkv.device())
        && dt_bias.device().same_device(mixed_qkv.device())
        && a_log_exp.device().same_device(mixed_qkv.device())
        && initial_state.device().same_device(mixed_qkv.device()))
    {
        return Ok(None);
    }
    let (mixed_qkv_storage, mixed_qkv_layout) = mixed_qkv.storage_and_layout();
    let (prev_conv_state_storage, prev_conv_state_layout) = prev_conv_state.storage_and_layout();
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let (a_beta_raw_storage, a_beta_raw_layout) = a_beta_raw.storage_and_layout();
    let (dt_bias_storage, dt_bias_layout) = dt_bias.storage_and_layout();
    let (a_log_exp_storage, a_log_exp_layout) = a_log_exp.storage_and_layout();
    let (initial_state_storage, initial_state_layout) = initial_state.storage_and_layout();
    let (
        Storage::Hip(mixed_qkv_storage),
        Storage::Hip(prev_conv_state_storage),
        Storage::Hip(weights_storage),
        Storage::Hip(a_beta_raw_storage),
        Storage::Hip(dt_bias_storage),
        Storage::Hip(a_log_exp_storage),
        Storage::Hip(initial_state_storage),
    ) = (
        &*mixed_qkv_storage,
        &*prev_conv_state_storage,
        &*weights_storage,
        &*a_beta_raw_storage,
        &*dt_bias_storage,
        &*a_log_exp_storage,
        &*initial_state_storage,
    ) else {
        return Ok(None);
    };
    if !(mixed_qkv_layout.is_contiguous()
        && prev_conv_state_layout.is_contiguous()
        && weights_layout.is_contiguous()
        && a_beta_raw_layout.is_contiguous()
        && dt_bias_layout.is_contiguous()
        && a_log_exp_layout.is_contiguous()
        && initial_state_layout.is_contiguous())
    {
        return Ok(None);
    }
    if initial_state.dtype() != DType::F32 {
        return Ok(None);
    }
    let (batch_size, _conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
    let (_, _, state_len) = prev_conv_state_layout.shape().dims3()?;
    if seq_len != 1 {
        return Ok(None);
    }
    let packed = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size * num_v_heads, 2 * head_k_dim + head_v_dim + 2],
        DType::F32,
        mixed_qkv.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(packed_buffer) = &packed.storage else {
        return Ok(None);
    };
    let ordinal = mixed_qkv.device().as_hip_device()?.ordinal();
    let prepare_status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_decode_prepare(
            hip::dtype_code(mixed_qkv.dtype())?,
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_len,
            kernel_size,
            head_repeat,
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            prev_conv_state_storage
                .raw_device_ptr_with_offset(prev_conv_state_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            a_beta_raw_storage.raw_device_ptr_with_offset(a_beta_raw_layout.start_offset())?
                as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())?
                as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            packed_buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if prepare_status != 0 {
        return Err(hip::hip_error("linear-decode-prepare-owned-device", prepare_status));
    }
    let output_width = num_v_heads * head_v_dim + num_v_heads * head_k_dim * head_v_dim;
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, output_width],
        DType::F32,
        mixed_qkv.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(out_buffer) = &out.storage else {
        return Ok(None);
    };
    let apply_status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_decode_apply(
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            packed_buffer.raw_device_ptr() as *const c_void,
            initial_state_storage.raw_device_ptr_with_offset(initial_state_layout.start_offset())?
                as *const c_void,
            out_buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if apply_status != 0 {
        return Err(hip::hip_error("linear-decode-apply-owned-device", apply_status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn linear_decode_step_hip_owned_device(
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
) -> Result<Option<HipTensor>> {
    let _ = (
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
    );
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_full_attention_prefill_hip_host_buffer(
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let ordinal = match query.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key.buffer.device.same_device(&query.buffer.device)
        && value.buffer.device.same_device(&query.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query.buffer.dtype) else {
        return Ok(None);
    };
    if query.buffer.dtype != key.buffer.dtype || query.buffer.dtype != value.buffer.dtype {
        return Ok(None);
    }
    let [batch_size, q_heads, q_len, head_dim] = *query.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [key_batch, kv_heads, kv_len, key_head_dim] = *key.buffer.shape.as_slice() else {
        return Ok(None);
    };
    let [value_batch, value_kv_heads, value_kv_len, value_head_dim] = *value.buffer.shape.as_slice()
    else {
        return Ok(None);
    };
    if key_batch != batch_size
        || value_batch != batch_size
        || value_kv_heads != kv_heads
        || value_kv_len != kv_len
        || key_head_dim != head_dim
        || value_head_dim != head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_size, q_heads, q_len, head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, DType::F32)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_full_attention_prefill(
            dtype_code,
            ordinal,
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
            query.raw_device_ptr(),
            key.raw_device_ptr(),
            value.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "full-attention-prefill-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: DType::F32,
            device: query.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_full_attention_prefill_hip_host_buffer(
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let _ = (query, key, value, num_kv_groups, scale, seqlen_offset);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn full_attention_prefill_hip_owned_device(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    if !(query.device().is_hip()
        && key.device().same_device(query.device())
        && value.device().same_device(query.device()))
    {
        return Ok(None);
    }
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (Storage::Hip(query_storage), Storage::Hip(key_storage), Storage::Hip(value_storage)) =
        (&*query_storage, &*key_storage, &*value_storage)
    else {
        return Ok(None);
    };
    if !(query_layout.is_contiguous() && key_layout.is_contiguous() && value_layout.is_contiguous()) {
        return Ok(None);
    }
    let [batch_size, q_heads, q_len, head_dim] = *query_layout.shape().dims() else {
        return Ok(None);
    };
    let [key_batch, kv_heads, kv_len, key_head_dim] = *key_layout.shape().dims() else {
        return Ok(None);
    };
    let [value_batch, value_kv_heads, value_kv_len, value_head_dim] = *value_layout.shape().dims()
    else {
        return Ok(None);
    };
    if query.dtype() != key.dtype()
        || query.dtype() != value.dtype()
        || key_batch != batch_size
        || value_batch != batch_size
        || value_kv_heads != kv_heads
        || value_kv_len != kv_len
        || key_head_dim != head_dim
        || value_head_dim != head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_size, q_heads, q_len, head_dim],
        DType::F32,
        query.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_full_attention_prefill(
            hip::dtype_code(query.dtype())?,
            query.device().as_hip_device()?.ordinal(),
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("full-attention-prefill-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn full_attention_prefill_hip_owned_device(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let _ = (query, key, value, num_kv_groups, scale, seqlen_offset);
    Ok(None)
}

fn embedding_lookup_hip_host_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_embedding_lookup_host_buffer(embeddings, indexes)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: embeddings.dtype(),
            device: embeddings.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn embedding_lookup_hip_owned_device(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let embeddings = embeddings.contiguous()?;
    let indexes = indexes.contiguous()?;
    if !(embeddings.device().is_hip() && indexes.device().same_device(embeddings.device())) {
        return Ok(None);
    }
    let (embeddings_storage, embeddings_layout) = embeddings.storage_and_layout();
    let (indexes_storage, indexes_layout) = indexes.storage_and_layout();
    let (Storage::Hip(embeddings_storage), Storage::Hip(indexes_storage)) =
        (&*embeddings_storage, &*indexes_storage)
    else {
        return Ok(None);
    };
    if !(embeddings_layout.is_contiguous() && indexes_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = embeddings.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(embeddings.dtype())?;
    let index_dtype_code = hip::index_dtype_code(indexes.dtype())?;
    let (vocab_size, hidden_size) = embeddings_layout.shape().dims2()?;
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(hidden_size);
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        shape,
        embeddings.dtype(),
        embeddings.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            vocab_size,
            hidden_size,
            embeddings_storage.raw_device_ptr_with_offset(embeddings_layout.start_offset())?
                as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("embedding-lookup-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn embedding_lookup_hip_owned_device(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (embeddings, indexes);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_embedding_lookup_hip_host_buffer(
    embeddings: &HipMappedHostBuffer,
    indexes: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match embeddings.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !indexes.buffer.device.same_device(&embeddings.buffer.device) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(embeddings.buffer.dtype) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.buffer.dtype) else {
        return Ok(None);
    };
    let [vocab_size, hidden_size] = <[usize; 2]>::try_from(embeddings.buffer.shape.as_slice())
        .map_err(|_| candle_core::Error::Msg("embedding-lookup embeddings rank".into()))?;
    let token_count = indexes.buffer.shape.iter().product::<usize>();
    let mut shape = indexes.buffer.shape.clone();
    shape.push(hidden_size);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, embeddings.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            vocab_size,
            hidden_size,
            embeddings.raw_device_ptr(),
            indexes.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("embedding-lookup-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: embeddings.buffer.dtype,
            device: embeddings.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_embedding_lookup_hip_host_buffer(
    embeddings: &HipMappedHostBuffer,
    indexes: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (embeddings, indexes);
    Ok(None)
}

fn immutable_embedding_lookup_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = hip_immutable_embedding_lookup_host_buffer(embedding, indexes)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: embedding.dtype(),
            device: indexes.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn immutable_embedding_lookup_hip_owned_device(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let indexes = indexes.contiguous()?;
    if !indexes.device().is_hip() {
        return Ok(None);
    }
    let (indexes_storage, indexes_layout) = indexes.storage_and_layout();
    let Storage::Hip(indexes_storage) = &*indexes_storage else {
        return Ok(None);
    };
    if !indexes_layout.is_contiguous() {
        return Ok(None);
    }
    let ordinal = indexes.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(embedding.dtype())?;
    let index_dtype_code = hip::index_dtype_code(indexes.dtype())?;
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(embedding.hidden_size());
    let embedding_ptr = embedding.registered_device_ptr(ordinal)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, embedding.dtype(), indexes.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            embedding.vocab_size(),
            embedding.hidden_size(),
            embedding_ptr as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("immutable-embedding-lookup-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn immutable_embedding_lookup_hip_owned_device(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (embedding, indexes);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_immutable_embedding_lookup_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match indexes.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(embedding.dtype()) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.buffer.dtype) else {
        return Ok(None);
    };
    let token_count = indexes.buffer.shape.iter().product::<usize>();
    let mut shape = indexes.buffer.shape.clone();
    shape.push(embedding.hidden_size());
    let embedding_ptr = embedding.registered_device_ptr(ordinal)?;
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, embedding.dtype())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            embedding.vocab_size(),
            embedding.hidden_size(),
            embedding_ptr as *const c_void,
            indexes.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "immutable-embedding-lookup-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: embedding.dtype(),
            device: indexes.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_immutable_embedding_lookup_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (embedding, indexes);
    Ok(None)
}

fn output_projection_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = immutable_output_projection_host_buffer(embedding, hidden_states)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: embedding.dtype(),
            device: hidden_states.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn output_projection_hip_owned_device(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let hidden_states = hidden_states.contiguous()?;
    if !hidden_states.device().is_hip() {
        return Ok(None);
    }
    let (hidden_storage, hidden_layout) = hidden_states.storage_and_layout();
    let Storage::Hip(hidden_storage) = &*hidden_storage else {
        return Ok(None);
    };
    if !hidden_layout.is_contiguous() {
        return Ok(None);
    }
    let ordinal = hidden_states.device().as_hip_device()?.ordinal();
    let dims = hidden_layout.shape().dims();
    let hidden_size = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("hidden state rank must be >= 1".into()))?;
    if hidden_size != embedding.hidden_size() {
        return Ok(None);
    }
    let rows = hidden_layout.shape().elem_count() / hidden_size;
    let mut shape = dims.to_vec();
    *shape.last_mut().expect("validated non-empty dims") = embedding.vocab_size();
    let weight_ptr = embedding.registered_device_ptr(ordinal)?;
    let out = HipDeviceBuffer::from_raw_hip_device_output(shape, embedding.dtype(), hidden_states.device())?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
            hip::dtype_code(embedding.dtype())?,
            ordinal,
            rows,
            embedding.hidden_size(),
            embedding.vocab_size(),
            hidden_storage.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
            weight_ptr,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("immutable-output-projection-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn output_projection_hip_owned_device(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (embedding, hidden_states);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_output_projection_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match hidden_states.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let dims = &hidden_states.buffer.shape;
    let hidden_size = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("hidden state rank must be >= 1".into()))?;
    if hidden_size != embedding.hidden_size() {
        return Ok(None);
    }
    let rows = dims.iter().product::<usize>() / hidden_size;
    let mut shape = dims.clone();
    *shape.last_mut().expect("validated non-empty dims") = embedding.vocab_size();
    let weight_ptr = embedding.registered_device_ptr(ordinal)?;
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, embedding.dtype())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
            hip::dtype_code(embedding.dtype())?,
            ordinal,
            rows,
            embedding.hidden_size(),
            embedding.vocab_size(),
            hidden_states.raw_device_ptr(),
            weight_ptr,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "immutable-output-projection-mapped-host-buffer",
            status,
        ));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: embedding.dtype(),
            device: hidden_states.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_output_projection_hip_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (embedding, hidden_states);
    Ok(None)
}

fn linear_prefill_conv_hip_host_buffer(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        linear_prefill_conv_pack_host_buffer(mixed_qkv, weights, seq_len, kernel_size)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: mixed_qkv.dtype(),
            device: mixed_qkv.device().clone(),
        },
    ))))
}

fn linear_stateful_conv_hip_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        linear_stateful_conv_host_buffer(mixed_qkv, prev_state, weights, kernel_size)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: mixed_qkv.dtype(),
            device: mixed_qkv.device().clone(),
        },
    ))))
}

fn linear_stateful_conv_value_decay_with_state_hip_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = linear_stateful_conv_value_decay_with_state_host_buffer(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: mixed_qkv.dtype(),
            device: mixed_qkv.device().clone(),
        },
    ))))
}

#[allow(clippy::too_many_arguments)]
fn linear_decode_step_hip_host_buffer(
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
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = linear_decode_step_host_buffer(
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
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: DType::F32,
            device: mixed_qkv.device().clone(),
        },
    ))))
}

fn full_attention_prefill_hip_host_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = full_attention_prefill_host_buffer(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: DType::F32,
            device: query.device().clone(),
        },
    ))))
}

fn delta_full_scan_pack_hip_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_full_scan_pack_host_buffer(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: query_scan.dtype(),
            device: query_scan.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_full_scan_pack_hip_owned_device(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let query_scan = query_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
    if !(query_scan.device().is_hip()
        && key_scan.device().same_device(query_scan.device())
        && exp_g_scan.device().same_device(query_scan.device())
        && k_cumdecay_scan.device().same_device(query_scan.device()))
    {
        return Ok(None);
    }
    let (query_storage, query_layout) = query_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_storage, exp_layout) = exp_g_scan.storage_and_layout();
    let (cum_storage, cum_layout) = k_cumdecay_scan.storage_and_layout();
    let (
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(exp_storage),
        Storage::Hip(cum_storage),
    ) = (&*query_storage, &*key_storage, &*exp_storage, &*cum_storage)
    else {
        return Ok(None);
    };
    if !(query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && exp_layout.is_contiguous()
        && cum_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = query_scan.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(query_scan.dtype())?;
    if query_scan.dtype() != key_scan.dtype()
        || query_scan.dtype() != exp_g_scan.dtype()
        || query_scan.dtype() != k_cumdecay_scan.dtype()
    {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_layout.shape().dims3()?;
    let (cum_bh, cum_chunks, cum_chunk_size, cum_k) = cum_layout.shape().dims4()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || cum_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || cum_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || cum_chunk_size != chunk_size
        || key_k != k_head_dim
        || cum_k != k_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks, chunk_size, 3 * k_head_dim + 1],
        query_scan.dtype(),
        query_scan.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan_pack(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_storage.raw_device_ptr_with_offset(exp_layout.start_offset())? as *const c_void,
            cum_storage.raw_device_ptr_with_offset(cum_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-pack-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_full_scan_pack_hip_owned_device(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (query_scan, key_scan, exp_g_scan, k_cumdecay_scan);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_full_scan_pack_hip_host_buffer(
    query_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
    k_cumdecay_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match query_scan.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.buffer.device.same_device(&query_scan.buffer.device)
        && exp_g_scan.buffer.device.same_device(&query_scan.buffer.device)
        && k_cumdecay_scan.buffer.device.same_device(&query_scan.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query_scan.buffer.dtype) else {
        return Ok(None);
    };
    if query_scan.buffer.dtype != key_scan.buffer.dtype
        || query_scan.buffer.dtype != exp_g_scan.buffer.dtype
        || query_scan.buffer.dtype != k_cumdecay_scan.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, num_chunks, chunk_size, k_head_dim] =
        <[usize; 4]>::try_from(query_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-pack query rank".into()))?;
    let [key_bh, key_chunks, key_chunk_size, key_k] =
        <[usize; 4]>::try_from(key_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-pack key rank".into()))?;
    let [exp_bh, exp_chunks, exp_chunk_size] =
        <[usize; 3]>::try_from(exp_g_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-pack exp_g rank".into()))?;
    let [cum_bh, cum_chunks, cum_chunk_size, cum_k] =
        <[usize; 4]>::try_from(k_cumdecay_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-pack k_cumdecay rank".into()))?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || cum_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || cum_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || cum_chunk_size != chunk_size
        || key_k != k_head_dim
        || cum_k != k_head_dim
    {
        return Ok(None);
    }
    let packed_width = 3 * k_head_dim + 1;
    let shape = vec![batch_heads, num_chunks, chunk_size, packed_width];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, query_scan.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan_pack(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            query_scan.raw_device_ptr(),
            key_scan.raw_device_ptr(),
            exp_g_scan.raw_device_ptr(),
            k_cumdecay_scan.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-pack-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: query_scan.buffer.dtype,
            device: query_scan.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_full_scan_pack_hip_host_buffer(
    query_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
    k_cumdecay_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (query_scan, key_scan, exp_g_scan, k_cumdecay_scan);
    Ok(None)
}

fn delta_full_scan_packed_hip_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_full_scan_packed_host_buffer(
        initial_state,
        packed_scan,
        local_attn_scan,
        value,
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_full_scan_packed_hip_owned_device(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let packed_scan = packed_scan.contiguous()?;
    let local_attn_scan = local_attn_scan.contiguous()?;
    let value = value.contiguous()?;
    if !(initial_state.device().is_hip()
        && packed_scan.device().same_device(initial_state.device())
        && local_attn_scan.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (packed_storage, packed_layout) = packed_scan.storage_and_layout();
    let (local_storage, local_layout) = local_attn_scan.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (
        Storage::Hip(initial_storage),
        Storage::Hip(packed_storage),
        Storage::Hip(local_storage),
        Storage::Hip(value_storage),
    ) = (
        &*initial_storage,
        &*packed_storage,
        &*local_storage,
        &*value_storage,
    ) else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && packed_layout.is_contiguous()
        && local_layout.is_contiguous()
        && value_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != packed_scan.dtype()
        || initial_state.dtype() != local_attn_scan.dtype()
        || initial_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
    let (local_bh, local_chunks, local_chunk_size, local_width) = local_layout.shape().dims4()?;
    let (value_bh, value_chunks, value_chunk_size, value_v) = value_layout.shape().dims4()?;
    if packed_bh != batch_heads
        || local_bh != batch_heads
        || value_bh != batch_heads
        || local_chunks != num_chunks
        || value_chunks != num_chunks
        || local_chunk_size != chunk_size
        || value_chunk_size != chunk_size
        || local_width != chunk_size
        || value_v != v_head_dim
        || packed_width != 3 * k_head_dim + 1
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan_packed(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())? as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())? as *const c_void,
            local_storage.raw_device_ptr_with_offset(local_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-packed-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_full_scan_packed_hip_owned_device(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, packed_scan, local_attn_scan, value);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_full_scan_packed_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    packed_scan: &HipMappedHostBuffer,
    local_attn_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_scan.buffer.device.same_device(&initial_state.buffer.device)
        && local_attn_scan.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != packed_scan.buffer.dtype
        || initial_state.buffer.dtype != local_attn_scan.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-packed initial_state rank".into()))?;
    let [packed_bh, num_chunks, chunk_size, packed_width] =
        <[usize; 4]>::try_from(packed_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-packed packed_scan rank".into()))?;
    let [local_bh, local_chunks, local_chunk_size, local_width] =
        <[usize; 4]>::try_from(local_attn_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-packed local_attn rank".into()))?;
    let [value_bh, value_chunks, value_chunk_size, value_v] =
        <[usize; 4]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan-packed value rank".into()))?;
    if packed_bh != batch_heads
        || local_bh != batch_heads
        || value_bh != batch_heads
        || local_chunks != num_chunks
        || value_chunks != num_chunks
        || local_chunk_size != chunk_size
        || value_chunk_size != chunk_size
        || local_width != chunk_size
        || value_v != v_head_dim
        || packed_width != 3 * k_head_dim + 1
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan_packed(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_state.raw_device_ptr(),
            packed_scan.raw_device_ptr(),
            local_attn_scan.raw_device_ptr(),
            value.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-packed-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_full_scan_packed_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    packed_scan: &HipMappedHostBuffer,
    local_attn_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, packed_scan, local_attn_scan, value);
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn delta_full_scan_hip_host_buffer(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_full_scan_host_buffer(
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
    )? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn delta_full_scan_hip_owned_device(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let weighted_key_scan = weighted_key_scan.contiguous()?;
    let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
    let q_state_scan = q_state_scan.contiguous()?;
    let local_attn_scan = local_attn_scan.contiguous()?;
    let state_decay_scan = state_decay_scan.contiguous()?;
    let value = value.contiguous()?;
    if !(initial_state.device().is_hip()
        && weighted_key_scan.device().same_device(initial_state.device())
        && k_cumdecay_scan.device().same_device(initial_state.device())
        && q_state_scan.device().same_device(initial_state.device())
        && local_attn_scan.device().same_device(initial_state.device())
        && state_decay_scan.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (weighted_storage, weighted_layout) = weighted_key_scan.storage_and_layout();
    let (cum_storage, cum_layout) = k_cumdecay_scan.storage_and_layout();
    let (q_state_storage, q_state_layout) = q_state_scan.storage_and_layout();
    let (local_storage, local_layout) = local_attn_scan.storage_and_layout();
    let (state_decay_storage, state_decay_layout) = state_decay_scan.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (
        Storage::Hip(initial_storage),
        Storage::Hip(weighted_storage),
        Storage::Hip(cum_storage),
        Storage::Hip(q_state_storage),
        Storage::Hip(local_storage),
        Storage::Hip(state_decay_storage),
        Storage::Hip(value_storage),
    ) = (
        &*initial_storage,
        &*weighted_storage,
        &*cum_storage,
        &*q_state_storage,
        &*local_storage,
        &*state_decay_storage,
        &*value_storage,
    ) else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && weighted_layout.is_contiguous()
        && cum_layout.is_contiguous()
        && q_state_layout.is_contiguous()
        && local_layout.is_contiguous()
        && state_decay_layout.is_contiguous()
        && value_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != weighted_key_scan.dtype()
        || initial_state.dtype() != k_cumdecay_scan.dtype()
        || initial_state.dtype() != q_state_scan.dtype()
        || initial_state.dtype() != local_attn_scan.dtype()
        || initial_state.dtype() != state_decay_scan.dtype()
        || initial_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (weighted_bh, num_chunks, chunk_size, weighted_width) = weighted_layout.shape().dims4()?;
    let (cum_bh, cum_chunks, cum_chunk_size, cum_width) = cum_layout.shape().dims4()?;
    let (q_state_bh, q_state_chunks, q_state_chunk_size, q_state_width) =
        q_state_layout.shape().dims4()?;
    let (local_bh, local_chunks, local_chunk_size, local_width) = local_layout.shape().dims4()?;
    let (state_decay_bh, state_decay_chunks) = state_decay_layout.shape().dims2()?;
    let (value_bh, value_chunks, value_chunk_size, value_v_head_dim) = value_layout.shape().dims4()?;
    if weighted_bh != batch_heads
        || cum_bh != batch_heads
        || q_state_bh != batch_heads
        || local_bh != batch_heads
        || state_decay_bh != batch_heads
        || value_bh != batch_heads
        || cum_chunks != num_chunks
        || q_state_chunks != num_chunks
        || local_chunks != num_chunks
        || state_decay_chunks != num_chunks
        || value_chunks != num_chunks
        || cum_chunk_size != chunk_size
        || q_state_chunk_size != chunk_size
        || local_chunk_size != chunk_size
        || value_chunk_size != chunk_size
        || weighted_width != k_head_dim
        || cum_width != k_head_dim
        || q_state_width != k_head_dim
        || local_width != chunk_size
        || value_v_head_dim != v_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())? as *const c_void,
            weighted_storage.raw_device_ptr_with_offset(weighted_layout.start_offset())? as *const c_void,
            cum_storage.raw_device_ptr_with_offset(cum_layout.start_offset())? as *const c_void,
            q_state_storage.raw_device_ptr_with_offset(q_state_layout.start_offset())? as *const c_void,
            local_storage.raw_device_ptr_with_offset(local_layout.start_offset())? as *const c_void,
            state_decay_storage.raw_device_ptr_with_offset(state_decay_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn delta_full_scan_hip_owned_device(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
    );
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
#[allow(clippy::too_many_arguments)]
fn mapped_delta_full_scan_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    weighted_key_scan: &HipMappedHostBuffer,
    k_cumdecay_scan: &HipMappedHostBuffer,
    q_state_scan: &HipMappedHostBuffer,
    local_attn_scan: &HipMappedHostBuffer,
    state_decay_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(weighted_key_scan.buffer.device.same_device(&initial_state.buffer.device)
        && k_cumdecay_scan.buffer.device.same_device(&initial_state.buffer.device)
        && q_state_scan.buffer.device.same_device(&initial_state.buffer.device)
        && local_attn_scan.buffer.device.same_device(&initial_state.buffer.device)
        && state_decay_scan.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != weighted_key_scan.buffer.dtype
        || initial_state.buffer.dtype != k_cumdecay_scan.buffer.dtype
        || initial_state.buffer.dtype != q_state_scan.buffer.dtype
        || initial_state.buffer.dtype != local_attn_scan.buffer.dtype
        || initial_state.buffer.dtype != state_decay_scan.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan initial_state rank".into()))?;
    let [weighted_key_bh, num_chunks, chunk_size, weighted_key_width] =
        <[usize; 4]>::try_from(weighted_key_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan weighted_key rank".into()))?;
    let [k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width] =
        <[usize; 4]>::try_from(k_cumdecay_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan k_cumdecay rank".into()))?;
    let [q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width] =
        <[usize; 4]>::try_from(q_state_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan q_state rank".into()))?;
    let [local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width] =
        <[usize; 4]>::try_from(local_attn_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan local_attn rank".into()))?;
    let [state_decay_bh, state_decay_num_chunks] =
        <[usize; 2]>::try_from(state_decay_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan state_decay rank".into()))?;
    let [value_bh, value_num_chunks, value_chunk_size, value_v_head_dim] =
        <[usize; 4]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-full-scan value rank".into()))?;
    if weighted_key_bh != batch_heads
        || k_cumdecay_bh != batch_heads
        || q_state_bh != batch_heads
        || local_attn_bh != batch_heads
        || state_decay_bh != batch_heads
        || value_bh != batch_heads
        || k_cumdecay_num_chunks != num_chunks
        || q_state_num_chunks != num_chunks
        || local_attn_num_chunks != num_chunks
        || state_decay_num_chunks != num_chunks
        || value_num_chunks != num_chunks
        || k_cumdecay_chunk_size != chunk_size
        || q_state_chunk_size != chunk_size
        || local_attn_chunk_size != chunk_size
        || value_chunk_size != chunk_size
        || weighted_key_width != k_head_dim
        || k_cumdecay_width != k_head_dim
        || q_state_width != k_head_dim
        || local_attn_width != chunk_size
        || value_v_head_dim != v_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_full_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_state.raw_device_ptr(),
            weighted_key_scan.raw_device_ptr(),
            k_cumdecay_scan.raw_device_ptr(),
            q_state_scan.raw_device_ptr(),
            local_attn_scan.raw_device_ptr(),
            state_decay_scan.raw_device_ptr(),
            value.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[allow(clippy::too_many_arguments)]
fn mapped_delta_full_scan_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    weighted_key_scan: &HipMappedHostBuffer,
    k_cumdecay_scan: &HipMappedHostBuffer,
    q_state_scan: &HipMappedHostBuffer,
    local_attn_scan: &HipMappedHostBuffer,
    state_decay_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
    );
    Ok(None)
}

fn delta_local_attn_scan_hip_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_local_attn_scan_host_buffer(query_scan, key_scan, exp_g_scan)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: query_scan.dtype(),
            device: query_scan.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_local_attn_scan_hip_owned_device(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let query_scan = query_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    if !(query_scan.device().is_hip()
        && key_scan.device().same_device(query_scan.device())
        && exp_g_scan.device().same_device(query_scan.device()))
    {
        return Ok(None);
    }
    let (query_storage, query_layout) = query_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_storage, exp_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(query_storage), Storage::Hip(key_storage), Storage::Hip(exp_storage)) =
        (&*query_storage, &*key_storage, &*exp_storage)
    else {
        return Ok(None);
    };
    if !(query_layout.is_contiguous() && key_layout.is_contiguous() && exp_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = query_scan.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(query_scan.dtype())?;
    if query_scan.dtype() != key_scan.dtype() || query_scan.dtype() != exp_g_scan.dtype() {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks, chunk_size, chunk_size],
        query_scan.dtype(),
        query_scan.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_local_attn_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_storage.raw_device_ptr_with_offset(exp_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-local-attn-scan-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_local_attn_scan_hip_owned_device(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (query_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_local_attn_scan_hip_host_buffer(
    query_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match query_scan.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.buffer.device.same_device(&query_scan.buffer.device)
        && exp_g_scan.buffer.device.same_device(&query_scan.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query_scan.buffer.dtype) else {
        return Ok(None);
    };
    if query_scan.buffer.dtype != key_scan.buffer.dtype
        || query_scan.buffer.dtype != exp_g_scan.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, num_chunks, chunk_size, k_head_dim] =
        <[usize; 4]>::try_from(query_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-local-attn-scan query rank".into()))?;
    let [key_bh, key_chunks, key_chunk_size, key_k] =
        <[usize; 4]>::try_from(key_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-local-attn-scan key rank".into()))?;
    let [exp_bh, exp_chunks, exp_chunk_size] =
        <[usize; 3]>::try_from(exp_g_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-local-attn-scan exp_g rank".into()))?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, query_scan.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_local_attn_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            query_scan.raw_device_ptr(),
            key_scan.raw_device_ptr(),
            exp_g_scan.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-local-attn-scan-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: query_scan.buffer.dtype,
            device: query_scan.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_local_attn_scan_hip_host_buffer(
    query_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (query_scan, key_scan, exp_g_scan);
    Ok(None)
}

fn delta_base_attn_scan_hip_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        delta_base_attn_scan_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: k_beta_scan.dtype(),
            device: k_beta_scan.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_base_attn_scan_hip_owned_device(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let k_beta_scan = k_beta_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    if !(k_beta_scan.device().is_hip()
        && key_scan.device().same_device(k_beta_scan.device())
        && exp_g_scan.device().same_device(k_beta_scan.device()))
    {
        return Ok(None);
    }
    let (k_beta_storage, k_beta_layout) = k_beta_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_storage, exp_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(k_beta_storage), Storage::Hip(key_storage), Storage::Hip(exp_storage)) =
        (&*k_beta_storage, &*key_storage, &*exp_storage)
    else {
        return Ok(None);
    };
    if !(k_beta_layout.is_contiguous() && key_layout.is_contiguous() && exp_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = k_beta_scan.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(k_beta_scan.dtype())?;
    if k_beta_scan.dtype() != key_scan.dtype() || k_beta_scan.dtype() != exp_g_scan.dtype() {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks, chunk_size, chunk_size],
        k_beta_scan.dtype(),
        k_beta_scan.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_base_attn_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            k_beta_storage.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_storage.raw_device_ptr_with_offset(exp_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-base-attn-scan-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_base_attn_scan_hip_owned_device(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_base_attn_scan_hip_host_buffer(
    k_beta_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match k_beta_scan.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.buffer.device.same_device(&k_beta_scan.buffer.device)
        && exp_g_scan.buffer.device.same_device(&k_beta_scan.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(k_beta_scan.buffer.dtype) else {
        return Ok(None);
    };
    if k_beta_scan.buffer.dtype != key_scan.buffer.dtype
        || k_beta_scan.buffer.dtype != exp_g_scan.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, num_chunks, chunk_size, k_head_dim] =
        <[usize; 4]>::try_from(k_beta_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-base-attn-scan k_beta rank".into()))?;
    let [key_bh, key_chunks, key_chunk_size, key_k] =
        <[usize; 4]>::try_from(key_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-base-attn-scan key rank".into()))?;
    let [exp_bh, exp_chunks, exp_chunk_size] =
        <[usize; 3]>::try_from(exp_g_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-base-attn-scan exp_g rank".into()))?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, k_beta_scan.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_base_attn_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            k_beta_scan.raw_device_ptr(),
            key_scan.raw_device_ptr(),
            exp_g_scan.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-base-attn-scan-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: k_beta_scan.buffer.dtype,
            device: k_beta_scan.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_base_attn_scan_hip_host_buffer(
    k_beta_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

fn delta_attn_solve_from_inputs_hip_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        delta_attn_solve_from_inputs_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: k_beta_scan.dtype(),
            device: k_beta_scan.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_attn_solve_from_inputs_hip_owned_device(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let k_beta_scan = k_beta_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    if !(k_beta_scan.device().is_hip()
        && key_scan.device().same_device(k_beta_scan.device())
        && exp_g_scan.device().same_device(k_beta_scan.device()))
    {
        return Ok(None);
    }
    let (k_beta_storage, k_beta_layout) = k_beta_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_storage, exp_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(k_beta_storage), Storage::Hip(key_storage), Storage::Hip(exp_storage)) =
        (&*k_beta_storage, &*key_storage, &*exp_storage)
    else {
        return Ok(None);
    };
    if !(k_beta_layout.is_contiguous() && key_layout.is_contiguous() && exp_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = k_beta_scan.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(k_beta_scan.dtype())?;
    if k_beta_scan.dtype() != key_scan.dtype() || k_beta_scan.dtype() != exp_g_scan.dtype() {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks, chunk_size, chunk_size],
        k_beta_scan.dtype(),
        k_beta_scan.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_attn_solve_from_inputs(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            k_beta_storage.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_storage.raw_device_ptr_with_offset(exp_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-attn-solve-from-inputs-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_attn_solve_from_inputs_hip_owned_device(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_attn_solve_from_inputs_hip_host_buffer(
    k_beta_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match k_beta_scan.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.buffer.device.same_device(&k_beta_scan.buffer.device)
        && exp_g_scan.buffer.device.same_device(&k_beta_scan.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(k_beta_scan.buffer.dtype) else {
        return Ok(None);
    };
    if k_beta_scan.buffer.dtype != key_scan.buffer.dtype
        || k_beta_scan.buffer.dtype != exp_g_scan.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, num_chunks, chunk_size, k_head_dim] =
        <[usize; 4]>::try_from(k_beta_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-attn-solve-from-inputs k_beta rank".into()))?;
    let [key_bh, key_chunks, key_chunk_size, key_k] =
        <[usize; 4]>::try_from(key_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-attn-solve-from-inputs key rank".into()))?;
    let [exp_bh, exp_chunks, exp_chunk_size] =
        <[usize; 3]>::try_from(exp_g_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-attn-solve-from-inputs exp_g rank".into()))?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, k_beta_scan.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_attn_solve_from_inputs(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            k_beta_scan.raw_device_ptr(),
            key_scan.raw_device_ptr(),
            exp_g_scan.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-attn-solve-from-inputs-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: k_beta_scan.buffer.dtype,
            device: k_beta_scan.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_attn_solve_from_inputs_hip_host_buffer(
    k_beta_scan: &HipMappedHostBuffer,
    key_scan: &HipMappedHostBuffer,
    exp_g_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

fn delta_attn_solve_scan_hip_host_buffer(base_attn_scan: &Tensor) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_attn_solve_scan_host_buffer(base_attn_scan)? else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: base_attn_scan.dtype(),
            device: base_attn_scan.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_attn_solve_scan_hip_owned_device(base_attn_scan: &Tensor) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let base_attn_scan = base_attn_scan.contiguous()?;
    if !base_attn_scan.device().is_hip() {
        return Ok(None);
    }
    let (base_storage, base_layout) = base_attn_scan.storage_and_layout();
    let Storage::Hip(base_storage) = &*base_storage else {
        return Ok(None);
    };
    if !base_layout.is_contiguous() {
        return Ok(None);
    }
    let ordinal = base_attn_scan.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(base_attn_scan.dtype())?;
    let (batch_heads, num_chunks, chunk_size, width) = base_layout.shape().dims4()?;
    if width != chunk_size {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks, chunk_size, chunk_size],
        base_attn_scan.dtype(),
        base_attn_scan.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_attn_solve_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            base_storage.raw_device_ptr_with_offset(base_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-attn-solve-scan-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_attn_solve_scan_hip_owned_device(base_attn_scan: &Tensor) -> Result<Option<HipTensor>> {
    let _ = base_attn_scan;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_attn_solve_scan_hip_host_buffer(
    base_attn_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match base_attn_scan.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(base_attn_scan.buffer.dtype) else {
        return Ok(None);
    };
    let [batch_heads, num_chunks, chunk_size, width] =
        <[usize; 4]>::try_from(base_attn_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-attn-solve-scan base_attn rank".into()))?;
    if width != chunk_size {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, base_attn_scan.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_attn_solve_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            base_attn_scan.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-attn-solve-scan-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: base_attn_scan.buffer.dtype,
            device: base_attn_scan.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_attn_solve_scan_hip_host_buffer(
    base_attn_scan: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = base_attn_scan;
    Ok(None)
}

fn delta_recurrent_prefill_hip_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        delta_recurrent_prefill_host_buffer(initial_state, query, key, value, beta, g)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_recurrent_prefill_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !(initial_state.device().is_hip()
        && query.device().same_device(initial_state.device())
        && key.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device())
        && beta.device().same_device(initial_state.device())
        && g.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (
        Storage::Hip(initial_storage),
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(value_storage),
        Storage::Hip(beta_storage),
        Storage::Hip(g_storage),
    ) = (
        &*initial_storage,
        &*query_storage,
        &*key_storage,
        &*value_storage,
        &*beta_storage,
        &*g_storage,
    ) else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && value_layout.is_contiguous()
        && beta_layout.is_contiguous()
        && g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (query_bh, seq_len, query_k) = query_layout.shape().dims3()?;
    let (key_bh, key_seq, key_k) = key_layout.shape().dims3()?;
    let (value_bh, value_seq, value_v) = value_layout.shape().dims3()?;
    let (beta_bh, beta_seq) = beta_layout.shape().dims2()?;
    let (g_bh, g_seq) = g_layout.shape().dims2()?;
    if query_bh != batch_heads
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_seq != seq_len
        || value_seq != seq_len
        || beta_seq != seq_len
        || g_seq != seq_len
        || query_k != k_head_dim
        || key_k != k_head_dim
        || value_v != v_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, seq_len + k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_recurrent_prefill(
            dtype_code,
            ordinal,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())? as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-recurrent-prefill-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_recurrent_prefill_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_recurrent_prefill_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.buffer.device.same_device(&initial_state.buffer.device)
        && key.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device)
        && beta.buffer.device.same_device(&initial_state.buffer.device)
        && g.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != query.buffer.dtype
        || initial_state.buffer.dtype != key.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
        || initial_state.buffer.dtype != beta.buffer.dtype
        || initial_state.buffer.dtype != g.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill initial_state rank".into()))?;
    let [query_bh, seq_len, query_k] =
        <[usize; 3]>::try_from(query.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill query rank".into()))?;
    let [key_bh, key_seq, key_k] =
        <[usize; 3]>::try_from(key.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill key rank".into()))?;
    let [value_bh, value_seq, value_v] =
        <[usize; 3]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill value rank".into()))?;
    let [beta_bh, beta_seq] =
        <[usize; 2]>::try_from(beta.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill beta rank".into()))?;
    let [g_bh, g_seq] = <[usize; 2]>::try_from(g.buffer.shape.as_slice())
        .map_err(|_| candle_core::Error::Msg("delta-recurrent-prefill g rank".into()))?;
    if query_bh != batch_heads
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_seq != seq_len
        || value_seq != seq_len
        || beta_seq != seq_len
        || g_seq != seq_len
        || query_k != k_head_dim
        || key_k != k_head_dim
        || value_v != v_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, seq_len + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_recurrent_prefill(
            dtype_code,
            ordinal,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            initial_state.raw_device_ptr(),
            query.raw_device_ptr(),
            key.raw_device_ptr(),
            value.raw_device_ptr(),
            beta.raw_device_ptr(),
            g.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-recurrent-prefill-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_recurrent_prefill_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

fn delta_chunk_scan_raw_hip_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        delta_chunk_scan_raw_host_buffer(initial_state, query, key, value, beta, g)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_chunk_scan_raw_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !(initial_state.device().is_hip()
        && query.device().same_device(initial_state.device())
        && key.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device())
        && beta.device().same_device(initial_state.device())
        && g.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (
        Storage::Hip(initial_storage),
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(value_storage),
        Storage::Hip(beta_storage),
        Storage::Hip(g_storage),
    ) = (
        &*initial_storage,
        &*query_storage,
        &*key_storage,
        &*value_storage,
        &*beta_storage,
        &*g_storage,
    ) else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && value_layout.is_contiguous()
        && beta_layout.is_contiguous()
        && g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
    let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
    let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
    let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
    let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
    if query_bh != batch_heads
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_num_chunks != num_chunks
        || value_num_chunks != num_chunks
        || beta_num_chunks != num_chunks
        || g_num_chunks != num_chunks
        || key_chunk != chunk_size
        || value_chunk != chunk_size
        || beta_chunk != chunk_size
        || g_chunk != chunk_size
        || query_k != k_head_dim
        || key_k != k_head_dim
        || value_v != v_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_scan_raw(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())? as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-scan-raw-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_chunk_scan_raw_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_chunk_scan_raw_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.buffer.device.same_device(&initial_state.buffer.device)
        && key.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device)
        && beta.buffer.device.same_device(&initial_state.buffer.device)
        && g.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != query.buffer.dtype
        || initial_state.buffer.dtype != key.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
        || initial_state.buffer.dtype != beta.buffer.dtype
        || initial_state.buffer.dtype != g.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw initial_state rank".into()))?;
    let [query_bh, num_chunks, chunk_size, query_k] =
        <[usize; 4]>::try_from(query.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw query rank".into()))?;
    let [key_bh, key_num_chunks, key_chunk, key_k] =
        <[usize; 4]>::try_from(key.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw key rank".into()))?;
    let [value_bh, value_num_chunks, value_chunk, value_v] =
        <[usize; 4]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw value rank".into()))?;
    let [beta_bh, beta_num_chunks, beta_chunk] =
        <[usize; 3]>::try_from(beta.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw beta rank".into()))?;
    let [g_bh, g_num_chunks, g_chunk] =
        <[usize; 3]>::try_from(g.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-scan-raw g rank".into()))?;
    if query_bh != batch_heads
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_num_chunks != num_chunks
        || value_num_chunks != num_chunks
        || beta_num_chunks != num_chunks
        || g_num_chunks != num_chunks
        || key_chunk != chunk_size
        || value_chunk != chunk_size
        || beta_chunk != chunk_size
        || g_chunk != chunk_size
        || query_k != k_head_dim
        || key_k != k_head_dim
        || value_v != v_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_scan_raw(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_state.raw_device_ptr(),
            query.raw_device_ptr(),
            key.raw_device_ptr(),
            value.raw_device_ptr(),
            beta.raw_device_ptr(),
            g.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-scan-raw-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_chunk_scan_raw_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

fn delta_chunk_single_prefill_hip_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) =
        delta_chunk_single_prefill_host_buffer(initial_state, query, key, value, beta, g)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_chunk_single_prefill_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !(initial_state.device().is_hip()
        && query.device().same_device(initial_state.device())
        && key.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device())
        && beta.device().same_device(initial_state.device())
        && g.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (
        Storage::Hip(_initial_storage),
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(value_storage),
        Storage::Hip(beta_storage),
        Storage::Hip(g_storage),
    ) = (
        &*initial_storage,
        &*query_storage,
        &*key_storage,
        &*value_storage,
        &*beta_storage,
        &*g_storage,
    ) else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && value_layout.is_contiguous()
        && beta_layout.is_contiguous()
        && g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let (initial_bh, initial_k_head_dim, initial_v_head_dim) = initial_layout.shape().dims3()?;
    let (batch_heads, chunk_size, k_head_dim) = query_layout.shape().dims3()?;
    let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
    let (value_bh, value_chunk, v_head_dim) = value_layout.shape().dims3()?;
    let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
    let (g_bh, g_chunk) = g_layout.shape().dims2()?;
    if initial_bh != batch_heads
        || initial_k_head_dim != k_head_dim
        || initial_v_head_dim != v_head_dim
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_chunk != chunk_size
        || value_chunk != chunk_size
        || beta_chunk != chunk_size
        || g_chunk != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, chunk_size + k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_single_prefill(
            dtype_code,
            ordinal,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-single-prefill-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_chunk_single_prefill_hip_owned_device(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_chunk_single_prefill_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.buffer.device.same_device(&initial_state.buffer.device)
        && key.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device)
        && beta.buffer.device.same_device(&initial_state.buffer.device)
        && g.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != query.buffer.dtype
        || initial_state.buffer.dtype != key.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
        || initial_state.buffer.dtype != beta.buffer.dtype
        || initial_state.buffer.dtype != g.buffer.dtype
    {
        return Ok(None);
    }
    let [initial_bh, initial_k_head_dim, initial_v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill initial_state rank".into()))?;
    let [batch_heads, chunk_size, k_head_dim] =
        <[usize; 3]>::try_from(query.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill query rank".into()))?;
    let [key_bh, key_chunk, key_k] =
        <[usize; 3]>::try_from(key.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill key rank".into()))?;
    let [value_bh, value_chunk, v_head_dim] =
        <[usize; 3]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill value rank".into()))?;
    let [beta_bh, beta_chunk] =
        <[usize; 2]>::try_from(beta.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill beta rank".into()))?;
    let [g_bh, g_chunk] =
        <[usize; 2]>::try_from(g.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-single-prefill g rank".into()))?;
    if initial_bh != batch_heads
        || initial_k_head_dim != k_head_dim
        || initial_v_head_dim != v_head_dim
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
        || key_chunk != chunk_size
        || value_chunk != chunk_size
        || beta_chunk != chunk_size
        || g_chunk != chunk_size
        || key_k != k_head_dim
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_single_prefill(
            dtype_code,
            ordinal,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            query.raw_device_ptr(),
            key.raw_device_ptr(),
            value.raw_device_ptr(),
            beta.raw_device_ptr(),
            g.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-single-prefill-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_chunk_single_prefill_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    query: &HipMappedHostBuffer,
    key: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
    beta: &HipMappedHostBuffer,
    g: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

fn delta_state_scan_hip_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_state_scan_host_buffer(initial_state, packed_scan, value)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: initial_state.dtype(),
            device: initial_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_state_scan_hip_owned_device(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let initial_state = initial_state.contiguous()?;
    let packed_scan = packed_scan.contiguous()?;
    let value = value.contiguous()?;
    if !(initial_state.device().is_hip()
        && packed_scan.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (packed_storage, packed_layout) = packed_scan.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (Storage::Hip(initial_storage), Storage::Hip(packed_storage), Storage::Hip(value_storage)) =
        (&*initial_storage, &*packed_storage, &*value_storage)
    else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous() && packed_layout.is_contiguous() && value_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = initial_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(initial_state.dtype())?;
    if initial_state.dtype() != packed_scan.dtype() || initial_state.dtype() != value.dtype() {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
    let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
        value_layout.shape().dims4()?;
    if packed_bh != batch_heads
        || value_bh != batch_heads
        || value_num_chunks != num_chunks
        || value_chunk_size != chunk_size
        || value_v_head_dim != v_head_dim
        || packed_width != 2 * k_head_dim + 1
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, num_chunks + 1, k_head_dim, v_head_dim],
        initial_state.dtype(),
        initial_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_state_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())? as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-state-scan-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_state_scan_hip_owned_device(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, packed_scan, value);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_state_scan_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    packed_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match initial_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_scan.buffer.device.same_device(&initial_state.buffer.device)
        && value.buffer.device.same_device(&initial_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.buffer.dtype) else {
        return Ok(None);
    };
    if initial_state.buffer.dtype != packed_scan.buffer.dtype
        || initial_state.buffer.dtype != value.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(initial_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-state-scan initial_state rank".into()))?;
    let [packed_bh, num_chunks, chunk_size, packed_width] =
        <[usize; 4]>::try_from(packed_scan.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-state-scan packed_scan rank".into()))?;
    let [value_bh, value_num_chunks, value_chunk_size, value_v_head_dim] =
        <[usize; 4]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-state-scan value rank".into()))?;
    if packed_bh != batch_heads
        || value_bh != batch_heads
        || value_num_chunks != num_chunks
        || value_chunk_size != chunk_size
        || value_v_head_dim != v_head_dim
        || packed_width != 2 * k_head_dim + 1
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, num_chunks + 1, k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, initial_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_state_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial_state.raw_device_ptr(),
            packed_scan.raw_device_ptr(),
            value.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-state-scan-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: initial_state.buffer.dtype,
            device: initial_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_state_scan_hip_host_buffer(
    initial_state: &HipMappedHostBuffer,
    packed_scan: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (initial_state, packed_scan, value);
    Ok(None)
}

fn delta_chunk_fused_hip_host_buffer(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let Some((bytes, shape)) = delta_chunk_fused_host_buffer(prev_state, packed_chunk, value)?
    else {
        return Ok(None);
    };
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: bytes.into(),
            shape,
            dtype: prev_state.dtype(),
            device: prev_state.device().clone(),
        },
    ))))
}

#[cfg(feature = "qwen35-minimal-hip")]
fn delta_chunk_fused_hip_owned_device(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    use candle_core::Storage;

    let prev_state = prev_state.contiguous()?;
    let packed_chunk = packed_chunk.contiguous()?;
    let value = value.contiguous()?;
    if !(prev_state.device().is_hip()
        && packed_chunk.device().same_device(prev_state.device())
        && value.device().same_device(prev_state.device()))
    {
        return Ok(None);
    }
    let (prev_storage, prev_layout) = prev_state.storage_and_layout();
    let (packed_storage, packed_layout) = packed_chunk.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (Storage::Hip(prev_storage), Storage::Hip(packed_storage), Storage::Hip(value_storage)) =
        (&*prev_storage, &*packed_storage, &*value_storage)
    else {
        return Ok(None);
    };
    if !(prev_layout.is_contiguous() && packed_layout.is_contiguous() && value_layout.is_contiguous()) {
        return Ok(None);
    }
    let ordinal = prev_state.device().as_hip_device()?.ordinal();
    let dtype_code = hip::dtype_code(prev_state.dtype())?;
    if prev_state.dtype() != packed_chunk.dtype() || prev_state.dtype() != value.dtype() {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
    let (packed_bh, chunk_size, packed_width) = packed_layout.shape().dims3()?;
    let (value_bh, value_chunk_size, value_v_head_dim) = value_layout.shape().dims3()?;
    if packed_bh != batch_heads
        || value_bh != batch_heads
        || value_chunk_size != chunk_size
        || value_v_head_dim != v_head_dim
        || packed_width != 3 * k_head_dim + 1
    {
        return Ok(None);
    }
    let out = HipDeviceBuffer::from_raw_hip_device_output(
        vec![batch_heads, 2 * chunk_size + k_head_dim, v_head_dim],
        prev_state.dtype(),
        prev_state.device(),
    )?;
    let HipDeviceStorage::OwnedDeviceBuffer(buffer) = &out.storage else {
        return Ok(None);
    };
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_fused(
            dtype_code,
            ordinal,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_storage.raw_device_ptr_with_offset(prev_layout.start_offset())? as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            buffer.raw_device_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-fused-owned-device", status));
    }
    Ok(Some(HipTensor::from_device_buffer(out)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_chunk_fused_hip_owned_device(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Option<HipTensor>> {
    let _ = (prev_state, packed_chunk, value);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn mapped_delta_chunk_fused_hip_host_buffer(
    prev_state: &HipMappedHostBuffer,
    packed_chunk: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let ordinal = match prev_state.buffer.device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_chunk.buffer.device.same_device(&prev_state.buffer.device)
        && value.buffer.device.same_device(&prev_state.buffer.device))
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(prev_state.buffer.dtype) else {
        return Ok(None);
    };
    if prev_state.buffer.dtype != packed_chunk.buffer.dtype
        || prev_state.buffer.dtype != value.buffer.dtype
    {
        return Ok(None);
    }
    let [batch_heads, k_head_dim, v_head_dim] =
        <[usize; 3]>::try_from(prev_state.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-fused prev_state rank".into()))?;
    let [packed_bh, chunk_size, packed_width] =
        <[usize; 3]>::try_from(packed_chunk.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-fused packed_chunk rank".into()))?;
    let [value_bh, value_chunk_size, value_v_head_dim] =
        <[usize; 3]>::try_from(value.buffer.shape.as_slice())
            .map_err(|_| candle_core::Error::Msg("delta-chunk-fused value rank".into()))?;
    if packed_bh != batch_heads
        || value_bh != batch_heads
        || value_chunk_size != chunk_size
        || value_v_head_dim != v_head_dim
        || packed_width != 3 * k_head_dim + 1
    {
        return Ok(None);
    }
    let shape = vec![batch_heads, 2 * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&shape, prev_state.buffer.dtype)];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_fused(
            dtype_code,
            ordinal,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_state.raw_device_ptr(),
            packed_chunk.raw_device_ptr(),
            value.raw_device_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-fused-mapped-host-buffer", status));
    }
    Ok(Some(HipTensor::from_device_buffer(host_result_device_buffer(
        HipHostBuffer {
            bytes: out.into(),
            shape,
            dtype: prev_state.buffer.dtype,
            device: prev_state.buffer.device.clone(),
        },
    ))))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn mapped_delta_chunk_fused_hip_host_buffer(
    prev_state: &HipMappedHostBuffer,
    packed_chunk: &HipMappedHostBuffer,
    value: &HipMappedHostBuffer,
) -> Result<Option<HipTensor>> {
    let _ = (prev_state, packed_chunk, value);
    Ok(None)
}

fn materialize_host_result_as_device_leaf(host: HipTensor) -> Result<HipTensor> {
    if let Some(buffer) = host.try_host_buffer()? {
        return Ok(HipTensor::from_device_buffer(
            host_result_device_buffer(buffer),
        ));
    }
    Ok(host)
}

fn host_result_device_buffer(buffer: HipHostBuffer) -> HipDeviceBuffer {
    if buffer.device.is_hip() {
        if let Ok(device) = HipOwnedDeviceBuffer::from_host_buffer(buffer.clone()) {
            return HipDeviceBuffer::from_owned_device_buffer(device);
        }
    }
    HipDeviceBuffer::from_materialized_host_buffer(buffer)
}

fn prepare_full_attention_output_host_buffer(
    attn_output: &HipHostBuffer,
    gate: &HipHostBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<HipHostBuffer> {
    if !HipNativeBuffer::supports_host_float_ops(attn_output.dtype)
        || !HipNativeBuffer::supports_host_float_ops(gate.dtype)
        || !HipNativeBuffer::supports_host_float_ops(hidden_dtype)
    {
        candle_core::bail!(
            "prepare_full_attention_output host path unsupported for dtypes {:?}, {:?} -> {:?}",
            attn_output.dtype,
            gate.dtype,
            hidden_dtype
        );
    }
    if attn_output.shape.len() != 4 {
        candle_core::bail!(
            "prepare_full_attention_output expects rank-4 attn output, got {:?}",
            attn_output.shape
        );
    }
    if gate.shape != [b_sz, q_len, attention_size] {
        candle_core::bail!(
            "prepare_full_attention_output gate shape mismatch: expected {:?}, got {:?}",
            vec![b_sz, q_len, attention_size],
            gate.shape
        );
    }
    let heads = attn_output.shape[1];
    let head_dim = attn_output.shape[3];
    if attn_output.shape[0] != b_sz || attn_output.shape[2] != q_len {
        candle_core::bail!(
            "prepare_full_attention_output attn shape mismatch: expected [{b_sz}, heads, {q_len}, head_dim], got {:?}",
            attn_output.shape
        );
    }
    if heads.saturating_mul(head_dim) != attention_size {
        candle_core::bail!(
            "prepare_full_attention_output attention size mismatch: heads={heads} head_dim={head_dim} attention_size={attention_size}"
        );
    }
    let out_shape = vec![b_sz, q_len, attention_size];
    let elem_count = HipNativeBuffer::elem_count(&out_shape);
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&out_shape, hidden_dtype)];
    for idx in 0..elem_count {
        let a = idx % attention_size;
        let q = (idx / attention_size) % q_len;
        let b = idx / (q_len * attention_size);
        let h = a / head_dim;
        let d = a % head_dim;
        let attn_idx = (((b * heads + h) * q_len + q) * head_dim) + d;
        let attn_val =
            HipNativeBuffer::read_host_float(attn_output.bytes.as_ref(), attn_output.dtype, attn_idx)?;
        let gate_val = HipNativeBuffer::read_host_float(gate.bytes.as_ref(), gate.dtype, idx)?;
        let silu_gate = 1.0 / (1.0 + (-gate_val).exp());
        HipNativeBuffer::write_host_float(&mut out, hidden_dtype, idx, attn_val * silu_gate)?;
    }
    Ok(HipHostBuffer {
        bytes: out.into(),
        shape: out_shape,
        dtype: hidden_dtype,
        device: attn_output.device.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_full_attention_inputs_host_buffers(
    q_and_gate: &HipHostBuffer,
    k_proj: &HipHostBuffer,
    v_proj: &HipHostBuffer,
    b_sz: usize,
    q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_norm_weight: &Tensor,
    q_norm_eps: f64,
    k_norm_weight: &Tensor,
    k_norm_eps: f64,
) -> Result<(HipHostBuffer, HipHostBuffer, HipHostBuffer, HipHostBuffer)> {
    if !HipNativeBuffer::supports_host_float_ops(q_and_gate.dtype)
        || !HipNativeBuffer::supports_host_float_ops(k_proj.dtype)
        || !HipNativeBuffer::supports_host_float_ops(v_proj.dtype)
    {
        candle_core::bail!(
            "full attention host path unsupported for dtypes {:?}, {:?}, {:?}",
            q_and_gate.dtype,
            k_proj.dtype,
            v_proj.dtype
        );
    }
    let q_width = num_heads * head_dim;
    let kv_width = num_kv_heads * head_dim;
    if q_and_gate.shape != [b_sz, q_len, q_width * 2] {
        candle_core::bail!(
            "q_and_gate shape mismatch: expected {:?}, got {:?}",
            vec![b_sz, q_len, q_width * 2],
            q_and_gate.shape
        );
    }
    if k_proj.shape != [b_sz, q_len, kv_width] || v_proj.shape != [b_sz, q_len, kv_width] {
        candle_core::bail!(
            "k/v proj shape mismatch: expected {:?}, got {:?} and {:?}",
            vec![b_sz, q_len, kv_width],
            k_proj.shape,
            v_proj.shape
        );
    }
    let outer = b_sz.saturating_mul(q_len);
    let mut query_bytes =
        vec![0u8; HipNativeBuffer::byte_len(&[b_sz, q_len, num_heads, head_dim], q_and_gate.dtype)];
    let mut gate_bytes = vec![0u8; HipNativeBuffer::byte_len(&[b_sz, q_len, q_width], q_and_gate.dtype)];
    let mut key_bytes =
        vec![0u8; HipNativeBuffer::byte_len(&[b_sz, q_len, num_kv_heads, head_dim], k_proj.dtype)];
    let mut value_bytes =
        vec![0u8; HipNativeBuffer::byte_len(&[b_sz, num_kv_heads, q_len, head_dim], v_proj.dtype)];
    for outer_idx in 0..outer.max(1) {
        let qg_base = outer_idx * q_width * 2;
        let kv_base = outer_idx * kv_width;
        let b = outer_idx / q_len.max(1);
        let q = outer_idx % q_len.max(1);
        for idx in 0..q_width {
            let query_val =
                HipNativeBuffer::read_host_float(q_and_gate.bytes.as_ref(), q_and_gate.dtype, qg_base + idx)?;
            HipNativeBuffer::write_host_float(
                &mut query_bytes,
                q_and_gate.dtype,
                outer_idx * q_width + idx,
                query_val,
            )?;
            let gate_val = HipNativeBuffer::read_host_float(
                q_and_gate.bytes.as_ref(),
                q_and_gate.dtype,
                qg_base + q_width + idx,
            )?;
            HipNativeBuffer::write_host_float(
                &mut gate_bytes,
                q_and_gate.dtype,
                outer_idx * q_width + idx,
                gate_val,
            )?;
        }
        for idx in 0..kv_width {
            let key_val = HipNativeBuffer::read_host_float(k_proj.bytes.as_ref(), k_proj.dtype, kv_base + idx)?;
            HipNativeBuffer::write_host_float(&mut key_bytes, k_proj.dtype, outer_idx * kv_width + idx, key_val)?;
            let head = idx / head_dim;
            let d = idx % head_dim;
            let value_val =
                HipNativeBuffer::read_host_float(v_proj.bytes.as_ref(), v_proj.dtype, kv_base + idx)?;
            let value_dst = (((b * num_kv_heads + head) * q_len + q) * head_dim) + d;
            HipNativeBuffer::write_host_float(&mut value_bytes, v_proj.dtype, value_dst, value_val)?;
        }
    }
    let query_pre = HipHostBuffer {
        bytes: query_bytes.into(),
        shape: vec![b_sz, q_len, num_heads, head_dim],
        dtype: q_and_gate.dtype,
        device: q_and_gate.device.clone(),
    };
    let key_pre = HipHostBuffer {
        bytes: key_bytes.into(),
        shape: vec![b_sz, q_len, num_kv_heads, head_dim],
        dtype: k_proj.dtype,
        device: k_proj.device.clone(),
    };
    let query_norm = query_pre.rms_norm(q_norm_weight, q_norm_eps, true)?;
    let key_norm = key_pre.rms_norm(k_norm_weight, k_norm_eps, true)?;
    let mut query_out =
        vec![0u8; HipNativeBuffer::byte_len(&[b_sz, num_heads, q_len, head_dim], query_norm.dtype)];
    let mut key_out =
        vec![0u8; HipNativeBuffer::byte_len(&[b_sz, num_kv_heads, q_len, head_dim], key_norm.dtype)];
    for idx in 0..HipNativeBuffer::elem_count(&query_norm.shape) {
        let d = idx % head_dim;
        let h = (idx / head_dim) % num_heads;
        let q = (idx / (head_dim * num_heads)) % q_len.max(1);
        let b = idx / (head_dim * num_heads * q_len.max(1));
        let dst = (((b * num_heads + h) * q_len + q) * head_dim) + d;
        let value = HipNativeBuffer::read_host_float(query_norm.bytes.as_ref(), query_norm.dtype, idx)?;
        HipNativeBuffer::write_host_float(&mut query_out, query_norm.dtype, dst, value)?;
    }
    for idx in 0..HipNativeBuffer::elem_count(&key_norm.shape) {
        let d = idx % head_dim;
        let h = (idx / head_dim) % num_kv_heads;
        let q = (idx / (head_dim * num_kv_heads)) % q_len.max(1);
        let b = idx / (head_dim * num_kv_heads * q_len.max(1));
        let dst = (((b * num_kv_heads + h) * q_len + q) * head_dim) + d;
        let value = HipNativeBuffer::read_host_float(key_norm.bytes.as_ref(), key_norm.dtype, idx)?;
        HipNativeBuffer::write_host_float(&mut key_out, key_norm.dtype, dst, value)?;
    }
    Ok((
        HipHostBuffer {
            bytes: query_out.into(),
            shape: vec![b_sz, num_heads, q_len, head_dim],
            dtype: query_norm.dtype,
            device: q_and_gate.device.clone(),
        },
        HipHostBuffer {
            bytes: gate_bytes.into(),
            shape: vec![b_sz, q_len, q_width],
            dtype: q_and_gate.dtype,
            device: q_and_gate.device.clone(),
        },
        HipHostBuffer {
            bytes: key_out.into(),
            shape: vec![b_sz, num_kv_heads, q_len, head_dim],
            dtype: key_norm.dtype,
            device: k_proj.device.clone(),
        },
        HipHostBuffer {
            bytes: value_bytes.into(),
            shape: vec![b_sz, num_kv_heads, q_len, head_dim],
            dtype: v_proj.dtype,
            device: v_proj.device.clone(),
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_full_attention_inputs_tensors_hip(
    q_and_gate: &HipTensor,
    k_proj: &HipTensor,
    v_proj: &HipTensor,
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
    if let (Some(q_and_gate), Some(k_proj), Some(v_proj)) = (
        q_and_gate.0 .0.direct_materialized_device_buffer(),
        k_proj.0 .0.direct_materialized_device_buffer(),
        v_proj.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (Some(q_and_gate_host), Some(k_proj_host), Some(v_proj_host)) = (
            q_and_gate.storage.as_host_buffer(),
            k_proj.storage.as_host_buffer(),
            v_proj.storage.as_host_buffer(),
        )
        {
            let (query_states, gate, key_states, value_states) =
                prepare_full_attention_inputs_host_buffers(
                    q_and_gate_host,
                    k_proj_host,
                    v_proj_host,
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
            return Ok((
                HipTensor::from_device_buffer(host_result_device_buffer(query_states)),
                HipTensor::from_device_buffer(host_result_device_buffer(gate)),
                HipTensor::from_device_buffer(host_result_device_buffer(key_states)),
                HipTensor::from_device_buffer(host_result_device_buffer(value_states)),
            ));
        }
        let q_and_gate = q_and_gate.reshape(vec![b_sz, q_len, num_heads, head_dim * 2])?;
        let last_dim = q_and_gate.dims().len() - 1;
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
    prepare_full_attention_inputs_tensors_hip(
        &q_and_gate,
        &k_proj,
        &v_proj,
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
        query_states.into_state_buffer()?,
        gate.into_state_buffer()?,
        key_states.into_state_buffer()?,
        value_states.into_state_buffer()?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_linear_attention_inputs_host_buffers(
    mixed_qkv: &HipHostBuffer,
    beta_raw: &HipHostBuffer,
    g: &HipHostBuffer,
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
) -> Result<(HipHostBuffer, HipHostBuffer, HipHostBuffer, HipHostBuffer, HipHostBuffer)> {
    if !HipNativeBuffer::supports_host_float_ops(mixed_qkv.dtype)
        || !HipNativeBuffer::supports_host_float_ops(beta_raw.dtype)
        || !HipNativeBuffer::supports_host_float_ops(g.dtype)
        || !HipNativeBuffer::supports_host_float_ops(compute_dtype)
    {
        candle_core::bail!(
            "linear attention host path unsupported for dtypes {:?}, {:?}, {:?} -> {:?}",
            mixed_qkv.dtype,
            beta_raw.dtype,
            g.dtype,
            compute_dtype
        );
    }
    if mixed_qkv.shape != [batch_size, seq_len, key_dim * 2 + value_dim] {
        candle_core::bail!(
            "linear attention mixed_qkv shape mismatch: expected {:?}, got {:?}",
            vec![batch_size, seq_len, key_dim * 2 + value_dim],
            mixed_qkv.shape
        );
    }
    let head_repeat = num_v_heads / num_k_heads;
    let out_k_heads = if repeat_kv_heads && head_repeat > 1 {
        num_k_heads * head_repeat
    } else {
        num_k_heads
    };
    let outer = batch_size.saturating_mul(seq_len);
    let source_stride = key_dim * 2 + value_dim;
    let query_shape = vec![batch_size, seq_len, out_k_heads, head_k_dim];
    let key_shape = vec![batch_size, seq_len, out_k_heads, head_k_dim];
    let value_shape = vec![batch_size, seq_len, num_v_heads, head_v_dim];
    let mut query_out = vec![0u8; HipNativeBuffer::byte_len(&query_shape, compute_dtype)];
    let mut key_out = vec![0u8; HipNativeBuffer::byte_len(&key_shape, compute_dtype)];
    let mut value_out = vec![0u8; HipNativeBuffer::byte_len(&value_shape, compute_dtype)];
    for outer_idx in 0..outer.max(1) {
        for head in 0..num_k_heads {
            let mut query_sum_sq = 0.0f64;
            let mut key_sum_sq = 0.0f64;
            for dim in 0..head_k_dim {
                let query_src_idx = outer_idx * source_stride + head * head_k_dim + dim;
                let key_src_idx = outer_idx * source_stride + key_dim + head * head_k_dim + dim;
                let query_val = HipNativeBuffer::read_host_float(
                    mixed_qkv.bytes.as_ref(),
                    mixed_qkv.dtype,
                    query_src_idx,
                )?;
                let key_val = HipNativeBuffer::read_host_float(
                    mixed_qkv.bytes.as_ref(),
                    mixed_qkv.dtype,
                    key_src_idx,
                )?;
                query_sum_sq += query_val * query_val;
                key_sum_sq += key_val * key_val;
            }
            let query_denom = (query_sum_sq + 1e-6).sqrt();
            let key_denom = (key_sum_sq + 1e-6).sqrt();
            let repeat_range = if repeat_kv_heads && head_repeat > 1 {
                (head * head_repeat)..((head + 1) * head_repeat)
            } else {
                head..(head + 1)
            };
            for dim in 0..head_k_dim {
                let query_src_idx = outer_idx * source_stride + head * head_k_dim + dim;
                let key_src_idx = outer_idx * source_stride + key_dim + head * head_k_dim + dim;
                let query_val = HipNativeBuffer::read_host_float(
                    mixed_qkv.bytes.as_ref(),
                    mixed_qkv.dtype,
                    query_src_idx,
                )? / query_denom;
                let key_val = HipNativeBuffer::read_host_float(
                    mixed_qkv.bytes.as_ref(),
                    mixed_qkv.dtype,
                    key_src_idx,
                )? / key_denom;
                for out_head in repeat_range.clone() {
                    let out_idx = (outer_idx * out_k_heads + out_head) * head_k_dim + dim;
                    HipNativeBuffer::write_host_float(&mut query_out, compute_dtype, out_idx, query_val)?;
                    HipNativeBuffer::write_host_float(&mut key_out, compute_dtype, out_idx, key_val)?;
                }
            }
        }
        for head in 0..num_v_heads {
            for dim in 0..head_v_dim {
                let src_idx = outer_idx * source_stride + key_dim * 2 + head * head_v_dim + dim;
                let out_idx = (outer_idx * num_v_heads + head) * head_v_dim + dim;
                let value =
                    HipNativeBuffer::read_host_float(mixed_qkv.bytes.as_ref(), mixed_qkv.dtype, src_idx)?;
                HipNativeBuffer::write_host_float(&mut value_out, compute_dtype, out_idx, value)?;
            }
        }
    }
    Ok((
        HipHostBuffer {
            bytes: query_out.into(),
            shape: query_shape,
            dtype: compute_dtype,
            device: mixed_qkv.device.clone(),
        },
        HipHostBuffer {
            bytes: key_out.into(),
            shape: key_shape,
            dtype: compute_dtype,
            device: mixed_qkv.device.clone(),
        },
        HipHostBuffer {
            bytes: value_out.into(),
            shape: value_shape,
            dtype: compute_dtype,
            device: mixed_qkv.device.clone(),
        },
        beta_raw.sigmoid()?.cast(compute_dtype)?,
        g.cast(compute_dtype)?,
    ))
}

fn dense_full_attention_fallback_host_buffers(
    query_states: &HipHostBuffer,
    key_states: &HipHostBuffer,
    value_states: &HipHostBuffer,
    attention_mask: Option<&HipHostBuffer>,
    scale: f64,
) -> Result<HipHostBuffer> {
    if query_states.dtype != key_states.dtype || query_states.dtype != value_states.dtype {
        candle_core::bail!(
            "dense fallback dtype mismatch: {:?}, {:?}, {:?}",
            query_states.dtype,
            key_states.dtype,
            value_states.dtype
        );
    }
    if !HipNativeBuffer::supports_host_float_ops(query_states.dtype) {
        candle_core::bail!("dense fallback unsupported for dtype {:?}", query_states.dtype);
    }
    if query_states.shape.len() != 4 || key_states.shape.len() != 4 || value_states.shape.len() != 4 {
        candle_core::bail!(
            "dense fallback expects rank-4 tensors, got {:?}, {:?}, {:?}",
            query_states.shape,
            key_states.shape,
            value_states.shape
        );
    }
    let (b, h, q_len, d) = (
        query_states.shape[0],
        query_states.shape[1],
        query_states.shape[2],
        query_states.shape[3],
    );
    let kv_len = key_states.shape[2];
    let value_d = value_states.shape[3];
    if key_states.shape[0] != b
        || key_states.shape[1] != h
        || key_states.shape[3] != d
        || value_states.shape[0] != b
        || value_states.shape[1] != h
        || value_states.shape[2] != kv_len
    {
        candle_core::bail!(
            "dense fallback shape mismatch: query={:?} key={:?} value={:?}",
            query_states.shape,
            key_states.shape,
            value_states.shape
        );
    }
    if let Some(mask) = attention_mask {
        let _ = HipNativeBuffer::broadcast_shape(
            &[b, h, q_len, kv_len],
            mask.shape.as_slice(),
            "dense full attention mask",
        )?;
    }
    let out_shape = vec![b, h, q_len, value_d];
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&out_shape, query_states.dtype)];
    let mut logits = vec![0.0f64; kv_len];
    for batch in 0..b {
        for head in 0..h {
            for q in 0..q_len {
                let mut max_logit = f64::NEG_INFINITY;
                for (k, slot) in logits.iter_mut().enumerate().take(kv_len) {
                    let mut dot = 0.0f64;
                    for dim in 0..d {
                        let q_idx = (((batch * h + head) * q_len + q) * d) + dim;
                        let k_idx = (((batch * h + head) * kv_len + k) * d) + dim;
                        let qv = HipNativeBuffer::read_host_float(
                            query_states.bytes.as_ref(),
                            query_states.dtype,
                            q_idx,
                        )?;
                        let kv = HipNativeBuffer::read_host_float(
                            key_states.bytes.as_ref(),
                            key_states.dtype,
                            k_idx,
                        )?;
                        dot += qv * kv;
                    }
                    let mut logit = dot * scale;
                    if let Some(mask) = attention_mask {
                        let attn_idx = (((batch * h + head) * q_len + q) * kv_len) + k;
                        let mask_idx = HipNativeBuffer::broadcast_elem_index(
                            attn_idx,
                            &[b, h, q_len, kv_len],
                            mask.shape.as_slice(),
                        );
                        logit += HipNativeBuffer::read_host_float(
                            mask.bytes.as_ref(),
                            mask.dtype,
                            mask_idx,
                        )?;
                    }
                    *slot = logit;
                    if logit > max_logit {
                        max_logit = logit;
                    }
                }
                let mut denom = 0.0f64;
                for slot in logits.iter_mut().take(kv_len) {
                    *slot = (*slot - max_logit).exp();
                    denom += *slot;
                }
                for value_idx in 0..value_d {
                    let mut acc = 0.0f64;
                    for (k, weight) in logits.iter().enumerate().take(kv_len) {
                        let v_idx = (((batch * h + head) * kv_len + k) * value_d) + value_idx;
                        let vv = HipNativeBuffer::read_host_float(
                            value_states.bytes.as_ref(),
                            value_states.dtype,
                            v_idx,
                        )?;
                        acc += (*weight / denom) * vv;
                    }
                    let out_idx = (((batch * h + head) * q_len + q) * value_d) + value_idx;
                    HipNativeBuffer::write_host_float(&mut out, query_states.dtype, out_idx, acc)?;
                }
            }
        }
    }
    Ok(HipHostBuffer {
        bytes: out.into(),
        shape: out_shape,
        dtype: query_states.dtype,
        device: query_states.device.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_linear_attention_inputs_tensors_hip(
    mixed_qkv: &HipTensor,
    beta_raw: &HipTensor,
    g: &HipTensor,
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
    if let (Some(mixed_qkv), Some(beta_raw), Some(g)) = (
        mixed_qkv.0 .0.direct_materialized_device_buffer(),
        beta_raw.0 .0.direct_materialized_device_buffer(),
        g.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (Some(mixed_qkv_host), Some(beta_raw_host), Some(g_host)) = (
            mixed_qkv.storage.as_host_buffer(),
            beta_raw.storage.as_host_buffer(),
            g.storage.as_host_buffer(),
        )
        {
            let (query, key, value, beta, g) = prepare_linear_attention_inputs_host_buffers(
                mixed_qkv_host,
                beta_raw_host,
                g_host,
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
            return Ok((
                HipTensor::from_device_buffer(host_result_device_buffer(query)),
                HipTensor::from_device_buffer(host_result_device_buffer(key)),
                HipTensor::from_device_buffer(host_result_device_buffer(value)),
                HipTensor::from_device_buffer(host_result_device_buffer(beta)),
                HipTensor::from_device_buffer(host_result_device_buffer(g)),
            ));
        }
        let last_dim = mixed_qkv.dims().len() - 1;
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
    prepare_linear_attention_inputs_tensors_hip(
        &mixed_qkv,
        &beta_raw,
        &g,
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

fn prepare_full_attention_output_hip(
    attn_output_hip: &HipTensor,
    gate_hip: &HipTensor,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<HipTensor> {
    if let (Some(attn_output), Some(gate)) = (
        attn_output_hip.0 .0.direct_materialized_device_buffer(),
        gate_hip.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (Some(attn_host), Some(gate_host)) = (
            attn_output.storage.as_host_buffer(),
            gate.storage.as_host_buffer(),
        )
        {
            return Ok(HipTensor::from_device_buffer(host_result_device_buffer(
                prepare_full_attention_output_host_buffer(
                    attn_host,
                    gate_host,
                    b_sz,
                    q_len,
                    attention_size,
                    hidden_dtype,
                )?,
            )));
        }
        return Ok(HipTensor::from_device_buffer(
            attn_output
                .transpose(1, 2)?
                .reshape(vec![b_sz, q_len, attention_size])?
                .to_dtype(hidden_dtype)?
                .broadcast_mul(&gate.sigmoid()?)?,
        ));
    }
    let attn_output = attn_output_hip
        .transpose(1, 2)?
        .reshape((b_sz, q_len, attention_size))?
        .to_dtype(hidden_dtype)?;
    let gate = gate_hip.sigmoid()?;
    attn_output.broadcast_mul(&gate)
}

pub(crate) fn prepare_full_attention_output(
    attn_output: &Tensor,
    gate: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    let attn_output_hip = HipTensor::from_scaffold_tensor(attn_output.clone());
    let gate_hip = HipTensor::from_state_buffer(gate);
    prepare_full_attention_output_hip(
        &attn_output_hip,
        &gate_hip,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )?
    .into_state_buffer()
}

pub(crate) fn prepare_full_attention_output_buffer(
    attn_output: &StateBuffer,
    gate: &StateBuffer,
    b_sz: usize,
    q_len: usize,
    attention_size: usize,
    hidden_dtype: DType,
) -> Result<StateBuffer> {
    let attn_output_hip = HipTensor::from_state_buffer(attn_output);
    let gate_hip = HipTensor::from_state_buffer(gate);
    prepare_full_attention_output_hip(
        &attn_output_hip,
        &gate_hip,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )?
    .into_state_buffer()
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
                prev_k.0 .0.direct_materialized_device_buffer(),
                prev_v.0 .0.direct_materialized_device_buffer(),
                key_states.0 .0.direct_materialized_device_buffer(),
                value_states.0 .0.direct_materialized_device_buffer(),
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

pub(crate) fn append_full_attention_kv_buffers(
    prev_k: Option<&StateBuffer>,
    prev_v: Option<&StateBuffer>,
    key_states: &Tensor,
    value_states: &Tensor,
) -> Result<(StateBuffer, StateBuffer)> {
    let (key_states, value_states) =
        append_full_attention_kv_hip(prev_k, prev_v, key_states, value_states)?;
    Ok((key_states.into_state_buffer()?, value_states.into_state_buffer()?))
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
        query_states.0 .0.direct_materialized_device_buffer(),
        key_states.0 .0.direct_materialized_device_buffer(),
        value_states.0 .0.direct_materialized_device_buffer(),
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

pub(crate) fn prepare_full_attention_kernel_inputs_with_buffer_kv(
    query_states: &StateBuffer,
    key_states: &StateBuffer,
    value_states: &StateBuffer,
) -> Result<(Tensor, Tensor, Tensor)> {
    let query_states = HipTensor::from_state_buffer(query_states);
    let key_states = HipTensor::from_state_buffer(key_states);
    let value_states = HipTensor::from_state_buffer(value_states);
    let (query_states, key_states, value_states) = if let (
        Some(query_device),
        Some(key_device),
        Some(value_device),
    ) = (
        query_states.0 .0.direct_materialized_device_buffer(),
        key_states.0 .0.direct_materialized_device_buffer(),
        value_states.0 .0.direct_materialized_device_buffer(),
    ) {
        (
            HipTensor::from_device_buffer(query_device.contiguous()?),
            HipTensor::from_device_buffer(key_device.contiguous()?),
            HipTensor::from_device_buffer(value_device.contiguous()?),
        )
    } else {
        (
            query_states.contiguous()?,
            key_states.contiguous()?,
            value_states.contiguous()?,
        )
    };
    Ok((
        query_states.into_tensor(),
        key_states.into_tensor(),
        value_states.into_tensor(),
    ))
}

pub(crate) fn prepare_full_attention_kernel_input_buffers_with_buffer_kv(
    query_states: &StateBuffer,
    key_states: &StateBuffer,
    value_states: &StateBuffer,
) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
    let query_states = HipTensor::from_state_buffer(query_states);
    let key_states = HipTensor::from_state_buffer(key_states);
    let value_states = HipTensor::from_state_buffer(value_states);
    let (query_states, key_states, value_states) = if let (
        Some(query_device),
        Some(key_device),
        Some(value_device),
    ) = (
        query_states.0 .0.direct_materialized_device_buffer(),
        key_states.0 .0.direct_materialized_device_buffer(),
        value_states.0 .0.direct_materialized_device_buffer(),
    ) {
        (
            HipTensor::from_device_buffer(query_device.contiguous()?),
            HipTensor::from_device_buffer(key_device.contiguous()?),
            HipTensor::from_device_buffer(value_device.contiguous()?),
        )
    } else {
        (
            query_states.contiguous()?,
            key_states.contiguous()?,
            value_states.contiguous()?,
        )
    };
    Ok((
        query_states.into_state_buffer()?,
        key_states.into_state_buffer()?,
        value_states.into_state_buffer()?,
    ))
}

pub(crate) fn rope_buffer(xs: &StateBuffer, cos: &Tensor, sin: &Tensor) -> Result<StateBuffer> {
    rope_hip(&HipTensor::from_state_buffer(xs), cos, sin)?.into_state_buffer()
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
        query_states.0 .0.direct_materialized_device_buffer(),
        key_states.0 .0.direct_materialized_device_buffer(),
        value_states.0 .0.direct_materialized_device_buffer(),
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

fn dense_full_attention_fallback_tensors_hip(
    query_states_hip: &HipTensor,
    key_states_hip: &HipTensor,
    value_states_hip: &HipTensor,
    attention_mask: Option<&HipTensor>,
    scale: f64,
) -> Result<HipTensor> {
    if let (Some(query_host), Some(key_host), Some(value_host)) = (
        query_states_hip.try_host_buffer()?,
        key_states_hip.try_host_buffer()?,
        value_states_hip.try_host_buffer()?,
    ) {
        let mask_host = attention_mask
            .map(|mask| mask.try_host_buffer())
            .transpose()?
            .flatten();
        return Ok(HipTensor::from_device_buffer(host_result_device_buffer(
            dense_full_attention_fallback_host_buffers(
                &query_host,
                &key_host,
                &value_host,
                mask_host.as_ref(),
                scale,
            )?,
        )));
    }
    if let (Some(query_states_f), Some(key_states_f), Some(value_states_f), mask_device) = (
        query_states_hip.0 .0.direct_materialized_device_buffer(),
        key_states_hip.0 .0.direct_materialized_device_buffer(),
        value_states_hip.0 .0.direct_materialized_device_buffer(),
        attention_mask
            .as_ref()
            .and_then(|mask| mask.0 .0.direct_materialized_device_buffer()),
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
    if let Some(mask) = attention_mask {
        attn_weights = attn_weights.broadcast_add(&mask)?;
    }
    let attn_weights = softmax_last_dim_hip(&attn_weights)?;
    attn_weights.matmul(&value_states_hip)
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
    dense_full_attention_fallback_tensors_hip(
        &query_states_hip,
        &key_states_hip,
        &value_states_hip,
        mask_hip.as_ref(),
        scale,
    )
}

fn softmax_last_dim_hip(xs: &HipTensor) -> Result<HipTensor> {
    let max = xs.max_keepdim(candle_core::D::Minus1)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(candle_core::D::Minus1)?;
    num.broadcast_div(&den)
}

fn softmax_last_dim_device_hip(xs: &HipDeviceBuffer) -> Result<HipDeviceBuffer> {
    let last_dim = xs.rank() - 1;
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

#[allow(clippy::too_many_arguments)]
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
    let attn_output = dense_full_attention_fallback_hip(
        query_states_f,
        key_states_f,
        value_states_f,
        attention_mask,
        scale,
    )?;
    prepare_full_attention_output_buffer(
        &attn_output.into_state_buffer()?,
        gate,
        b_sz,
        q_len,
        attention_size,
        hidden_dtype,
    )
}

pub(crate) fn zeros(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<HipTensor> {
    Ok(HipTensor(HipStorage::zeros(dims, dtype, device)?))
}

pub(crate) fn zeros_state(dims: Vec<usize>, dtype: DType, device: &Device) -> Result<StateBuffer> {
    zeros(dims, dtype, device)?.into_state_buffer()
}

pub(crate) fn copy_state_into_scratch(
    src: &StateBuffer,
    scratch: &StateBuffer,
) -> Result<StateBuffer> {
    if src.dtype() != scratch.dtype() {
        candle_core::bail!(
            "HIP scratch dtype mismatch: src={:?} scratch={:?}",
            src.dtype(),
            scratch.dtype(),
        );
    }
    if src.tensor().dims() != scratch.tensor().dims() {
        candle_core::bail!(
            "HIP scratch shape mismatch: src={:?} scratch={:?}",
            src.tensor().dims(),
            scratch.tensor().dims(),
        );
    }
    let src_hip = HipTensor::from_state_buffer(src);
    let scratch_hip = HipTensor::from_state_buffer(scratch);
    if let (Some(src_buffer), Some(dst_buffer)) = (
        src_hip.0 .0.direct_materialized_device_buffer(),
        scratch_hip.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (
            Some((src_ordinal, src_dtype, src_shape, src_ptr)),
            Some((dst_ordinal, dst_dtype, dst_shape, dst_ptr)),
        ) = (
            src_buffer.standard_contiguous_launch_spec()?,
            dst_buffer.standard_contiguous_launch_spec()?,
        ) {
            if src_ordinal == dst_ordinal && src_dtype == dst_dtype && src_shape == dst_shape {
                hip::copy_device_to_device(
                    dst_ordinal,
                    dst_ptr as *mut c_void,
                    src_ptr,
                    HipNativeBuffer::byte_len(&src_shape, src_dtype),
                )?;
                return scratch_hip.into_state_buffer();
            }
        }
    }
    Ok(src.clone())
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
        let uploaded = buffer.upload_to_device_buffer()?;
        assert!(!uploaded.is_materialized());
        let roundtrip = HipTensor::from_device_buffer(uploaded);
        assert!(matches!(roundtrip.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(roundtrip)?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_views_roundtrip_to_host_without_materialization() -> Result<()> {
        let buffer = HipHostBuffer {
            bytes: [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                .into(),
            shape: vec![2, 2],
            dtype: DType::F32,
            device: Device::Cpu,
        };
        let uploaded = buffer
            .upload_to_device_buffer()?
            .reshape(vec![1, 4])?
            .narrow(1, 1, 2)?;
        assert!(!uploaded.is_materialized());
        let host = uploaded.try_host_buffer()?.expect("pending upload should stay host-extractable");
        assert_eq!(host.shape(), &[1, 2]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_cat_stays_pending() -> Result<()> {
        let lhs = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer");
        let rhs = host_f32_tensor(&[1, 2], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer");
        let lhs = lhs.upload_to_device_buffer()?;
        let rhs = rhs.upload_to_device_buffer()?;
        let out = HipDeviceBuffer::cat(&[&lhs, &rhs], 1)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[1, 4]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_pad_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.pad_with_zeros(1, 1, 1)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[1, 4]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![0.0, 1.0, 2.0, 0.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_cast_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.to_dtype(DType::F16)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.dtype(), DType::F16);
        let roundtrip = HipTensor::from_device_buffer(out);
        assert_eq!(values_f32(roundtrip.to_dtype(DType::F32)?)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_exp_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[0.0, 1.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.exp()?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[1, 2]);
        let vals = host_buffer_values_f32(&host)?;
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - std::f32::consts::E).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_broadcast_add_stays_pending() -> Result<()> {
        let lhs = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let rhs = host_f32_tensor(&[1, 2], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = lhs.broadcast_add(&rhs)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host_buffer_values_f32(&host)?, vec![4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_recip_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[2.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.recip()?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host_buffer_values_f32(&host)?, vec![0.5, 0.25]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_sum_keepdim_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.sum_keepdim(1)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[2, 1]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_max_keepdim_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.max_keepdim(1)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[2, 1]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_sigmoid_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[0.0, 1.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.sigmoid()?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        assert!((vals[0] - 0.5).abs() < 1e-5);
        assert!((vals[1] - 0.7310586).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_mul_scalar_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[1.5, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.mul_scalar(2.0)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host_buffer_values_f32(&host)?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_l2norm_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[1, 2], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.l2norm(0.0)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        assert!((vals[0] - 0.6).abs() < 1e-5);
        assert!((vals[1] - 0.8).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_rms_norm_stays_pending() -> Result<()> {
        let device = Device::Cpu;
        let src = host_f32_tensor(&[1, 2], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let weight = Tensor::from_vec(vec![1f32, 1.0], (2,), &device)?;
        let out = src.rms_norm(&weight, 0.0, false)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        let denom = ((9.0 + 16.0) / 2.0f32).sqrt();
        assert!((vals[0] - (3.0 / denom)).abs() < 1e-5);
        assert!((vals[1] - (4.0 / denom)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_rms_norm_gated_stays_pending() -> Result<()> {
        let device = Device::Cpu;
        let hidden = host_f32_tensor(&[1, 2], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let gate = host_f32_tensor(&[1, 2], &[0.5, -1.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let weight = Tensor::from_vec(vec![1f32, 1.0], (2,), &device)?;
        let out = hidden.rms_norm_gated(&gate, &weight, 0.0)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        let denom = ((9.0 + 16.0) / 2.0f32).sqrt();
        let silu0 = 0.5f32 / (1.0 + (-0.5f32).exp());
        let silu1 = -1.0f32 / (1.0 + 1.0f32.exp());
        assert!((vals[0] - ((3.0 / denom) * silu0)).abs() < 1e-5);
        assert!((vals[1] - ((4.0 / denom) * silu1)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_swiglu_mul_stays_pending() -> Result<()> {
        let gate = host_f32_tensor(&[1, 2], &[0.5, -1.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let up = host_f32_tensor(&[1, 2], &[2.0, 3.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = gate.swiglu_mul(&up)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        let silu0 = 0.5f32 / (1.0 + (-0.5f32).exp());
        let silu1 = -1.0f32 / (1.0 + 1.0f32.exp());
        assert!((vals[0] - (silu0 * 2.0)).abs() < 1e-5);
        assert!((vals[1] - (silu1 * 3.0)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_value_decay_stays_pending() -> Result<()> {
        let a = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let dt_bias = host_f32_tensor(&[1, 2], &[0.25, -0.5])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let a_log_exp = host_f32_tensor(&[1, 2], &[0.5, 1.5])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = a.value_decay(&dt_bias, &a_log_exp)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        let vals = host_buffer_values_f32(&host)?;
        let expected0 = -(((1.0f32 + 0.25).exp() + 1.0).ln() * 0.5);
        let expected1 = -(((2.0f32 - 0.5).exp() + 1.0).ln() * 1.5);
        assert!((vals[0] - expected0).abs() < 1e-5);
        assert!((vals[1] - expected1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_cumsum_last_dim_stays_pending() -> Result<()> {
        let src = host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = src.cumsum_last_dim()?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[2, 2]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 3.0, 3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_matmul_stays_pending() -> Result<()> {
        let lhs = host_f32_tensor(&[1, 2], &[1.0, 2.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let rhs = host_f32_tensor(&[2, 1], &[3.0, 4.0])
            .try_host_buffer()?
            .expect("host buffer")
            .upload_to_device_buffer()?;
        let out = lhs.matmul(&rhs)?;
        assert!(!out.is_materialized());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host.shape(), &[1, 1]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![11.0]);
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
    fn device_leaf_cpu_tensor_uses_host_storage() -> Result<()> {
        let device = Device::Cpu;
        let buffer = HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        let host = buffer.try_host_buffer()?.expect("host-backed storage");
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_exp_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let buffer = HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![0f32, 1.0],
            (2,),
            &device,
        )?);

        let out = buffer.exp()?;

        assert!(matches!(out.storage, HipDeviceStorage::HostBuffer(_)));
        let host = out.try_host_buffer()?.expect("host-backed storage");
        let values = host_buffer_values_f32(&host)?;
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - std::f32::consts::E).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_matmul_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);
        let rhs = HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![5f32, 6.0, 7.0, 8.0],
            (2, 2),
            &device,
        )?);

        let out = lhs.matmul(&rhs)?;

        assert!(matches!(out.storage, HipDeviceStorage::HostBuffer(_)));
        let host = out.try_host_buffer()?.expect("host-backed storage");
        assert_eq!(host_buffer_values_f32(&host)?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_view_materialization_stays_host_side() -> Result<()> {
        let device = Device::Cpu;
        let buffer = HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?)
        .transpose(0, 1)?
        .narrow(1, 0, 1)?;

        let host = buffer
            .materialize_host_buffer_with_views()?
            .expect("host-side view materialization");

        assert_eq!(host.shape(), &[2, 1]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_expand_materialization_stays_host_side() -> Result<()> {
        let device = Device::Cpu;
        let buffer = HipDeviceBuffer::from_tensor(Tensor::from_vec(vec![1f32, 2.0], (1, 2), &device)?)
            .expand(vec![3, 2])?;

        let host = buffer
            .materialize_host_buffer_with_views()?
            .expect("host-side expand materialization");

        assert_eq!(host.shape(), &[3, 2]);
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_view_ops_stay_lazy() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let tensor = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(tensor))
            .reshape((1, 2, 2))?
            .transpose(0, 1)?
            .narrow(1, 0, 1)?;

        let buffer = tensor
            .0
            .0
            .direct_device_buffer()
            .expect("device-backed view expected");
        assert!(buffer.has_pending_views());
        assert_eq!(buffer.dims(), &[2, 1, 2]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_repeat_heads_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2, 1), &device)?;
        let tensor =
            repeat_heads_hip(&HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(tensor)), 2)?;
        let buffer = tensor
            .0
            .0
            .direct_device_buffer()
            .expect("device-backed repeat expected");
        assert!(buffer.has_pending_views());
        assert_eq!(buffer.dims(), &[1, 2, 4, 1]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_repeat_kv_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2, 1), &device)?;
        let tensor =
            repeat_kv_hip(&HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(tensor)), 2)?;
        let buffer = tensor
            .0
            .0
            .direct_device_buffer()
            .expect("device-backed repeat expected");
        assert!(buffer.has_pending_views());
        assert_eq!(buffer.dims(), &[1, 4, 2, 1]);
        assert_eq!(values_f32(tensor)?, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_state_scan_chunk_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(
            (0..8).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 2, 2, 2),
            &device,
        )?;
        let tensor =
            HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(tensor)).select(1, 1)?;
        let buffer = tensor
            .0
            .0
            .direct_device_buffer()
            .expect("device-backed select expected");
        assert!(buffer.has_pending_views());
        assert_eq!(buffer.dims(), &[1, 2, 2]);
        assert_eq!(values_f32(tensor)?, vec![4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_cat_materializes_logical_views_correctly() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;
        let rhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![5f32, 6.0, 7.0, 8.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;
        let out = HipTensor::cat(&[&lhs, &rhs], 1)?;
        assert!(!matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_cat_of_materialized_buffers_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0],
            (1, 2),
            &device,
        )?));
        let rhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![3f32, 4.0],
            (1, 2),
            &device,
        )?));
        let out = HipTensor::cat(&[&lhs, &rhs], 0)?;
        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pad_materializes_logical_views_correctly() -> Result<()> {
        let device = Device::Cpu;
        let tensor = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?
        .pad_with_zeros(1, 1, 0)?;
        assert!(!matches!(tensor.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(tensor)?, vec![0.0, 1.0, 3.0, 0.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pad_of_materialized_buffer_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let tensor = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?))
        .pad_with_zeros(1, 1, 0)?;
        assert!(matches!(tensor.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(tensor)?, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
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
    fn device_leaf_host_storage_prepare_depthwise_conv_input_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let prev_state = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![7f32, 8.0],
            (1, 1, 2),
            &device,
        )?))
        .into_state_buffer()?;
        let mixed_qkv = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![9f32],
            (1, 1, 1),
            &device,
        )?));

        let (prepared, next_state) =
            prepare_depthwise_conv_input_hip(Some(&prev_state), &mixed_qkv.into_tensor(), 3)?;

        let prepared_buffer = prepared
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(prepared_buffer.storage, HipDeviceStorage::HostBuffer(_)));
        let next_state = next_state.expect("next state");
        let next_state_buffer = next_state
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(next_state_buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(prepared)?, vec![7.0, 8.0, 9.0]);
        assert_eq!(values_f32(next_state)?, vec![8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_update_depthwise_conv_state_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let prev_state = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![7f32, 8.0],
            (1, 1, 2),
            &device,
        )?))
        .into_state_buffer()?;
        let mixed_qkv = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![9f32],
            (1, 1, 1),
            &device,
        )?));

        let state =
            update_depthwise_conv_state_hip(Some(&prev_state), &mixed_qkv.into_tensor(), 3)?
                .expect("state");

        let buffer = state
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(state)?, vec![8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_pack_delta_state_scan_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let weighted_key_scan = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0], (1, 1, 1, 2), &device)?,
        ));
        let k_cumdecay_scan = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![3f32], (1, 1, 1, 1), &device)?,
        ));
        let state_decay_feature = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![4f32], (1, 1, 1, 1), &device)?,
        ));

        let packed = pack_delta_state_scan_hip(
            &weighted_key_scan.into_tensor(),
            &k_cumdecay_scan.into_tensor(),
            &state_decay_feature.into_tensor(),
        )?;

        let buffer = packed
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
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
    fn device_leaf_host_storage_unpack_linear_decode_output_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let fused = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 4),
            &device,
        )?))
        .into_state_buffer()?;

        let (core_attn_out, recurrent_state) =
            unpack_linear_decode_output_hip(&fused, 1, 1, 2, 1, 1, 2)?;

        for tensor in [&core_attn_out, &recurrent_state] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
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
    fn device_leaf_host_storage_dense_full_attention_fallback_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let query_states = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 0.0],
            (1, 1, 1, 2),
            &device,
        )?));
        let key_states = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 0.0],
            (1, 1, 1, 2),
            &device,
        )?));
        let value_states = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![7f32, 8.0],
            (1, 1, 1, 2),
            &device,
        )?));

        let out = dense_full_attention_fallback_tensors_hip(
            &query_states,
            &key_states,
            &value_states,
            None,
            1.0,
        )?;

        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(out)?, vec![7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_dense_full_attention_fallback_stays_host_extractable() -> Result<()> {
        let query_states = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1, 2], &[1.0, 0.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let key_states = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1, 2], &[1.0, 0.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let value_states = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1, 2], &[7.0, 8.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );

        let out = dense_full_attention_fallback_tensors_hip(
            &query_states,
            &key_states,
            &value_states,
            None,
            1.0,
        )?;

        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host_buffer_values_f32(&host)?, vec![7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_prepare_full_attention_output_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let attn_output = Tensor::from_vec(vec![2f32, 4.0], (1, 1, 1, 2), &device)?;
        let gate = StateBuffer::from_tensor(Tensor::from_vec(vec![0f32, 0.0], (1, 1, 2), &device)?)?;

        let out = prepare_full_attention_output(&attn_output, &gate, 1, 1, 2, DType::F32)?;
        let out = HipTensor::from_state_buffer(&out);

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_prepare_full_attention_output_stays_pending() -> Result<()> {
        let attn_output = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1, 2], &[2.0, 4.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let gate = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 2], &[0.0, 0.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );

        let out = prepare_full_attention_output_hip(&attn_output, &gate, 1, 1, 2, DType::F32)?;

        assert!(out.try_host_buffer()?.is_some());
        let host = out.try_host_buffer()?.expect("pending upload host bytes");
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_prepare_full_attention_output_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let attn_output = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![2f32, 4.0], (1, 1, 1, 2), &device)?,
        ));
        let gate = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![0f32, 0.0], (1, 1, 2), &device)?,
        ));

        let out = prepare_full_attention_output_hip(&attn_output, &gate, 1, 1, 2, DType::F32)?;
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");

        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        let host = out.try_host_buffer()?.expect("host-backed output");
        assert_eq!(host_buffer_values_f32(&host)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_prepare_full_attention_output_reorders_heads_correctly() -> Result<()> {
        let device = Device::Cpu;
        let attn_output = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2, 1), &device)?,
        ));
        let gate = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![0f32, 0.0, 0.0, 0.0], (1, 2, 2), &device)?,
        ));

        let out = prepare_full_attention_output_hip(&attn_output, &gate, 1, 2, 2, DType::F32)?;

        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        let host = out.try_host_buffer()?.expect("host-backed output");
        assert_eq!(host_buffer_values_f32(&host)?, vec![0.5, 1.5, 1.0, 2.0]);
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

        assert_eq!(query_states.0.shape(), vec![1, 1, 1, 2]);
        assert_eq!(gate.0.shape(), vec![1, 1, 2]);
        assert_eq!(key_states.0.shape(), vec![1, 1, 1, 2]);
        assert_eq!(value_states.0.shape(), vec![1, 1, 1, 2]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_prepare_full_attention_inputs_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let q_and_gate = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 4), &device)?,
        ));
        let k_proj = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![5f32, 6.0], (1, 1, 2), &device)?,
        ));
        let v_proj = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![7f32, 8.0], (1, 1, 2), &device)?,
        ));
        let q_norm_weight = Tensor::ones((2,), DType::F32, &device)?;
        let k_norm_weight = Tensor::ones((2,), DType::F32, &device)?;

        let (query_states, gate, key_states, value_states) = prepare_full_attention_inputs_tensors_hip(
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

        for tensor in [&query_states, &gate, &key_states, &value_states] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
        assert_eq!(values_f32(gate)?, vec![3.0, 4.0]);
        assert_eq!(values_f32(value_states)?, vec![7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_prepare_full_attention_inputs_stays_host_extractable() -> Result<()> {
        let device = Device::Cpu;
        let q_and_gate = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let k_proj = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 2], &[5.0, 6.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let v_proj = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 2], &[7.0, 8.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let q_norm_weight = Tensor::ones((2,), DType::F32, &device)?;
        let k_norm_weight = Tensor::ones((2,), DType::F32, &device)?;

        let (query_states, gate, key_states, value_states) = prepare_full_attention_inputs_tensors_hip(
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

        assert!(gate.try_host_buffer()?.is_some());
        assert!(value_states.try_host_buffer()?.is_some());
        assert_eq!(query_states.0.shape(), vec![1, 1, 1, 2]);
        assert_eq!(gate.0.shape(), vec![1, 1, 2]);
        assert_eq!(key_states.0.shape(), vec![1, 1, 1, 2]);
        assert_eq!(value_states.0.shape(), vec![1, 1, 1, 2]);
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
    fn device_leaf_pending_upload_delta_chunk_recurrent_read_stays_host_extractable() -> Result<()> {
        let prev_state = HipTensor::from_device_buffer(
            host_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let k_cumdecay_chunk = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 2], &[1.0, 0.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let q_state_chunk = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 2], &[0.0, 1.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let value_chunk = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 2], &[10.0, 20.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );

        let (v_new, attn_inter) = delta_chunk_recurrent_read_tensors_hip(
            &prev_state,
            &k_cumdecay_chunk,
            &q_state_chunk,
            &value_chunk,
        )?;

        assert_eq!(host_buffer_values_f32(&v_new.try_host_buffer()?.expect("host"))?, vec![9.0, 18.0]);
        assert_eq!(host_buffer_values_f32(&attn_inter.try_host_buffer()?.expect("host"))?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_mix_chunk_attention_stays_host_extractable() -> Result<()> {
        let attn = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1], &[2.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let attn_inter = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 2], &[1.0, 2.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let value_chunk = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 2], &[3.0, 4.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );

        let mixed = mix_chunk_attention_tensors_hip(&attn, &attn_inter, &value_chunk)?;

        assert_eq!(host_buffer_values_f32(&mixed.try_host_buffer()?.expect("host"))?, vec![7.0, 10.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_delta_chunk_recurrent_read_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let prev_state = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &device)?,
        ));
        let k_cumdecay_chunk = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 0.0], (1, 2), &device)?,
        ));
        let q_state_chunk = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![0f32, 1.0], (1, 2), &device)?,
        ));
        let value_chunk = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![10f32, 20.0], (1, 2), &device)?,
        ));

        let (v_new, attn_inter) = delta_chunk_recurrent_read_tensors_hip(
            &prev_state,
            &k_cumdecay_chunk,
            &q_state_chunk,
            &value_chunk,
        )?;

        for tensor in [&v_new, &attn_inter] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
        assert_eq!(host_buffer_values_f32(&v_new.try_host_buffer()?.expect("host"))?, vec![9.0, 18.0]);
        assert_eq!(host_buffer_values_f32(&attn_inter.try_host_buffer()?.expect("host"))?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_mix_chunk_attention_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let attn = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![2f32], (1, 1), &device)?,
        ));
        let attn_inter = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0], (1, 2), &device)?,
        ));
        let value_chunk = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![3f32, 4.0], (1, 2), &device)?,
        ));

        let mixed = mix_chunk_attention_tensors_hip(&attn, &attn_inter, &value_chunk)?;

        let buffer = mixed
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(host_buffer_values_f32(&mixed.try_host_buffer()?.expect("host"))?, vec![7.0, 10.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_delta_state_update_stays_host_extractable() -> Result<()> {
        let device = Device::Cpu;
        let prev_state_scaled =
            HipTensor::from_scaffold_tensor(Tensor::from_vec(vec![1f32, 2.0], (1, 1, 2), &device)?);
        let weighted_key =
            HipTensor::from_scaffold_tensor(Tensor::from_vec(vec![3f32, 4.0], (1, 1, 2), &device)?);
        let value = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![5f32, 6.0],
            (1, 1, 2),
            &device,
        )?));

        let out =
            delta_state_update_tensors_hip(&prev_state_scaled, &weighted_key, &value, false)?;

        assert_eq!(
            host_buffer_values_f32(&out.try_host_buffer()?.expect("host"))?,
            vec![16.0, 20.0, 21.0, 26.0]
        );
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_unpack_linear_prefill_output_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let fused = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            (1, 7),
            &device,
        )?))
        .into_state_buffer()?;

        let (mixed_qkv, g, conv_state) =
            unpack_linear_prefill_output_hip(&fused, 1, 1, 2, 1, 2)?;

        for tensor in [&mixed_qkv, &g, &conv_state] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
        assert_eq!(values_f32(mixed_qkv)?, vec![1.0, 2.0]);
        assert_eq!(values_f32(g)?, vec![3.0]);
        assert_eq!(values_f32(conv_state)?, vec![4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_unpack_scan_fused_output_and_state_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let fused = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 2, 2),
            &device,
        )?))
        .into_state_buffer()?;

        let (output, recurrent_state) = unpack_scan_fused_output_and_state_hip(
            &fused,
            1,
            1,
            1,
            1,
            2,
            1,
            DType::F32,
        )?;

        for tensor in [&output, &recurrent_state] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
        assert_eq!(values_f32(output)?, vec![1.0, 2.0]);
        assert_eq!(values_f32(recurrent_state)?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_unpack_chunk_fused_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let fused = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0, 5.0],
            (1, 5),
            &device,
        )?))
        .into_state_buffer()?;

        let (attn, local, q_state) = unpack_chunk_fused_hip(&fused, 2, 1)?;

        for tensor in [&attn, &local, &q_state] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
        assert_eq!(values_f32(attn)?, vec![1.0, 2.0]);
        assert_eq!(values_f32(local)?, vec![3.0, 4.0]);
        assert_eq!(values_f32(q_state)?, vec![5.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_pack_delta_chunk_fused_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let weighted_key = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0], (1, 1, 2), &device)?,
        ));
        let k_cumdecay = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![3f32], (1, 1, 1), &device)?,
        ));
        let q_state = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![4f32], (1, 1, 1), &device)?,
        ));
        let state_decay = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![5f32], (1, 1, 1), &device)?,
        ));

        let packed = pack_delta_chunk_fused_hip(
            &weighted_key.into_tensor(),
            &k_cumdecay.into_tensor(),
            &q_state.into_tensor(),
            &state_decay.into_tensor(),
        )?;

        let buffer = packed
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(packed)?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_state_scan_chunk_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let state_scan = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 2, 2),
            &device,
        )?))
        .into_state_buffer()?;

        let chunk = state_scan_chunk(&state_scan, 1)?;
        let chunk = HipTensor::from_state_buffer(&chunk);

        let buffer = chunk
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(chunk)?, vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_state_scan_next_chunk_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let state_scan = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 2, 2),
            &device,
        )?))
        .into_state_buffer()?;

        let chunk = state_scan_next_chunk(&state_scan, 1)?;
        let chunk = HipTensor::from_state_buffer(&chunk);

        let buffer = chunk
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(chunk)?, vec![3.0, 4.0]);
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
    fn device_leaf_pending_upload_prepare_linear_attention_inputs_stays_host_extractable() -> Result<()> {
        let mixed_qkv = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let beta_raw = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1], &[0.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let g = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1], &[1.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );

        let (query, key, value, beta, g) = prepare_linear_attention_inputs_tensors_hip(
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

        let query_vals = values_f32(query.clone())?;
        let key_vals = values_f32(key.clone())?;
        let value_vals = values_f32(value.clone())?;
        let beta_vals = values_f32(beta.clone())?;
        let g_vals = values_f32(g.clone())?;
        let expected_query = [1.0 / 5.0f32.sqrt(), 2.0 / 5.0f32.sqrt()];
        let expected_key = [3.0 / 25.0f32.sqrt(), 4.0 / 25.0f32.sqrt()];
        for (got, expected) in query_vals.iter().zip(expected_query.iter()) {
            assert!((got - expected).abs() < 1e-5);
        }
        for (got, expected) in key_vals.iter().zip(expected_key.iter()) {
            assert!((got - expected).abs() < 1e-5);
        }
        assert_eq!(value_vals, vec![5.0, 6.0]);
        assert_eq!(beta_vals, vec![0.5]);
        assert_eq!(g_vals, vec![1.0]);
        assert!(query.try_host_buffer()?.is_some());
        assert!(key.try_host_buffer()?.is_some());
        assert!(value.try_host_buffer()?.is_some());
        assert!(beta.try_host_buffer()?.is_some());
        assert!(g.try_host_buffer()?.is_some());
        Ok(())
    }

    #[test]
    fn device_leaf_host_storage_prepare_linear_attention_inputs_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let mixed_qkv = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 1, 6), &device)?,
        ));
        let beta_raw = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![0f32], (1, 1, 1), &device)?,
        ));
        let g = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(
            Tensor::from_vec(vec![1f32], (1, 1, 1), &device)?,
        ));

        let (query, key, value, beta, g) = prepare_linear_attention_inputs_tensors_hip(
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

        let query_vals = values_f32(query.clone())?;
        let key_vals = values_f32(key.clone())?;
        let value_vals = values_f32(value.clone())?;
        let beta_vals = values_f32(beta.clone())?;
        let g_vals = values_f32(g.clone())?;
        let expected_query = [1.0 / 5.0f32.sqrt(), 2.0 / 5.0f32.sqrt()];
        let expected_key = [3.0 / 25.0f32.sqrt(), 4.0 / 25.0f32.sqrt()];
        for (got, expected) in query_vals.iter().zip(expected_query.iter()) {
            assert!((got - expected).abs() < 1e-5);
        }
        for (got, expected) in key_vals.iter().zip(expected_key.iter()) {
            assert!((got - expected).abs() < 1e-5);
        }
        assert_eq!(value_vals, vec![5.0, 6.0]);
        assert_eq!(beta_vals, vec![0.5]);
        assert_eq!(g_vals, vec![1.0]);
        for tensor in [&query, &key, &value, &beta, &g] {
            assert!(tensor.try_host_buffer()?.is_some());
            if let Some(buffer) = tensor.0 .0.direct_materialized_device_buffer() {
                assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
            }
        }
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
    fn device_leaf_generic_matmul_of_views_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;
        let rhs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![5f32, 6.0, 7.0, 8.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;

        let out = lhs.matmul(&rhs)?;

        assert!(!matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![23.0, 31.0, 34.0, 46.0]);
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
    fn device_leaf_generic_sigmoid_of_view_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![0f32, 1.0, -1.0, 2.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;

        let out = xs.sigmoid()?;

        assert!(!matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!((vals[0] - 0.5).abs() < 1e-6);
        assert!((vals[1] - 0.26894143).abs() < 1e-5);
        assert!((vals[2] - 0.7310586).abs() < 1e-5);
        assert!((vals[3] - 0.8807971).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_exp_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![0f32, 1.0, -1.0],
            (3,),
            &device,
        )?);

        let out = xs.exp()?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((vals[2] - (-1.0f32).exp()).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_recip_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![2f32, -4.0, 0.5],
            (3,),
            &device,
        )?);

        let out = xs.recip()?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!((vals[0] - 0.5).abs() < 1e-6);
        assert!((vals[1] + 0.25).abs() < 1e-6);
        assert!((vals[2] - 2.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_sqrt_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 4.0, 9.0],
            (3,),
            &device,
        )?);

        let out = xs.sqrt()?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_broadcast_add_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let lhs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);
        let rhs =
            HipTensor::from_scaffold_tensor(Tensor::from_vec(vec![10f32, 20.0], (1, 2), &device)?);

        let out = lhs.broadcast_add(&rhs)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![11.0, 22.0, 13.0, 24.0]);
        Ok(())
    }

    fn device_leaf_generic_mul_scalar_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, -2.0, 4.0],
            (3,),
            &device,
        )?);

        let out = xs.mul_scalar(0.5)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![0.5, -1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_log_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, std::f32::consts::E, 4.0],
            (3,),
            &device,
        )?);

        let out = xs.log()?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let vals = values_f32(out)?;
        assert!(vals[0].abs() < 1e-6);
        assert!((vals[1] - 1.0).abs() < 1e-5);
        assert!((vals[2] - (4.0f32).ln()).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_sum_keepdim_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);

        let out = xs.sum_keepdim(candle_core::D::Minus1)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_max_keepdim_stays_device_backed_via_kernel() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?);

        let out = xs.max_keepdim(candle_core::D::Minus1)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(values_f32(out)?, vec![2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_sigmoid_stays_device_backed_via_kernel() -> Result<()> {
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
    fn device_leaf_generic_cast_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_scaffold_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0],
            (3,),
            &device,
        )?);

        let out = xs.to_dtype(DType::F16)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(out.0.dtype(), DType::F16);
        assert_eq!(values_f32(out.to_dtype(DType::F32)?)?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_generic_cast_of_view_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (2, 2),
            &device,
        )?))
        .transpose(0, 1)?;

        let out = xs.to_dtype(DType::F64)?;

        assert!(!matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        assert_eq!(out.0.dtype(), DType::F64);
        assert_eq!(values_f32(out.to_dtype(DType::F32)?)?, vec![1.0, 3.0, 2.0, 4.0]);
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
    fn device_leaf_host_storage_rope_stays_host_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_device_buffer(HipDeviceBuffer::from_tensor(Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0],
            (1, 1, 1, 4),
            &device,
        )?));
        let cos = Tensor::from_vec(vec![0f32, 1.0], (1, 2), &device)?;
        let sin = Tensor::from_vec(vec![1f32, 0.0], (1, 2), &device)?;

        let out = rope_hip(&xs, &cos, &sin)?;

        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        assert_eq!(values_f32(out)?, vec![-2.0, 1.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_pending_upload_rope_stays_pending() -> Result<()> {
        let device = Device::Cpu;
        let xs = HipTensor::from_device_buffer(
            host_f32_tensor(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0])
                .try_host_buffer()?
                .expect("host buffer")
                .upload_to_device_buffer()?,
        );
        let cos = Tensor::from_vec(vec![0f32, 1.0], (1, 2), &device)?;
        let sin = Tensor::from_vec(vec![1f32, 0.0], (1, 2), &device)?;

        let out = rope_hip(&xs, &cos, &sin)?;

        let buffer = out
            .0
            .0
            .direct_device_buffer()
            .expect("device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::PendingHostUpload(_)));
        assert_eq!(host_buffer_values_f32(&out.try_host_buffer()?.expect("host"))?, vec![-2.0, 1.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn device_leaf_causal_mask_stays_device_backed() -> Result<()> {
        let out = causal_mask(&Device::Cpu, DType::F32, 1, 3, 2)?;
        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_cumsum_last_dim_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![1f32, 2.0, 3.0, -1.0, 4.0, 2.0], (2, 3), &device)?;

        let out = cumsum_last_dim(&xs)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
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
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
        Ok(())
    }

    #[test]
    fn device_leaf_rms_norm_stays_device_backed() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![3f32, 4.0, 5.0, 12.0], (2, 2), &device)?;
        let weight = Tensor::ones((2,), DType::F32, &device)?;

        let out = rms_norm(&xs, &weight, 1e-6, true)?;

        assert!(matches!(out.0 .0.expr, HipNativeExpr::DeviceBuffer(_)));
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
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
        let buffer = out
            .0
            .0
            .direct_materialized_device_buffer()
            .expect("materialized device leaf");
        assert!(matches!(buffer.storage, HipDeviceStorage::HostBuffer(_)));
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
    if let Some(device_out) = embedding_lookup_hip_owned_device(embeddings, indexes)? {
        return Ok(device_out);
    }
    let embeddings_hip = HipTensor::from_scaffold_tensor(embeddings.clone());
    let indexes_hip = HipTensor::from_scaffold_tensor(indexes.clone());
    if let (Some(embeddings), Some(indexes)) = (
        embeddings_hip.try_materialized_device_buffer()?,
        indexes_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(embeddings_mapped),
            HipDeviceStorage::MappedHostBuffer(indexes_mapped),
        ) = (&embeddings.storage, &indexes.storage)
        {
            if let Some(out) =
                mapped_embedding_lookup_hip_host_buffer(embeddings_mapped, indexes_mapped)?
            {
                return Ok(out);
            }
        }
    }
    if let Some(host) = embedding_lookup_hip_host_buffer(embeddings, indexes)? {
        return Ok(host);
    }
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
    if let Some(device_out) = immutable_embedding_lookup_hip_owned_device(embedding, indexes)? {
        return Ok(device_out);
    }
    let indexes_hip = HipTensor::from_scaffold_tensor(indexes.clone());
    if let Some(indexes) = indexes_hip.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(indexes_mapped) = &indexes.storage {
            if let Some(out) =
                mapped_immutable_embedding_lookup_hip_host_buffer(embedding, indexes_mapped)?
            {
                return Ok(out);
            }
        }
    }
    if let Some(host) = immutable_embedding_lookup_hip_host_buffer(embedding, indexes)? {
        return Ok(host);
    }
    Ok(from_kernel_tensor(hip_immutable_embedding_lookup(
        embedding, indexes,
    )?))
}

pub(crate) fn output_projection(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<HipTensor> {
    if let Some(device_out) = output_projection_hip_owned_device(embedding, hidden_states)? {
        return Ok(device_out);
    }
    let hidden_states_hip = HipTensor::from_scaffold_tensor(hidden_states.clone());
    if let Some(hidden_states) = hidden_states_hip.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(hidden_states_mapped) = &hidden_states.storage {
            if let Some(out) =
                mapped_output_projection_hip_host_buffer(embedding, hidden_states_mapped)?
            {
                return Ok(out);
            }
        }
    }
    if let Some(host) = output_projection_hip_host_buffer(embedding, hidden_states)? {
        return Ok(host);
    }
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
    if let Some(device_out) = rms_norm_hip_owned_device(xs, weight, eps, add_unit_offset)? {
        return Ok(device_out);
    }
    if let Some(host) = rms_norm_hip_host_buffer(xs, weight, eps, add_unit_offset)? {
        return Ok(host);
    }
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    if xs_hip.try_materialized_device_buffer()?.is_some() {
        return rms_norm_hip(&xs_hip, weight, eps, add_unit_offset);
    }
    if let Some(host) = rms_norm_host(&xs_hip, weight, eps, add_unit_offset)? {
        return materialize_host_result_as_device_leaf(host);
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
    if let Some(device_out) = rms_norm_gated_hip_owned_device(hidden_states, gate, weight, eps)? {
        return Ok(device_out);
    }
    if let Some(host) = rms_norm_gated_hip_host_buffer(hidden_states, gate, weight, eps)? {
        return Ok(host);
    }
    let hidden_states_hip = HipTensor::from_scaffold_tensor(hidden_states.clone());
    let gate_hip = HipTensor::from_scaffold_tensor(gate.clone());
    if let (Some(hidden_states), Some(gate)) = (
        hidden_states_hip.try_materialized_device_buffer()?,
        gate_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(hidden_mapped),
            HipDeviceStorage::MappedHostBuffer(gate_mapped),
        ) = (&hidden_states.storage, &gate.storage)
        {
            if let Some(out) =
                mapped_rms_norm_gated_hip_host_buffer(hidden_mapped, gate_mapped, weight, eps)?
            {
                return Ok(out);
            }
        }
        if hidden_states.storage.as_host_buffer().is_some() && gate.storage.as_host_buffer().is_some()
        {
            return Ok(HipTensor::from_device_buffer(
                hidden_states.rms_norm_gated(&gate, weight, eps)?,
            ));
        }
        return Ok(HipTensor::from_device_buffer(
            hidden_states.rms_norm_gated(&gate, weight, eps)?,
        ));
    }
    if let Some(host) = rms_norm_gated_host(&hidden_states_hip, &gate_hip, weight, eps)? {
        return materialize_host_result_as_device_leaf(host);
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
    if let Some(device_out) = swiglu_mul_hip_owned_device(gate, up)? {
        return Ok(device_out);
    }
    if let Some(host) = swiglu_mul_hip_host_buffer(gate, up)? {
        return Ok(host);
    }
    let gate_hip = HipTensor::from_scaffold_tensor(gate.clone());
    let up_hip = HipTensor::from_scaffold_tensor(up.clone());
    if let (Some(gate), Some(up)) = (
        gate_hip.try_materialized_device_buffer()?,
        up_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(gate_mapped),
            HipDeviceStorage::MappedHostBuffer(up_mapped),
        ) = (&gate.storage, &up.storage)
        {
            if let Some(out) = mapped_swiglu_mul_hip_host_buffer(gate_mapped, up_mapped)? {
                return Ok(out);
            }
        }
        if gate.storage.as_host_buffer().is_some() && up.storage.as_host_buffer().is_some() {
            return Ok(HipTensor::from_device_buffer(gate.swiglu_mul(&up)?));
        }
        return Ok(HipTensor::from_device_buffer(gate.swiglu_mul(&up)?));
    }
    if let Some(host) = swiglu_mul_host(&gate_hip, &up_hip)? {
        return materialize_host_result_as_device_leaf(host);
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
    if let Some(device_out) =
        causal_mask_hip_owned_device(device, dtype, batch_size, tgt_len, seqlen_offset)?
    {
        return Ok(device_out);
    }
    if let Some(host) =
        causal_mask_hip_host_buffer(device, dtype, batch_size, tgt_len, seqlen_offset)?
    {
        return Ok(host);
    }
    if device.is_hip() {
        return Ok(from_kernel_tensor(hip_causal_mask(
            device,
            dtype,
            batch_size,
            tgt_len,
            seqlen_offset,
        )?));
    }
    if let Some(host) = causal_mask_host(device, dtype, batch_size, tgt_len, seqlen_offset)? {
        return materialize_host_result_as_device_leaf(host);
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
    if let Some(host) = cumsum_last_dim_hip_host_buffer(xs)? {
        return Ok(host);
    }
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    if let Some(xs) = xs_hip.try_materialized_device_buffer()? {
        if let Some(out) = owned_cumsum_last_dim_hip_device_buffer(&xs)? {
            return Ok(out);
        }
        if let HipDeviceStorage::MappedHostBuffer(mapped) = &xs.storage {
            if let Some(out) = mapped_cumsum_last_dim_hip_host_buffer(mapped)? {
                return Ok(out);
            }
        }
        return Ok(HipTensor::from_device_buffer(xs.cumsum_last_dim()?));
    }
    if let Some(host) = cumsum_last_dim_host(&xs_hip)? {
        return materialize_host_result_as_device_leaf(host);
    }
    Ok(from_kernel_tensor(hip_cumsum_last_dim(xs)?))
}

pub(crate) fn cumsum_last_dim_buffer(xs: &StateBuffer) -> Result<StateBuffer> {
    cumsum_last_dim(xs.tensor())?.into_state_buffer()
}

pub(crate) fn l2norm(xs: &Tensor, eps: f64) -> Result<HipTensor> {
    if let Some(host) = l2norm_hip_host_buffer(xs, eps)? {
        return Ok(host);
    }
    let xs_hip = HipTensor::from_scaffold_tensor(xs.clone());
    if let Some(xs) = xs_hip.try_materialized_device_buffer()? {
        return Ok(HipTensor::from_device_buffer(xs.l2norm(eps)?));
    }
    Ok(l2norm_hip(&xs_hip, eps)?)
}

pub(crate) fn l2norm_buffer(xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
    l2norm(xs.tensor(), eps)?.into_state_buffer()
}

pub(crate) fn value_decay(a: &Tensor, dt_bias: &Tensor, a_log_exp: &Tensor) -> Result<HipTensor> {
    if let Some(device_out) = value_decay_hip_owned_device(a, dt_bias, a_log_exp)? {
        return Ok(device_out);
    }
    if let Some(host) = value_decay_hip_host_buffer(a, dt_bias, a_log_exp)? {
        return Ok(host);
    }
    let a_hip = HipTensor::from_scaffold_tensor(a.clone());
    let dt_bias_hip = HipTensor::from_scaffold_tensor(dt_bias.clone());
    let a_log_exp_hip = HipTensor::from_scaffold_tensor(a_log_exp.clone());
    if let (Some(a), Some(dt_bias), Some(a_log_exp)) = (
        a_hip.0 .0.direct_materialized_device_buffer(),
        dt_bias_hip.0 .0.direct_materialized_device_buffer(),
        a_log_exp_hip.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(a_mapped),
            HipDeviceStorage::MappedHostBuffer(dt_bias_mapped),
            HipDeviceStorage::MappedHostBuffer(a_log_exp_mapped),
        ) = (&a.storage, &dt_bias.storage, &a_log_exp.storage)
        {
            if let Some(out) = mapped_value_decay_hip_host_buffer(
                a_mapped,
                dt_bias_mapped,
                a_log_exp_mapped,
            )? {
                return Ok(out);
            }
        }
        if a.storage.as_host_buffer().is_some()
            && dt_bias.storage.as_host_buffer().is_some()
            && a_log_exp.storage.as_host_buffer().is_some()
        {
            return Ok(HipTensor::from_device_buffer(
                a.value_decay(dt_bias, a_log_exp)?,
            ));
        }
        return Ok(HipTensor::from_device_buffer(a.value_decay(dt_bias, a_log_exp)?));
    }
    if let Some(host) = value_decay_host(&a_hip, &dt_bias_hip, &a_log_exp_hip)? {
        return materialize_host_result_as_device_leaf(host);
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

fn rope_host_buffer(
    xs: &HipHostBuffer,
    cos: &HipHostBuffer,
    sin: &HipHostBuffer,
    b_sz: usize,
    n_head: usize,
    seq_len: usize,
    n_embd: usize,
) -> Result<HipHostBuffer> {
    if !HipNativeBuffer::supports_host_float_ops(xs.dtype)
        || !HipNativeBuffer::supports_host_float_ops(cos.dtype)
        || !HipNativeBuffer::supports_host_float_ops(sin.dtype)
    {
        candle_core::bail!(
            "rope host path unsupported for dtypes {:?}, {:?}, {:?}",
            xs.dtype,
            cos.dtype,
            sin.dtype
        );
    }
    let cos_rank = cos.shape.len();
    let sin_rank = sin.shape.len();
    if !matches!(cos_rank, 2 | 3) || !matches!(sin_rank, 2 | 3) {
        candle_core::bail!(
            "rope host cos/sin rank mismatch: {:?} {:?}",
            cos.shape,
            sin.shape
        );
    }
    let half = n_embd / 2;
    let mut out = vec![0u8; HipNativeBuffer::byte_len(&xs.shape, xs.dtype)];
    for b in 0..b_sz {
        for h in 0..n_head {
            for t in 0..seq_len {
                for i in 0..half {
                    let x_idx = (((b * n_head + h) * seq_len + t) * n_embd) + (2 * i);
                    let x0 =
                        HipNativeBuffer::read_host_float(xs.bytes.as_ref(), xs.dtype, x_idx)?;
                    let x1 = HipNativeBuffer::read_host_float(
                        xs.bytes.as_ref(),
                        xs.dtype,
                        x_idx + 1,
                    )?;
                    let cos_idx = if cos_rank == 2 {
                        t * half + i
                    } else {
                        ((b * cos.shape[1] + t) * half) + i
                    };
                    let sin_idx = if sin_rank == 2 {
                        t * half + i
                    } else {
                        ((b * sin.shape[1] + t) * half) + i
                    };
                    let c =
                        HipNativeBuffer::read_host_float(cos.bytes.as_ref(), cos.dtype, cos_idx)?;
                    let s =
                        HipNativeBuffer::read_host_float(sin.bytes.as_ref(), sin.dtype, sin_idx)?;
                    HipNativeBuffer::write_host_float(
                        &mut out,
                        xs.dtype,
                        x_idx,
                        x0 * c - x1 * s,
                    )?;
                    HipNativeBuffer::write_host_float(
                        &mut out,
                        xs.dtype,
                        x_idx + 1,
                        x0 * s + x1 * c,
                    )?;
                }
            }
        }
    }
    Ok(HipHostBuffer {
        bytes: out.into(),
        shape: xs.shape.clone(),
        dtype: xs.dtype,
        device: xs.device.clone(),
    })
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
    if let Some(xs_buffer) = xs.try_materialized_device_buffer()? {
        if let (Some(xs_host), Some(cos_host), Some(sin_host)) = (
            xs_buffer.try_host_buffer()?,
            cos.try_host_buffer()?,
            sin.try_host_buffer()?,
        ) {
            let out = rope_host_buffer(&xs_host, &cos_host, &sin_host, b_sz, n_head, seq_len, n_embd)?;
            return Ok(HipTensor::from_device_buffer(
                xs_buffer.from_host_computed_buffer_like(out),
            ));
        }
    }
    if let (Some(xs), Some(cos), Some(sin)) = (
        xs.try_materialized_device_buffer()?,
        cos.try_materialized_device_buffer()?,
        sin.try_materialized_device_buffer()?,
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
        let last_dim = x.dims().len() - 1;
        let x0 = x.narrow(last_dim, 0, 1)?;
        let x1 = x.narrow(last_dim, 1, 1)?;
        let y0 = x0.broadcast_mul(&cos)?.broadcast_sub(&x1.broadcast_mul(&sin)?)?;
        let y1 = x0.broadcast_mul(&sin)?.broadcast_add(&x1.broadcast_mul(&cos)?)?;
        return Ok(HipTensor::from_device_buffer(
            HipDeviceBuffer::cat(&[&y0, &y1], y0.dims().len() - 1)?
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
    HipTensor::cat(&[&y0, &y1], y0.rank() - 1)?
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
    if let Some(device_out) = linear_prefill_conv_hip_owned_device(mixed_qkv, weights, seq_len, kernel_size)? {
        return Ok(device_out);
    }
    let mixed_qkv_hip = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    if let Some(mixed_qkv) = mixed_qkv_hip.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(mapped) = &mixed_qkv.storage {
            if let Some(out) =
                mapped_linear_prefill_conv_hip_host_buffer(mapped, weights, seq_len, kernel_size)?
            {
                return Ok(out);
            }
        }
    }
    if let Some(host) = linear_prefill_conv_hip_host_buffer(mixed_qkv, weights, seq_len, kernel_size)? {
        return Ok(host);
    }
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
    if let Some(device_out) = linear_stateful_conv_hip_owned_device(mixed_qkv, prev_state, weights, kernel_size)? {
        return Ok(device_out);
    }
    let mixed_qkv_hip = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let prev_state_hip = HipTensor::from_scaffold_tensor(prev_state.clone());
    if let (Some(mixed_qkv), Some(prev_state)) = (
        mixed_qkv_hip.try_materialized_device_buffer()?,
        prev_state_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(mixed_qkv_mapped),
            HipDeviceStorage::MappedHostBuffer(prev_state_mapped),
        ) = (&mixed_qkv.storage, &prev_state.storage)
        {
            if let Some(out) = mapped_linear_stateful_conv_hip_host_buffer(
                mixed_qkv_mapped,
                prev_state_mapped,
                weights,
                kernel_size,
            )? {
                return Ok(out);
            }
        }
    }
    if let Some(host) =
        linear_stateful_conv_hip_host_buffer(mixed_qkv, prev_state, weights, kernel_size)?
    {
        return Ok(host);
    }
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
    if let Some(device_out) = linear_decode_step_hip_owned_device(
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
    )? {
        return Ok(device_out);
    }
    let mixed_qkv_hip = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let prev_conv_state_hip = HipTensor::from_scaffold_tensor(prev_conv_state.clone());
    let a_beta_raw_hip = HipTensor::from_scaffold_tensor(a_beta_raw.clone());
    let dt_bias_hip = HipTensor::from_scaffold_tensor(dt_bias.clone());
    let a_log_exp_hip = HipTensor::from_scaffold_tensor(a_log_exp.clone());
    let initial_state_hip = HipTensor::from_scaffold_tensor(initial_state.clone());
    if let (
        Some(mixed_qkv),
        Some(prev_conv_state),
        Some(a_beta_raw),
        Some(dt_bias),
        Some(a_log_exp),
        Some(initial_state),
    ) = (
        mixed_qkv_hip.try_materialized_device_buffer()?,
        prev_conv_state_hip.try_materialized_device_buffer()?,
        a_beta_raw_hip.try_materialized_device_buffer()?,
        dt_bias_hip.try_materialized_device_buffer()?,
        a_log_exp_hip.try_materialized_device_buffer()?,
        initial_state_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(mixed_qkv_mapped),
            HipDeviceStorage::MappedHostBuffer(prev_conv_state_mapped),
            HipDeviceStorage::MappedHostBuffer(a_beta_raw_mapped),
            HipDeviceStorage::MappedHostBuffer(dt_bias_mapped),
            HipDeviceStorage::MappedHostBuffer(a_log_exp_mapped),
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
        ) = (
            &mixed_qkv.storage,
            &prev_conv_state.storage,
            &a_beta_raw.storage,
            &dt_bias.storage,
            &a_log_exp.storage,
            &initial_state.storage,
        ) {
            if let Some(out) = mapped_linear_decode_step_hip_host_buffer(
                mixed_qkv_mapped,
                prev_conv_state_mapped,
                weights,
                a_beta_raw_mapped,
                dt_bias_mapped,
                a_log_exp_mapped,
                initial_state_mapped,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                kernel_size,
                head_repeat,
            )? {
                return Ok(out);
            }
        }
    }
    if let Some(host) = linear_decode_step_hip_host_buffer(
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
    )? {
        return Ok(host);
    }
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
    if let Some(device_out) = linear_stateful_conv_value_decay_with_state_hip_owned_device(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )? {
        return Ok(device_out);
    }
    let mixed_qkv_hip = HipTensor::from_scaffold_tensor(mixed_qkv.clone());
    let prev_state_hip = HipTensor::from_scaffold_tensor(prev_state.clone());
    let a_hip = HipTensor::from_scaffold_tensor(a.clone());
    let dt_bias_hip = HipTensor::from_scaffold_tensor(dt_bias.clone());
    let a_log_exp_hip = HipTensor::from_scaffold_tensor(a_log_exp.clone());
    if let (Some(mixed_qkv), Some(prev_state), Some(a), Some(dt_bias), Some(a_log_exp)) = (
        mixed_qkv_hip.try_materialized_device_buffer()?,
        prev_state_hip.try_materialized_device_buffer()?,
        a_hip.try_materialized_device_buffer()?,
        dt_bias_hip.try_materialized_device_buffer()?,
        a_log_exp_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(mixed_qkv_mapped),
            HipDeviceStorage::MappedHostBuffer(prev_state_mapped),
            HipDeviceStorage::MappedHostBuffer(a_mapped),
            HipDeviceStorage::MappedHostBuffer(dt_bias_mapped),
            HipDeviceStorage::MappedHostBuffer(a_log_exp_mapped),
        ) = (
            &mixed_qkv.storage,
            &prev_state.storage,
            &a.storage,
            &dt_bias.storage,
            &a_log_exp.storage,
        ) {
            if let Some(out) = mapped_linear_stateful_conv_value_decay_with_state_hip_host_buffer(
                mixed_qkv_mapped,
                prev_state_mapped,
                weights,
                a_mapped,
                dt_bias_mapped,
                a_log_exp_mapped,
                kernel_size,
            )? {
                return Ok(out);
            }
        }
    }
    if let Some(host) = linear_stateful_conv_value_decay_with_state_hip_host_buffer(
        mixed_qkv,
        prev_state,
        weights,
        a,
        dt_bias,
        a_log_exp,
        kernel_size,
    )? {
        return Ok(host);
    }
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
    if let Some(device_out) =
        full_attention_prefill_hip_owned_device(query, key, value, num_kv_groups, scale, seqlen_offset)?
    {
        return Ok(device_out);
    }
    let query_hip = HipTensor::from_scaffold_tensor(query.clone());
    let key_hip = HipTensor::from_scaffold_tensor(key.clone());
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (Some(query), Some(key), Some(value)) = (
        query_hip.try_materialized_device_buffer()?,
        key_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(query_mapped),
            HipDeviceStorage::MappedHostBuffer(key_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (&query.storage, &key.storage, &value.storage)
        {
            if let Some(out) = mapped_full_attention_prefill_hip_host_buffer(
                query_mapped,
                key_mapped,
                value_mapped,
                num_kv_groups,
                scale,
                seqlen_offset,
            )? {
                return Ok(out);
            }
        }
    }
    if let Some(host) = full_attention_prefill_hip_host_buffer(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )? {
        return Ok(host);
    }
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
    if let Some(device_out) =
        full_attention_prefill_hip_owned_device(query, key, value, num_kv_groups, scale, seqlen_offset)?
    {
        return Ok(device_out);
    }
    let query_hip = HipTensor::from_scaffold_tensor(query.clone());
    let key_hip = HipTensor::from_scaffold_tensor(key.clone());
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (Some(query), Some(key), Some(value)) = (
        query_hip.try_materialized_device_buffer()?,
        key_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(query_mapped),
            HipDeviceStorage::MappedHostBuffer(key_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (&query.storage, &key.storage, &value.storage)
        {
            if let Some(out) = mapped_full_attention_prefill_hip_host_buffer(
                query_mapped,
                key_mapped,
                value_mapped,
                num_kv_groups,
                scale,
                seqlen_offset,
            )? {
                return Ok(out);
            }
        }
    }
    if let Some(host) = full_attention_prefill_hip_host_buffer(
        query,
        key,
        value,
        num_kv_groups,
        scale,
        seqlen_offset,
    )? {
        return Ok(host);
    }
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
    if let Some(device_out) = delta_recurrent_prefill_hip_owned_device(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )? {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let query_scan_hip = HipTensor::from_scaffold_tensor(query_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let value_scan_hip = HipTensor::from_scaffold_tensor(value_scan.clone());
    let beta_scan_hip = HipTensor::from_scaffold_tensor(beta_scan.clone());
    let g_scan_hip = HipTensor::from_scaffold_tensor(g_scan.clone());
    if let (
        Some(initial_state),
        Some(query_scan),
        Some(key_scan),
        Some(value_scan),
        Some(beta_scan),
        Some(g_scan),
    ) = (
        initial_state_hip.try_materialized_device_buffer()?,
        query_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        value_scan_hip.try_materialized_device_buffer()?,
        beta_scan_hip.try_materialized_device_buffer()?,
        g_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(query_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(value_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(beta_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(g_scan_mapped),
        ) = (
            &initial_state.storage,
            &query_scan.storage,
            &key_scan.storage,
            &value_scan.storage,
            &beta_scan.storage,
            &g_scan.storage,
        ) {
            if let Some(out) = mapped_delta_recurrent_prefill_hip_host_buffer(
                initial_state_mapped,
                query_scan_mapped,
                key_scan_mapped,
                value_scan_mapped,
                beta_scan_mapped,
                g_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_recurrent_prefill_hip_host_buffer(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )? {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) = delta_chunk_single_prefill_hip_owned_device(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )? {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let query_hip = HipTensor::from_scaffold_tensor(query.clone());
    let key_hip = HipTensor::from_scaffold_tensor(key.clone());
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    let beta_hip = HipTensor::from_scaffold_tensor(beta.clone());
    let g_hip = HipTensor::from_scaffold_tensor(g.clone());
    if let (Some(initial_state), Some(query), Some(key), Some(value), Some(beta), Some(g)) = (
        initial_state_hip.try_materialized_device_buffer()?,
        query_hip.try_materialized_device_buffer()?,
        key_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
        beta_hip.try_materialized_device_buffer()?,
        g_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(query_mapped),
            HipDeviceStorage::MappedHostBuffer(key_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
            HipDeviceStorage::MappedHostBuffer(beta_mapped),
            HipDeviceStorage::MappedHostBuffer(g_mapped),
        ) = (
            &initial_state.storage,
            &query.storage,
            &key.storage,
            &value.storage,
            &beta.storage,
            &g.storage,
        ) {
            if let Some(out) = mapped_delta_chunk_single_prefill_hip_host_buffer(
                initial_state_mapped,
                query_mapped,
                key_mapped,
                value_mapped,
                beta_mapped,
                g_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_chunk_single_prefill_hip_host_buffer(
        initial_state.tensor(),
        query,
        key,
        value,
        beta,
        g,
    )? {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) = delta_chunk_scan_raw_hip_owned_device(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )? {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let query_scan_hip = HipTensor::from_scaffold_tensor(query_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let value_scan_hip = HipTensor::from_scaffold_tensor(value_scan.clone());
    let beta_scan_hip = HipTensor::from_scaffold_tensor(beta_scan.clone());
    let g_scan_hip = HipTensor::from_scaffold_tensor(g_scan.clone());
    if let (
        Some(initial_state),
        Some(query_scan),
        Some(key_scan),
        Some(value_scan),
        Some(beta_scan),
        Some(g_scan),
    ) = (
        initial_state_hip.try_materialized_device_buffer()?,
        query_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        value_scan_hip.try_materialized_device_buffer()?,
        beta_scan_hip.try_materialized_device_buffer()?,
        g_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(query_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(value_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(beta_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(g_scan_mapped),
        ) = (
            &initial_state.storage,
            &query_scan.storage,
            &key_scan.storage,
            &value_scan.storage,
            &beta_scan.storage,
            &g_scan.storage,
        ) {
            if let Some(out) = mapped_delta_chunk_scan_raw_hip_host_buffer(
                initial_state_mapped,
                query_scan_mapped,
                key_scan_mapped,
                value_scan_mapped,
                beta_scan_mapped,
                g_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_chunk_scan_raw_hip_host_buffer(
        initial_state.tensor(),
        query_scan,
        key_scan,
        value_scan,
        beta_scan,
        g_scan,
    )? {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) =
        delta_base_attn_scan_hip_owned_device(k_beta_scan, key_scan, exp_g_scan)?
    {
        return device_out.into_state_buffer();
    }
    let k_beta_scan_hip = HipTensor::from_scaffold_tensor(k_beta_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let exp_g_scan_hip = HipTensor::from_scaffold_tensor(exp_g_scan.clone());
    if let (Some(k_beta_scan), Some(key_scan), Some(exp_g_scan)) = (
        k_beta_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        exp_g_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(k_beta_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(exp_g_scan_mapped),
        ) = (&k_beta_scan.storage, &key_scan.storage, &exp_g_scan.storage)
        {
            if let Some(out) = mapped_delta_base_attn_scan_hip_host_buffer(
                k_beta_scan_mapped,
                key_scan_mapped,
                exp_g_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) =
        delta_base_attn_scan_hip_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    {
        return host.into_state_buffer();
    }
    from_kernel_tensor(delta_base_attn_scan(k_beta_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_from_inputs_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    if let Some(device_out) =
        delta_attn_solve_from_inputs_hip_owned_device(k_beta_scan, key_scan, exp_g_scan)?
    {
        return device_out.into_state_buffer();
    }
    let k_beta_scan_hip = HipTensor::from_scaffold_tensor(k_beta_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let exp_g_scan_hip = HipTensor::from_scaffold_tensor(exp_g_scan.clone());
    if let (Some(k_beta_scan), Some(key_scan), Some(exp_g_scan)) = (
        k_beta_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        exp_g_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(k_beta_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(exp_g_scan_mapped),
        ) = (&k_beta_scan.storage, &key_scan.storage, &exp_g_scan.storage)
        {
            if let Some(out) = mapped_delta_attn_solve_from_inputs_hip_host_buffer(
                k_beta_scan_mapped,
                key_scan_mapped,
                exp_g_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) =
        delta_attn_solve_from_inputs_hip_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    {
        return host.into_state_buffer();
    }
    from_kernel_tensor(delta_attn_solve_from_inputs(
        k_beta_scan,
        key_scan,
        exp_g_scan,
    )?)
        .into_state_buffer()
}

pub(crate) fn delta_attn_solve_scan_buffer(base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
    if let Some(device_out) = delta_attn_solve_scan_hip_owned_device(base_attn_scan.tensor())? {
        return device_out.into_state_buffer();
    }
    let base_attn_scan_hip = HipTensor::from_state_buffer(base_attn_scan);
    if let Some(base_attn_scan) = base_attn_scan_hip.try_materialized_device_buffer()? {
        if let HipDeviceStorage::MappedHostBuffer(base_attn_scan_mapped) = &base_attn_scan.storage {
            if let Some(out) =
                mapped_delta_attn_solve_scan_hip_host_buffer(base_attn_scan_mapped)?
            {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_attn_solve_scan_hip_host_buffer(base_attn_scan.tensor())? {
        return host.into_state_buffer();
    }
    from_kernel_tensor(delta_attn_solve_scan(base_attn_scan.tensor())?)
        .into_state_buffer()
}

pub(crate) fn delta_local_attn_scan_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<StateBuffer> {
    if let Some(device_out) =
        delta_local_attn_scan_hip_owned_device(query_scan, key_scan, exp_g_scan)?
    {
        return device_out.into_state_buffer();
    }
    let query_scan_hip = HipTensor::from_scaffold_tensor(query_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let exp_g_scan_hip = HipTensor::from_scaffold_tensor(exp_g_scan.clone());
    if let (Some(query_scan), Some(key_scan), Some(exp_g_scan)) = (
        query_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        exp_g_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(query_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(exp_g_scan_mapped),
        ) = (&query_scan.storage, &key_scan.storage, &exp_g_scan.storage)
        {
            if let Some(out) = mapped_delta_local_attn_scan_hip_host_buffer(
                query_scan_mapped,
                key_scan_mapped,
                exp_g_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_local_attn_scan_hip_host_buffer(query_scan, key_scan, exp_g_scan)? {
        return host.into_state_buffer();
    }
    from_kernel_tensor(delta_local_attn_scan(query_scan, key_scan, exp_g_scan)?)
        .into_state_buffer()
}

pub(crate) fn delta_full_scan_pack_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<StateBuffer> {
    if let Some(device_out) = delta_full_scan_pack_hip_owned_device(
        query_scan,
        key_scan,
        exp_g_scan,
        k_cumdecay_scan,
    )? {
        return device_out.into_state_buffer();
    }
    let query_scan_hip = HipTensor::from_scaffold_tensor(query_scan.clone());
    let key_scan_hip = HipTensor::from_scaffold_tensor(key_scan.clone());
    let exp_g_scan_hip = HipTensor::from_scaffold_tensor(exp_g_scan.clone());
    let k_cumdecay_scan_hip = HipTensor::from_scaffold_tensor(k_cumdecay_scan.clone());
    if let (Some(query_scan), Some(key_scan), Some(exp_g_scan), Some(k_cumdecay_scan)) = (
        query_scan_hip.try_materialized_device_buffer()?,
        key_scan_hip.try_materialized_device_buffer()?,
        exp_g_scan_hip.try_materialized_device_buffer()?,
        k_cumdecay_scan_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(query_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(exp_g_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(k_cumdecay_scan_mapped),
        ) = (
            &query_scan.storage,
            &key_scan.storage,
            &exp_g_scan.storage,
            &k_cumdecay_scan.storage,
        ) {
            if let Some(out) = mapped_delta_full_scan_pack_hip_host_buffer(
                query_scan_mapped,
                key_scan_mapped,
                exp_g_scan_mapped,
                k_cumdecay_scan_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) =
        delta_full_scan_pack_hip_host_buffer(query_scan, key_scan, exp_g_scan, k_cumdecay_scan)?
    {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) = delta_full_scan_packed_hip_owned_device(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )? {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let packed_scan_hip = HipTensor::from_state_buffer(packed_scan);
    let local_attn_scan_hip = HipTensor::from_state_buffer(local_attn_scan);
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (Some(initial_state), Some(packed_scan), Some(local_attn_scan), Some(value)) = (
        initial_state_hip.try_materialized_device_buffer()?,
        packed_scan_hip.try_materialized_device_buffer()?,
        local_attn_scan_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(packed_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(local_attn_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (
            &initial_state.storage,
            &packed_scan.storage,
            &local_attn_scan.storage,
            &value.storage,
        ) {
            if let Some(out) = mapped_delta_full_scan_packed_hip_host_buffer(
                initial_state_mapped,
                packed_scan_mapped,
                local_attn_scan_mapped,
                value_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_full_scan_packed_hip_host_buffer(
        initial_state.tensor(),
        packed_scan.tensor(),
        local_attn_scan.tensor(),
        value,
    )? {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) = delta_full_scan_hip_owned_device(
        initial_state.tensor(),
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan.tensor(),
        state_decay_scan,
        value,
    )? {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let weighted_key_scan_hip = HipTensor::from_scaffold_tensor(weighted_key_scan.clone());
    let k_cumdecay_scan_hip = HipTensor::from_scaffold_tensor(k_cumdecay_scan.clone());
    let q_state_scan_hip = HipTensor::from_scaffold_tensor(q_state_scan.clone());
    let local_attn_scan_hip = HipTensor::from_state_buffer(local_attn_scan);
    let state_decay_scan_hip = HipTensor::from_scaffold_tensor(state_decay_scan.clone());
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (
        Some(initial_state),
        Some(weighted_key_scan),
        Some(k_cumdecay_scan),
        Some(q_state_scan),
        Some(local_attn_scan),
        Some(state_decay_scan),
        Some(value),
    ) = (
        initial_state_hip.try_materialized_device_buffer()?,
        weighted_key_scan_hip.try_materialized_device_buffer()?,
        k_cumdecay_scan_hip.try_materialized_device_buffer()?,
        q_state_scan_hip.try_materialized_device_buffer()?,
        local_attn_scan_hip.try_materialized_device_buffer()?,
        state_decay_scan_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(weighted_key_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(k_cumdecay_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(q_state_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(local_attn_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(state_decay_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (
            &initial_state.storage,
            &weighted_key_scan.storage,
            &k_cumdecay_scan.storage,
            &q_state_scan.storage,
            &local_attn_scan.storage,
            &state_decay_scan.storage,
            &value.storage,
        ) {
            if let Some(out) = mapped_delta_full_scan_hip_host_buffer(
                initial_state_mapped,
                weighted_key_scan_mapped,
                k_cumdecay_scan_mapped,
                q_state_scan_mapped,
                local_attn_scan_mapped,
                state_decay_scan_mapped,
                value_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) = delta_full_scan_hip_host_buffer(
        initial_state.tensor(),
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan.tensor(),
        state_decay_scan,
        value,
    )? {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) =
        delta_state_scan_hip_owned_device(initial_state.tensor(), packed_scan.tensor(), value)?
    {
        return device_out.into_state_buffer();
    }
    let initial_state_hip = HipTensor::from_state_buffer(initial_state);
    let packed_scan_hip = HipTensor::from_state_buffer(packed_scan);
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (Some(initial_state), Some(packed_scan), Some(value)) = (
        initial_state_hip.try_materialized_device_buffer()?,
        packed_scan_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(initial_state_mapped),
            HipDeviceStorage::MappedHostBuffer(packed_scan_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (&initial_state.storage, &packed_scan.storage, &value.storage)
        {
            if let Some(out) = mapped_delta_state_scan_hip_host_buffer(
                initial_state_mapped,
                packed_scan_mapped,
                value_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) =
        delta_state_scan_hip_host_buffer(initial_state.tensor(), packed_scan.tensor(), value)?
    {
        return host.into_state_buffer();
    }
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
    if let Some(device_out) =
        delta_chunk_fused_hip_owned_device(prev_state.tensor(), packed_chunk.tensor(), value)?
    {
        return device_out.into_state_buffer();
    }
    let prev_state_hip = HipTensor::from_state_buffer(prev_state);
    let packed_chunk_hip = HipTensor::from_state_buffer(packed_chunk);
    let value_hip = HipTensor::from_scaffold_tensor(value.clone());
    if let (Some(prev_state), Some(packed_chunk), Some(value)) = (
        prev_state_hip.try_materialized_device_buffer()?,
        packed_chunk_hip.try_materialized_device_buffer()?,
        value_hip.try_materialized_device_buffer()?,
    ) {
        if let (
            HipDeviceStorage::MappedHostBuffer(prev_state_mapped),
            HipDeviceStorage::MappedHostBuffer(packed_chunk_mapped),
            HipDeviceStorage::MappedHostBuffer(value_mapped),
        ) = (&prev_state.storage, &packed_chunk.storage, &value.storage)
        {
            if let Some(out) = mapped_delta_chunk_fused_hip_host_buffer(
                prev_state_mapped,
                packed_chunk_mapped,
                value_mapped,
            )? {
                return out.into_state_buffer();
            }
        }
    }
    if let Some(host) =
        delta_chunk_fused_hip_host_buffer(prev_state.tensor(), packed_chunk.tensor(), value)?
    {
        return host.into_state_buffer();
    }
    from_kernel_tensor(delta_chunk_fused(
        prev_state.tensor(),
        packed_chunk.tensor(),
        value,
    )?)
    .into_state_buffer()
}

pub(crate) fn unpack_delta_chunk_step_output(
    fused: &StateBuffer,
    chunk_size: usize,
    k_head_dim: usize,
) -> Result<(StateBuffer, StateBuffer)> {
    let fused = HipTensor::from_state_buffer(fused);
    if let Some(fused) = fused.0 .0.direct_materialized_device_buffer() {
        if let Some(fused_host) = fused.storage.as_host_buffer() {
            let output = HipTensor::from_device_buffer(host_result_device_buffer(
                fused_host.narrow_copy(1, 0, chunk_size)?,
            ))
            .into_state_buffer()?;
            let recurrent_state = HipTensor::from_device_buffer(host_result_device_buffer(
                fused_host.narrow_copy(1, chunk_size, k_head_dim)?,
            ))
            .into_state_buffer()?;
            return Ok((output, recurrent_state));
        }
        let output = HipTensor::from_device_buffer(fused.narrow(1, 0, chunk_size)?).into_state_buffer()?;
        let recurrent_state = HipTensor::from_device_buffer(
            fused.narrow(1, chunk_size, k_head_dim)?.contiguous()?,
        )
        .into_state_buffer()?;
        return Ok((output, recurrent_state));
    }
    let output = fused.narrow(1, 0, chunk_size)?.into_state_buffer()?;
    let recurrent_state = fused
        .narrow(1, chunk_size, k_head_dim)?
        .contiguous()?
        .into_state_buffer()?;
    Ok((output, recurrent_state))
}

fn delta_chunk_recurrent_read_tensors_hip(
    prev_state: &HipTensor,
    k_cumdecay_chunk: &HipTensor,
    q_state_chunk: &HipTensor,
    value_chunk: &HipTensor,
) -> Result<(HipTensor, HipTensor)> {
    if let (Some(prev_state), Some(k_cumdecay_chunk), Some(q_state_chunk), Some(value_chunk)) = (
        prev_state.0 .0.direct_materialized_device_buffer(),
        k_cumdecay_chunk.0 .0.direct_materialized_device_buffer(),
        q_state_chunk.0 .0.direct_materialized_device_buffer(),
        value_chunk.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (
            Some(prev_state_host),
            Some(k_cumdecay_host),
            Some(q_state_host),
            Some(value_host),
        ) = (
            prev_state.storage.as_host_buffer(),
            k_cumdecay_chunk.storage.as_host_buffer(),
            q_state_chunk.storage.as_host_buffer(),
            value_chunk.storage.as_host_buffer(),
        ) {
            let v_prime = k_cumdecay_host.matmul(prev_state_host)?;
            let v_new = HipHostBuffer::broadcast_sub(value_host, &v_prime)?;
            let attn_inter = q_state_host.matmul(prev_state_host)?;
            return Ok((
                HipTensor::from_device_buffer(host_result_device_buffer(v_new)),
                HipTensor::from_device_buffer(host_result_device_buffer(attn_inter)),
            ));
        }
        let v_prime = k_cumdecay_chunk.matmul(prev_state)?;
        let v_new = value_chunk.broadcast_sub(&v_prime)?;
        let attn_inter = q_state_chunk.matmul(prev_state)?;
        return Ok((HipTensor::from_device_buffer(v_new), HipTensor::from_device_buffer(attn_inter)));
    }
    let v_prime = k_cumdecay_chunk.matmul(&prev_state)?;
    let v_new = value_chunk.broadcast_sub(&v_prime)?;
    let attn_inter = q_state_chunk.matmul(&prev_state)?;
    Ok((v_new, attn_inter))
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
    let (v_new, attn_inter) = delta_chunk_recurrent_read_tensors_hip(
        &prev_state,
        &k_cumdecay_chunk,
        &q_state_chunk,
        &value_chunk,
    )?;
    Ok((v_new.into_state_buffer()?, attn_inter.into_state_buffer()?))
}

fn mix_chunk_attention_tensors_hip(
    attn: &HipTensor,
    attn_inter: &HipTensor,
    value_chunk: &HipTensor,
) -> Result<HipTensor> {
    if let (Some(attn), Some(attn_inter), Some(value_chunk)) = (
        attn.0 .0.direct_materialized_device_buffer(),
        attn_inter.0 .0.direct_materialized_device_buffer(),
        value_chunk.0 .0.direct_materialized_device_buffer(),
    ) {
        if let (Some(attn_host), Some(attn_inter_host), Some(value_host)) = (
            attn.storage.as_host_buffer(),
            attn_inter.storage.as_host_buffer(),
            value_chunk.storage.as_host_buffer(),
        )
        {
            let attn_value = attn_host.matmul(value_host)?;
            let mixed = HipHostBuffer::broadcast_add(attn_inter_host, &attn_value)?;
            return Ok(HipTensor::from_device_buffer(host_result_device_buffer(mixed)));
        }
        let attn_value = attn.matmul(value_chunk)?;
        let mixed = attn_inter.broadcast_add(&attn_value)?;
        return Ok(HipTensor::from_device_buffer(mixed));
    }
    let attn_value = attn.matmul(&value_chunk)?;
    let mixed = attn_inter.broadcast_add(&attn_value)?;
    Ok(mixed)
}

pub(crate) fn mix_chunk_attention(
    attn: &Tensor,
    attn_inter: &StateBuffer,
    value_chunk: &StateBuffer,
) -> Result<StateBuffer> {
    let attn = HipTensor::from_scaffold_tensor(attn.clone());
    let attn_inter = HipTensor::from_state_buffer(attn_inter);
    let value_chunk = HipTensor::from_state_buffer(value_chunk);
    mix_chunk_attention_tensors_hip(&attn, &attn_inter, &value_chunk)?.into_state_buffer()
}

fn delta_state_update_tensors_hip(
    prev_state_scaled: &HipTensor,
    weighted_key: &HipTensor,
    value: &HipTensor,
    use_kernel: bool,
) -> Result<HipTensor> {
    let _ = use_kernel;
    weighted_key
        .transpose(2, 1)?
        .matmul(value)?
        .broadcast_add(prev_state_scaled)
}

pub(crate) fn delta_state_update_buffer(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &StateBuffer,
    use_kernel: bool,
) -> Result<StateBuffer> {
    let prev_state_scaled = HipTensor::from_scaffold_tensor(prev_state_scaled.clone());
    let weighted_key = HipTensor::from_scaffold_tensor(weighted_key.clone());
    let value = HipTensor::from_state_buffer(value);
    delta_state_update_tensors_hip(&prev_state_scaled, &weighted_key, &value, use_kernel)?
        .into_state_buffer()
}
