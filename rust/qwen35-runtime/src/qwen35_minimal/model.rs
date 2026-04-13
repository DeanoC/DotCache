#![allow(unexpected_cfgs)]

use super::backend_buffer_api;
use super::backend_buffer_api::Qwen35BackendBufferApi;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
use super::frontend::{
    build_prepared_embedding_source, debug_full_prefill_kernel_compare_enabled,
    immutable_embedding_enabled, immutable_linear_enabled, max_abs_delta, prepared_linear_b,
    prepared_linear_no_bias, profile_elapsed, profile_start, EmbeddingSource, Mlp,
    OutputProjectionSource, Qwen35RmsNorm, RotaryEmbedding,
};
use super::full_attention::FullAttention;
#[cfg(any(feature = "hf", test))]
use super::frontend::embedding;
#[cfg(feature = "qwen35-minimal-hip")]
use super::hip;
use super::ops;
use super::prepared::PreparedTensorSource;
use super::linear_attention::GatedDeltaNet;
pub use super::types::{
    CacheState, Config, ExternalFullAttention, FullAttentionCacheState, LayerCacheState,
    LinearAttentionBenchResult, LinearAttentionLayerSpec, LinearAttentionTrace, RuntimeProfile,
    StateBuffer, TextConfig,
};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::{linear_b, linear_no_bias};
use super::with_tracing::Linear;
use candle::{DType, Device, DeviceLocation, Module, Result, Tensor, D};
use candle_core as candle;
use crate::PreparedQwen35DirectMetadata;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) use super::frontend::{
    hip_rms_norm, hip_rms_norm_gated, hip_rms_norm_gated_host_buffer,
    hip_rms_norm_host_buffer, ImmutableEmbedding,
};
pub(crate) use super::hip_wrappers::{
    hip_add_scalar_host_buffer, hip_broadcast_add_host_buffer, hip_broadcast_div_host_buffer,
    hip_broadcast_mul_host_buffer, hip_broadcast_sub_host_buffer, hip_cast_host_buffer,
    hip_causal_mask, hip_causal_mask_host_buffer, hip_cumsum_last_dim,
    hip_cumsum_last_dim_host_buffer, hip_embedding_lookup, hip_embedding_lookup_host_buffer,
    hip_exp_host_buffer, hip_immutable_embedding_lookup,
    hip_immutable_embedding_lookup_host_buffer, hip_l2norm, hip_l2norm_host_buffer,
    linear_attention_chunk_size, softplus,
    hip_log_host_buffer, hip_matmul_host_buffer, hip_max_keepdim_host_buffer,
    hip_mul_scalar_host_buffer, hip_recip_host_buffer, hip_sigmoid_host_buffer,
    hip_sqrt_host_buffer, hip_sum_keepdim_host_buffer, hip_swiglu_mul, hip_swiglu_mul_host_buffer,
    hip_value_decay, hip_value_decay_host_buffer, immutable_output_projection,
    immutable_output_projection_host_buffer,
};
pub(super) fn elapsed_millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

pub(super) fn hip_output_bytes_to_cpu_storage(
    dtype: DType,
    output: Vec<u8>,
) -> Result<candle::CpuStorage> {
    match dtype {
        DType::F16 => {
            if output.len() % 2 != 0 {
                candle::bail!("invalid f16 output byte length {}", output.len());
            }
            let output = output
                .chunks_exact(2)
                .map(|chunk| half::f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
                .collect();
            Ok(<half::f16 as candle::WithDType>::to_cpu_storage_owned(output))
        }
        DType::BF16 => {
            if output.len() % 2 != 0 {
                candle::bail!("invalid bf16 output byte length {}", output.len());
            }
            let output = output
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
                .collect();
            Ok(<half::bf16 as candle::WithDType>::to_cpu_storage_owned(output))
        }
        DType::F32 => {
            if output.len() % 4 != 0 {
                candle::bail!("invalid f32 output byte length {}", output.len());
            }
            let output = output
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(<f32 as candle::WithDType>::to_cpu_storage_owned(output))
        }
        other => candle::bail!("unsupported HIP host output dtype {other:?}"),
    }
}

pub(super) fn hip_tensor_from_host_bytes<S: Into<candle::Shape>>(
    device: &Device,
    dtype: DType,
    shape: S,
    output: Vec<u8>,
) -> Result<Tensor> {
    let hip_device = match device {
        Device::Hip(device) => device.clone(),
        _ => candle::bail!("hip tensor construction requires a hip device"),
    };
    let storage = candle::Storage::Hip(candle::HipStorage::wrap_cpu_storage(
        hip_output_bytes_to_cpu_storage(dtype, output)?,
        hip_device,
    ));
    Ok(Tensor::from_storage(
        storage,
        shape,
        candle::op::BackpropOp::none(),
        false,
    ))
}

pub(super) fn trace_hip_wrapper_fallback(op: &str, tensor: &Tensor) {
    if std::env::var_os("DOTCACHE_HIP_TRACE_CANDLE_FALLBACK")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        eprintln!(
            "hip-wrapper-fallback op={} dtype={:?} shape={:?} device={:?}",
            op,
            tensor.dtype(),
            tensor.dims(),
            tensor.device().location()
        );
    }
}

pub(super) fn repeat_kv(xs: Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(xs);
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![&xs; repeats];
    Tensor::cat(&repeated, 2)?.reshape((b_sz, kv_heads * repeats, seq_len, head_dim))
}

#[cfg(test)]
fn l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    backend_ops::l2norm(xs, eps)
}

fn delta_chunk_step_2d_enabled() -> bool {
    match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_2D_KERNEL") {
        Ok(value)
            if matches!(
                value.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            ) =>
        {
            false
        }
        Ok(_) => true,
        Err(_) => true,
    }
}

fn delta_chunk_step_windowed_2d_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_2D_KERNEL").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(super) fn use_delta_chunk_windowed_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 2048
        && chunk_size <= 24)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

pub(super) fn use_linear_prefill_packed_kernel(device: &Device, sequence_length: usize) -> bool {
    if sequence_length < 2048 {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_LINEAR_PACKED_PREFILL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_LINEAR_PACKED_PREFILL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        DeviceLocation::Hip { .. } => false,
        _ => false,
    }
}

pub(super) fn use_hip_short_linear_prefill_recurrent(device: &Device, sequence_length: usize) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && sequence_length > 1
        && sequence_length <= linear_attention_chunk_size(device, sequence_length)
        && matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_SHORT_LINEAR_PREFILL_RECURRENT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

pub(super) fn use_hip_combined_linear_prefill(device: &Device, sequence_length: usize) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && sequence_length > 1
        && !matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_PREFILL").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
}

pub(super) fn use_hip_combined_linear_decode(device: &Device, sequence_length: usize) -> bool {
    // Keep the combined decode path opt-in on this UMA ROCm host: it cuts transfer
    // traffic substantially, but the custom kernels are still slower than the split
    // path here. This remains worth testing on larger discrete ROCm systems where
    // transfer cost is materially higher.
    matches!(device.location(), DeviceLocation::Hip { .. })
        && sequence_length == 1
        && matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_DECODE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

pub(super) fn use_hip_chunk_single_prefill_kernel(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
    chunk_size: usize,
) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && num_chunks == 1
        && sequence_length > 1
        && sequence_length <= chunk_size
        && chunk_size <= 64
        && !matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_CHUNK_SINGLE_PREFILL").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
}

pub(super) fn use_hip_multi_chunk_scan_prefill_kernel(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
    chunk_size: usize,
) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && num_chunks > 1
        && num_chunks <= 4
        && sequence_length > chunk_size
        && chunk_size <= 64
        && matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_MULTI_CHUNK_SCAN_PREFILL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

pub(super) fn use_full_attention_prefill_megakernel(
    device: &Device,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    seqlen_offset: usize,
) -> bool {
    if kv_len != q_len + seqlen_offset {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => {
            head_dim <= 128
                && matches!(
                    std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL").as_deref(),
                    Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
                )
        }
        DeviceLocation::Hip { .. } => {
            match std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        _ => false,
    }
}

pub(super) fn use_full_attention_decode_megakernel(
    device: &Device,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    seqlen_offset: usize,
) -> bool {
    if q_len != 1 || kv_len != seqlen_offset + 1 {
        return false;
    }

    match device.location() {
        DeviceLocation::Cuda { .. } => {
            head_dim <= 128
                && matches!(
                    std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL").as_deref(),
                    Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
                )
        }
        DeviceLocation::Hip { .. } => {
            match std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        _ => false,
    }
}

pub(super) fn use_delta_chunk_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 4096
        && chunk_size <= 16)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_CHUNK_SCAN_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearPrefillConvPack {
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
}

impl candle::CustomOp2 for LinearPrefillConvPack {
    fn name(&self) -> &'static str {
        "linear-prefill-conv-pack"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-prefill-conv-pack has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        mixed_qkv: &candle::CudaStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::CudaStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device().clone();
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let mixed_qkv = mixed_qkv.as_cuda_slice::<$ty>()?;
                let mixed_qkv = match mixed_qkv_layout.contiguous_offsets() {
                    Some((o1, o2)) => mixed_qkv.slice(o1..o2),
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let weights = weights.as_cuda_slice::<$ty>()?;
                let weights = match weights_layout.contiguous_offsets() {
                    Some((o1, o2)) => weights.slice(o1..o2),
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    self.batch_size,
                    self.conv_dim,
                    self.total_len,
                    self.seq_len,
                    self.kernel_size
                );
                builder.arg(&mixed_qkv);
                builder.arg(&weights);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, output_shape.clone()))
            }};
        }

        match mixed_qkv.dtype() {
            DType::F16 => launch!(half::f16, "linear_prefill_conv_pack_f16"),
            DType::F32 => launch!(f32, "linear_prefill_conv_pack_f32"),
            DType::BF16 => launch!(half::bf16, "linear_prefill_conv_pack_bf16"),
            other => candle::bail!("linear-prefill-conv-pack unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        mixed_qkv: &candle::MetalStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::MetalStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device();
        let storage_dtype = mixed_qkv.dtype();
        let dtype = match storage_dtype {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("linear-prefill-conv-pack unsupported dtype {other:?}"),
        };
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let output = device.new_buffer(elem_count, storage_dtype, "linear-prefill-conv-pack")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("linear-prefill-conv-pack");
        let mixed_qkv = candle_metal_kernels::BufferOffset {
            buffer: mixed_qkv.buffer(),
            offset_in_bytes: mixed_qkv_layout.start_offset() * mixed_qkv.dtype().size_in_bytes(),
        };
        let weights = candle_metal_kernels::BufferOffset {
            buffer: weights.buffer(),
            offset_in_bytes: weights_layout.start_offset() * weights.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_linear_prefill_conv_pack(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            self.batch_size,
            self.conv_dim,
            self.total_len,
            self.seq_len,
            self.kernel_size,
            mixed_qkv,
            weights,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage = candle::MetalStorage::new(output, device.clone(), elem_count, storage_dtype);
        Ok((storage, output_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device().clone();
        let storage_dtype = mixed_qkv.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_prefill_conv_pack(
                dtype_code,
                device.ordinal(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.seq_len,
                self.kernel_size,
                mixed_qkv.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                    as *const c_void,
                weights.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

pub(crate) fn linear_prefill_conv_pack(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = linear_prefill_conv_pack_host_buffer(mixed_qkv, weights, seq_len, kernel_size)? {
        return hip_tensor_from_host_bytes(mixed_qkv.device(), mixed_qkv.dtype(), shape, output);
    }
    let (batch_size, conv_dim, total_len) = mixed_qkv.dims3()?;
    trace_hip_wrapper_fallback("linear_prefill_conv_pack", &mixed_qkv);
    mixed_qkv.apply_op2_no_bwd(
        weights,
        &LinearPrefillConvPack {
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn linear_prefill_conv_pack_host_buffer(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weights = weights.contiguous()?;
    let ordinal = match mixed_qkv.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !weights.device().same_device(mixed_qkv.device()) {
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
    if weights_conv_dim != conv_dim || weights_kernel_size != kernel_size {
        return Ok(None);
    }
    if total_len < seq_len + kernel_size.saturating_sub(1) {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(mixed_qkv.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_size, seq_len, conv_dim];
    let mut out = vec![
        0u8;
        batch_size
            .saturating_mul(seq_len)
            .saturating_mul(conv_dim)
            .saturating_mul(mixed_qkv.dtype().size_in_bytes())
    ];
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
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "dotcache-hip-linear-prefill-conv-pack-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn linear_prefill_conv_pack_host_buffer(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (mixed_qkv, weights, seq_len, kernel_size);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct LinearStatefulConv {
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    state_len: usize,
    kernel_size: usize,
}

impl candle::CustomOp3 for LinearStatefulConv {
    fn name(&self) -> &'static str {
        "linear-stateful-conv"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-stateful-conv has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        prev_state: &candle::HipStorage,
        prev_state_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous()
            && prev_state_layout.is_contiguous()
            && weights_layout.is_contiguous())
        {
            candle::bail!("linear-stateful-conv requires contiguous inputs")
        }

        let (batch_size, conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
        let (state_batch, state_conv_dim, state_len) = prev_state_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || seq_len != self.seq_len
            || state_batch != self.batch_size
            || state_conv_dim != self.conv_dim
            || state_len != self.state_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-stateful-conv shape mismatch: mixed_qkv={:?} prev_state={:?} weights={:?} expected=({}, {}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                prev_state_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size
            )
        }
        if mixed_qkv.dtype() != prev_state.dtype() || mixed_qkv.dtype() != weights.dtype() {
            candle::bail!(
                "linear-stateful-conv requires matching dtypes, got mixed_qkv={:?} prev_state={:?} weights={:?}",
                mixed_qkv.dtype(),
                prev_state.dtype(),
                weights.dtype()
            )
        }

        let device = mixed_qkv.device().clone();
        let storage_dtype = mixed_qkv.dtype();
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_stateful_conv(
                candle::hip::qwen35_dtype_code(storage_dtype)?,
                device.ordinal(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size,
                mixed_qkv.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                    as *const c_void,
                prev_state.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                    as *const c_void,
                weights.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

pub(crate) fn linear_stateful_conv_hip(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        linear_stateful_conv_host_buffer(&mixed_qkv, &prev_state, &weights, kernel_size)?
    {
        return hip_tensor_from_host_bytes(mixed_qkv.device(), mixed_qkv.dtype(), shape, output);
    }
    let (batch_size, conv_dim, seq_len) = mixed_qkv.dims3()?;
    let (state_batch, state_conv_dim, state_len) = prev_state.dims3()?;
    if state_batch != batch_size || state_conv_dim != conv_dim {
        candle::bail!(
            "linear-stateful-conv state mismatch: mixed_qkv={:?} prev_state={:?}",
            mixed_qkv.dims(),
            prev_state.dims()
        )
    }
    trace_hip_wrapper_fallback("linear_stateful_conv_hip", &mixed_qkv);
    mixed_qkv.apply_op3_no_bwd(
        &prev_state,
        &weights,
        &LinearStatefulConv {
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn linear_stateful_conv_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let ordinal = match mixed_qkv.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_state.device().same_device(mixed_qkv.device())
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
    {
        return Ok(None);
    }
    if mixed_qkv.dtype() != prev_state.dtype() || mixed_qkv.dtype() != weights.dtype() {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(mixed_qkv.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_size, seq_len, conv_dim];
    let mut out = vec![
        0u8;
        batch_size
            .saturating_mul(seq_len)
            .saturating_mul(conv_dim)
            .saturating_mul(mixed_qkv.dtype().size_in_bytes())
    ];
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
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            prev_state_storage.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "linear-stateful-conv-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn linear_stateful_conv_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (mixed_qkv, prev_state, weights, kernel_size);
    Ok(None)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct LinearStatefulConvValueDecay {
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
}

#[cfg(test)]
impl candle::CustomOp6 for LinearStatefulConvValueDecay {
    fn name(&self) -> &'static str {
        "linear-stateful-conv-value-decay"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-stateful-conv-value-decay has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        prev_state: &candle::HipStorage,
        prev_state_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
        a: &candle::HipStorage,
        a_layout: &candle::Layout,
        dt_bias: &candle::HipStorage,
        dt_bias_layout: &candle::Layout,
        a_log_exp: &candle::HipStorage,
        a_log_exp_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous()
            && prev_state_layout.is_contiguous()
            && weights_layout.is_contiguous()
            && a_layout.is_contiguous()
            && dt_bias_layout.is_contiguous()
            && a_log_exp_layout.is_contiguous())
        {
            candle::bail!("linear-stateful-conv-value-decay requires contiguous inputs")
        }

        let (batch_size, conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
        let (state_batch, state_conv_dim, state_len) = prev_state_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        let (a_batch, a_seq_len, a_heads) = a_layout.shape().dims3()?;
        let dt_bias_elems = dt_bias_layout.shape().elem_count();
        let a_log_exp_elems = a_log_exp_layout.shape().elem_count();
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || seq_len != self.seq_len
            || state_batch != self.batch_size
            || state_conv_dim != self.conv_dim
            || state_len != self.state_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
            || a_batch != self.batch_size
            || a_seq_len != self.seq_len
            || a_heads != self.num_heads
            || dt_bias_elems != self.num_heads
            || a_log_exp_elems != self.num_heads
        {
            candle::bail!(
                "linear-stateful-conv-value-decay shape mismatch mixed_qkv={:?} prev_state={:?} weights={:?} a={:?} dt_bias={:?} a_log_exp={:?} expected=({}, {}, {}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                prev_state_layout.shape().dims(),
                weights_layout.shape().dims(),
                a_layout.shape().dims(),
                dt_bias_layout.shape().dims(),
                a_log_exp_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size,
                self.num_heads
            )
        }
        if mixed_qkv.dtype() != prev_state.dtype()
            || mixed_qkv.dtype() != weights.dtype()
            || mixed_qkv.dtype() != a.dtype()
            || mixed_qkv.dtype() != dt_bias.dtype()
            || mixed_qkv.dtype() != a_log_exp.dtype()
        {
            candle::bail!(
                "linear-stateful-conv-value-decay requires matching dtypes, got mixed_qkv={:?} prev_state={:?} weights={:?} a={:?} dt_bias={:?} a_log_exp={:?}",
                mixed_qkv.dtype(),
                prev_state.dtype(),
                weights.dtype(),
                a.dtype(),
                dt_bias.dtype(),
                a_log_exp.dtype()
            )
        }

        let device = mixed_qkv.device().clone();
        let storage_dtype = mixed_qkv.dtype();
        let output_shape = candle::Shape::from((
            self.batch_size,
            self.seq_len,
            self.conv_dim + self.num_heads,
        ));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_stateful_conv_value_decay(
                candle::hip::qwen35_dtype_code(storage_dtype)?,
                device.ordinal(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size,
                self.num_heads,
                mixed_qkv.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                    as *const c_void,
                prev_state.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                    as *const c_void,
                weights.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
                a.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
                dt_bias.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
                a_log_exp.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearStatefulConvValueDecayWithState {
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
}

impl candle::CustomOp6 for LinearStatefulConvValueDecayWithState {
    fn name(&self) -> &'static str {
        "linear-stateful-conv-value-decay-with-state"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-stateful-conv-value-decay-with-state has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        prev_state: &candle::HipStorage,
        prev_state_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
        a: &candle::HipStorage,
        a_layout: &candle::Layout,
        dt_bias: &candle::HipStorage,
        dt_bias_layout: &candle::Layout,
        a_log_exp: &candle::HipStorage,
        a_log_exp_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous()
            && prev_state_layout.is_contiguous()
            && weights_layout.is_contiguous()
            && a_layout.is_contiguous()
            && dt_bias_layout.is_contiguous()
            && a_log_exp_layout.is_contiguous())
        {
            candle::bail!("linear-stateful-conv-value-decay-with-state requires contiguous inputs")
        }

        let (batch_size, conv_dim, seq_len) = mixed_qkv_layout.shape().dims3()?;
        let (state_batch, state_conv_dim, state_len) = prev_state_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        let (a_batch, a_seq_len, a_heads) = a_layout.shape().dims3()?;
        let dt_bias_elems = dt_bias_layout.shape().elem_count();
        let a_log_exp_elems = a_log_exp_layout.shape().elem_count();
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || seq_len != self.seq_len
            || state_batch != self.batch_size
            || state_conv_dim != self.conv_dim
            || state_len != self.state_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
            || a_batch != self.batch_size
            || a_seq_len != self.seq_len
            || a_heads != self.num_heads
            || dt_bias_elems != self.num_heads
            || a_log_exp_elems != self.num_heads
        {
            candle::bail!(
                "linear-stateful-conv-value-decay-with-state shape mismatch mixed_qkv={:?} prev_state={:?} weights={:?} a={:?} dt_bias={:?} a_log_exp={:?} expected=({}, {}, {}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                prev_state_layout.shape().dims(),
                weights_layout.shape().dims(),
                a_layout.shape().dims(),
                dt_bias_layout.shape().dims(),
                a_log_exp_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size,
                self.num_heads
            )
        }
        if mixed_qkv.dtype() != prev_state.dtype()
            || mixed_qkv.dtype() != weights.dtype()
            || mixed_qkv.dtype() != a.dtype()
            || mixed_qkv.dtype() != dt_bias.dtype()
            || mixed_qkv.dtype() != a_log_exp.dtype()
        {
            candle::bail!(
                "linear-stateful-conv-value-decay-with-state requires matching dtypes, got mixed_qkv={:?} prev_state={:?} weights={:?} a={:?} dt_bias={:?} a_log_exp={:?}",
                mixed_qkv.dtype(),
                prev_state.dtype(),
                weights.dtype(),
                a.dtype(),
                dt_bias.dtype(),
                a_log_exp.dtype()
            )
        }

        let device = mixed_qkv.device().clone();
        let storage_dtype = mixed_qkv.dtype();
        let flat_width =
            self.seq_len * (self.conv_dim + self.num_heads) + self.conv_dim * self.state_len;
        let output_shape = candle::Shape::from((self.batch_size, flat_width));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_stateful_conv_value_decay_with_state(
                candle::hip::qwen35_dtype_code(storage_dtype)?,
                device.ordinal(),
                self.batch_size,
                self.conv_dim,
                self.seq_len,
                self.state_len,
                self.kernel_size,
                self.num_heads,
                mixed_qkv.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                    as *const c_void,
                prev_state.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                    as *const c_void,
                weights.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
                a.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
                dt_bias.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
                a_log_exp.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

#[cfg(test)]
fn linear_stateful_conv_value_decay_hip(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a = a.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let (batch_size, conv_dim, seq_len) = mixed_qkv.dims3()?;
    let (state_batch, state_conv_dim, state_len) = prev_state.dims3()?;
    let (a_batch, a_seq_len, num_heads) = a.dims3()?;
    if state_batch != batch_size || state_conv_dim != conv_dim {
        candle::bail!(
            "linear-stateful-conv-value-decay state mismatch: mixed_qkv={:?} prev_state={:?}",
            mixed_qkv.dims(),
            prev_state.dims()
        )
    }
    if a_batch != batch_size || a_seq_len != seq_len {
        candle::bail!(
            "linear-stateful-conv-value-decay a mismatch: mixed_qkv={:?} a={:?}",
            mixed_qkv.dims(),
            a.dims()
        )
    }
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = linear_stateful_conv_value_decay_host_buffer(
        &mixed_qkv,
        &prev_state,
        &weights,
        &a,
        &dt_bias,
        &a_log_exp,
        kernel_size,
    )? {
        return hip_tensor_from_host_bytes(mixed_qkv.device(), mixed_qkv.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("linear_stateful_conv_value_decay_hip", &mixed_qkv);
    mixed_qkv.apply_op6_no_bwd(
        &prev_state,
        &weights,
        &a,
        &dt_bias,
        &a_log_exp,
        &LinearStatefulConvValueDecay {
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
#[cfg(all(test, feature = "qwen35-minimal-hip"))]
fn linear_stateful_conv_value_decay_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a = a.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let ordinal = match mixed_qkv.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_state.device().same_device(mixed_qkv.device())
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
    {
        return Ok(None);
    }
    if mixed_qkv.dtype() != prev_state.dtype()
        || mixed_qkv.dtype() != weights.dtype()
        || mixed_qkv.dtype() != a.dtype()
        || mixed_qkv.dtype() != dt_bias.dtype()
        || mixed_qkv.dtype() != a_log_exp.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(mixed_qkv.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_size, seq_len, conv_dim + num_heads];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(mixed_qkv.dtype().size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_linear_stateful_conv_value_decay(
            dtype_code,
            ordinal,
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
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())?
                as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "linear-stateful-conv-value-decay-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
#[cfg(all(test, not(feature = "qwen35-minimal-hip")))]
fn linear_stateful_conv_value_decay_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, kernel_size);
    Ok(None)
}

pub(crate) fn linear_stateful_conv_value_decay_with_state_hip(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a = a.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let (batch_size, conv_dim, seq_len) = mixed_qkv.dims3()?;
    let (state_batch, state_conv_dim, state_len) = prev_state.dims3()?;
    let (a_batch, a_seq_len, num_heads) = a.dims3()?;
    if state_batch != batch_size || state_conv_dim != conv_dim {
        candle::bail!(
            "linear-stateful-conv-value-decay-with-state mismatch: mixed_qkv={:?} prev_state={:?}",
            mixed_qkv.dims(),
            prev_state.dims()
        )
    }
    if a_batch != batch_size || a_seq_len != seq_len {
        candle::bail!(
            "linear-stateful-conv-value-decay-with-state a mismatch: mixed_qkv={:?} a={:?}",
            mixed_qkv.dims(),
            a.dims()
        )
    }
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = linear_stateful_conv_value_decay_with_state_host_buffer(
        &mixed_qkv,
        &prev_state,
        &weights,
        &a,
        &dt_bias,
        &a_log_exp,
        kernel_size,
    )? {
        return hip_tensor_from_host_bytes(mixed_qkv.device(), mixed_qkv.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("linear_stateful_conv_value_decay_with_state_hip", &mixed_qkv);
    mixed_qkv.apply_op6_no_bwd(
        &prev_state,
        &weights,
        &a,
        &dt_bias,
        &a_log_exp,
        &LinearStatefulConvValueDecayWithState {
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn linear_stateful_conv_value_decay_with_state_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_state = prev_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a = a.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let ordinal = match mixed_qkv.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_state.device().same_device(mixed_qkv.device())
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
    {
        return Ok(None);
    }
    if mixed_qkv.dtype() != prev_state.dtype()
        || mixed_qkv.dtype() != weights.dtype()
        || mixed_qkv.dtype() != a.dtype()
        || mixed_qkv.dtype() != dt_bias.dtype()
        || mixed_qkv.dtype() != a_log_exp.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(mixed_qkv.dtype()) else {
        return Ok(None);
    };
    let flat_width = seq_len * (conv_dim + num_heads) + conv_dim * state_len;
    let shape = vec![batch_size, flat_width];
    let mut out = vec![
        0u8;
        batch_size
            .saturating_mul(flat_width)
            .saturating_mul(mixed_qkv.dtype().size_in_bytes())
    ];
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
            mixed_qkv_storage.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                as *const c_void,
            prev_state_storage.raw_device_ptr_with_offset(prev_state_layout.start_offset())?
                as *const c_void,
            weights_storage.raw_device_ptr_with_offset(weights_layout.start_offset())?
                as *const c_void,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())?
                as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "linear-stateful-conv-value-decay-with-state-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn linear_stateful_conv_value_decay_with_state_host_buffer(
    mixed_qkv: &Tensor,
    prev_state: &Tensor,
    weights: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
    kernel_size: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, kernel_size);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct LinearDecodePrepare {
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    state_len: usize,
    kernel_size: usize,
    head_repeat: usize,
}

impl candle::CustomOp6 for LinearDecodePrepare {
    fn name(&self) -> &'static str {
        "linear-decode-prepare"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-decode-prepare has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        prev_conv_state: &candle::HipStorage,
        prev_conv_state_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
        a_beta_raw: &candle::HipStorage,
        a_beta_raw_layout: &candle::Layout,
        dt_bias: &candle::HipStorage,
        dt_bias_layout: &candle::Layout,
        a_log_exp: &candle::HipStorage,
        a_log_exp_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous()
            && prev_conv_state_layout.is_contiguous()
            && weights_layout.is_contiguous()
            && a_beta_raw_layout.is_contiguous()
            && dt_bias_layout.is_contiguous()
            && a_log_exp_layout.is_contiguous())
        {
            candle::bail!("linear-decode-prepare requires contiguous inputs")
        }
        let device = mixed_qkv.device().clone();
        let packed_width = 2 * self.head_k_dim + self.head_v_dim + 2;
        let output_shape = candle::Shape::from((self.batch_size * self.num_v_heads, packed_width));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(DType::F32.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_decode_prepare(
                candle::hip::qwen35_dtype_code(mixed_qkv.dtype())?,
                device.ordinal(),
                self.batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.state_len,
                self.kernel_size,
                self.head_repeat,
                mixed_qkv.raw_device_ptr_with_offset(mixed_qkv_layout.start_offset())?
                    as *const c_void,
                prev_conv_state.raw_device_ptr_with_offset(prev_conv_state_layout.start_offset())?
                    as *const c_void,
                weights.raw_device_ptr_with_offset(weights_layout.start_offset())? as *const c_void,
                a_beta_raw.raw_device_ptr_with_offset(a_beta_raw_layout.start_offset())?
                    as *const c_void,
                dt_bias.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
                a_log_exp.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(DType::F32, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearDecodeApply {
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
}

impl candle::CustomOp2 for LinearDecodeApply {
    fn name(&self) -> &'static str {
        "linear-decode-apply"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-decode-apply has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        packed: &candle::HipStorage,
        packed_layout: &candle::Layout,
        initial_state: &candle::HipStorage,
        initial_state_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(packed_layout.is_contiguous() && initial_state_layout.is_contiguous()) {
            candle::bail!("linear-decode-apply requires contiguous inputs")
        }
        if packed.dtype() != DType::F32 || initial_state.dtype() != DType::F32 {
            candle::bail!("linear-decode-apply requires F32 packed/state inputs")
        }
        let device = packed.device().clone();
        let value_dim = self.num_v_heads * self.head_v_dim;
        let output_shape = candle::Shape::from((
            self.batch_size,
            value_dim + self.num_v_heads * self.head_k_dim * self.head_v_dim,
        ));
        let elem_count = output_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(DType::F32.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_linear_decode_apply(
                device.ordinal(),
                self.batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                packed.raw_device_ptr_with_offset(packed_layout.start_offset())? as *const c_void,
                initial_state.raw_device_ptr_with_offset(initial_state_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(DType::F32, output)?,
                device,
            ),
            output_shape,
        ))
    }
}

pub(crate) fn linear_decode_step_hip(
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
    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_conv_state = prev_conv_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a_beta_raw = a_beta_raw.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let initial_state = initial_state.contiguous()?;
    let (batch_size, _conv_dim, seq_len) = mixed_qkv.dims3()?;
    let (_, _, state_len) = prev_conv_state.dims3()?;
    if seq_len != 1 {
        candle::bail!("linear-decode-step expects seq_len=1, got {seq_len}")
    }
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = linear_decode_step_host_buffer(
        &mixed_qkv,
        &prev_conv_state,
        &weights,
        &a_beta_raw,
        &dt_bias,
        &a_log_exp,
        &initial_state,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        kernel_size,
        head_repeat,
    )? {
        return hip_tensor_from_host_bytes(mixed_qkv.device(), DType::F32, shape, output);
    }
    trace_hip_wrapper_fallback("linear_decode_step_prepare", &mixed_qkv);
    let packed = mixed_qkv.apply_op6_no_bwd(
        &prev_conv_state,
        &weights,
        &a_beta_raw,
        &dt_bias,
        &a_log_exp,
        &LinearDecodePrepare {
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_len,
            kernel_size,
            head_repeat,
        },
    )?;
    trace_hip_wrapper_fallback("linear_decode_step_apply", &packed);
    packed.apply_op2_no_bwd(
        &initial_state,
        &LinearDecodeApply {
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn linear_decode_step_host_buffer(
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
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let mixed_qkv = mixed_qkv.contiguous()?;
    let prev_conv_state = prev_conv_state.contiguous()?;
    let weights = weights.contiguous()?;
    let a_beta_raw = a_beta_raw.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_log_exp = a_log_exp.contiguous()?;
    let initial_state = initial_state.contiguous()?;
    let ordinal = match mixed_qkv.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(prev_conv_state.device().same_device(mixed_qkv.device())
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
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(mixed_qkv.dtype()) else {
        return Ok(None);
    };
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
            packed_device_ptr as *mut c_void,
        )
    };
    if prepare_status != 0 {
        hip::unregister_host_mapping(packed_host_ptr);
        return Err(hip::hip_error(
            "linear-decode-prepare-host-buffer",
            prepare_status,
        ));
    }
    let output_width = num_v_heads * head_v_dim + num_v_heads * head_k_dim * head_v_dim;
    let output_shape = vec![batch_size, output_width];
    let mut out =
        vec![0u8; batch_size.saturating_mul(output_width).saturating_mul(DType::F32.size_in_bytes())];
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
            initial_state_storage.raw_device_ptr_with_offset(initial_state_layout.start_offset())?
                as *const c_void,
            out_device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(out_host_ptr);
    hip::unregister_host_mapping(packed_host_ptr);
    if apply_status != 0 {
        return Err(hip::hip_error(
            "linear-decode-apply-host-buffer",
            apply_status,
        ));
    }
    Ok(Some((out, output_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn linear_decode_step_host_buffer(
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
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
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

#[derive(Debug, Clone, Copy)]
struct FullAttentionPrefillMegakernel {
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
}

impl candle::CustomOp3 for FullAttentionPrefillMegakernel {
    fn name(&self) -> &'static str {
        "full-attention-prefill-megakernel"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("full-attention-prefill-megakernel has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device().clone();
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();
        let total_rows = self.batch_size * self.q_heads * self.q_len;
        let cfg = LaunchConfig::for_num_elems(total_rows as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    self.batch_size,
                    self.q_heads,
                    self.kv_heads,
                    self.q_len,
                    self.kv_len,
                    self.head_dim,
                    self.num_kv_groups,
                    self.scale,
                    self.seqlen_offset
                );
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match query.dtype() {
            DType::F16 => launch!(half::f16, "full_attention_prefill_f16"),
            DType::F32 => launch!(f32, "full_attention_prefill_f32"),
            DType::BF16 => launch!(half::bf16, "full_attention_prefill_bf16"),
            other => candle::bail!("full-attention-prefill-megakernel unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device();
        let dtype = match query.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("full-attention-prefill-megakernel unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            query.dtype(),
            "full-attention-prefill-megakernel",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("full-attention-prefill-megakernel");
        candle_metal_kernels::call_full_attention_prefill(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            self.batch_size,
            self.q_heads,
            self.kv_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            self.num_kv_groups,
            self.scale,
            self.seqlen_offset,
            candle_metal_kernels::BufferOffset {
                buffer: query.buffer(),
                offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: key.buffer(),
                offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: value.buffer(),
                offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(MetalError::from)?;
        Ok((
            candle::MetalStorage::new(output, device.clone(), elem_count, query.dtype()),
            out_shape,
        ))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device().clone();
        let storage_dtype = query.dtype();
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(DType::F32.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let query_ptr = query.raw_device_ptr_with_offset(query_layout.start_offset())?;
        let key_ptr = key.raw_device_ptr_with_offset(key_layout.start_offset())?;
        let value_ptr = value.raw_device_ptr_with_offset(value_layout.start_offset())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_full_attention_prefill(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                self.batch_size,
                self.q_heads,
                self.kv_heads,
                self.q_len,
                self.kv_len,
                self.head_dim,
                self.num_kv_groups,
                self.scale,
                self.seqlen_offset,
                query_ptr as *const c_void,
                key_ptr as *const c_void,
                value_ptr as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(DType::F32, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn full_attention_prefill_megakernel(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Tensor> {
    let (batch_size, q_heads, q_len, head_dim) = query.dims4()?;
    let (_, kv_heads, kv_len, value_head_dim) = value.dims4()?;
    if value_head_dim != head_dim {
        candle::bail!(
            "full-attention-prefill-megakernel requires matching head dims, got q={} v={}",
            head_dim,
            value_head_dim
        )
    }
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        full_attention_prefill_host_buffer(query, key, value, num_kv_groups, scale, seqlen_offset)?
    {
        return hip_tensor_from_host_bytes(query.device(), DType::F32, shape, output);
    }
    trace_hip_wrapper_fallback("full_attention_prefill_megakernel", &query);
    query.apply_op3_no_bwd(
        key,
        value,
        &FullAttentionPrefillMegakernel {
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn full_attention_prefill_host_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let ordinal = match query.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key.device().same_device(query.device()) && value.device().same_device(query.device())) {
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
    if !(query_layout.is_contiguous() && key_layout.is_contiguous() && value_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
    let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
    let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
        value_layout.shape().dims4()?;
    if key_batch != batch_size
        || value_batch != batch_size
        || value_kv_heads != kv_heads
        || value_kv_len != kv_len
        || key_head_dim != head_dim
        || value_head_dim != head_dim
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_size, q_heads, q_len, head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(DType::F32.size_in_bytes())
    ];
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
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "full-attention-prefill-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn full_attention_prefill_host_buffer(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (query, key, value, num_kv_groups, scale, seqlen_offset);
    Ok(None)
}

pub(crate) fn full_attention_decode_megakernel(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Tensor> {
    let (_, _, q_len, _) = query.dims4()?;
    if q_len != 1 {
        candle::bail!("full-attention-decode-megakernel requires q_len == 1, got {q_len}")
    }
    full_attention_prefill_megakernel(query, key, value, num_kv_groups, scale, seqlen_offset)
}

#[cfg(test)]
fn paged_attention_decode_fallback(
    queries: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    let (batch_queries, head_dim) = queries.dims2()?;
    let (kv_len, _) = key.dims2()?;
    let query = queries
        .contiguous()?
        .reshape((1, batch_queries, 1, head_dim))?
        .to_dtype(DType::F32)?;
    let key = key
        .contiguous()?
        .reshape((1, 1, kv_len, head_dim))?
        .to_dtype(DType::F32)?;
    let value = value
        .contiguous()?
        .reshape((1, 1, kv_len, head_dim))?
        .to_dtype(DType::F32)?;
    let key = repeat_kv(key, batch_queries)?.contiguous()?;
    let value = repeat_kv(value, batch_queries)?.contiguous()?;
    let key_t = key.transpose(2, 3)?.contiguous()?;
    let attn_weights = ops::softmax_last_dim(
        &((query.matmul(&key_t)?) * (1.0f64 / (head_dim as f64).sqrt()))?,
    )?;
    Ok(attn_weights
        .matmul(&value)?
        .reshape((batch_queries, head_dim))?)
}

#[cfg(test)]
pub fn paged_attention_decode_megakernel(
    queries: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    let (batch_queries, head_dim) = queries.dims2()?;
    let (kv_len, key_head_dim) = key.dims2()?;
    let (value_kv_len, value_head_dim) = value.dims2()?;
    if key_head_dim != head_dim || value_head_dim != head_dim || value_kv_len != kv_len {
        candle::bail!(
            "paged-attention-decode-megakernel shape mismatch: query={:?} key={:?} value={:?}",
            queries.dims(),
            key.dims(),
            value.dims()
        )
    }
    if batch_queries == 0 {
        candle::bail!("paged-attention-decode-megakernel requires at least one query row")
    }
    if matches!(queries.device().location(), DeviceLocation::Cuda { .. }) && head_dim > 128 {
        return paged_attention_decode_fallback(queries, key, value);
    }
    let query = queries
        .contiguous()?
        .reshape((1, batch_queries, 1, head_dim))?;
    let key = key.contiguous()?.reshape((1, 1, kv_len, head_dim))?;
    let value = value.contiguous()?.reshape((1, 1, kv_len, head_dim))?;
    Ok(full_attention_prefill_megakernel(
        &query,
        &key,
        &value,
        batch_queries,
        1.0f32 / (head_dim as f32).sqrt(),
        kv_len.saturating_sub(1),
    )?
    .reshape((batch_queries, head_dim))?)
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct DeltaStateUpdate;

#[cfg(feature = "metal")]
impl candle::CustomOp3 for DeltaStateUpdate {
    fn name(&self) -> &'static str {
        "delta-state-update"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-state-update has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        weighted_key: &candle::MetalStorage,
        weighted_key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-update requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (wk_batch_heads, chunk_size, wk_k_head_dim) = weighted_key_layout.shape().dims3()?;
        let (value_batch_heads, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims3()?;
        if wk_batch_heads != batch_heads
            || value_batch_heads != batch_heads
            || wk_k_head_dim != k_head_dim
            || value_v_head_dim != v_head_dim
            || value_chunk_size != chunk_size
        {
            candle::bail!(
                "delta-state-update shape mismatch: prev={:?} weighted_key={:?} value={:?}",
                prev_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-state-update unsupported dtype {other:?}"),
        };
        let elem_count = prev_layout.shape().elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-state-update")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-state-update");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let wk = candle_metal_kernels::BufferOffset {
            buffer: weighted_key.buffer(),
            offset_in_bytes: weighted_key_layout.start_offset()
                * weighted_key.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_state_update(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            wk,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, prev_layout.shape().clone()))
    }
}

pub(crate) fn delta_state_update(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &Tensor,
    _use_kernel: bool,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if _use_kernel && matches!(prev_state_scaled.device().location(), DeviceLocation::Metal { .. }) {
        prev_state_scaled.apply_op3_no_bwd(weighted_key, value, &DeltaStateUpdate)
    }
    weighted_key
        .transpose(2, 1)?
        .matmul(value)?
        .broadcast_add(prev_state_scaled)
}

#[derive(Debug, Clone, Copy)]
struct DeltaStateScan;

impl candle::CustomOp3 for DeltaStateScan {
    fn name(&self) -> &'static str {
        "delta-state-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-state-scan has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::CudaStorage,
        packed_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
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
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let packed_scan = packed_scan.as_cuda_slice::<$ty>()?;
                let packed_scan = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => packed_scan.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&packed_scan);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_state_scan_f16"),
            DType::F32 => launch!(f32, "delta_state_scan_f32"),
            DType::BF16 => launch!(half::bf16, "delta_state_scan_bf16"),
            other => candle::bail!("delta-state-scan unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::MetalStorage,
        packed_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
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
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-state-scan unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, initial_state.dtype(), "delta-state-scan")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-state-scan");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let packed = candle_metal_kernels::BufferOffset {
            buffer: packed_scan.buffer(),
            offset_in_bytes: packed_layout.start_offset() * packed_scan.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_state_scan(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            packed,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::HipStorage,
        packed_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
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
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_state_scan(
                dtype_code,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                v_head_dim,
                initial_state.raw_device_ptr_with_offset(initial_layout.start_offset())?
                    as *const c_void,
                packed_scan.raw_device_ptr_with_offset(packed_layout.start_offset())?
                    as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_state_scan(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = delta_state_scan_host_buffer(initial_state, packed_scan, value)? {
        return hip_tensor_from_host_bytes(initial_state.device(), initial_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_state_scan", initial_state);
    initial_state.apply_op3_no_bwd(packed_scan, value, &DeltaStateScan)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_state_scan_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let packed_scan = packed_scan.contiguous()?;
    let value = value.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_scan.device().same_device(initial_state.device())
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
    if !(initial_layout.is_contiguous() && packed_layout.is_contiguous() && value_layout.is_contiguous())
    {
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
        || initial_state.dtype() != packed_scan.dtype()
        || initial_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks + 1, k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())?
                as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())?
                as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-state-scan-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_state_scan_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (initial_state, packed_scan, value);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkFused;

impl candle::CustomOp3 for DeltaChunkFused {
    fn name(&self) -> &'static str {
        "delta-chunk-fused"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-fused has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::CudaStorage,
        packed_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let packed_chunk = packed_chunk.as_cuda_slice::<$ty>()?;
                let packed_chunk = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => packed_chunk.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, chunk_size, k_head_dim, v_head_dim);
                builder.arg(&prev_state);
                builder.arg(&packed_chunk);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_fused_f16"),
            DType::F32 => launch!(f32, "delta_chunk_fused_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_fused_bf16"),
            other => candle::bail!("delta-chunk-fused unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::MetalStorage,
        packed_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-fused unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-fused")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-fused");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let packed = candle_metal_kernels::BufferOffset {
            buffer: packed_chunk.buffer(),
            offset_in_bytes: packed_layout.start_offset() * packed_chunk.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_fused(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            packed,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::HipStorage,
        packed_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_chunk_fused(
                dtype_code,
                device.ordinal(),
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_state.raw_device_ptr_with_offset(prev_layout.start_offset())? as *const c_void,
                packed_chunk.raw_device_ptr_with_offset(packed_layout.start_offset())?
                    as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_chunk_fused(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = delta_chunk_fused_host_buffer(prev_state, packed_chunk, value)? {
        return hip_tensor_from_host_bytes(prev_state.device(), prev_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_chunk_fused", prev_state);
    prev_state.apply_op3_no_bwd(packed_chunk, value, &DeltaChunkFused)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_chunk_fused_host_buffer(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let prev_state = prev_state.contiguous()?;
    let packed_chunk = packed_chunk.contiguous()?;
    let value = value.contiguous()?;
    let ordinal = match prev_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_chunk.device().same_device(prev_state.device())
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
    if !(prev_layout.is_contiguous() && packed_layout.is_contiguous() && value_layout.is_contiguous())
    {
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
        || prev_state.dtype() != packed_chunk.dtype()
        || prev_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(prev_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, 2 * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(prev_state.dtype().size_in_bytes())
    ];
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
            prev_storage.raw_device_ptr_with_offset(prev_layout.start_offset())? as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())?
                as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-fused-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_chunk_fused_host_buffer(
    prev_state: &Tensor,
    packed_chunk: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (prev_state, packed_chunk, value);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaRecurrentPrefill;

impl candle::CustomOp6 for DeltaRecurrentPrefill {
    fn name(&self) -> &'static str {
        "delta-recurrent-prefill"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-recurrent-prefill has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
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
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, seq_len, k_head_dim, v_head_dim);
                builder.arg(&initial_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_recurrent_prefill_f16"),
            DType::F32 => launch!(f32, "delta_recurrent_prefill_f32"),
            DType::BF16 => launch!(half::bf16, "delta_recurrent_prefill_bf16"),
            other => candle::bail!("delta-recurrent-prefill unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
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
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-recurrent-prefill unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, initial_state.dtype(), "delta-recurrent-prefill")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-recurrent-prefill");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_recurrent_prefill(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            initial,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
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
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_recurrent_prefill(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state.raw_device_ptr_with_offset(initial_layout.start_offset())?
                    as *const c_void,
                query.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                key.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                beta.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
                g.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_recurrent_prefill(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_recurrent_prefill_host_buffer(initial_state, query, key, value, beta, g)?
    {
        return hip_tensor_from_host_bytes(initial_state.device(), initial_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_recurrent_prefill", initial_state);
    initial_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaRecurrentPrefill)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_recurrent_prefill_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.device().same_device(initial_state.device())
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
    )
    else {
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
        || initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, seq_len + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())?
                as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-recurrent-prefill-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_recurrent_prefill_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkSinglePrefill;

impl candle::CustomOp6 for DeltaChunkSinglePrefill {
    fn name(&self) -> &'static str {
        "delta-chunk-single-prefill"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-single-prefill has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-single-prefill requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-single-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_chunk_single_prefill(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                query.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                key.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                beta.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
                g.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_chunk_single_prefill(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_chunk_single_prefill_host_buffer(initial_state, query, key, value, beta, g)?
    {
        return hip_tensor_from_host_bytes(initial_state.device(), initial_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_chunk_single_prefill", initial_state);
    initial_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkSinglePrefill)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_chunk_single_prefill_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.device().same_device(initial_state.device())
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
    )
    else {
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
        || initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "delta-chunk-single-prefill-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_chunk_single_prefill_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkStepRaw;

impl candle::CustomOp6 for DeltaChunkStepRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-step-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-step-raw has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, chunk_size, k_head_dim, v_head_dim);
                builder.arg(&prev_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_step_f16"),
            DType::F32 => launch!(f32, "delta_chunk_step_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_step_bf16"),
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-step-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-step-raw");
        let prev_offset = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let query_offset = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key_offset = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value_offset = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta_offset = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g_offset = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        let use_split_kernel = matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_SPLIT_KERNEL").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );
        if use_split_kernel {
            let v_new_shape = candle::Shape::from_dims(&[batch_heads, chunk_size, v_head_dim]);
            let v_new_elem_count = v_new_shape.elem_count();
            let v_new_output =
                device.new_buffer(v_new_elem_count, prev_state.dtype(), "delta-chunk-v-new")?;
            candle_metal_kernels::call_delta_chunk_readout_split(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
                &v_new_output,
            )
            .map_err(MetalError::from)?;
            candle_metal_kernels::call_delta_chunk_state_update_raw(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                candle_metal_kernels::BufferOffset {
                    buffer: prev_state.buffer(),
                    offset_in_bytes: prev_layout.start_offset()
                        * prev_state.dtype().size_in_bytes(),
                },
                candle_metal_kernels::BufferOffset {
                    buffer: key.buffer(),
                    offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
                },
                candle_metal_kernels::BufferOffset {
                    buffer: &v_new_output,
                    offset_in_bytes: 0,
                },
                candle_metal_kernels::BufferOffset {
                    buffer: g.buffer(),
                    offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
                },
                chunk_size,
                &output,
            )
            .map_err(MetalError::from)?;
            let storage =
                candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
            return Ok((storage, out_shape));
        }
        let use_2d_kernel = delta_chunk_step_2d_enabled() && chunk_size <= 16;
        if use_2d_kernel {
            candle_metal_kernels::call_delta_chunk_step_2d(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
            )
            .map_err(MetalError::from)?;
        } else {
            candle_metal_kernels::call_delta_chunk_step(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
            )
            .map_err(MetalError::from)?;
        }
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let mut output = vec![$zero; elem_count];
                let host_ptr = output.as_mut_ptr() as *const c_void;
                let device_ptr =
                    hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len() * std::mem::size_of::<$ty>())?;
                let status = unsafe {
                    hip::ffi::dotcache_qwen35_hip_delta_chunk_step(
                        hip::dtype_code(storage_dtype)?,
                        device.ordinal(),
                        batch_heads,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        prev_state.raw_device_ptr_with_offset(prev_layout.start_offset())?
                            as *const c_void,
                        query.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                        key.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                        value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                        beta.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
                        g.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
                        device_ptr as *mut c_void,
                    )
                };
                hip::unregister_host_mapping(host_ptr);
                if status != 0 {
                    return Err(hip::hip_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        }
    }
}

pub(super) fn delta_chunk_step_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_chunk_step_raw_host_buffer(prev_state, query, key, value, beta, g)?
    {
        return hip_tensor_from_host_bytes(prev_state.device(), prev_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_chunk_step_raw", prev_state);
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkStepRaw)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(super) fn delta_chunk_step_raw_host_buffer(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let prev_state = prev_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let ordinal = match prev_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.device().same_device(prev_state.device())
        && key.device().same_device(prev_state.device())
        && value.device().same_device(prev_state.device())
        && beta.device().same_device(prev_state.device())
        && g.device().same_device(prev_state.device()))
    {
        return Ok(None);
    }
    let (prev_storage, prev_layout) = prev_state.storage_and_layout();
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (
        Storage::Hip(prev_storage),
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(value_storage),
        Storage::Hip(beta_storage),
        Storage::Hip(g_storage),
    ) = (
        &*prev_storage,
        &*query_storage,
        &*key_storage,
        &*value_storage,
        &*beta_storage,
        &*g_storage,
    ) else {
        return Ok(None);
    };
    if !(prev_layout.is_contiguous()
        && query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && value_layout.is_contiguous()
        && beta_layout.is_contiguous()
        && g_layout.is_contiguous())
    {
        return Ok(None);
    }

    let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
    let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
    let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
    let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
    let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
    let (g_bh, g_chunk) = g_layout.shape().dims2()?;
    if query_bh != batch_heads
        || key_bh != batch_heads
        || value_bh != batch_heads
        || beta_bh != batch_heads
        || g_bh != batch_heads
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

    let dtype = prev_state.dtype();
    if !(dtype == query.dtype()
        && dtype == key.dtype()
        && dtype == value.dtype()
        && dtype == beta.dtype()
        && dtype == g.dtype())
    {
        return Ok(None);
    }

    let out_shape = vec![batch_heads, chunk_size + k_head_dim, v_head_dim];
    let mut output = vec![0u8; out_shape.iter().product::<usize>() * dtype.size_in_bytes()];
    let host_ptr = output.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, output.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_chunk_step(
            hip::dtype_code(dtype)?,
            ordinal,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_storage.raw_device_ptr_with_offset(prev_layout.start_offset())? as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-step-raw-host-buffer", status));
    }
    Ok(Some((output, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_chunk_step_raw_host_buffer(
    _prev_state: &Tensor,
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _beta: &Tensor,
    _g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkStepWindowedRaw;

impl candle::CustomOp6 for DeltaChunkStepWindowedRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-step-windowed-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-step-windowed-raw has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
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
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&prev_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_step_windowed_f16"),
            DType::F32 => launch!(f32, "delta_chunk_step_windowed_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_step_windowed_bf16"),
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
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
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        };
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            prev_state.dtype(),
            "delta-chunk-step-windowed-raw",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-step-windowed-raw");

        let elem_bytes = prev_state.dtype().size_in_bytes();
        let query_chunk_elems = chunk_size * k_head_dim;
        let value_chunk_elems = chunk_size * v_head_dim;
        let scalar_chunk_elems = chunk_size;
        let query_bh_stride = num_chunks * query_chunk_elems;
        let value_bh_stride = num_chunks * value_chunk_elems;
        let scalar_bh_stride = num_chunks * scalar_chunk_elems;
        let initial_prev_offset = prev_layout.start_offset() * elem_bytes;
        let output_state_offset = total_tokens * v_head_dim * elem_bytes;
        let output_state_bh_stride = total_rows * v_head_dim;

        for chunk_idx in 0..num_chunks {
            let query_offset =
                (query_layout.start_offset() + chunk_idx * query_chunk_elems) * elem_bytes;
            let key_offset =
                (key_layout.start_offset() + chunk_idx * query_chunk_elems) * elem_bytes;
            let value_offset =
                (value_layout.start_offset() + chunk_idx * value_chunk_elems) * elem_bytes;
            let scalar_offset =
                (beta_layout.start_offset() + chunk_idx * scalar_chunk_elems) * elem_bytes;
            let g_scalar_offset =
                (g_layout.start_offset() + chunk_idx * scalar_chunk_elems) * elem_bytes;
            let prev = if chunk_idx == 0 {
                candle_metal_kernels::BufferOffset {
                    buffer: prev_state.buffer(),
                    offset_in_bytes: initial_prev_offset,
                }
            } else {
                candle_metal_kernels::BufferOffset {
                    buffer: &output,
                    offset_in_bytes: output_state_offset,
                }
            };
            let prev_state_bh_stride = if chunk_idx == 0 {
                k_head_dim * v_head_dim
            } else {
                output_state_bh_stride
            };
            let query = candle_metal_kernels::BufferOffset {
                buffer: query.buffer(),
                offset_in_bytes: query_offset,
            };
            let key = candle_metal_kernels::BufferOffset {
                buffer: key.buffer(),
                offset_in_bytes: key_offset,
            };
            let value = candle_metal_kernels::BufferOffset {
                buffer: value.buffer(),
                offset_in_bytes: value_offset,
            };
            let beta = candle_metal_kernels::BufferOffset {
                buffer: beta.buffer(),
                offset_in_bytes: scalar_offset,
            };
            let g = candle_metal_kernels::BufferOffset {
                buffer: g.buffer(),
                offset_in_bytes: g_scalar_offset,
            };
            if delta_chunk_step_windowed_2d_enabled() {
                candle_metal_kernels::call_delta_chunk_step_windowed_2d(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    dtype,
                    batch_heads,
                    chunk_size,
                    k_head_dim,
                    v_head_dim,
                    prev_state_bh_stride,
                    query_bh_stride,
                    value_bh_stride,
                    scalar_bh_stride,
                    total_rows,
                    chunk_idx * chunk_size,
                    total_tokens,
                    prev,
                    query,
                    key,
                    value,
                    beta,
                    g,
                    &output,
                )
                .map_err(MetalError::from)?;
            } else {
                candle_metal_kernels::call_delta_chunk_step_windowed(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    dtype,
                    batch_heads,
                    chunk_size,
                    k_head_dim,
                    v_head_dim,
                    prev_state_bh_stride,
                    query_bh_stride,
                    value_bh_stride,
                    scalar_bh_stride,
                    total_rows,
                    chunk_idx * chunk_size,
                    total_tokens,
                    prev,
                    query,
                    key,
                    value,
                    beta,
                    g,
                    &output,
                )
                .map_err(MetalError::from)?;
            }
        }

        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
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
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let mut output = vec![$zero; elem_count];
                let host_ptr = output.as_mut_ptr() as *const c_void;
                let device_ptr =
                    hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len() * std::mem::size_of::<$ty>())?;
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_chunk_windowed(
                        dtype_code,
                        device.ordinal(),
                        batch_heads,
                        num_chunks,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        prev_state.raw_device_ptr_with_offset(prev_layout.start_offset())?
                            as *const c_void,
                        query.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                        key.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                        value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                        beta.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
                        g.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
                        device_ptr as *mut c_void,
                    )
                };
                hip::unregister_host_mapping(host_ptr);
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        }
    }
}

pub(super) fn delta_chunk_step_windowed_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_chunk_step_windowed_raw_host_buffer(prev_state, query, key, value, beta, g)?
    {
        return hip_tensor_from_host_bytes(prev_state.device(), prev_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_chunk_step_windowed_raw", prev_state);
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkStepWindowedRaw)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(super) fn delta_chunk_step_windowed_raw_host_buffer(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let prev_state = prev_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let ordinal = match prev_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.device().same_device(prev_state.device())
        && key.device().same_device(prev_state.device())
        && value.device().same_device(prev_state.device())
        && beta.device().same_device(prev_state.device())
        && g.device().same_device(prev_state.device()))
    {
        return Ok(None);
    }
    let (prev_storage, prev_layout) = prev_state.storage_and_layout();
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (
        Storage::Hip(prev_storage),
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(value_storage),
        Storage::Hip(beta_storage),
        Storage::Hip(g_storage),
    ) = (
        &*prev_storage,
        &*query_storage,
        &*key_storage,
        &*value_storage,
        &*beta_storage,
        &*g_storage,
    ) else {
        return Ok(None);
    };
    if !(prev_layout.is_contiguous()
        && query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && value_layout.is_contiguous()
        && beta_layout.is_contiguous()
        && g_layout.is_contiguous())
    {
        return Ok(None);
    }

    let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
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

    let dtype = prev_state.dtype();
    if !(dtype == query.dtype()
        && dtype == key.dtype()
        && dtype == value.dtype()
        && dtype == beta.dtype()
        && dtype == g.dtype())
    {
        return Ok(None);
    }

    let total_tokens = num_chunks * chunk_size;
    let out_shape = vec![batch_heads, total_tokens + k_head_dim, v_head_dim];
    let mut output = vec![0u8; out_shape.iter().product::<usize>() * dtype.size_in_bytes()];
    let host_ptr = output.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, output.len())?;
    let status = unsafe {
        candle::hip::ffi::qwen35_hip_delta_chunk_windowed(
            candle::hip::qwen35_dtype_code(dtype)?,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_storage.raw_device_ptr_with_offset(prev_layout.start_offset())? as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(candle::hip::qwen35_error(
            "delta-chunk-step-windowed-raw-host-buffer",
            status,
        ));
    }
    Ok(Some((output, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn delta_chunk_step_windowed_raw_host_buffer(
    _prev_state: &Tensor,
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _beta: &Tensor,
    _g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    Ok(None)
}

#[allow(dead_code)]
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct DeltaChunkReadoutRaw;

#[cfg(test)]
impl candle::CustomOp6 for DeltaChunkReadoutRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-readout-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-readout-raw has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-readout-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-readout-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-readout-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, 2 * chunk_size, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-readout-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-readout-raw");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_readout(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }
}

#[cfg(test)]
fn delta_chunk_readout_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkReadoutRaw)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct DeltaChunkStateUpdateRaw;

#[cfg(test)]
impl candle::CustomOp4 for DeltaChunkStateUpdateRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-state-update-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-state-update-raw has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        v_new: &candle::MetalStorage,
        v_new_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && key_layout.is_contiguous()
            && v_new_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-state-update-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (key_bh, chunk_size, key_k) = key_layout.shape().dims3()?;
        let (v_new_bh, v_new_chunk, v_new_v) = v_new_layout.shape().dims3()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if key_bh != batch_heads
            || v_new_bh != batch_heads
            || g_bh != batch_heads
            || v_new_chunk != chunk_size
            || g_chunk != chunk_size
            || key_k != k_head_dim
            || v_new_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-state-update-raw shape mismatch: prev={:?} key={:?} v_new={:?} g={:?}",
                prev_layout.shape().dims(),
                key_layout.shape().dims(),
                v_new_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-state-update-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            prev_state.dtype(),
            "delta-chunk-state-update-raw",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-state-update-raw");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let v_new = candle_metal_kernels::BufferOffset {
            buffer: v_new.buffer(),
            offset_in_bytes: v_new_layout.start_offset() * v_new.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_state_update_raw(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            key,
            v_new,
            g,
            0,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }
}

#[cfg(test)]
fn delta_chunk_state_update_raw(
    prev_state: &Tensor,
    key: &Tensor,
    v_new: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op4_no_bwd(key, v_new, g, &DeltaChunkStateUpdateRaw)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkScanRaw;

impl candle::CustomOp6 for DeltaChunkScanRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-scan-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-scan-raw has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_scan_raw_f16"),
            DType::F32 => launch!(f32, "delta_chunk_scan_raw_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_scan_raw_bf16"),
            other => candle::bail!("delta-chunk-scan-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-scan-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, initial_state.dtype(), "delta-chunk-scan-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-scan-raw");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_scan_raw(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
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
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_chunk_scan_raw(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                v_head_dim,
                initial_state.raw_device_ptr_with_offset(initial_layout.start_offset())?
                    as *const c_void,
                query.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                key.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                beta.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
                g.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_chunk_scan_raw(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_chunk_scan_raw_host_buffer(initial_state, query, key, value, beta, g)?
    {
        return hip_tensor_from_host_bytes(initial_state.device(), initial_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_chunk_scan_raw", initial_state);
    initial_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkScanRaw)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_chunk_scan_raw_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(query.device().same_device(initial_state.device())
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
    )
    else {
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
        || initial_state.dtype() != query.dtype()
        || initial_state.dtype() != key.dtype()
        || initial_state.dtype() != value.dtype()
        || initial_state.dtype() != beta.dtype()
        || initial_state.dtype() != g.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())?
                as *const c_void,
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            beta_storage.raw_device_ptr_with_offset(beta_layout.start_offset())? as *const c_void,
            g_storage.raw_device_ptr_with_offset(g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-chunk-scan-raw-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_chunk_scan_raw_host_buffer(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (initial_state, query, key, value, beta, g);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaFullScan;

impl candle::CustomOp7 for DeltaFullScan {
    fn name(&self) -> &'static str {
        "delta-full-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
        _s7: &candle::CpuStorage,
        _l7: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-full-scan has no cpu implementation")
    }

    #[cfg(any(feature = "candle-cuda", feature = "qwen35-minimal-cuda"))]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::CudaStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::CudaStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::CudaStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::CudaStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::CudaStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
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
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let weighted_key_scan = weighted_key_scan.as_cuda_slice::<$ty>()?;
                let weighted_key_scan = match weighted_key_layout.contiguous_offsets() {
                    Some((o1, o2)) => weighted_key_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let k_cumdecay_scan = k_cumdecay_scan.as_cuda_slice::<$ty>()?;
                let k_cumdecay_scan = match k_cumdecay_layout.contiguous_offsets() {
                    Some((o1, o2)) => k_cumdecay_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let q_state_scan = q_state_scan.as_cuda_slice::<$ty>()?;
                let q_state_scan = match q_state_layout.contiguous_offsets() {
                    Some((o1, o2)) => q_state_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let local_attn_scan = local_attn_scan.as_cuda_slice::<$ty>()?;
                let local_attn_scan = match local_attn_layout.contiguous_offsets() {
                    Some((o1, o2)) => local_attn_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let state_decay_scan = state_decay_scan.as_cuda_slice::<$ty>()?;
                let state_decay_scan = match state_decay_layout.contiguous_offsets() {
                    Some((o1, o2)) => state_decay_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&weighted_key_scan);
                builder.arg(&k_cumdecay_scan);
                builder.arg(&q_state_scan);
                builder.arg(&local_attn_scan);
                builder.arg(&state_decay_scan);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_full_scan_f16"),
            DType::F32 => launch!(f32, "delta_full_scan_f32"),
            DType::BF16 => launch!(half::bf16, "delta_full_scan_bf16"),
            other => candle::bail!("delta-full-scan unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::MetalStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::MetalStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::MetalStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::MetalStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::MetalStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
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
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-full-scan unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, initial_state.dtype(), "delta-full-scan")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-full-scan");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let weighted_key = candle_metal_kernels::BufferOffset {
            buffer: weighted_key_scan.buffer(),
            offset_in_bytes: weighted_key_layout.start_offset()
                * weighted_key_scan.dtype().size_in_bytes(),
        };
        let k_cumdecay = candle_metal_kernels::BufferOffset {
            buffer: k_cumdecay_scan.buffer(),
            offset_in_bytes: k_cumdecay_layout.start_offset()
                * k_cumdecay_scan.dtype().size_in_bytes(),
        };
        let q_state = candle_metal_kernels::BufferOffset {
            buffer: q_state_scan.buffer(),
            offset_in_bytes: q_state_layout.start_offset() * q_state_scan.dtype().size_in_bytes(),
        };
        let local_attn = candle_metal_kernels::BufferOffset {
            buffer: local_attn_scan.buffer(),
            offset_in_bytes: local_attn_layout.start_offset()
                * local_attn_scan.dtype().size_in_bytes(),
        };
        let state_decay = candle_metal_kernels::BufferOffset {
            buffer: state_decay_scan.buffer(),
            offset_in_bytes: state_decay_layout.start_offset()
                * state_decay_scan.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_full_scan(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            weighted_key,
            k_cumdecay,
            q_state,
            local_attn,
            state_decay,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::HipStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::HipStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::HipStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::HipStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::HipStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
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
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_full_scan(
                dtype_code,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                v_head_dim,
                initial_state.raw_device_ptr_with_offset(initial_layout.start_offset())?
                    as *const c_void,
                weighted_key_scan.raw_device_ptr_with_offset(weighted_key_layout.start_offset())?
                    as *const c_void,
                k_cumdecay_scan.raw_device_ptr_with_offset(k_cumdecay_layout.start_offset())?
                    as *const c_void,
                q_state_scan.raw_device_ptr_with_offset(q_state_layout.start_offset())?
                    as *const c_void,
                local_attn_scan.raw_device_ptr_with_offset(local_attn_layout.start_offset())?
                    as *const c_void,
                state_decay_scan.raw_device_ptr_with_offset(state_decay_layout.start_offset())?
                    as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_full_scan(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = delta_full_scan_host_buffer(
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
    )? {
        return hip_tensor_from_host_bytes(initial_state.device(), initial_state.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_full_scan", initial_state);
    initial_state.apply_op7_no_bwd(
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
        &DeltaFullScan,
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_full_scan_host_buffer(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let weighted_key_scan = weighted_key_scan.contiguous()?;
    let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
    let q_state_scan = q_state_scan.contiguous()?;
    let local_attn_scan = local_attn_scan.contiguous()?;
    let state_decay_scan = state_decay_scan.contiguous()?;
    let value = value.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(weighted_key_scan.device().same_device(initial_state.device())
        && k_cumdecay_scan.device().same_device(initial_state.device())
        && q_state_scan.device().same_device(initial_state.device())
        && local_attn_scan.device().same_device(initial_state.device())
        && state_decay_scan.device().same_device(initial_state.device())
        && value.device().same_device(initial_state.device()))
    {
        return Ok(None);
    }
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (weighted_key_storage, weighted_key_layout) = weighted_key_scan.storage_and_layout();
    let (k_cumdecay_storage, k_cumdecay_layout) = k_cumdecay_scan.storage_and_layout();
    let (q_state_storage, q_state_layout) = q_state_scan.storage_and_layout();
    let (local_attn_storage, local_attn_layout) = local_attn_scan.storage_and_layout();
    let (state_decay_storage, state_decay_layout) = state_decay_scan.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (
        Storage::Hip(initial_storage),
        Storage::Hip(weighted_key_storage),
        Storage::Hip(k_cumdecay_storage),
        Storage::Hip(q_state_storage),
        Storage::Hip(local_attn_storage),
        Storage::Hip(state_decay_storage),
        Storage::Hip(value_storage),
    ) = (
        &*initial_storage,
        &*weighted_key_storage,
        &*k_cumdecay_storage,
        &*q_state_storage,
        &*local_attn_storage,
        &*state_decay_storage,
        &*value_storage,
    )
    else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && weighted_key_layout.is_contiguous()
        && k_cumdecay_layout.is_contiguous()
        && q_state_layout.is_contiguous()
        && local_attn_layout.is_contiguous()
        && state_decay_layout.is_contiguous()
        && value_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
    let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
        weighted_key_layout.shape().dims4()?;
    let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
        k_cumdecay_layout.shape().dims4()?;
    let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
        q_state_layout.shape().dims4()?;
    let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
        local_attn_layout.shape().dims4()?;
    let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
    let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
        value_layout.shape().dims4()?;
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
        || initial_state.dtype() != weighted_key_scan.dtype()
        || initial_state.dtype() != k_cumdecay_scan.dtype()
        || initial_state.dtype() != q_state_scan.dtype()
        || initial_state.dtype() != local_attn_scan.dtype()
        || initial_state.dtype() != state_decay_scan.dtype()
        || initial_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = candle::hip::qwen35_dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())?
                as *const c_void,
            weighted_key_storage.raw_device_ptr_with_offset(weighted_key_layout.start_offset())?
                as *const c_void,
            k_cumdecay_storage.raw_device_ptr_with_offset(k_cumdecay_layout.start_offset())?
                as *const c_void,
            q_state_storage.raw_device_ptr_with_offset(q_state_layout.start_offset())?
                as *const c_void,
            local_attn_storage.raw_device_ptr_with_offset(local_attn_layout.start_offset())?
                as *const c_void,
            state_decay_storage.raw_device_ptr_with_offset(state_decay_layout.start_offset())?
                as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_full_scan_host_buffer(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
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

#[derive(Debug, Clone, Copy)]
struct DeltaLocalAttnScan;

impl candle::CustomOp3 for DeltaLocalAttnScan {
    fn name(&self) -> &'static str {
        "delta-local-attn-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-local-attn-scan has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        query_scan: &candle::HipStorage,
        query_layout: &candle::Layout,
        key_scan: &candle::HipStorage,
        key_layout: &candle::Layout,
        exp_g_scan: &candle::HipStorage,
        exp_g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && exp_g_layout.is_contiguous())
        {
            candle::bail!("delta-local-attn-scan requires contiguous inputs")
        }

        let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
        let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
        let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
        if key_bh != batch_heads
            || exp_bh != batch_heads
            || key_chunks != num_chunks
            || exp_chunks != num_chunks
            || key_chunk_size != chunk_size
            || exp_chunk_size != chunk_size
            || key_k != k_head_dim
        {
            candle::bail!(
                "delta-local-attn-scan shape mismatch: query={:?} key={:?} exp_g={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                exp_g_layout.shape().dims()
            )
        }
        if query_scan.dtype() != key_scan.dtype() || query_scan.dtype() != exp_g_scan.dtype() {
            candle::bail!(
                "delta-local-attn-scan requires matching dtypes, got query={:?} key={:?} exp_g={:?}",
                query_scan.dtype(),
                key_scan.dtype(),
                exp_g_scan.dtype()
            )
        }

        let device = query_scan.device().clone();
        let storage_dtype = query_scan.dtype();
        let out_shape = candle::Shape::from_dims(&[batch_heads, num_chunks, chunk_size, chunk_size]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_local_attn_scan(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                query_scan.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                key_scan.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                exp_g_scan.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_local_attn_scan(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = delta_local_attn_scan_host_buffer(query_scan, key_scan, exp_g_scan)? {
        return hip_tensor_from_host_bytes(query_scan.device(), query_scan.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_local_attn_scan", query_scan);
    query_scan.apply_op3_no_bwd(key_scan, exp_g_scan, &DeltaLocalAttnScan)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_local_attn_scan_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let query_scan = query_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    let ordinal = match query_scan.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.device().same_device(query_scan.device())
        && exp_g_scan.device().same_device(query_scan.device()))
    {
        return Ok(None);
    }
    let (query_storage, query_layout) = query_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_g_storage, exp_g_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(query_storage), Storage::Hip(key_storage), Storage::Hip(exp_g_storage)) =
        (&*query_storage, &*key_storage, &*exp_g_storage)
    else {
        return Ok(None);
    };
    if !(query_layout.is_contiguous() && key_layout.is_contiguous() && exp_g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
        || query_scan.dtype() != key_scan.dtype()
        || query_scan.dtype() != exp_g_scan.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query_scan.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(query_scan.dtype().size_in_bytes())
    ];
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
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_g_storage.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-local-attn-scan-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_local_attn_scan_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (query_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaBaseAttnScan;

impl candle::CustomOp3 for DeltaBaseAttnScan {
    fn name(&self) -> &'static str {
        "delta-base-attn-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-base-attn-scan has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        k_beta_scan: &candle::HipStorage,
        k_beta_layout: &candle::Layout,
        key_scan: &candle::HipStorage,
        key_layout: &candle::Layout,
        exp_g_scan: &candle::HipStorage,
        exp_g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(k_beta_layout.is_contiguous()
            && key_layout.is_contiguous()
            && exp_g_layout.is_contiguous())
        {
            candle::bail!("delta-base-attn-scan requires contiguous inputs")
        }

        let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
        let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
        let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
        if key_bh != batch_heads
            || exp_bh != batch_heads
            || key_chunks != num_chunks
            || exp_chunks != num_chunks
            || key_chunk_size != chunk_size
            || exp_chunk_size != chunk_size
            || key_k != k_head_dim
        {
            candle::bail!(
                "delta-base-attn-scan shape mismatch: k_beta={:?} key={:?} exp_g={:?}",
                k_beta_layout.shape().dims(),
                key_layout.shape().dims(),
                exp_g_layout.shape().dims()
            )
        }
        if k_beta_scan.dtype() != key_scan.dtype() || k_beta_scan.dtype() != exp_g_scan.dtype() {
            candle::bail!(
                "delta-base-attn-scan requires matching dtypes, got k_beta={:?} key={:?} exp_g={:?}",
                k_beta_scan.dtype(),
                key_scan.dtype(),
                exp_g_scan.dtype()
            )
        }

        let device = k_beta_scan.device().clone();
        let storage_dtype = k_beta_scan.dtype();
        let out_shape = candle::Shape::from_dims(&[batch_heads, num_chunks, chunk_size, chunk_size]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_base_attn_scan(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                k_beta_scan.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
                key_scan.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                exp_g_scan.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_base_attn_scan(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_base_attn_scan_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    {
        return hip_tensor_from_host_bytes(k_beta_scan.device(), k_beta_scan.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_base_attn_scan", k_beta_scan);
    k_beta_scan.apply_op3_no_bwd(key_scan, exp_g_scan, &DeltaBaseAttnScan)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_base_attn_scan_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let k_beta_scan = k_beta_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    let ordinal = match k_beta_scan.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.device().same_device(k_beta_scan.device())
        && exp_g_scan.device().same_device(k_beta_scan.device()))
    {
        return Ok(None);
    }
    let (k_beta_storage, k_beta_layout) = k_beta_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_g_storage, exp_g_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(k_beta_storage), Storage::Hip(key_storage), Storage::Hip(exp_g_storage)) =
        (&*k_beta_storage, &*key_storage, &*exp_g_storage)
    else {
        return Ok(None);
    };
    if !(k_beta_layout.is_contiguous() && key_layout.is_contiguous() && exp_g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
        || k_beta_scan.dtype() != key_scan.dtype()
        || k_beta_scan.dtype() != exp_g_scan.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(k_beta_scan.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(k_beta_scan.dtype().size_in_bytes())
    ];
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
            k_beta_storage.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_g_storage.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-base-attn-scan-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_base_attn_scan_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaAttnSolveScan;

impl candle::CustomOp1 for DeltaAttnSolveScan {
    fn name(&self) -> &'static str {
        "delta-attn-solve-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-attn-solve-scan has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        base_attn_scan: &candle::HipStorage,
        base_attn_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !base_attn_layout.is_contiguous() {
            candle::bail!("delta-attn-solve-scan requires contiguous input")
        }

        let (batch_heads, num_chunks, chunk_size, width) = base_attn_layout.shape().dims4()?;
        if width != chunk_size {
            candle::bail!(
                "delta-attn-solve-scan shape mismatch: base_attn={:?}",
                base_attn_layout.shape().dims()
            )
        }

        let device = base_attn_scan.device().clone();
        let storage_dtype = base_attn_scan.dtype();
        let out_shape = candle::Shape::from_dims(&[batch_heads, num_chunks, chunk_size, chunk_size]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_attn_solve_scan(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                base_attn_scan.raw_device_ptr_with_offset(base_attn_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_attn_solve_scan(base_attn_scan: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = delta_attn_solve_scan_host_buffer(base_attn_scan)? {
        return hip_tensor_from_host_bytes(
            base_attn_scan.device(),
            base_attn_scan.dtype(),
            shape,
            output,
        );
    }
    trace_hip_wrapper_fallback("delta_attn_solve_scan", base_attn_scan);
    base_attn_scan.apply_op1_no_bwd(&DeltaAttnSolveScan)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_attn_solve_scan_host_buffer(
    base_attn_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let base_attn_scan = base_attn_scan.contiguous()?;
    let ordinal = match base_attn_scan.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (base_storage, base_layout) = base_attn_scan.storage_and_layout();
    let Storage::Hip(base_storage) = &*base_storage else {
        return Ok(None);
    };
    if !base_layout.is_contiguous() {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, width) = base_layout.shape().dims4()?;
    if width != chunk_size {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(base_attn_scan.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(base_attn_scan.dtype().size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_delta_attn_solve_scan(
            dtype_code,
            ordinal,
            batch_heads,
            num_chunks,
            chunk_size,
            base_storage.raw_device_ptr_with_offset(base_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-attn-solve-scan-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_attn_solve_scan_host_buffer(
    base_attn_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = base_attn_scan;
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaAttnSolveFromInputs;

impl candle::CustomOp3 for DeltaAttnSolveFromInputs {
    fn name(&self) -> &'static str {
        "delta-attn-solve-from-inputs"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-attn-solve-from-inputs has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        k_beta_scan: &candle::HipStorage,
        k_beta_layout: &candle::Layout,
        key_scan: &candle::HipStorage,
        key_layout: &candle::Layout,
        exp_g_scan: &candle::HipStorage,
        exp_g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !k_beta_layout.is_contiguous() || !key_layout.is_contiguous() || !exp_g_layout.is_contiguous() {
            candle::bail!("delta-attn-solve-from-inputs requires contiguous inputs")
        }

        let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
        let (key_batch_heads, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
        let (exp_batch_heads, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
        if key_batch_heads != batch_heads
            || exp_batch_heads != batch_heads
            || key_chunks != num_chunks
            || exp_chunks != num_chunks
            || key_chunk_size != chunk_size
            || exp_chunk_size != chunk_size
            || key_k != k_head_dim
        {
            candle::bail!(
                "delta-attn-solve-from-inputs shape mismatch: k_beta={:?} key={:?} exp_g={:?}",
                k_beta_layout.shape().dims(),
                key_layout.shape().dims(),
                exp_g_layout.shape().dims()
            )
        }
        if k_beta_scan.dtype() != key_scan.dtype() || k_beta_scan.dtype() != exp_g_scan.dtype() {
            candle::bail!(
                "delta-attn-solve-from-inputs requires matching dtypes, got k_beta={:?} key={:?} exp_g={:?}",
                k_beta_scan.dtype(),
                key_scan.dtype(),
                exp_g_scan.dtype()
            )
        }

        let device = k_beta_scan.device().clone();
        let storage_dtype = k_beta_scan.dtype();
        let out_shape = candle::Shape::from_dims(&[batch_heads, num_chunks, chunk_size, chunk_size]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_attn_solve_from_inputs(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                k_beta_scan.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
                key_scan.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                exp_g_scan.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_attn_solve_from_inputs(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_attn_solve_from_inputs_host_buffer(k_beta_scan, key_scan, exp_g_scan)?
    {
        return hip_tensor_from_host_bytes(k_beta_scan.device(), k_beta_scan.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_attn_solve_from_inputs", k_beta_scan);
    k_beta_scan.apply_op3_no_bwd(key_scan, exp_g_scan, &DeltaAttnSolveFromInputs)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_attn_solve_from_inputs_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let k_beta_scan = k_beta_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    let ordinal = match k_beta_scan.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.device().same_device(k_beta_scan.device())
        && exp_g_scan.device().same_device(k_beta_scan.device()))
    {
        return Ok(None);
    }
    let (k_beta_storage, k_beta_layout) = k_beta_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_g_storage, exp_g_layout) = exp_g_scan.storage_and_layout();
    let (Storage::Hip(k_beta_storage), Storage::Hip(key_storage), Storage::Hip(exp_g_storage)) =
        (&*k_beta_storage, &*key_storage, &*exp_g_storage)
    else {
        return Ok(None);
    };
    if !(k_beta_layout.is_contiguous() && key_layout.is_contiguous() && exp_g_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = k_beta_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
    if key_bh != batch_heads
        || exp_bh != batch_heads
        || key_chunks != num_chunks
        || exp_chunks != num_chunks
        || key_chunk_size != chunk_size
        || exp_chunk_size != chunk_size
        || key_k != k_head_dim
        || k_beta_scan.dtype() != key_scan.dtype()
        || k_beta_scan.dtype() != exp_g_scan.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(k_beta_scan.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks, chunk_size, chunk_size];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(k_beta_scan.dtype().size_in_bytes())
    ];
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
            k_beta_storage.raw_device_ptr_with_offset(k_beta_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_g_storage.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "delta-attn-solve-from-inputs-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_attn_solve_from_inputs_host_buffer(
    k_beta_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (k_beta_scan, key_scan, exp_g_scan);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaFullScanPack;

impl candle::CustomOp4 for DeltaFullScanPack {
    fn name(&self) -> &'static str {
        "delta-full-scan-pack"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-full-scan-pack has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        query_scan: &candle::HipStorage,
        query_layout: &candle::Layout,
        key_scan: &candle::HipStorage,
        key_layout: &candle::Layout,
        exp_g_scan: &candle::HipStorage,
        exp_g_layout: &candle::Layout,
        k_cumdecay_scan: &candle::HipStorage,
        k_cumdecay_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && exp_g_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan-pack requires contiguous inputs")
        }

        let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
        let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
        let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
        let (cum_bh, cum_chunks, cum_chunk_size, cum_k) = k_cumdecay_layout.shape().dims4()?;
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
            candle::bail!(
                "delta-full-scan-pack shape mismatch: query={:?} key={:?} exp_g={:?} k_cumdecay={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                exp_g_layout.shape().dims(),
                k_cumdecay_layout.shape().dims()
            )
        }
        if query_scan.dtype() != key_scan.dtype()
            || query_scan.dtype() != exp_g_scan.dtype()
            || query_scan.dtype() != k_cumdecay_scan.dtype()
        {
            candle::bail!(
                "delta-full-scan-pack requires matching dtypes, got query={:?} key={:?} exp_g={:?} k_cumdecay={:?}",
                query_scan.dtype(),
                key_scan.dtype(),
                exp_g_scan.dtype(),
                k_cumdecay_scan.dtype()
            )
        }

        let device = query_scan.device().clone();
        let storage_dtype = query_scan.dtype();
        let packed_width = 3 * k_head_dim + 1;
        let out_shape = candle::Shape::from_dims(&[batch_heads, num_chunks, chunk_size, packed_width]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_full_scan_pack(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                query_scan.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
                key_scan.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
                exp_g_scan.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
                k_cumdecay_scan.raw_device_ptr_with_offset(k_cumdecay_layout.start_offset())?
                    as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_full_scan_pack(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_full_scan_pack_host_buffer(query_scan, key_scan, exp_g_scan, k_cumdecay_scan)?
    {
        return hip_tensor_from_host_bytes(query_scan.device(), query_scan.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("delta_full_scan_pack", query_scan);
    query_scan.apply_op4_no_bwd(key_scan, exp_g_scan, k_cumdecay_scan, &DeltaFullScanPack)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_full_scan_pack_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let query_scan = query_scan.contiguous()?;
    let key_scan = key_scan.contiguous()?;
    let exp_g_scan = exp_g_scan.contiguous()?;
    let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
    let ordinal = match query_scan.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(key_scan.device().same_device(query_scan.device())
        && exp_g_scan.device().same_device(query_scan.device())
        && k_cumdecay_scan.device().same_device(query_scan.device()))
    {
        return Ok(None);
    }
    let (query_storage, query_layout) = query_scan.storage_and_layout();
    let (key_storage, key_layout) = key_scan.storage_and_layout();
    let (exp_g_storage, exp_g_layout) = exp_g_scan.storage_and_layout();
    let (k_cumdecay_storage, k_cumdecay_layout) = k_cumdecay_scan.storage_and_layout();
    let (
        Storage::Hip(query_storage),
        Storage::Hip(key_storage),
        Storage::Hip(exp_g_storage),
        Storage::Hip(k_cumdecay_storage),
    ) = (
        &*query_storage,
        &*key_storage,
        &*exp_g_storage,
        &*k_cumdecay_storage,
    )
    else {
        return Ok(None);
    };
    if !(query_layout.is_contiguous()
        && key_layout.is_contiguous()
        && exp_g_layout.is_contiguous()
        && k_cumdecay_layout.is_contiguous())
    {
        return Ok(None);
    }
    let (batch_heads, num_chunks, chunk_size, k_head_dim) = query_layout.shape().dims4()?;
    let (key_bh, key_chunks, key_chunk_size, key_k) = key_layout.shape().dims4()?;
    let (exp_bh, exp_chunks, exp_chunk_size) = exp_g_layout.shape().dims3()?;
    let (cum_bh, cum_chunks, cum_chunk_size, cum_k) = k_cumdecay_layout.shape().dims4()?;
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
        || query_scan.dtype() != key_scan.dtype()
        || query_scan.dtype() != exp_g_scan.dtype()
        || query_scan.dtype() != k_cumdecay_scan.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(query_scan.dtype()) else {
        return Ok(None);
    };
    let packed_width = 3 * k_head_dim + 1;
    let shape = vec![batch_heads, num_chunks, chunk_size, packed_width];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(query_scan.dtype().size_in_bytes())
    ];
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
            query_storage.raw_device_ptr_with_offset(query_layout.start_offset())? as *const c_void,
            key_storage.raw_device_ptr_with_offset(key_layout.start_offset())? as *const c_void,
            exp_g_storage.raw_device_ptr_with_offset(exp_g_layout.start_offset())? as *const c_void,
            k_cumdecay_storage.raw_device_ptr_with_offset(k_cumdecay_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-pack-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_full_scan_pack_host_buffer(
    query_scan: &Tensor,
    key_scan: &Tensor,
    exp_g_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (query_scan, key_scan, exp_g_scan, k_cumdecay_scan);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct DeltaFullScanPacked;

impl candle::CustomOp4 for DeltaFullScanPacked {
    fn name(&self) -> &'static str {
        "delta-full-scan-packed"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-full-scan-packed has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::HipStorage,
        packed_layout: &candle::Layout,
        local_attn_scan: &candle::HipStorage,
        local_attn_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::{BackendDevice, BackendStorage};
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan-packed requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
        let (local_bh, local_chunks, local_chunk_size, local_width) =
            local_attn_layout.shape().dims4()?;
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
            candle::bail!(
                "delta-full-scan-packed shape mismatch: initial={:?} packed={:?} local_attn={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }
        if initial_state.dtype() != packed_scan.dtype()
            || initial_state.dtype() != local_attn_scan.dtype()
            || initial_state.dtype() != value.dtype()
        {
            candle::bail!(
                "delta-full-scan-packed requires matching dtypes, got initial={:?} packed={:?} local_attn={:?} value={:?}",
                initial_state.dtype(),
                packed_scan.dtype(),
                local_attn_scan.dtype(),
                value.dtype()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_delta_full_scan_packed(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                batch_heads,
                num_chunks,
                chunk_size,
                k_head_dim,
                v_head_dim,
                initial_state.raw_device_ptr_with_offset(initial_layout.start_offset())?
                    as *const c_void,
                packed_scan.raw_device_ptr_with_offset(packed_layout.start_offset())?
                    as *const c_void,
                local_attn_scan.raw_device_ptr_with_offset(local_attn_layout.start_offset())?
                    as *const c_void,
                value.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn delta_full_scan_packed(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        delta_full_scan_packed_host_buffer(initial_state, packed_scan, local_attn_scan, value)?
    {
        return hip_tensor_from_host_bytes(
            initial_state.device(),
            initial_state.dtype(),
            shape,
            output,
        );
    }
    trace_hip_wrapper_fallback("delta_full_scan_packed", initial_state);
    initial_state.apply_op4_no_bwd(packed_scan, local_attn_scan, value, &DeltaFullScanPacked)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn delta_full_scan_packed_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let initial_state = initial_state.contiguous()?;
    let packed_scan = packed_scan.contiguous()?;
    let local_attn_scan = local_attn_scan.contiguous()?;
    let value = value.contiguous()?;
    let ordinal = match initial_state.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(packed_scan.device().same_device(initial_state.device())
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
    )
    else {
        return Ok(None);
    };
    if !(initial_layout.is_contiguous()
        && packed_layout.is_contiguous()
        && local_layout.is_contiguous()
        && value_layout.is_contiguous())
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
        || initial_state.dtype() != packed_scan.dtype()
        || initial_state.dtype() != local_attn_scan.dtype()
        || initial_state.dtype() != value.dtype()
    {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(initial_state.dtype()) else {
        return Ok(None);
    };
    let shape = vec![batch_heads, num_chunks * chunk_size + k_head_dim, v_head_dim];
    let mut out = vec![
        0u8;
        shape
            .iter()
            .product::<usize>()
            .saturating_mul(initial_state.dtype().size_in_bytes())
    ];
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
            initial_storage.raw_device_ptr_with_offset(initial_layout.start_offset())?
                as *const c_void,
            packed_storage.raw_device_ptr_with_offset(packed_layout.start_offset())?
                as *const c_void,
            local_storage.raw_device_ptr_with_offset(local_layout.start_offset())?
                as *const c_void,
            value_storage.raw_device_ptr_with_offset(value_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("delta-full-scan-packed-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn delta_full_scan_packed_host_buffer(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    local_attn_scan: &Tensor,
    value: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (initial_state, packed_scan, local_attn_scan, value);
    Ok(None)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeltaNetScanMode {
    Flat3d,
    HoistedDecays,
    PrebatchedLocal,
    TorchLike,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeltaNetExecutionPolicy {
    pub(super) scan_mode: DeltaNetScanMode,
    pub(super) use_flattened_solve: bool,
}

fn parse_delta_net_scan_mode(raw_value: &str) -> Option<DeltaNetScanMode> {
    match raw_value.trim() {
        "flat3d" => Some(DeltaNetScanMode::Flat3d),
        "hoisted-decays" => Some(DeltaNetScanMode::HoistedDecays),
        "prebatched-local" => Some(DeltaNetScanMode::PrebatchedLocal),
        "torch-like" => Some(DeltaNetScanMode::TorchLike),
        _ => None,
    }
}

fn debug_delta_scan_policy(sequence_length: usize, policy: DeltaNetExecutionPolicy) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if std::env::var("CANDLE_QWEN35_DEBUG_DELTA_SCAN").is_ok()
        && !LOGGED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "qwen3.5 delta scan policy: sequence_length={} mode={:?} flattened_solve={}",
            sequence_length, policy.scan_mode, policy.use_flattened_solve
        );
    }
}

fn recommended_delta_net_execution_policy(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
) -> DeltaNetExecutionPolicy {
    let long_metal_context = matches!(device.location(), DeviceLocation::Metal { .. })
        && (sequence_length >= 2048 || num_chunks >= 64);
    DeltaNetExecutionPolicy {
        scan_mode: if long_metal_context {
            DeltaNetScanMode::PrebatchedLocal
        } else {
            DeltaNetScanMode::Flat3d
        },
        use_flattened_solve: long_metal_context,
    }
}

pub(super) fn delta_net_execution_policy(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
) -> DeltaNetExecutionPolicy {
    let mut policy = recommended_delta_net_execution_policy(device, sequence_length, num_chunks);
    if let Ok(raw_value) = std::env::var("CANDLE_QWEN35_DELTA_SCAN_MODE") {
        if let Some(mode) = parse_delta_net_scan_mode(&raw_value) {
            policy.scan_mode = mode;
        }
    }
    debug_delta_scan_policy(sequence_length, policy);
    policy
}

pub(super) fn parse_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub(super) fn full_attention_blockwise_tiles(
    device: &Device,
    q_len: usize,
    kv_len: usize,
) -> Option<(usize, usize)> {
    if !matches!(device.location(), DeviceLocation::Metal { .. }) || q_len <= 1 || kv_len <= 1 {
        return None;
    }
    let enabled = matches!(
        std::env::var("CANDLE_QWEN35_FULL_BLOCKWISE_ATTN").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    );
    if !enabled {
        return None;
    }
    let q_block = parse_usize_env("CANDLE_QWEN35_FULL_ATTN_Q_BLOCK").unwrap_or(128);
    let k_block = parse_usize_env("CANDLE_QWEN35_FULL_ATTN_K_BLOCK").unwrap_or(512);
    Some((q_block.min(q_len), k_block.min(kv_len)))
}

pub(super) fn full_attention_sdpa_q_block(device: &Device, q_len: usize) -> Option<usize> {
    if !matches!(device.location(), DeviceLocation::Metal { .. }) || q_len <= 1 {
        return None;
    }
    let enabled = matches!(
        std::env::var("CANDLE_QWEN35_FULL_SDPA_CHUNKED").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    );
    if !enabled {
        return None;
    }
    Some(
        parse_usize_env("CANDLE_QWEN35_FULL_SDPA_Q_BLOCK")
            .unwrap_or(128)
            .min(q_len),
    )
}

pub(super) fn use_full_attention_torchlike_eager(device: &Device) -> bool {
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(
            std::env::var("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        )
}

pub(super) fn delta_net_compute_dtype(scan_mode: DeltaNetScanMode, initial_dtype: DType) -> DType {
    match scan_mode {
        DeltaNetScanMode::TorchLike => DType::F32,
        _ => initial_dtype,
    }
}

#[derive(Debug, Clone)]
enum LayerKind {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    layer_type: String,
    token_mixer: LayerKind,
    mlp: Mlp,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
}

impl DecoderLayer {
    #[cfg(any(feature = "hf", test))]
    fn new(
        cfg: &TextConfig,
        layer_idx: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: WeightBuilder,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "linear_attention".to_string());
        let token_mixer = match layer_type.as_str() {
            "linear_attention" => LayerKind::Linear(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?),
            "full_attention" => {
                LayerKind::Full(FullAttention::new(cfg, rotary_emb, vb.pp("self_attn"))?)
            }
            other => candle::bail!("unsupported qwen3.5 layer type {other:?}"),
        };
        Ok(Self {
            layer_type,
            token_mixer,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn from_prepared(
        cfg: &TextConfig,
        layer_idx: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        source: &PreparedTensorSource,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "linear_attention".to_string());
        let token_mixer = match layer_type.as_str() {
            "linear_attention" => {
                LayerKind::Linear(GatedDeltaNet::from_prepared(cfg, &source.pp("linear_attn"))?)
            }
            "full_attention" => LayerKind::Full(FullAttention::from_prepared(
                cfg,
                rotary_emb,
                &source.pp("self_attn"),
            )?),
            other => candle::bail!("unsupported qwen3.5 layer type {other:?}"),
        };
        Ok(Self {
            layer_type,
            token_mixer,
            mlp: Mlp::from_prepared(cfg, &source.pp("mlp"))?,
            input_layernorm: Qwen35RmsNorm::from_prepared(
                cfg.rms_norm_eps,
                &source.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen35RmsNorm::from_prepared(
                cfg.rms_norm_eps,
                &source.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward_profiled_with_external(
        &mut self,
        layer_id: usize,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = xs.device();
        let mut profile = RuntimeProfile::default();
        let residual = xs.clone();
        let xs_norm = self.input_layernorm.forward_buffer(xs)?;
        let xs = match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => {
                let (xs, layer_profile) =
                    linear_attn.forward_profiled_buffer(&xs_norm, attention_mask)?;
                profile.add_assign(&layer_profile);
                xs
            }
            LayerKind::Full(self_attn) => {
                let (xs, layer_profile) = self_attn.forward_profiled_with_external_buffer(
                    &xs_norm,
                    attention_mask,
                    seqlen_offset,
                    layer_id,
                    external_full_attention,
                )?;
                profile.add_assign(&layer_profile);
                xs
            }
        };
        let backend = backend_buffer_api::for_device(device);
        let xs = backend.add(&residual, &xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward_buffer(&xs)?;
        let mlp_start = profile_start(device)?;
        let xs = self.mlp.forward_buffer(&xs)?;
        profile.mlp_millis += profile_elapsed(mlp_start, device)?;
        Ok((backend.add(&residual, &xs)?, profile))
    }

    fn forward_profiled(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let mut no_external = None;
        self.forward_profiled_with_external(
            usize::MAX,
            xs,
            attention_mask,
            seqlen_offset,
            &mut no_external,
        )
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        self.forward_profiled(xs, attention_mask, seqlen_offset)
            .map(|(output, _)| output)
    }

    fn forward_profiled_direct_decode_v1(
        &mut self,
        layer_id: usize,
        xs: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let mut profile = RuntimeProfile::default();
        let residual = xs.clone();
        let xs_norm = self.input_layernorm.forward_buffer(xs)?;
        let xs = match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => {
                let (xs, layer_profile) = linear_attn.forward_profiled_direct_decode_v1(&xs_norm)?;
                profile.add_assign(&layer_profile);
                xs
            }
            LayerKind::Full(self_attn) => {
                let (xs, layer_profile) =
                    self_attn.forward_profiled_direct_decode_v1(&xs_norm, seqlen_offset, layer_id)?;
                profile.add_assign(&layer_profile);
                xs
            }
        };
        let xs = backend.add(&residual, &xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward_buffer(&xs)?;
        let mlp_start = profile_start(device)?;
        let xs = self.mlp.forward_buffer(&xs)?;
        profile.mlp_millis += profile_elapsed(mlp_start, device)?;
        Ok((backend.add(&residual, &xs)?, profile))
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => linear_attn.clear_kv_cache(),
            LayerKind::Full(self_attn) => self_attn.clear_kv_cache(),
        }
    }

    fn cache_state(&self) -> LayerCacheState {
        match &self.token_mixer {
            LayerKind::Linear(linear_attn) => LayerCacheState::Linear(linear_attn.cache_state()),
            LayerKind::Full(self_attn) => LayerCacheState::Full(self_attn.cache_state()),
        }
    }

    fn restore_cache_state(&mut self, state: &LayerCacheState) -> Result<()> {
        match (&mut self.token_mixer, state) {
            (LayerKind::Linear(linear_attn), LayerCacheState::Linear(state)) => {
                linear_attn.restore_cache_state(state);
                Ok(())
            }
            (LayerKind::Full(self_attn), LayerCacheState::Full(state)) => {
                self_attn.restore_cache_state(state);
                Ok(())
            }
            (LayerKind::Linear(_), LayerCacheState::Full(_)) => {
                candle::bail!("cannot restore full-attention cache into linear-attention layer")
            }
            (LayerKind::Full(_), LayerCacheState::Linear(_)) => {
                candle::bail!("cannot restore linear-attention cache into full-attention layer")
            }
        }
    }

    pub fn layer_type(&self) -> &str {
        &self.layer_type
    }

    fn deferred_linear_count(&self) -> usize {
        self.mlp.deferred_linear_count()
            + match &self.token_mixer {
                LayerKind::Linear(linear_attn) => linear_attn.deferred_linear_count(),
                LayerKind::Full(_) => 0,
            }
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: EmbeddingSource,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    device: Device,
    dtype: DType,
    immutable_embedding_requested: bool,
    immutable_embedding_active: bool,
    immutable_embedding_fallback_reason: Option<String>,
    immutable_linear_requested: bool,
    deferred_linear_count: usize,
}

impl TextModel {
    #[cfg(any(feature = "hf", test))]
    pub fn new(cfg: &TextConfig, vb: WeightBuilder) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(&cfg, vb.device(), vb.dtype())?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                &cfg,
                layer_idx,
                rotary_emb.clone(),
                vb_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens: EmbeddingSource::Materialized(embed_tokens),
            layers,
            norm: Qwen35RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            immutable_embedding_requested: false,
            immutable_embedding_active: false,
            immutable_embedding_fallback_reason: None,
            immutable_linear_requested: false,
            deferred_linear_count: 0,
        })
    }

    pub(crate) fn from_prepared(
        cfg: &TextConfig,
        source: PreparedTensorSource,
    ) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let model_source = source.pp("model").pp("language_model");
        let immutable_embedding_requested = immutable_embedding_enabled();
        let (embed_tokens, immutable_embedding_active, immutable_embedding_fallback_reason) =
            build_prepared_embedding_source(
                &model_source.pp("embed_tokens"),
                cfg.hidden_size,
                immutable_embedding_requested,
            )?;
        let dtype = embed_tokens.dtype();
        let rotary_emb = Arc::new(RotaryEmbedding::new(&cfg, source.device(), dtype)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_source = model_source.pp("layers");
        let immutable_linear_requested = immutable_linear_enabled(&cfg);
        let mut deferred_linear_count = 0usize;
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::from_prepared(
                &cfg,
                layer_idx,
                rotary_emb.clone(),
                &layers_source.pp(layer_idx),
            )?;
            deferred_linear_count += layer.deferred_linear_count();
            layers.push(layer);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: Qwen35RmsNorm::from_prepared(cfg.rms_norm_eps, &model_source.pp("norm"))?,
            device: source.device().clone(),
            dtype,
            immutable_embedding_requested,
            immutable_embedding_active,
            immutable_embedding_fallback_reason,
            immutable_linear_requested,
            deferred_linear_count,
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        backend_buffer_api::for_device(&self.device)
            .causal_mask(&self.device, self.dtype, b_size, tgt_len, seqlen_offset)
    }

    pub fn linear_attention_layer_ids(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(layer_id, layer)| {
                (layer.layer_type() == "linear_attention").then_some(layer_id)
            })
            .collect()
    }

    pub fn linear_attention_layer_spec(
        &self,
        layer_id: usize,
    ) -> Result<LinearAttentionLayerSpec> {
        let layer = self.layers.get(layer_id).ok_or_else(|| {
            candle::Error::Msg(format!(
                "linear-attention layer {} is out of range for {} layers",
                layer_id,
                self.layers.len()
            ))
        })?;
        match &layer.token_mixer {
            LayerKind::Linear(linear) => Ok(LinearAttentionLayerSpec {
                layer_id,
                conv_dim: linear.conv_dim(),
                num_v_heads: linear.num_v_heads,
                num_k_heads: linear.num_k_heads,
                head_k_dim: linear.head_k_dim,
                head_v_dim: linear.head_v_dim,
                key_dim: linear.key_dim,
                value_dim: linear.value_dim,
                state_len: linear.conv_kernel_size.saturating_sub(1),
                kernel_size: linear.conv_kernel_size,
            }),
            LayerKind::Full(_) => candle::bail!(
                "layer {} is {:?}, expected linear_attention",
                layer_id,
                layer.layer_type()
            ),
        }
    }

    pub fn immutable_linear_requested(&self) -> bool {
        self.immutable_linear_requested
    }

    pub fn deferred_linear_count(&self) -> usize {
        self.deferred_linear_count
    }

    pub fn hidden_states_from_input_ids(&self, input_ids: &Tensor) -> Result<StateBuffer> {
        self.embed_tokens.forward_buffer(input_ids)
    }

    fn materialized_embeddings(&self) -> Option<&Tensor> {
        self.embed_tokens.embeddings()
    }

    fn immutable_embedding_requested(&self) -> bool {
        self.immutable_embedding_requested
    }

    fn immutable_embedding_active(&self) -> bool {
        self.immutable_embedding_active
    }

    fn immutable_embedding_fallback_reason(&self) -> Option<&str> {
        self.immutable_embedding_fallback_reason.as_deref()
    }

    fn immutable_embedding_runtime_mode(&self) -> &'static str {
        self.embed_tokens.runtime_mode()
    }

    pub fn full_attention_layer_ids(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(layer_id, layer)| {
                (layer.layer_type() == "full_attention").then_some(layer_id)
            })
            .collect()
    }

    pub fn bench_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        repeats: usize,
    ) -> Result<LinearAttentionBenchResult> {
        if repeats == 0 {
            candle::bail!("linear-attention bench requires repeats > 0");
        }
        if target_layer >= self.layers.len() {
            candle::bail!(
                "linear-attention bench target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
            );
        }
        if self.layers[target_layer].layer_type() != "linear_attention" {
            candle::bail!(
                "linear-attention bench target layer {} is {:?}, expected \"linear_attention\"",
                target_layer,
                self.layers[target_layer].layer_type()
            );
        }

        self.clear_kv_cache();
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        let mut xs = self.hidden_states_from_input_ids(input_ids)?;
        for layer in self.layers.iter_mut().take(target_layer) {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, _) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            xs = next_xs;
        }
        self.clear_kv_cache();

        let mut total_profile = RuntimeProfile::default();
        let mut total_millis_acc = 0.0;
        let mut best_total_millis = f64::INFINITY;
        let mut best_profile = RuntimeProfile::default();
        let mut iteration_total_millis = Vec::with_capacity(repeats);
        let device = input_ids.device();
        for _ in 0..repeats {
            self.layers[target_layer].clear_kv_cache();
            let iteration_start = profile_start(device)?;
            let (_, profile) =
                self.layers[target_layer].forward_profiled(&xs, None, seqlen_offset)?;
            let total_millis = profile_elapsed(iteration_start, device)?;
            iteration_total_millis.push(total_millis);
            total_millis_acc += total_millis;
            total_profile.add_assign(&profile);
            if total_millis < best_total_millis {
                best_total_millis = total_millis;
                best_profile = profile.clone();
            }
        }
        self.clear_kv_cache();

        Ok(LinearAttentionBenchResult {
            layer_id: target_layer,
            sequence_length: seq_len,
            repeats,
            mean_total_millis: total_millis_acc / repeats as f64,
            best_total_millis,
            iteration_total_millis,
            mean_profile: total_profile.scaled(1.0 / repeats as f64),
            best_profile,
        })
    }

    pub fn trace_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<LinearAttentionTrace> {
        if target_layer >= self.layers.len() {
            candle::bail!(
                "linear-attention trace target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
            );
        }
        if self.layers[target_layer].layer_type() != "linear_attention" {
            candle::bail!(
                "linear-attention trace target layer {} is {:?}, expected \"linear_attention\"",
                target_layer,
                self.layers[target_layer].layer_type()
            );
        }

        self.clear_kv_cache();
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        let mut xs = self.hidden_states_from_input_ids(input_ids)?;
        for layer in self.layers.iter_mut().take(target_layer) {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, _) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            xs = next_xs;
        }

        let target = self
            .layers
            .get_mut(target_layer)
            .expect("target layer index already validated");
        let (layer_output, recurrent_state, profile) = match &mut target.token_mixer {
            LayerKind::Linear(linear_attn) => linear_attn.trace_profiled_buffer(&xs, None)?,
            LayerKind::Full(_) => unreachable!("target layer is validated as linear attention"),
        };
        self.clear_kv_cache();

        Ok(LinearAttentionTrace {
            layer_id: target_layer,
            sequence_length: seq_len,
            layer_output,
            recurrent_state,
            profile,
        })
    }

    pub fn forward_profiled_with_linear_traces(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        target_layers: &[usize],
    ) -> Result<(Tensor, Vec<LinearAttentionTrace>, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = self.hidden_states_from_input_ids(input_ids)?;
        let mut traces = Vec::new();
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let should_trace = target_layers.contains(&layer_id);
            let (next_xs, maybe_trace, layer_profile) = match &mut layer.token_mixer {
                LayerKind::Linear(linear_attn) if should_trace => {
                    let backend = backend_buffer_api::for_device(xs.device());
                    let xs_norm = layer.input_layernorm.forward_buffer(&xs)?;
                    let (layer_output, recurrent_state, layer_profile) =
                        linear_attn.trace_profiled_buffer(&xs_norm, None)?;
                    linear_attn.recurrent_state = Some(recurrent_state.clone());
                    let attn_residual = backend.add(&xs, &layer_output)?;
                    let post_norm = layer.post_attention_layernorm.forward_buffer(&attn_residual)?;
                    let mlp_start = profile_start(device)?;
                    let mlp_out = layer.mlp.forward_buffer(&post_norm)?;
                    let mut profile = layer_profile;
                    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
                    let next_xs = backend.add(&attn_residual, &mlp_out)?;
                    (
                        next_xs,
                        Some(LinearAttentionTrace {
                            layer_id,
                            sequence_length: seq_len,
                            layer_output,
                            recurrent_state,
                            profile: profile.clone(),
                        }),
                        profile,
                    )
                }
                _ => {
                    let (next_xs, layer_profile) =
                        layer.forward_profiled(&xs, mask, seqlen_offset)?;
                    (next_xs, None, layer_profile)
                }
            };
            if let Some(trace) = maybe_trace {
                traces.push(trace);
            }
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?.clone_tensor(), traces, profile))
    }

    pub fn forward_profiled_with_external_full_attention(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        external_full_attention: &mut dyn ExternalFullAttention,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = self.hidden_states_from_input_ids(input_ids)?;
        let mut external = Some(external_full_attention);
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, layer_profile) = layer.forward_profiled_with_external(
                layer_id,
                &xs,
                mask,
                seqlen_offset,
                &mut external,
            )?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?.clone_tensor(), profile))
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = self.hidden_states_from_input_ids(input_ids)?;
        for layer in self.layers.iter_mut() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, layer_profile) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?.clone_tensor(), profile))
    }

    pub fn forward_hidden_states_profiled(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let (b_size, seq_len, _) = hidden_states.dims3()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = hidden_states.clone();
        for layer in self.layers.iter_mut() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, layer_profile) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?, profile))
    }

    pub(crate) fn forward_hidden_states_profiled_direct_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        if self.layers.len() != metadata.num_hidden_layers {
            candle::bail!(
                "direct-hip-v1 layer count mismatch: model={} metadata={}",
                self.layers.len(),
                metadata.num_hidden_layers
            );
        }
        if metadata.layers.len() != self.layers.len() {
            candle::bail!(
                "direct-hip-v1 metadata layer schedule mismatch: metadata={} model={}",
                metadata.layers.len(),
                self.layers.len()
            );
        }
        let device = hidden_states.device();
        let (b_size, seq_len, _) = hidden_states.dims3()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = hidden_states.clone();
        for (layer_idx, (layer, layer_meta)) in self
            .layers
            .iter_mut()
            .zip(metadata.layers.iter())
            .enumerate()
        {
            if layer_meta.layer_idx != layer_idx {
                candle::bail!(
                    "direct-hip-v1 metadata index mismatch at layer {}: got {}",
                    layer_idx,
                    layer_meta.layer_idx
                );
            }
            if layer.layer_type() != layer_meta.layer_type {
                candle::bail!(
                    "direct-hip-v1 layer type mismatch at layer {}: model={} metadata={}",
                    layer_idx,
                    layer.layer_type(),
                    layer_meta.layer_type
                );
            }
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, layer_profile) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?, profile))
    }

    pub(crate) fn validate_direct_hip_metadata(
        &self,
        metadata: &PreparedQwen35DirectMetadata,
    ) -> Result<()> {
        if self.layers.len() != metadata.num_hidden_layers {
            candle::bail!(
                "direct-hip-v1 decode layer count mismatch: model={} metadata={}",
                self.layers.len(),
                metadata.num_hidden_layers
            );
        }
        if metadata.layers.len() != self.layers.len() {
            candle::bail!(
                "direct-hip-v1 decode metadata layer schedule mismatch: metadata={} model={}",
                metadata.layers.len(),
                self.layers.len()
            );
        }
        Ok(())
    }

    pub(crate) fn direct_decode_linear_phase_profiled_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        start_layer_idx: usize,
        end_layer_idx: usize,
        xs: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        self.validate_direct_hip_metadata(metadata)?;
        let mut profile = RuntimeProfile::default();
        let mut xs = xs.clone();
        let layer_count = self.layers.len();
        for layer_idx in start_layer_idx..end_layer_idx {
            let layer_meta = metadata.layers.get(layer_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "direct-hip-v1 decode metadata missing linear layer {}",
                    layer_idx
                ))
            })?;
            if layer_meta.layer_idx != layer_idx {
                candle::bail!(
                    "direct-hip-v1 decode metadata index mismatch at linear layer {}: got {}",
                    layer_idx,
                    layer_meta.layer_idx
                );
            }
            if layer_meta.layer_type != "linear_attention" {
                candle::bail!(
                    "direct-hip-v1 linear decode phase expected linear_attention at layer {}, got {}",
                    layer_idx,
                    layer_meta.layer_type
                );
            }
            let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "direct-hip-v1 linear decode layer index {} out of range for {} layers",
                    layer_idx,
                    layer_count
                ))
            })?;
            if layer.layer_type() != layer_meta.layer_type {
                candle::bail!(
                    "direct-hip-v1 linear decode layer type mismatch at layer {}: model={} metadata={}",
                    layer_idx,
                    layer.layer_type(),
                    layer_meta.layer_type
                );
            }
            let (next_xs, layer_profile) =
                layer.forward_profiled_direct_decode_v1(layer_idx, &xs, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((xs, profile))
    }

    pub(crate) fn direct_decode_full_phase_profiled_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        start_layer_idx: usize,
        end_layer_idx: usize,
        xs: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        self.validate_direct_hip_metadata(metadata)?;
        let mut profile = RuntimeProfile::default();
        let mut xs = xs.clone();
        let layer_count = self.layers.len();
        for layer_idx in start_layer_idx..end_layer_idx {
            let layer_meta = metadata.layers.get(layer_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "direct-hip-v1 decode metadata missing full-attention layer {}",
                    layer_idx
                ))
            })?;
            if layer_meta.layer_idx != layer_idx {
                candle::bail!(
                    "direct-hip-v1 decode metadata index mismatch at full-attention layer {}: got {}",
                    layer_idx,
                    layer_meta.layer_idx
                );
            }
            if layer_meta.layer_type != "full_attention" {
                candle::bail!(
                    "direct-hip-v1 full decode phase expected full_attention at layer {}, got {}",
                    layer_idx,
                    layer_meta.layer_type
                );
            }
            let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "direct-hip-v1 full decode layer index {} out of range for {} layers",
                    layer_idx,
                    layer_count
                ))
            })?;
            if layer.layer_type() != layer_meta.layer_type {
                candle::bail!(
                    "direct-hip-v1 full decode layer type mismatch at layer {}: model={} metadata={}",
                    layer_idx,
                    layer.layer_type(),
                    layer_meta.layer_type
                );
            }
            let (next_xs, layer_profile) =
                layer.forward_profiled_direct_decode_v1(layer_idx, &xs, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((xs, profile))
    }

    pub(crate) fn finalize_direct_decode_hidden_hip_v1(
        &mut self,
        xs: &StateBuffer,
    ) -> Result<StateBuffer> {
        self.norm.forward_buffer(xs)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_profiled(input_ids, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    pub fn cache_state(&self) -> CacheState {
        CacheState {
            layers: self.layers.iter().map(DecoderLayer::cache_state).collect(),
        }
    }

    pub fn restore_cache_state(&mut self, state: &CacheState) -> Result<()> {
        if state.layers.len() != self.layers.len() {
            candle::bail!(
                "cache state layer count mismatch: model={} state={}",
                self.layers.len(),
                state.layers.len()
            );
        }
        for (layer, layer_state) in self.layers.iter_mut().zip(state.layers.iter()) {
            layer.restore_cache_state(layer_state)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    language_model: TextModel,
    lm_head: OutputProjectionSource,
}

impl ModelForCausalLM {
    #[cfg(any(feature = "hf", test))]
    pub(crate) fn new(cfg: &Config, vb: WeightBuilder) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let language_model = TextModel::new(&cfg.text_config, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            OutputProjectionSource::Materialized(linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?)
        } else {
            OutputProjectionSource::Materialized(Linear::new(
                language_model
                    .materialized_embeddings()
                    .expect("direct loader uses eager embedding")
                    .clone(),
                None,
            ))
        };
        Ok(Self {
            language_model,
            lm_head,
        })
    }

    pub(crate) fn from_prepared(cfg: &Config, source: PreparedTensorSource) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let language_model = TextModel::from_prepared(&cfg.text_config, source.clone())?;
        let lm_head = if source.contains_tensor("lm_head.weight") {
            OutputProjectionSource::Materialized(prepared_linear_no_bias(&source.pp("lm_head"))?)
        } else if immutable_embedding_enabled() && language_model.immutable_embedding_active() {
            match &language_model.embed_tokens {
                EmbeddingSource::Immutable(embedding) => {
                    OutputProjectionSource::TiedImmutable(embedding.clone())
                }
                EmbeddingSource::Materialized(_) => OutputProjectionSource::Materialized(Linear::new(
                    language_model
                        .materialized_embeddings()
                        .expect("materialized embedding should be available")
                        .clone(),
                    None,
                )),
            }
        } else {
            OutputProjectionSource::Materialized(Linear::new(
                language_model
                    .materialized_embeddings()
                    .expect("tied lm_head requires eager embedding")
                    .clone(),
                None,
            ))
        };
        Ok(Self {
            language_model,
            lm_head,
        })
    }

    pub fn forward_profiled_with_external_full_attention(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        external_full_attention: &mut dyn ExternalFullAttention,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, mut profile) = self
            .language_model
            .forward_profiled_with_external_full_attention(
                input_ids,
                seqlen_offset,
                external_full_attention,
            )?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&backend.tensor_to_buffer(hidden_states)?)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits.clone_tensor(), profile))
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, mut profile) = self
            .language_model
            .forward_profiled(input_ids, seqlen_offset)?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&backend.tensor_to_buffer(hidden_states)?)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits.clone_tensor(), profile))
    }

    pub fn hidden_states_from_input_ids(&self, input_ids: &Tensor) -> Result<StateBuffer> {
        self.language_model.hidden_states_from_input_ids(input_ids)
    }

    pub fn immutable_embedding_requested(&self) -> bool {
        self.language_model.immutable_embedding_requested()
    }

    pub fn immutable_embedding_active(&self) -> bool {
        self.language_model.immutable_embedding_active()
    }

    pub fn immutable_embedding_fallback_reason(&self) -> Option<&str> {
        self.language_model.immutable_embedding_fallback_reason()
    }

    pub fn immutable_embedding_runtime_mode(&self) -> &'static str {
        self.language_model.immutable_embedding_runtime_mode()
    }

    pub fn immutable_linear_requested(&self) -> bool {
        self.language_model.immutable_linear_requested()
    }

    pub fn deferred_linear_count(&self) -> usize {
        self.language_model.deferred_linear_count()
    }

    pub fn forward_hidden_states_profiled(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, mut profile) =
            self.language_model
                .forward_hidden_states_profiled(hidden_states, seqlen_offset)?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&hidden_states)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits, profile))
    }

    pub(crate) fn forward_hidden_states_profiled_direct_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, mut profile) = self
            .language_model
            .forward_hidden_states_profiled_direct_hip_v1(
                metadata,
                hidden_states,
                seqlen_offset,
            )?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&hidden_states)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits, profile))
    }

    pub(crate) fn validate_direct_hip_metadata(
        &self,
        metadata: &PreparedQwen35DirectMetadata,
    ) -> Result<()> {
        self.language_model.validate_direct_hip_metadata(metadata)
    }

    pub(crate) fn direct_decode_linear_phase_profiled_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        start_layer_idx: usize,
        end_layer_idx: usize,
        xs: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        self.language_model.direct_decode_linear_phase_profiled_hip_v1(
            metadata,
            start_layer_idx,
            end_layer_idx,
            xs,
            seqlen_offset,
        )
    }

    pub(crate) fn direct_decode_full_phase_profiled_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        start_layer_idx: usize,
        end_layer_idx: usize,
        xs: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        self.language_model.direct_decode_full_phase_profiled_hip_v1(
            metadata,
            start_layer_idx,
            end_layer_idx,
            xs,
            seqlen_offset,
        )
    }

    pub(crate) fn finalize_direct_decode_logits_hip_v1(
        &mut self,
        hidden_states: &StateBuffer,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let backend = backend_buffer_api::for_device(device);
        let output_start = profile_start(device)?;
        let hidden_states = self
            .language_model
            .finalize_direct_decode_hidden_hip_v1(hidden_states)?;
        let logits = backend.slice_last_token(&hidden_states)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        let mut profile = RuntimeProfile::default();
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits, profile))
    }

    pub fn forward_profiled_with_linear_traces(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        target_layers: &[usize],
    ) -> Result<(Tensor, Vec<LinearAttentionTrace>, RuntimeProfile)> {
        let device = input_ids.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, traces, mut profile) = self
            .language_model
            .forward_profiled_with_linear_traces(input_ids, seqlen_offset, target_layers)?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&backend.tensor_to_buffer(hidden_states)?)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits.clone_tensor(), traces, profile))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_profiled(input_ids, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn forward_hidden_states(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        self.forward_hidden_states_profiled(hidden_states, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn linear_attention_layer_ids(&self) -> Vec<usize> {
        self.language_model.linear_attention_layer_ids()
    }

    pub fn linear_attention_layer_spec(
        &self,
        layer_id: usize,
    ) -> Result<LinearAttentionLayerSpec> {
        self.language_model.linear_attention_layer_spec(layer_id)
    }

    pub fn full_attention_layer_ids(&self) -> Vec<usize> {
        self.language_model.full_attention_layer_ids()
    }

    pub fn bench_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        repeats: usize,
    ) -> Result<LinearAttentionBenchResult> {
        self.language_model.bench_linear_attention_layer(
            input_ids,
            target_layer,
            seqlen_offset,
            repeats,
        )
    }

    pub fn trace_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<LinearAttentionTrace> {
        self.language_model
            .trace_linear_attention_layer(input_ids, target_layer, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }

    pub fn cache_state(&self) -> CacheState {
        self.language_model.cache_state()
    }

    pub fn restore_cache_state(&mut self, state: &CacheState) -> Result<()> {
        self.language_model.restore_cache_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "qwen35-minimal-hip", feature = "qwen35-minimal-cuda"))]
    use std::ffi::OsString;
    #[cfg(any(feature = "qwen35-minimal-hip", feature = "qwen35-minimal-cuda"))]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    #[cfg(any(feature = "qwen35-minimal-hip", feature = "qwen35-minimal-cuda"))]
    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let delta = (lhs - rhs).abs();
            assert!(
                delta <= tol,
                "mismatch at {idx}: lhs={lhs} rhs={rhs} delta={delta} tol={tol}"
            );
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_test_guard() -> MutexGuard<'static, ()> {
        hip_env_lock().lock().unwrap_or_else(|err| err.into_inner())
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    fn cuda_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    fn cuda_test_guard() -> MutexGuard<'static, ()> {
        cuda_env_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct HipPersistentPrefillEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl HipPersistentPrefillEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL";

        fn clear() -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::remove_var(Self::KEY);
            }
            Self { saved }
        }

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for HipPersistentPrefillEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    struct CudaFullPrefillEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    impl CudaFullPrefillEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    impl Drop for CudaFullPrefillEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct HipChunkSinglePrefillEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl HipChunkSinglePrefillEnvGuard {
        const KEY: &'static str = "DOTCACHE_QWEN35_HIP_CHUNK_SINGLE_PREFILL";

        fn clear() -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::remove_var(Self::KEY);
            }
            Self { saved }
        }

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for HipChunkSinglePrefillEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct HipMultiChunkScanPrefillEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl HipMultiChunkScanPrefillEnvGuard {
        const KEY: &'static str = "DOTCACHE_QWEN35_HIP_MULTI_CHUNK_SCAN_PREFILL";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for HipMultiChunkScanPrefillEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct LinearChunkSizeEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl LinearChunkSizeEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_LINEAR_CHUNK_SIZE";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for LinearChunkSizeEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct DeltaScanModeEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl DeltaScanModeEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_DELTA_SCAN_MODE";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for DeltaScanModeEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct DeltaKernelMinSequenceEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl DeltaKernelMinSequenceEnvGuard {
        const KEY: &'static str = "DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for DeltaKernelMinSequenceEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct DeltaStateScanKernelEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl DeltaStateScanKernelEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for DeltaStateScanKernelEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct DeltaChunkFusedKernelEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl DeltaChunkFusedKernelEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for DeltaChunkFusedKernelEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    struct DeltaFullKernelEnvGuard {
        saved: Option<OsString>,
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl DeltaFullKernelEnvGuard {
        const KEY: &'static str = "CANDLE_QWEN35_DELTA_FULL_KERNEL";

        fn set(value: &str) -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::set_var(Self::KEY, value);
            }
            Self { saved }
        }

        fn clear() -> Self {
            let saved = std::env::var_os(Self::KEY);
            unsafe {
                std::env::remove_var(Self::KEY);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    impl Drop for DeltaFullKernelEnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = self.saved.as_ref() {
                    std::env::set_var(Self::KEY, value);
                } else {
                    std::env::remove_var(Self::KEY);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn tiny_linear_text_config() -> TextConfig {
        TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            hidden_act: Activation::Silu,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            attention_bias: false,
            attention_dropout: 0.0,
            head_dim: 8,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            layer_types: vec!["linear_attention".to_string()],
            rope_parameters: None,
        }
    }

    #[cfg(any(feature = "qwen35-minimal-hip", feature = "qwen35-minimal-cuda"))]
    fn hip_full_attention_prefill_sample(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, usize, f32, usize, Vec<f32>)> {
        let batch_size = 1usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let q_len = 3usize;
        let kv_len = 5usize;
        let head_dim = 4usize;
        let num_kv_groups = 2usize;
        let scale = 0.5f32;
        let seqlen_offset = 2usize;

        let query_data = vec![
            0.2f32, 0.0, 0.1, -0.1, 0.1, 0.3, -0.2, 0.0, 0.4, -0.1, 0.0, 0.2, -0.2, 0.1, 0.0, 0.3,
            0.2, -0.3, 0.1, 0.0, 0.0, 0.2, 0.2, -0.1,
        ];
        let key_data = vec![
            0.1f32, 0.0, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.4, -0.3, 0.2, 0.1, 0.0,
            0.1, 0.1, -0.1, 0.2,
        ];
        let value_data = vec![
            0.0f32, 0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.2, 0.4, 0.1, -0.3, 0.0, -0.1, 0.3, 0.2, -0.2,
            0.2, 0.0, 0.1, 0.4,
        ];

        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_size, q_heads, q_len, head_dim),
            device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?;

        let mut expected = Vec::with_capacity(batch_size * q_heads * q_len * head_dim);
        for b in 0..batch_size {
            for q_head in 0..q_heads {
                let kv_head = q_head / num_kv_groups;
                for q_pos in 0..q_len {
                    let causal_limit = kv_len.min(seqlen_offset + q_pos + 1);
                    let query_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
                    let q_row = &query_data[query_offset..query_offset + head_dim];
                    let key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
                    let value_head_offset = key_head_offset;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut denom = 0.0f32;
                    let mut out_row = vec![0.0f32; head_dim];
                    for k_pos in 0..causal_limit {
                        let key_offset = key_head_offset + k_pos * head_dim;
                        let value_offset = value_head_offset + k_pos * head_dim;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_row[d] * key_data[key_offset + d];
                        }
                        score *= scale;

                        if !max_score.is_finite() {
                            max_score = score;
                            denom = 1.0;
                            out_row.copy_from_slice(
                                &value_data[value_offset..value_offset + head_dim],
                            );
                            continue;
                        }

                        let new_max = max_score.max(score);
                        let prev_scale = (max_score - new_max).exp();
                        let curr_scale = (score - new_max).exp();
                        denom = denom * prev_scale + curr_scale;
                        for d in 0..head_dim {
                            out_row[d] =
                                out_row[d] * prev_scale + curr_scale * value_data[value_offset + d];
                        }
                        max_score = new_max;
                    }

                    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    for value in out_row {
                        expected.push(value * inv_denom);
                    }
                }
            }
        }

        Ok((
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
            expected,
        ))
    }

    #[cfg(any(feature = "qwen35-minimal-hip", feature = "qwen35-minimal-cuda"))]
    fn hip_full_attention_prefill_sample_qwen35_like(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, usize, f32, usize, Vec<f32>)> {
        let batch_size = 1usize;
        let q_heads = 8usize;
        let kv_heads = 2usize;
        let q_len = 4usize;
        let kv_len = 4usize;
        let head_dim = 256usize;
        let num_kv_groups = 4usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seqlen_offset = 0usize;

        let query_data = (0..batch_size * q_heads * q_len * head_dim)
            .map(|idx| {
                let raw = ((idx * 17 + 11) % 97) as f32 - 48.0;
                raw / 3.5
            })
            .collect::<Vec<_>>();
        let key_data = (0..batch_size * kv_heads * kv_len * head_dim)
            .map(|idx| {
                let raw = ((idx * 13 + 7) % 89) as f32 - 44.0;
                raw / 3.0
            })
            .collect::<Vec<_>>();
        let value_data = (0..batch_size * kv_heads * kv_len * head_dim)
            .map(|idx| {
                let raw = ((idx * 19 + 5) % 101) as f32 - 50.0;
                raw / 4.0
            })
            .collect::<Vec<_>>();

        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_size, q_heads, q_len, head_dim),
            device,
        )?
        .to_dtype(DType::F16)?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?
        .to_dtype(DType::F16)?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?
        .to_dtype(DType::F16)?;

        let mut expected = Vec::with_capacity(batch_size * q_heads * q_len * head_dim);
        for b in 0..batch_size {
            for q_head in 0..q_heads {
                let kv_head = q_head / num_kv_groups;
                for q_pos in 0..q_len {
                    let causal_limit = kv_len.min(seqlen_offset + q_pos + 1);
                    let query_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
                    let q_row = &query_data[query_offset..query_offset + head_dim];
                    let key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
                    let value_head_offset = key_head_offset;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut denom = 0.0f32;
                    let mut out_row = vec![0.0f32; head_dim];
                    for k_pos in 0..causal_limit {
                        let key_offset = key_head_offset + k_pos * head_dim;
                        let value_offset = value_head_offset + k_pos * head_dim;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_row[d] * key_data[key_offset + d];
                        }
                        score *= scale;

                        if !max_score.is_finite() {
                            max_score = score;
                            denom = 1.0;
                            out_row.copy_from_slice(
                                &value_data[value_offset..value_offset + head_dim],
                            );
                            continue;
                        }

                        let new_max = max_score.max(score);
                        let prev_scale = (max_score - new_max).exp();
                        let curr_scale = (score - new_max).exp();
                        denom = denom * prev_scale + curr_scale;
                        for d in 0..head_dim {
                            out_row[d] =
                                out_row[d] * prev_scale + curr_scale * value_data[value_offset + d];
                        }
                        max_score = new_max;
                    }

                    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    for value in out_row {
                        expected.push(value * inv_denom);
                    }
                }
            }
        }

        Ok((
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
            expected,
        ))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_full_attention_decode_sample(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, usize, f32, usize, Vec<f32>)> {
        let batch_size = 1usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let q_len = 1usize;
        let kv_len = 5usize;
        let head_dim = 4usize;
        let num_kv_groups = 2usize;
        let scale = 0.5f32;
        let seqlen_offset = 4usize;

        let query_data = vec![0.2f32, -0.3, 0.1, 0.0, 0.0, 0.2, 0.2, -0.1];
        let key_data = vec![
            0.1f32, 0.0, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.4, -0.3, 0.2, 0.1, 0.0,
            0.1, 0.1, -0.1, 0.2,
        ];
        let value_data = vec![
            0.0f32, 0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.2, 0.4, 0.1, -0.3, 0.0, -0.1, 0.3, 0.2, -0.2,
            0.2, 0.0, 0.1, 0.4,
        ];

        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_size, q_heads, q_len, head_dim),
            device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            device,
        )?;

        let mut expected = Vec::with_capacity(batch_size * q_heads * q_len * head_dim);
        for b in 0..batch_size {
            for q_head in 0..q_heads {
                let kv_head = q_head / num_kv_groups;
                for q_pos in 0..q_len {
                    let causal_limit = kv_len.min(seqlen_offset + q_pos + 1);
                    let query_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
                    let q_row = &query_data[query_offset..query_offset + head_dim];
                    let key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
                    let value_head_offset = key_head_offset;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut denom = 0.0f32;
                    let mut out_row = vec![0.0f32; head_dim];
                    for k_pos in 0..causal_limit {
                        let key_offset = key_head_offset + k_pos * head_dim;
                        let value_offset = value_head_offset + k_pos * head_dim;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_row[d] * key_data[key_offset + d];
                        }
                        score *= scale;

                        if !max_score.is_finite() {
                            max_score = score;
                            denom = 1.0;
                            out_row.copy_from_slice(
                                &value_data[value_offset..value_offset + head_dim],
                            );
                            continue;
                        }

                        let new_max = max_score.max(score);
                        let prev_scale = (max_score - new_max).exp();
                        let curr_scale = (score - new_max).exp();
                        denom = denom * prev_scale + curr_scale;
                        for d in 0..head_dim {
                            out_row[d] =
                                out_row[d] * prev_scale + curr_scale * value_data[value_offset + d];
                        }
                        max_score = new_max;
                    }

                    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    for value in out_row {
                        expected.push(value * inv_denom);
                    }
                }
            }
        }

        Ok((
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
            expected,
        ))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_rms_norm_sample(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Vec<f32>, Vec<f32>)> {
        let xs_data = vec![
            0.5f32, -1.0, 0.25, 1.5, -0.75, 0.2, 0.9, -0.4, 1.1, -0.6, 0.3, 0.8,
        ];
        let gate_data = vec![
            -0.2f32, 0.7, 0.5, -1.1, 0.3, -0.4, 0.9, 0.1, -0.8, 0.6, 0.2, -0.5,
        ];
        let weight_data = vec![0.1f32, -0.2, 0.3, 0.4];
        let shape = (1usize, 3usize, 4usize);
        let eps = 1e-6f64;

        let xs = Tensor::from_vec(xs_data.clone(), shape, device)?.to_dtype(DType::F16)?;
        let gate = Tensor::from_vec(gate_data.clone(), shape, device)?.to_dtype(DType::F16)?;
        let weight = Tensor::from_vec(weight_data.clone(), 4usize, device)?.to_dtype(DType::F16)?;

        let mut expected_norm = Vec::with_capacity(xs_data.len());
        let mut expected_gated = Vec::with_capacity(xs_data.len());
        for row in 0..3 {
            let row_slice = &xs_data[row * 4..(row + 1) * 4];
            let gate_slice = &gate_data[row * 4..(row + 1) * 4];
            let mean_sq = row_slice.iter().map(|x| x * x).sum::<f32>() / 4.0;
            let inv_rms = 1.0f32 / (mean_sq + eps as f32).sqrt();
            for col in 0..4 {
                let normed = row_slice[col] * inv_rms;
                expected_norm.push(normed * (weight_data[col] + 1.0));
                let gate_x = gate_slice[col];
                let silu = gate_x / (1.0 + (-gate_x).exp());
                expected_gated.push(normed * weight_data[col] * silu);
            }
        }

        Ok((xs, gate, weight, expected_norm, expected_gated))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_l2norm_sample(device: &Device) -> Result<(Tensor, Vec<f32>)> {
        let xs_data = vec![
            0.5f32, -1.0, 0.25, 1.5, -0.75, 0.2, 0.9, -0.4, 1.1, -0.6, 0.3, 0.8,
        ];
        let shape = (1usize, 3usize, 4usize);
        let eps = 1e-6f32;
        let xs = Tensor::from_vec(xs_data.clone(), shape, device)?.to_dtype(DType::F16)?;

        let mut expected = Vec::with_capacity(xs_data.len());
        for row in 0..3 {
            let row_slice = &xs_data[row * 4..(row + 1) * 4];
            let norm = (row_slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            for value in row_slice {
                expected.push(*value / norm);
            }
        }
        Ok((xs, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_swiglu_mul_sample(device: &Device) -> Result<(Tensor, Tensor, Vec<f32>)> {
        let gate_data = vec![
            0.5f32, -1.0, 0.25, 1.5, -0.75, 0.2, 0.9, -0.4, 1.1, -0.6, 0.3, 0.8,
        ];
        let up_data = vec![
            1.2f32, -0.4, 0.8, 0.3, -1.1, 0.6, 0.5, 2.0, 0.7, -0.9, 1.3, 0.2,
        ];
        let shape = (1usize, 3usize, 4usize);
        let gate = Tensor::from_vec(gate_data.clone(), shape, device)?.to_dtype(DType::F16)?;
        let up = Tensor::from_vec(up_data.clone(), shape, device)?.to_dtype(DType::F16)?;

        let mut expected = Vec::with_capacity(gate_data.len());
        for (gate_x, up_x) in gate_data.iter().zip(up_data.iter()) {
            let silu = *gate_x / (1.0 + (-*gate_x).exp());
            expected.push(silu * *up_x);
        }
        Ok((gate, up, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_embedding_lookup_sample(device: &Device) -> Result<(Tensor, Tensor, Vec<f32>)> {
        let embeddings_data = vec![
            0.1f32, 0.2, 0.3, 0.4, //
            1.0, 1.1, 1.2, 1.3, //
            2.0, 2.1, 2.2, 2.3, //
            3.0, 3.1, 3.2, 3.3,
        ];
        let index_data = vec![2u32, 0, 3, 1];
        let embeddings = Tensor::from_vec(embeddings_data.clone(), (4usize, 4usize), device)?
            .to_dtype(DType::F16)?;
        let indexes = Tensor::from_vec(index_data.clone(), (2usize, 2usize), device)?;
        let mut expected = Vec::with_capacity(index_data.len() * 4);
        for token in index_data {
            let row = token as usize;
            expected.extend_from_slice(&embeddings_data[row * 4..(row + 1) * 4]);
        }
        Ok((embeddings, indexes, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_causal_mask_expected(
        batch_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Vec<f32> {
        let kv_len = tgt_len + seqlen_offset;
        let mut expected = Vec::with_capacity(batch_size * tgt_len * kv_len);
        for _batch in 0..batch_size {
            for row in 0..tgt_len {
                for col in 0..kv_len {
                    expected.push(if col <= seqlen_offset + row {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        expected
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_cumsum_last_dim_sample(device: &Device) -> Result<(Tensor, Vec<f32>)> {
        let xs_data = vec![
            0.5f32, -1.0, 0.25, 1.5, //
            -0.75, 0.2, 0.9, -0.4, //
            1.1, -0.6, 0.3, 0.8,
        ];
        let xs = Tensor::from_vec(xs_data.clone(), (1usize, 3usize, 4usize), device)?
            .to_dtype(DType::F16)?;
        let mut expected = Vec::with_capacity(xs_data.len());
        for row in xs_data.chunks_exact(4) {
            let mut acc = 0.0f32;
            for value in row {
                acc += *value;
                expected.push(acc);
            }
        }
        Ok((xs, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_value_decay_sample(device: &Device) -> Result<(Tensor, Tensor, Tensor, Vec<f32>)> {
        let a_data = vec![
            0.5f32, -1.0, 0.25, 1.5, -0.75, 0.2, 0.9, -0.4, 1.1, -0.6, 0.3, 0.8,
        ];
        let dt_bias_data = vec![0.1f32, -0.2, 0.3, 0.4];
        let a_log_exp_data = vec![0.7f32, 0.8, 0.9, 1.1];
        let shape = (1usize, 3usize, 4usize);
        let a = Tensor::from_vec(a_data.clone(), shape, device)?.to_dtype(DType::F16)?;
        let dt_bias = Tensor::from_vec(dt_bias_data.clone(), (1usize, 1usize, 4usize), device)?
            .to_dtype(DType::F16)?;
        let a_log_exp = Tensor::from_vec(a_log_exp_data.clone(), (1usize, 1usize, 4usize), device)?
            .to_dtype(DType::F16)?;

        let mut expected = Vec::with_capacity(a_data.len());
        for value in a_data.chunks_exact(4) {
            for head in 0..4 {
                let shifted = value[head] + dt_bias_data[head];
                let softplus = (shifted.exp() + 1.0).ln();
                expected.push(-softplus * a_log_exp_data[head]);
            }
        }
        Ok((a, dt_bias, a_log_exp, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_linear_stateful_conv_sample(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Vec<f32>)> {
        let batch_size = 1usize;
        let conv_dim = 2usize;
        let seq_len = 4usize;
        let state_len = 2usize;
        let kernel_size = 3usize;
        let mixed_qkv_data = vec![
            0.3f32, 0.4, 0.5, 0.6, //
            -0.1, 0.0, 0.1, 0.2,
        ];
        let prev_state_data = vec![
            0.1f32, 0.2, //
            -0.3, -0.2,
        ];
        let weight_data = vec![
            0.5f32, -0.25, 0.75, //
            -0.4, 0.3, 0.2,
        ];
        let mixed_qkv = Tensor::from_vec(
            mixed_qkv_data.clone(),
            (batch_size, conv_dim, seq_len),
            device,
        )?;
        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_size, conv_dim, state_len),
            device,
        )?;
        let weights = Tensor::from_vec(weight_data.clone(), (conv_dim, kernel_size), device)?;

        let mut expected = Vec::with_capacity(batch_size * seq_len * conv_dim);
        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..conv_dim {
                    let mixed_base = b * conv_dim * seq_len + c * seq_len;
                    let state_base = b * conv_dim * state_len + c * state_len;
                    let weight_base = c * kernel_size;
                    let mut acc = 0.0f32;
                    for tap in 0..kernel_size {
                        let src = t as isize + tap as isize - (kernel_size as isize - 1);
                        let x = if src >= 0 {
                            mixed_qkv_data[mixed_base + src as usize]
                        } else {
                            prev_state_data[state_base + (state_len as isize + src) as usize]
                        };
                        acc += x * weight_data[weight_base + tap];
                    }
                    expected.push(acc / (1.0 + (-acc).exp()));
                }
            }
        }
        Ok((mixed_qkv, prev_state, weights, expected))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_linear_stateful_conv_value_decay_sample(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Vec<f32>)> {
        let (mixed_qkv, prev_state, weights, conv_expected) = hip_linear_stateful_conv_sample(device)?;
        let a_data = vec![
            0.5f32, -1.0, //
            0.25, 1.5, //
            -0.75, 0.2, //
            0.9, -0.4,
        ];
        let dt_bias_data = vec![0.1f32, -0.2];
        let a_log_exp_data = vec![0.7f32, 0.8];
        let a = Tensor::from_vec(a_data.clone(), (1usize, 4usize, 2usize), device)?
            .to_dtype(mixed_qkv.dtype())?;
        let dt_bias =
            Tensor::from_vec(dt_bias_data.clone(), (1usize, 1usize, 2usize), device)?
                .to_dtype(mixed_qkv.dtype())?;
        let a_log_exp =
            Tensor::from_vec(a_log_exp_data.clone(), (1usize, 1usize, 2usize), device)?
                .to_dtype(mixed_qkv.dtype())?;

        let mut expected = Vec::with_capacity(4 * 4);
        for t in 0..4 {
            expected.push(conv_expected[t * 2]);
            expected.push(conv_expected[t * 2 + 1]);
            for head in 0..2 {
                let shifted = a_data[t * 2 + head] + dt_bias_data[head];
                let softplus = (shifted.exp() + 1.0).ln();
                expected.push(-softplus * a_log_exp_data[head]);
            }
        }
        Ok((mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, expected))
    }


    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_linear_decode_step_sample(
        device: &Device,
    ) -> Result<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<f32>,
    )> {
        let batch_size = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 2usize;
        let head_repeat = 2usize;
        let num_k_heads = num_v_heads / head_repeat;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let state_len = 2usize;
        let kernel_size = 3usize;

        let mixed_qkv_data = vec![
            0.10f32, -0.20, 0.30, 0.05, -0.10, 0.25, 0.40, -0.15,
        ];
        let prev_conv_state_data = vec![
            0.05f32, 0.02, -0.10, 0.03, 0.12, -0.07, -0.02, 0.04, 0.08, -0.05, -0.03, 0.09,
            0.11, -0.06, -0.08, 0.01,
        ];
        let weight_data = vec![
            0.5f32, -0.25, 0.75, -0.4, 0.3, 0.2, 0.35, -0.1, 0.45, 0.1, 0.25, -0.2, 0.6, -0.15,
            0.4, -0.3, 0.2, 0.5, 0.15, -0.05, 0.3, -0.25, 0.4, 0.2,
        ];
        let a_beta_raw_data = vec![0.5f32, -0.6, 0.2, -0.3];
        let dt_bias_data = vec![0.1f32, -0.2];
        let a_log_exp_data = vec![0.7f32, 0.8];
        let initial_state_data = vec![
            0.10f32, -0.20, 0.05, 0.30, -0.15, 0.08, 0.12, -0.04,
        ];

        let mixed_qkv =
            Tensor::from_vec(mixed_qkv_data.clone(), (batch_size, conv_dim, 1usize), device)?
                .to_dtype(DType::F16)?;
        let prev_conv_state = Tensor::from_vec(
            prev_conv_state_data.clone(),
            (batch_size, conv_dim, state_len),
            device,
        )?
        .to_dtype(DType::F16)?;
        let weights = Tensor::from_vec(weight_data.clone(), (conv_dim, kernel_size), device)?
            .to_dtype(DType::F16)?;
        let a_beta_raw =
            Tensor::from_vec(a_beta_raw_data.clone(), (batch_size, 1usize, 4usize), device)?
                .to_dtype(DType::F16)?;
        let dt_bias =
            Tensor::from_vec(dt_bias_data.clone(), (1usize, 1usize, num_v_heads), device)?
                .to_dtype(DType::F16)?;
        let a_log_exp =
            Tensor::from_vec(a_log_exp_data.clone(), (1usize, 1usize, num_v_heads), device)?
                .to_dtype(DType::F16)?;
        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_size, num_v_heads, head_k_dim, head_v_dim),
            device,
        )?;

        let conv_channel = |channel: usize| -> f32 {
            let mut acc = 0.0f32;
            let state_base = channel * state_len;
            let weight_base = channel * kernel_size;
            for tap in 0..kernel_size {
                let x = if tap + 1 == kernel_size {
                    mixed_qkv_data[channel]
                } else {
                    prev_conv_state_data[state_base + tap]
                };
                acc += x * weight_data[weight_base + tap];
            }
            acc / (1.0 + (-acc).exp())
        };

        let mut expected = vec![0.0f32; value_dim + num_v_heads * head_k_dim * head_v_dim];
        for v_head in 0..num_v_heads {
            let k_head = v_head / head_repeat;
            let q_base = k_head * head_k_dim;
            let k_base = key_dim + k_head * head_k_dim;
            let mut q = vec![0.0f32; head_k_dim];
            let mut k = vec![0.0f32; head_k_dim];
            let mut q_sq = 0.0f32;
            let mut k_sq = 0.0f32;
            for k_idx in 0..head_k_dim {
                q[k_idx] = conv_channel(q_base + k_idx);
                k[k_idx] = conv_channel(k_base + k_idx);
                q_sq += q[k_idx] * q[k_idx];
                k_sq += k[k_idx] * k[k_idx];
            }
            let q_inv = 1.0f32 / (q_sq + 1e-6).sqrt();
            let k_inv = 1.0f32 / (k_sq + 1e-6).sqrt();
            let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
            let a_raw = a_beta_raw_data[v_head];
            let beta_raw = a_beta_raw_data[num_v_heads + v_head];
            let beta = 1.0 / (1.0 + (-beta_raw).exp());
            let g = -((a_raw + dt_bias_data[v_head]).exp() + 1.0).ln() * a_log_exp_data[v_head];
            let g_exp = g.exp();

            for v_idx in 0..head_v_dim {
                let value_raw = conv_channel(key_dim * 2 + v_head * head_v_dim + v_idx);
                let mut state = vec![0.0f32; head_k_dim];
                for k_idx in 0..head_k_dim {
                    let flat = ((v_head * head_k_dim + k_idx) * head_v_dim) + v_idx;
                    state[k_idx] = initial_state_data[flat] * g_exp;
                }
                let mut kv_mem = 0.0f32;
                for k_idx in 0..head_k_dim {
                    kv_mem += state[k_idx] * (k[k_idx] * k_inv);
                }
                let delta = (value_raw - kv_mem) * beta;
                let mut out_value = 0.0f32;
                for k_idx in 0..head_k_dim {
                    state[k_idx] += (k[k_idx] * k_inv) * delta;
                    out_value += state[k_idx] * (q[k_idx] * q_inv * q_scale);
                    let flat = value_dim + ((v_head * head_k_dim + k_idx) * head_v_dim) + v_idx;
                    expected[flat] = state[k_idx];
                }
                expected[v_head * head_v_dim + v_idx] = out_value;
            }
        }

        Ok((
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            initial_state,
            expected,
        ))
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_prefill_conv_pack_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_size = 1usize;
        let conv_dim = 2usize;
        let total_len = 6usize;
        let seq_len = 4usize;
        let kernel_size = 3usize;

        let mixed_qkv_data = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, // channel 0
            -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, // channel 1
        ];
        let weight_data = vec![
            0.5f32, -0.25, 0.75, // channel 0
            -0.4, 0.3, 0.2, // channel 1
        ];
        let mixed_qkv = Tensor::from_vec(
            mixed_qkv_data.clone(),
            (batch_size, conv_dim, total_len),
            &device,
        )?;
        let weights = Tensor::from_vec(weight_data.clone(), (conv_dim, kernel_size), &device)?;
        let output = linear_prefill_conv_pack(&mixed_qkv, &weights, seq_len, kernel_size)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = Vec::with_capacity(batch_size * seq_len * conv_dim);
        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..conv_dim {
                    let input_base = b * conv_dim * total_len + c * total_len;
                    let weight_base = c * kernel_size;
                    let mut acc = 0.0f32;
                    for tap in 0..kernel_size {
                        acc +=
                            mixed_qkv_data[input_base + t + tap] * weight_data[weight_base + tap];
                    }
                    expected.push(acc / (1.0 + (-acc).exp()));
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_prefill_conv_pack_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_size = 1usize;
        let conv_dim = 2usize;
        let total_len = 6usize;
        let seq_len = 4usize;
        let kernel_size = 3usize;

        let mixed_qkv = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3,
            ],
            (batch_size, conv_dim, total_len),
            &device,
        )?;
        let weights = Tensor::from_vec(
            vec![0.5f32, -0.25, 0.75, -0.4, 0.3, 0.2],
            (conv_dim, kernel_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = linear_prefill_conv_pack(&mixed_qkv, &weights, seq_len, kernel_size)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), mixed_qkv.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_prefill_conv_pack_single_step_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_size = 1usize;
        let conv_dim = 2usize;
        let total_len = 3usize;
        let seq_len = 1usize;
        let kernel_size = 3usize;

        let mixed_qkv = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, -0.1, 0.0, 0.2],
            (batch_size, conv_dim, total_len),
            &device,
        )?;
        let weights = Tensor::from_vec(
            vec![0.5f32, -0.25, 0.75, -0.4, 0.3, 0.2],
            (conv_dim, kernel_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = linear_prefill_conv_pack(&mixed_qkv, &weights, seq_len, kernel_size)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), mixed_qkv.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_prefill_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, expected) =
            hip_full_attention_prefill_sample(&device)?;
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_prefill_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, _) =
            hip_full_attention_prefill_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_prefill_persistent_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::set("1");
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, expected) =
            hip_full_attention_prefill_sample(&device)?;
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_prefill_persistent_matches_legacy() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, _) =
            hip_full_attention_prefill_sample(&device)?;

        let _legacy_env_guard = HipPersistentPrefillEnvGuard::clear();
        let legacy = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
        drop(_legacy_env_guard);

        let _persistent_env_guard = HipPersistentPrefillEnvGuard::set("1");
        let persistent = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;

        assert_close(&legacy, &persistent, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_prefill_qwen35_like_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, expected) =
            hip_full_attention_prefill_sample_qwen35_like(&device)?;
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_close(&output, &expected, 1.5e-1);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_decode_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, expected) =
            hip_full_attention_decode_sample(&device)?;
        let output = full_attention_decode_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    #[test]
    fn cuda_full_attention_prefill_matches_reference() -> Result<()> {
        let _guard = cuda_test_guard();
        let _env_guard = CudaFullPrefillEnvGuard::set("1");
        let device = Device::new_cuda(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, expected) =
            hip_full_attention_prefill_sample(&device)?;
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-cuda")]
    #[test]
    fn cuda_paged_attention_decode_falls_back_for_large_head_dim() -> Result<()> {
        let _guard = cuda_test_guard();
        let _env_guard = CudaFullPrefillEnvGuard::set("1");
        let device = Device::new_cuda(0)?;
        let batch_queries = 2usize;
        let kv_len = 5usize;
        let head_dim = 256usize;
        let query_data = (0..batch_queries * head_dim)
            .map(|idx| (idx % 17) as f32 * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let key_data = (0..kv_len * head_dim)
            .map(|idx| (idx % 19) as f32 * 0.015625 - 0.125)
            .collect::<Vec<_>>();
        let value_data = (0..kv_len * head_dim)
            .map(|idx| (idx % 23) as f32 * 0.046875 - 0.5)
            .collect::<Vec<_>>();
        let queries = Tensor::from_vec(query_data, (batch_queries, head_dim), &device)?;
        let key = Tensor::from_vec(key_data, (kv_len, head_dim), &device)?;
        let value = Tensor::from_vec(value_data, (kv_len, head_dim), &device)?;
        let expected = paged_attention_decode_fallback(&queries, &key, &value)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let output = paged_attention_decode_megakernel(&queries, &key, &value)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_attention_decode_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let _env_guard = HipPersistentPrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        let (query, key, value, num_kv_groups, scale, seqlen_offset, _) =
            hip_full_attention_decode_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = full_attention_decode_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_stateful_conv_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (mixed_qkv, prev_state, weights, expected) = hip_linear_stateful_conv_sample(&device)?;
        let output = linear_stateful_conv_hip(&mixed_qkv, &prev_state, &weights, 3)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_stateful_conv_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (mixed_qkv, prev_state, weights, _expected) = hip_linear_stateful_conv_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = linear_stateful_conv_hip(&mixed_qkv, &prev_state, &weights, 3)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), mixed_qkv.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_stateful_conv_value_decay_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, expected) =
            hip_linear_stateful_conv_value_decay_sample(&device)?;
        let output = linear_stateful_conv_value_decay_hip(
            &mixed_qkv,
            &prev_state,
            &weights,
            &a,
            &dt_bias,
            &a_log_exp,
            3,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_stateful_conv_value_decay_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (mixed_qkv, prev_state, weights, a, dt_bias, a_log_exp, _expected) =
            hip_linear_stateful_conv_value_decay_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = linear_stateful_conv_value_decay_hip(
            &mixed_qkv,
            &prev_state,
            &weights,
            &a,
            &dt_bias,
            &a_log_exp,
            3,
        )?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), mixed_qkv.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_decode_step_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            initial_state,
            expected,
        ) = hip_linear_decode_step_sample(&device)?;
        let output = linear_decode_step_hip(
            &mixed_qkv,
            &prev_conv_state,
            &weights,
            &a_beta_raw,
            &dt_bias,
            &a_log_exp,
            &initial_state,
            2,
            2,
            2,
            3,
            2,
        )?
        .to_vec2::<f32>()?;
        assert_close(&output[0], &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_linear_decode_step_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            initial_state,
            _expected,
        ) = hip_linear_decode_step_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = linear_decode_step_hip(
            &mixed_qkv,
            &prev_conv_state,
            &weights,
            &a_beta_raw,
            &dt_bias,
            &a_log_exp,
            &initial_state,
            2,
            2,
            2,
            3,
            2,
        )?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_rms_norm_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, gate, weight, expected_norm, expected_gated) = hip_rms_norm_sample(&device)?;

        let norm = hip_rms_norm(&xs, &weight, 1e-6, true)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&norm, &expected_norm, 5e-3);

        let gated = hip_rms_norm_gated(&xs, &gate, &weight, 1e-6)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&gated, &expected_gated, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_rms_norm_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, gate, weight, _expected_norm, _expected_gated) = hip_rms_norm_sample(&device)?;

        candle::hip::reset_transfer_counters();
        let norm = hip_rms_norm(&xs, &weight, 1e-6, true)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(norm.dtype(), xs.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);

        candle::hip::reset_transfer_counters();
        let gated = hip_rms_norm_gated(&xs, &gate, &weight, 1e-6)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(gated.dtype(), xs.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_l2norm_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, expected) = hip_l2norm_sample(&device)?;
        let output = l2norm(&xs, 1e-6)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_l2norm_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, _expected) = hip_l2norm_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = l2norm(&xs, 1e-6)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), xs.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_swiglu_mul_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (gate, up, expected) = hip_swiglu_mul_sample(&device)?;
        let output = hip_swiglu_mul(&gate, &up)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_swiglu_mul_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (gate, up, _expected) = hip_swiglu_mul_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = hip_swiglu_mul(&gate, &up)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), gate.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_embedding_lookup_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (embeddings, indexes, expected) = hip_embedding_lookup_sample(&device)?;
        let output = hip_embedding_lookup(&embeddings, &indexes)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_embedding_lookup_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (embeddings, indexes, _expected) = hip_embedding_lookup_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = hip_embedding_lookup(&embeddings, &indexes)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), embeddings.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_causal_mask_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_size = 2usize;
        let tgt_len = 4usize;
        let seqlen_offset = 3usize;
        let output = hip_causal_mask(&device, DType::F16, batch_size, tgt_len, seqlen_offset)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = hip_causal_mask_expected(batch_size, tgt_len, seqlen_offset);
        assert_eq!(output.len(), expected.len());
        for (out, exp) in output.iter().zip(expected.iter()) {
            if exp.is_infinite() {
                assert!(out.is_infinite() && out.is_sign_negative());
            } else {
                assert!((out - exp).abs() <= 1e-6);
            }
        }
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_causal_mask_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        candle::hip::reset_transfer_counters();
        let output = hip_causal_mask(&device, DType::F16, 2, 4, 3)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_cumsum_last_dim_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, expected) = hip_cumsum_last_dim_sample(&device)?;
        let output = hip_cumsum_last_dim(&xs)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_cumsum_last_dim_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (xs, _expected) = hip_cumsum_last_dim_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = hip_cumsum_last_dim(&xs)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), xs.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_value_decay_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (a, dt_bias, a_log_exp, expected) = hip_value_decay_sample(&device)?;
        let output = hip_value_decay(&a, &dt_bias, &a_log_exp)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&output, &expected, 5e-3);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_value_decay_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let (a, dt_bias, a_log_exp, _expected) = hip_value_decay_sample(&device)?;
        candle::hip::reset_transfer_counters();
        let output = hip_value_decay(&a, &dt_bias, &a_log_exp)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), a.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_recurrent_prefill_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let seq_len = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.1f32, -0.2, 0.05, 0.3];
        let query_data = vec![0.2f32, -0.1, 0.0, 0.3, -0.2, 0.4];
        let key_data = vec![0.1f32, 0.2, -0.3, 0.5, 0.4, -0.2];
        let value_data = vec![0.3f32, -0.1, 0.2, 0.4, -0.2, 0.1];
        let beta_data = vec![0.5f32, 0.25, 0.75];
        let g_data = vec![0.0f32, -0.2, 0.1];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, seq_len, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(beta_data.clone(), (batch_heads, seq_len), &device)?;
        let g = Tensor::from_vec(g_data.clone(), (batch_heads, seq_len), &device)?;

        let output = delta_recurrent_prefill(&initial_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (seq_len + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (seq_len + k_head_dim) * v_head_dim;
                for t in 0..seq_len {
                    let g_t = g_data[bh * seq_len + t].exp();
                    let key_row = bh * seq_len * k_head_dim + t * k_head_dim;
                    let value_row = bh * seq_len * v_head_dim + t * v_head_dim;
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * seq_len + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + seq_len * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_recurrent_prefill_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let seq_len = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.05, 0.3],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.0, 0.3, -0.2, 0.4],
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.5, 0.4, -0.2],
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![0.3f32, -0.1, 0.2, 0.4, -0.2, 0.1],
            (batch_heads, seq_len, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(vec![0.5f32, 0.25, 0.75], (batch_heads, seq_len), &device)?;
        let g = Tensor::from_vec(vec![0.0f32, -0.2, 0.1], (batch_heads, seq_len), &device)?;

        candle::hip::reset_transfer_counters();
        let output = delta_recurrent_prefill(&initial_state, &query, &key, &value, &beta, &g)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), initial_state.dtype());
        assert!(
            counters.host_to_device_bytes <= 32,
            "unexpected recurrent-prefill H2D traffic: {counters:?}"
        );
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_single_prefill_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 4usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let query_data = vec![0.2f32, -0.1, 0.0, 0.3, -0.2, 0.4, 0.1, -0.25];
        let key_data = vec![0.1f32, 0.2, -0.3, 0.5, 0.4, -0.2, -0.15, 0.05];
        let value_data = vec![0.3f32, -0.1, 0.2, 0.4, -0.2, 0.1, 0.05, -0.15];
        let beta_data = vec![0.5f32, 0.25, 0.75, 0.6];
        let g_data = vec![0.0f32, -0.2, 0.1, -0.15];

        let initial_state =
            Tensor::zeros((batch_heads, k_head_dim, v_head_dim), DType::F32, &device)?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(beta_data.clone(), (batch_heads, chunk_size), &device)?;
        let g = Tensor::from_vec(g_data.clone(), (batch_heads, chunk_size), &device)?;

        let output = delta_chunk_single_prefill(&initial_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut prefix_g = vec![0.0f32; chunk_size];
        let mut acc_g = 0.0f32;
        for t in 0..chunk_size {
            acc_g += g_data[t];
            prefix_g[t] = acc_g;
        }

        let mut expected = vec![0.0f32; batch_heads * (chunk_size + k_head_dim) * v_head_dim];
        for v_idx in 0..v_head_dim {
            for i in 0..chunk_size {
                let row_i_k = i * k_head_dim;
                let mut out_i = 0.0f32;
                for j in 0..=i {
                    let row_j_k = j * k_head_dim;
                    let mut dot = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        dot += query_data[row_i_k + k_idx] * key_data[row_j_k + k_idx];
                    }
                    let local = dot * (prefix_g[i] - prefix_g[j]).exp();
                    out_i += local * value_data[j * v_head_dim + v_idx];
                }
                expected[i * v_head_dim + v_idx] = out_i;
            }

            let state_out = chunk_size * v_head_dim;
            for k_idx in 0..k_head_dim {
                let mut state = 0.0f32;
                for t in 0..chunk_size {
                    let raw_g_t = g_data[t];
                    state += key_data[t * k_head_dim + k_idx]
                        * (g_data[chunk_size - 1] - raw_g_t).exp()
                        * value_data[t * v_head_dim + v_idx];
                }
                expected[state_out + k_idx * v_head_dim + v_idx] = state;
            }
        }

        assert_close(&output, &expected, 1e-4);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_scan_raw_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.05, 0.3],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.0, 0.3, -0.2, 0.4, 0.1, -0.25, 0.15, 0.05, -0.3, 0.2],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.5, 0.4, -0.2, -0.15, 0.05, 0.25, -0.35, -0.05, 0.45],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![0.3f32, -0.1, 0.2, 0.4, -0.2, 0.1, 0.05, -0.15, 0.35, -0.25, -0.1, 0.2],
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(
            vec![0.5f32, 0.25, 0.75, 0.6, 0.4, 0.55],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let g = Tensor::from_vec(
            vec![0.0f32, -0.2, 0.1, -0.15, 0.05, -0.1],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_chunk_scan_raw(&initial_state, &query, &key, &value, &beta, &g)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), initial_state.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_gated_deltanet_multi_chunk_scan_prefill_matches_baseline_and_reduces_host_staging(
    ) -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let _chunk_size = LinearChunkSizeEnvGuard::set("8");
        let _single_chunk = HipChunkSinglePrefillEnvGuard::set("0");
        let cfg = tiny_linear_text_config();
        let varmap = candle_nn::VarMap::new();
        let vb = WeightBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GatedDeltaNet::new(&cfg, vb)?;
        let hidden_data: Vec<f32> = (0..(12usize * cfg.hidden_size))
            .map(|idx| (((idx % 13) as f32) - 6.0) * 0.05)
            .collect();
        let hidden_states =
            Tensor::from_vec(hidden_data, (1usize, 12usize, cfg.hidden_size), &device)?;

        let mut baseline_model = model.clone();
        let _multi_chunk_off = HipMultiChunkScanPrefillEnvGuard::set("0");
        candle::hip::reset_transfer_counters();
        let (baseline_out, _baseline_state, _baseline_profile) =
            baseline_model.trace_profiled(&hidden_states, None)?;
        let baseline_counters = candle::hip::transfer_counters();
        drop(_multi_chunk_off);

        let mut gated_model = model.clone();
        let _multi_chunk_on = HipMultiChunkScanPrefillEnvGuard::set("1");
        candle::hip::reset_transfer_counters();
        let (gated_out, _gated_state, _gated_profile) =
            gated_model.trace_profiled(&hidden_states, None)?;
        let gated_counters = candle::hip::transfer_counters();

        let baseline_out = baseline_out
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gated_out = gated_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&gated_out, &baseline_out, 1e-4);
        assert!(
            gated_counters.host_to_device_bytes < baseline_counters.host_to_device_bytes,
            "expected gated multi-chunk scan to reduce H2D traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        assert!(
            gated_counters.device_to_host_bytes < baseline_counters.device_to_host_bytes,
            "expected gated multi-chunk scan to reduce D2H traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        Ok(())
    }

    #[test]
    fn delta_decay_mask_avoids_masked_overflow_nans() -> Result<()> {
        let device = Device::Cpu;
        let g = Tensor::from_vec(
            vec![0.0f32, -200.0, -400.0, -600.0],
            (1usize, 1usize, 1usize, 4usize),
            &device,
        )?;
        let lower = Tensor::tril2(4, DType::F32, &device)?.reshape((1usize, 1usize, 1usize, 4usize, 4usize))?;
        let decay_mask = g
            .unsqueeze(4)?
            .broadcast_sub(&g.unsqueeze(3)?)?
            .broadcast_mul(&lower)?
            .exp()?
            .broadcast_mul(&lower)?;
        let values = decay_mask.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|value| value.is_finite()),
            "decay mask contained non-finite values: {values:?}"
        );
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_prebatched_local_state_scan_and_chunk_fused_match_baseline_and_reduce_host_staging(
    ) -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let _chunk_size = LinearChunkSizeEnvGuard::set("8");
        let _scan_mode = DeltaScanModeEnvGuard::set("prebatched-local");
        let _min_seq = DeltaKernelMinSequenceEnvGuard::set("1");
        let _single_chunk = HipChunkSinglePrefillEnvGuard::set("0");
        let _multi_chunk = HipMultiChunkScanPrefillEnvGuard::set("0");
        let cfg = tiny_linear_text_config();
        let varmap = candle_nn::VarMap::new();
        let vb = WeightBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GatedDeltaNet::new(&cfg, vb)?;
        let hidden_data: Vec<f32> = (0..(12usize * cfg.hidden_size))
            .map(|idx| (((idx % 17) as f32) - 8.0) * 0.04)
            .collect();
        let hidden_states =
            Tensor::from_vec(hidden_data, (1usize, 12usize, cfg.hidden_size), &device)?;

        let _state_scan_off = DeltaStateScanKernelEnvGuard::set("0");
        let _chunk_fused_off = DeltaChunkFusedKernelEnvGuard::set("0");
        let _full_off = DeltaFullKernelEnvGuard::set("0");
        let mut baseline_model = model.clone();
        candle::hip::reset_transfer_counters();
        let (baseline_out, _baseline_state, _baseline_profile) =
            baseline_model.trace_profiled(&hidden_states, None)?;
        let baseline_counters = candle::hip::transfer_counters();
        drop(_full_off);
        drop(_chunk_fused_off);
        drop(_state_scan_off);

        let _state_scan_on = DeltaStateScanKernelEnvGuard::set("1");
        let _chunk_fused_on = DeltaChunkFusedKernelEnvGuard::set("1");
        let _full_off = DeltaFullKernelEnvGuard::set("0");
        let mut gated_model = model.clone();
        candle::hip::reset_transfer_counters();
        let (gated_out, _gated_state, _gated_profile) =
            gated_model.trace_profiled(&hidden_states, None)?;
        let gated_counters = candle::hip::transfer_counters();

        let baseline_out = baseline_out
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gated_out = gated_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&gated_out, &baseline_out, 1e-4);
        assert!(
            gated_counters.host_to_device_bytes < baseline_counters.host_to_device_bytes,
            "expected state-scan/chunk-fused path to reduce H2D traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        assert!(
            gated_counters.device_to_host_bytes < baseline_counters.device_to_host_bytes,
            "expected state-scan/chunk-fused path to reduce D2H traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_prebatched_local_full_scan_matches_baseline_and_reduces_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let _chunk_size = LinearChunkSizeEnvGuard::set("8");
        let _scan_mode = DeltaScanModeEnvGuard::set("prebatched-local");
        let _min_seq = DeltaKernelMinSequenceEnvGuard::set("1");
        let _single_chunk = HipChunkSinglePrefillEnvGuard::set("0");
        let _multi_chunk = HipMultiChunkScanPrefillEnvGuard::set("0");
        let cfg = tiny_linear_text_config();
        let varmap = candle_nn::VarMap::new();
        let vb = WeightBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GatedDeltaNet::new(&cfg, vb)?;
        let hidden_data: Vec<f32> = (0..(12usize * cfg.hidden_size))
            .map(|idx| (((idx % 19) as f32) - 9.0) * 0.03)
            .collect();
        let hidden_states =
            Tensor::from_vec(hidden_data, (1usize, 12usize, cfg.hidden_size), &device)?;

        let _state_scan_off = DeltaStateScanKernelEnvGuard::set("0");
        let _chunk_fused_off = DeltaChunkFusedKernelEnvGuard::set("0");
        let _full_off = DeltaFullKernelEnvGuard::set("0");
        let mut baseline_model = model.clone();
        candle::hip::reset_transfer_counters();
        let (baseline_out, _baseline_state, _baseline_profile) =
            baseline_model.trace_profiled(&hidden_states, None)?;
        let baseline_counters = candle::hip::transfer_counters();
        drop(_full_off);
        drop(_chunk_fused_off);
        drop(_state_scan_off);

        let _state_scan_off = DeltaStateScanKernelEnvGuard::set("0");
        let _chunk_fused_off = DeltaChunkFusedKernelEnvGuard::set("0");
        let _full_on = DeltaFullKernelEnvGuard::set("1");
        let mut gated_model = model.clone();
        candle::hip::reset_transfer_counters();
        let (gated_out, _gated_state, _gated_profile) =
            gated_model.trace_profiled(&hidden_states, None)?;
        let gated_counters = candle::hip::transfer_counters();

        let baseline_out = baseline_out
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gated_out = gated_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&gated_out, &baseline_out, 1e-4);
        assert!(
            gated_counters.host_to_device_bytes < baseline_counters.host_to_device_bytes,
            "expected full-scan path to reduce H2D traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        assert!(
            gated_counters.device_to_host_bytes < baseline_counters.device_to_host_bytes,
            "expected full-scan path to reduce D2H traffic: baseline={baseline_counters:?} gated={gated_counters:?}"
        );
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_scan_gate_defaults_on_for_prebatched_local() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = DeltaFullKernelEnvGuard::clear();
        let device = Device::new_hip(0)?;
        assert!(use_delta_full_scan_kernel(
            &device,
            DeltaNetScanMode::PrebatchedLocal,
            4096
        ));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_full_scan_gate_honors_opt_out() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = DeltaFullKernelEnvGuard::set("0");
        let device = Device::new_hip(0)?;
        assert!(!use_delta_full_scan_kernel(
            &device,
            DeltaNetScanMode::PrebatchedLocal,
            4096
        ));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_exact_multi_chunk_full_scan_gate_defaults_on() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = DeltaFullKernelEnvGuard::clear();
        let device = Device::new_hip(0)?;
        assert!(use_hip_exact_multi_chunk_full_scan_prefill(
            &device,
            DeltaNetScanMode::PrebatchedLocal,
            18,
            3,
            8
        ));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_exact_multi_chunk_full_scan_gate_honors_opt_out() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = DeltaFullKernelEnvGuard::set("0");
        let device = Device::new_hip(0)?;
        assert!(!use_hip_exact_multi_chunk_full_scan_prefill(
            &device,
            DeltaNetScanMode::PrebatchedLocal,
            18,
            3,
            8
        ));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_chunk_single_prefill_gate_defaults_on() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = HipChunkSinglePrefillEnvGuard::clear();
        let device = Device::new_hip(0)?;
        assert!(use_hip_chunk_single_prefill_kernel(&device, 27, 1, 64));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_chunk_single_prefill_gate_honors_opt_out() -> Result<()> {
        let _guard = hip_test_guard();
        let _env = HipChunkSinglePrefillEnvGuard::set("0");
        let device = Device::new_hip(0)?;
        assert!(!use_hip_chunk_single_prefill_kernel(&device, 27, 1, 64));
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_step_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let prev_state_data = vec![0.15f32, -0.05, 0.2, 0.1];
        let query_data = vec![0.1f32, 0.3, -0.2, 0.4, 0.5, -0.1];
        let key_data = vec![0.2f32, -0.1, 0.0, 0.25, -0.3, 0.15];
        let value_data = vec![0.4f32, 0.2, -0.1, 0.3, 0.05, -0.2];
        let beta_data = vec![0.6f32, 0.5, 0.4];
        let g_data = vec![-0.1f32, 0.0, 0.2];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(beta_data.clone(), (batch_heads, chunk_size), &device)?;
        let g = Tensor::from_vec(g_data.clone(), (batch_heads, chunk_size), &device)?;

        let output = delta_chunk_step_raw(&prev_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (chunk_size + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (chunk_size + k_head_dim) * v_head_dim;
                for t in 0..chunk_size {
                    let g_t = g_data[bh * chunk_size + t].exp();
                    let key_row = bh * chunk_size * k_head_dim + t * k_head_dim;
                    let value_row = bh * chunk_size * v_head_dim + t * v_head_dim;
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * chunk_size + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + chunk_size * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_step_windowed_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let prev_state_data = vec![0.1f32, -0.1, 0.2, 0.05];
        let query_data = vec![0.2f32, 0.1, -0.3, 0.4, 0.0, 0.25, 0.5, -0.2];
        let key_data = vec![0.05f32, 0.2, -0.1, 0.3, 0.4, -0.2, 0.15, 0.1];
        let value_data = vec![0.3f32, -0.2, 0.1, 0.5, -0.1, 0.2, 0.4, 0.0];
        let beta_data = vec![0.5f32, 0.25, 0.75, 0.4];
        let g_data = vec![0.0f32, -0.3, 0.2, 0.1];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(
            beta_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let g = Tensor::from_vec(
            g_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_chunk_step_windowed_raw(&prev_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let total_tokens = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (total_tokens + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (total_tokens + k_head_dim) * v_head_dim;
                for t in 0..total_tokens {
                    let key_row = bh * total_tokens * k_head_dim + t * k_head_dim;
                    let value_row = bh * total_tokens * v_head_dim + t * v_head_dim;
                    let g_t = g_data[bh * total_tokens + t].exp();
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * total_tokens + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + total_tokens * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_scan_raw_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.05f32, 0.1, -0.2, 0.15];
        let query_data = vec![0.1f32, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3];
        let key_data = vec![0.2f32, 0.0, -0.15, 0.35, 0.25, -0.2, 0.1, 0.05];
        let value_data = vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.3];
        let beta_data = vec![0.4f32, 0.7, 0.5, 0.6];
        let g_data = vec![-0.2f32, 0.0, 0.1, -0.1];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(
            beta_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let g = Tensor::from_vec(
            g_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_chunk_scan_raw(&initial_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let total_tokens = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (total_tokens + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (total_tokens + k_head_dim) * v_head_dim;
                for t in 0..total_tokens {
                    let key_row = bh * total_tokens * k_head_dim + t * k_head_dim;
                    let value_row = bh * total_tokens * v_head_dim + t * v_head_dim;
                    let g_t = g_data[bh * total_tokens + t].exp();
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * total_tokens + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + total_tokens * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_state_scan_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 2 * k_head_dim + 1;

        let initial_state_data = vec![0.1f32, -0.2, 0.05, 0.15];
        let packed_scan_data = vec![
            0.2f32, -0.1, 0.05, 0.3, 0.9, -0.2, 0.4, 0.1, -0.05, 0.9, 0.3, 0.1, -0.15, 0.2, 0.8,
            0.05, -0.25, 0.2, 0.1, 0.8,
        ];
        let value_data = vec![0.4f32, 0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.25];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_scan = Tensor::from_vec(
            packed_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_state_scan(&initial_state, &packed_scan, &value)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (num_chunks + 1) * k_head_dim * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    let idx = bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx;
                    state[k_idx] = initial_state_data[idx];
                    expected[idx] = state[k_idx];
                }
                for chunk in 0..num_chunks {
                    let packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
                    let value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
                    let state_decay = packed_scan_data[packed_chunk_base + 2 * k_head_dim];
                    let mut update = vec![0.0f32; k_head_dim];
                    for t in 0..chunk_size {
                        let packed_row = packed_chunk_base + t * packed_width;
                        let value_row = value_chunk_base + t * v_head_dim;
                        let mut v_prime = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            v_prime +=
                                packed_scan_data[packed_row + k_head_dim + k_idx] * state[k_idx];
                        }
                        let v_new = value_data[value_row + v_idx] - v_prime;
                        for k_idx in 0..k_head_dim {
                            update[k_idx] += packed_scan_data[packed_row + k_idx] * v_new;
                        }
                    }
                    let out_chunk_base =
                        ((bh * (num_chunks + 1)) + (chunk + 1)) * k_head_dim * v_head_dim;
                    for k_idx in 0..k_head_dim {
                        state[k_idx] = state_decay * state[k_idx] + update[k_idx];
                        expected[out_chunk_base + k_idx * v_head_dim + v_idx] = state[k_idx];
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_state_scan_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 2 * k_head_dim + 1;

        let initial_state = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.05, 0.15],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_scan = Tensor::from_vec(
            vec![
                0.2f32, -0.1, 0.05, 0.3, 0.9, -0.2, 0.4, 0.1, -0.05, 0.9, 0.3, 0.1, -0.15, 0.2,
                0.8, 0.05, -0.25, 0.2, 0.1, 0.8,
            ],
            (batch_heads, num_chunks, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![0.4f32, 0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.25],
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_state_scan(&initial_state, &packed_scan, &value)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), initial_state.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_fused_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 3 * k_head_dim + 1;

        let prev_state_data = vec![0.1f32, 0.2, -0.05, 0.15];
        let packed_chunk_data = vec![
            0.2f32, -0.1, 0.05, 0.3, 0.1, -0.2, 0.85, -0.15, 0.25, 0.2, -0.05, -0.1, 0.15, 0.85,
        ];
        let value_data = vec![0.35f32, -0.1, 0.05, 0.4];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_chunk = Tensor::from_vec(
            packed_chunk_data.clone(),
            (batch_heads, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_chunk_fused(&prev_state, &packed_chunk, &value)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (2 * chunk_size + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let packed_base = bh * chunk_size * packed_width;
                let value_base = bh * chunk_size * v_head_dim;
                let out_base = bh * (2 * chunk_size + k_head_dim) * v_head_dim;
                let mut v_new = vec![0.0f32; chunk_size];
                let mut attn_inter = vec![0.0f32; chunk_size];
                for t in 0..chunk_size {
                    let packed_row = packed_base + t * packed_width;
                    let mut v_prime = 0.0f32;
                    let mut attn = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        v_prime +=
                            packed_chunk_data[packed_row + k_head_dim + k_idx] * state[k_idx];
                        attn +=
                            packed_chunk_data[packed_row + 2 * k_head_dim + k_idx] * state[k_idx];
                    }
                    v_new[t] = value_data[value_base + t * v_head_dim + v_idx] - v_prime;
                    attn_inter[t] = attn;
                    expected[out_base + t * v_head_dim + v_idx] = v_new[t];
                    expected[out_base + (chunk_size + t) * v_head_dim + v_idx] = attn_inter[t];
                }
                let state_decay = packed_chunk_data[packed_base + 3 * k_head_dim];
                for k_idx in 0..k_head_dim {
                    let mut update = 0.0f32;
                    for t in 0..chunk_size {
                        let packed_row = packed_base + t * packed_width;
                        update += packed_chunk_data[packed_row + k_idx] * v_new[t];
                    }
                    expected[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
                        state_decay * state[k_idx] + update;
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_chunk_fused_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 3 * k_head_dim + 1;

        let prev_state = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.05, 0.15],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_chunk = Tensor::from_vec(
            vec![
                0.2f32, -0.1, 0.05, 0.3, 0.1, -0.2, 0.85, -0.15, 0.25, 0.2, -0.05, -0.1, 0.15,
                0.85,
            ],
            (batch_heads, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![0.35f32, -0.1, 0.05, 0.4],
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_chunk_fused(&prev_state, &packed_chunk, &value)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), prev_state.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_full_scan_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.15f32, -0.05, 0.2, 0.1];
        let weighted_key_data = vec![0.2f32, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25, -0.05];
        let k_cumdecay_data = vec![0.1f32, 0.25, -0.2, 0.05, 0.15, -0.1, 0.05, 0.2];
        let q_state_data = vec![0.05f32, -0.15, 0.2, 0.1, -0.1, 0.3, 0.15, -0.05];
        let local_attn_data = vec![0.2f32, 0.1, -0.1, 0.3, 0.05, -0.2, 0.25, 0.15];
        let state_decay_data = vec![0.85f32, 0.9];
        let value_data = vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.35];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let weighted_key_scan = Tensor::from_vec(
            weighted_key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let k_cumdecay_scan = Tensor::from_vec(
            k_cumdecay_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let q_state_scan = Tensor::from_vec(
            q_state_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let local_attn_scan = Tensor::from_vec(
            local_attn_data.clone(),
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;
        let state_decay_scan =
            Tensor::from_vec(state_decay_data.clone(), (batch_heads, num_chunks), &device)?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_full_scan(
            &initial_state,
            &weighted_key_scan,
            &k_cumdecay_scan,
            &q_state_scan,
            &local_attn_scan,
            &state_decay_scan,
            &value,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let token_count = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (token_count + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let scan_base = bh * num_chunks * chunk_size * k_head_dim;
                let local_base = bh * num_chunks * chunk_size * chunk_size;
                let decay_base = bh * num_chunks;
                let value_base = bh * token_count * v_head_dim;
                let out_base = bh * (token_count + k_head_dim) * v_head_dim;
                let mut v_new = vec![0.0f32; chunk_size];
                let mut attn_inter = vec![0.0f32; chunk_size];
                for chunk in 0..num_chunks {
                    let chunk_scan = scan_base + chunk * chunk_size * k_head_dim;
                    let chunk_local = local_base + chunk * chunk_size * chunk_size;
                    let chunk_value = value_base + chunk * chunk_size * v_head_dim;
                    for t in 0..chunk_size {
                        let row = chunk_scan + t * k_head_dim;
                        let mut v_prime = 0.0f32;
                        let mut attn = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            v_prime += k_cumdecay_data[row + k_idx] * state[k_idx];
                            attn += q_state_data[row + k_idx] * state[k_idx];
                        }
                        v_new[t] = value_data[chunk_value + t * v_head_dim + v_idx] - v_prime;
                        attn_inter[t] = attn;
                    }
                    for t in 0..chunk_size {
                        let row = chunk_local + t * chunk_size;
                        let mut local = 0.0f32;
                        for s in 0..chunk_size {
                            local += local_attn_data[row + s] * v_new[s];
                        }
                        expected[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                            attn_inter[t] + local;
                    }
                    let state_decay = state_decay_data[decay_base + chunk];
                    for k_idx in 0..k_head_dim {
                        let mut update = 0.0f32;
                        for t in 0..chunk_size {
                            let row = chunk_scan + t * k_head_dim;
                            update += weighted_key_data[row + k_idx] * v_new[t];
                        }
                        state[k_idx] = state_decay * state[k_idx] + update;
                    }
                }
                let state_out = out_base + token_count * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_full_scan_packed_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.15f32, -0.05, 0.2, 0.1];
        let query_scan_data = vec![0.05f32, -0.15, 0.2, 0.1, -0.1, 0.3, 0.15, -0.05];
        let key_scan_data = vec![0.2f32, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25, -0.05];
        let exp_g_scan_data = vec![0.9f32, 1.1, 0.85, 1.2];
        let k_cumdecay_data = vec![0.1f32, 0.25, -0.2, 0.05, 0.15, -0.1, 0.05, 0.2];
        let local_attn_data = vec![0.2f32, 0.1, -0.1, 0.3, 0.05, -0.2, 0.25, 0.15];
        let value_data = vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.35];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query_scan = Tensor::from_vec(
            query_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            key_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            exp_g_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let k_cumdecay_scan = Tensor::from_vec(
            k_cumdecay_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let local_attn_scan = Tensor::from_vec(
            local_attn_data.clone(),
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        let packed_scan =
            delta_full_scan_pack(&query_scan, &key_scan, &exp_g_scan, &k_cumdecay_scan)?;
        let output = delta_full_scan_packed(&initial_state, &packed_scan, &local_attn_scan, &value)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let exp_g_last = exp_g_scan.i((.., .., chunk_size - 1))?;
        let state_decay_scan = exp_g_last.contiguous()?;
        let chunk_decay_scan = exp_g_last
            .unsqueeze(D::Minus1)?
            .broadcast_div(&exp_g_scan)?
            .unsqueeze(D::Minus1)?;
        let weighted_key_scan = key_scan.broadcast_mul(&chunk_decay_scan)?;
        let q_state_scan = query_scan.broadcast_mul(&exp_g_scan.unsqueeze(D::Minus1)?)?;
        let reference = delta_full_scan(
            &initial_state,
            &weighted_key_scan.contiguous()?,
            &k_cumdecay_scan.contiguous()?,
            &q_state_scan.contiguous()?,
            &local_attn_scan.contiguous()?,
            &state_decay_scan.contiguous()?,
            &value.contiguous()?,
        )?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;

        assert_close(&output, &reference, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_full_scan_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state = Tensor::from_vec(
            vec![0.15f32, -0.05, 0.2, 0.1],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let weighted_key_scan = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25, -0.05],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let k_cumdecay_scan = Tensor::from_vec(
            vec![0.1f32, 0.25, -0.2, 0.05, 0.15, -0.1, 0.05, 0.2],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let q_state_scan = Tensor::from_vec(
            vec![0.05f32, -0.15, 0.2, 0.1, -0.1, 0.3, 0.15, -0.05],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let local_attn_scan = Tensor::from_vec(
            vec![0.2f32, 0.1, -0.1, 0.3, 0.05, -0.2, 0.25, 0.15],
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;
        let state_decay_scan =
            Tensor::from_vec(vec![0.85f32, 0.9], (batch_heads, num_chunks), &device)?;
        let value = Tensor::from_vec(
            vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.35],
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_full_scan(
            &initial_state,
            &weighted_key_scan,
            &k_cumdecay_scan,
            &q_state_scan,
            &local_attn_scan,
            &state_decay_scan,
            &value,
        )?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), initial_state.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_full_scan_packed_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state = Tensor::from_vec(
            vec![0.15f32, -0.05, 0.2, 0.1],
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query_scan = Tensor::from_vec(
            vec![0.05f32, -0.15, 0.2, 0.1, -0.1, 0.3, 0.15, -0.05],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25, -0.05],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            vec![0.9f32, 1.1, 0.85, 1.2],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let k_cumdecay_scan = Tensor::from_vec(
            vec![0.1f32, 0.25, -0.2, 0.05, 0.15, -0.1, 0.05, 0.2],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let local_attn_scan = Tensor::from_vec(
            vec![0.2f32, 0.1, -0.1, 0.3, 0.05, -0.2, 0.25, 0.15],
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.35],
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let packed_scan =
            delta_full_scan_pack(&query_scan, &key_scan, &exp_g_scan, &k_cumdecay_scan)?;
        let output =
            delta_full_scan_packed(&initial_state, &packed_scan, &local_attn_scan, &value)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(packed_scan.dtype(), query_scan.dtype());
        assert_eq!(output.dtype(), initial_state.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_local_attn_scan_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;

        let query_scan_data = vec![
            0.1f32, -0.2, 0.05, 0.3, -0.1, 0.25, 0.2, 0.15, -0.05, 0.4, 0.12, -0.18,
        ];
        let key_scan_data = vec![
            0.05f32, 0.2, -0.1, 0.15, 0.25, -0.05, 0.3, 0.1, -0.2, 0.35, 0.08, -0.12,
        ];
        let exp_g_scan_data = vec![0.8f32, 1.0, 1.3, 0.9, 1.1, 1.4];

        let query_scan = Tensor::from_vec(
            query_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            key_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            exp_g_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_local_attn_scan(&query_scan, &key_scan, &exp_g_scan)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * num_chunks * chunk_size * chunk_size];
        for bh in 0..batch_heads {
            for chunk in 0..num_chunks {
                let exp_base = (bh * num_chunks + chunk) * chunk_size;
                let qk_base = exp_base * k_head_dim;
                let out_base = (bh * num_chunks + chunk) * chunk_size * chunk_size;
                for t in 0..chunk_size {
                    let exp_t = exp_g_scan_data[exp_base + t];
                    for s in 0..=t {
                        let mut dot = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            dot += query_scan_data[qk_base + t * k_head_dim + k_idx]
                                * key_scan_data[qk_base + s * k_head_dim + k_idx];
                        }
                        expected[out_base + t * chunk_size + s] =
                            dot * (exp_t / exp_g_scan_data[exp_base + s]);
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_local_attn_scan_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;

        let query_scan = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.05, 0.3, -0.1, 0.25, 0.2, 0.15, -0.05, 0.4, 0.12, -0.18],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            vec![0.05f32, 0.2, -0.1, 0.15, 0.25, -0.05, 0.3, 0.1, -0.2, 0.35, 0.08, -0.12],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            vec![0.8f32, 1.0, 1.3, 0.9, 1.1, 1.4],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_local_attn_scan(&query_scan, &key_scan, &exp_g_scan)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), query_scan.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_base_attn_scan_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;

        let k_beta_data = vec![
            0.08f32, -0.12, 0.05, 0.18, -0.09, 0.14, 0.16, 0.11, -0.04, 0.22, 0.1, -0.07,
        ];
        let key_scan_data = vec![
            0.05f32, 0.2, -0.1, 0.15, 0.25, -0.05, 0.3, 0.1, -0.2, 0.35, 0.08, -0.12,
        ];
        let exp_g_scan_data = vec![0.8f32, 1.0, 1.3, 0.9, 1.1, 1.4];

        let k_beta_scan = Tensor::from_vec(
            k_beta_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            key_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            exp_g_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_base_attn_scan(&k_beta_scan, &key_scan, &exp_g_scan)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * num_chunks * chunk_size * chunk_size];
        for bh in 0..batch_heads {
            for chunk in 0..num_chunks {
                let exp_base = (bh * num_chunks + chunk) * chunk_size;
                let qk_base = exp_base * k_head_dim;
                let out_base = (bh * num_chunks + chunk) * chunk_size * chunk_size;
                for t in 0..chunk_size {
                    let exp_t = exp_g_scan_data[exp_base + t];
                    for s in 0..t {
                        let mut dot = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            dot += k_beta_data[qk_base + t * k_head_dim + k_idx]
                                * key_scan_data[qk_base + s * k_head_dim + k_idx];
                        }
                        expected[out_base + t * chunk_size + s] =
                            -dot * (exp_t / exp_g_scan_data[exp_base + s]);
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_base_attn_scan_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;

        let k_beta_scan = Tensor::from_vec(
            vec![0.08f32, -0.12, 0.05, 0.18, -0.09, 0.14, 0.16, 0.11, -0.04, 0.22, 0.1, -0.07],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            vec![0.05f32, 0.2, -0.1, 0.15, 0.25, -0.05, 0.3, 0.1, -0.2, 0.35, 0.08, -0.12],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            vec![0.8f32, 1.0, 1.3, 0.9, 1.1, 1.4],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_base_attn_scan(&k_beta_scan, &key_scan, &exp_g_scan)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), k_beta_scan.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_attn_solve_scan_matches_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 4usize;

        let base_attn_data = vec![
            0.0f32, 0.0, 0.0, 0.0,
            -0.2, 0.0, 0.0, 0.0,
            0.1, -0.05, 0.0, 0.0,
            -0.08, 0.03, -0.02, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -0.12, 0.0, 0.0, 0.0,
            0.04, -0.03, 0.0, 0.0,
            -0.02, 0.05, -0.01, 0.0,
        ];
        let base_attn_scan = Tensor::from_vec(
            base_attn_data.clone(),
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;

        let output = delta_attn_solve_scan(&base_attn_scan)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * num_chunks * chunk_size * chunk_size];
        for bh in 0..batch_heads {
            for chunk in 0..num_chunks {
                let base = (bh * num_chunks + chunk) * chunk_size * chunk_size;
                let mut rows = vec![0.0f32; chunk_size * chunk_size];
                for i in 1..chunk_size {
                    for j in 0..i {
                        let row_val = base_attn_data[base + i * chunk_size + j];
                        let mut correction = 0.0f32;
                        for k in 0..i {
                            correction += base_attn_data[base + i * chunk_size + k]
                                * rows[k * chunk_size + j];
                        }
                        rows[i * chunk_size + j] = row_val + correction;
                    }
                }
                for i in 0..chunk_size {
                    for j in 0..chunk_size {
                        let mut value = rows[i * chunk_size + j];
                        if i == j {
                            value += 1.0;
                        }
                        expected[base + i * chunk_size + j] = value;
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_attn_solve_scan_avoids_host_staging() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 4usize;

        let base_attn_scan = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0,
                -0.2, 0.0, 0.0, 0.0,
                0.1, -0.05, 0.0, 0.0,
                -0.08, 0.03, -0.02, 0.0,
                0.0, 0.0, 0.0, 0.0,
                -0.12, 0.0, 0.0, 0.0,
                0.04, -0.03, 0.0, 0.0,
                -0.02, 0.05, -0.01, 0.0,
            ],
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;

        candle::hip::reset_transfer_counters();
        let output = delta_attn_solve_scan(&base_attn_scan)?;
        let counters = candle::hip::transfer_counters();
        assert_eq!(output.dtype(), base_attn_scan.dtype());
        assert_eq!(counters.host_to_device_bytes, 0);
        assert_eq!(counters.device_to_host_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    #[test]
    fn hip_delta_attn_solve_from_inputs_matches_composed_reference() -> Result<()> {
        let _guard = hip_test_guard();
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 4usize;
        let k_head_dim = 3usize;

        let k_beta_scan = Tensor::from_vec(
            vec![
                0.08f32, -0.12, 0.03, 0.05, 0.18, -0.04, -0.09, 0.14, 0.07, 0.11, -0.06, 0.02,
                0.16, 0.11, -0.04, 0.22, 0.10, -0.07, -0.03, 0.09, 0.05, 0.18, -0.02, 0.04,
            ],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key_scan = Tensor::from_vec(
            vec![
                0.05f32, 0.20, -0.08, -0.10, 0.15, 0.04, 0.25, -0.05, 0.06, 0.30, 0.10, -0.02,
                -0.20, 0.35, 0.12, 0.08, -0.12, 0.05, 0.14, 0.09, -0.03, -0.06, 0.17, 0.11,
            ],
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let exp_g_scan = Tensor::from_vec(
            vec![0.8f32, 1.0, 1.3, 1.6, 0.9, 1.1, 1.4, 1.7],
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let base_attn = delta_base_attn_scan(&k_beta_scan, &key_scan, &exp_g_scan)?;
        let expected = delta_attn_solve_scan(&base_attn)?;
        let output = delta_attn_solve_from_inputs(&k_beta_scan, &key_scan, &exp_g_scan)?;

        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn parses_nested_text_config() {
        let config: Config = serde_json::from_str(
            r#"{
                "text_config": {
                    "vocab_size": 16,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25
                    }
                }
            }"#,
        )
        .unwrap();
        let config = config.normalized();
        assert_eq!(config.text_config.layer_types.len(), 4);
        assert_eq!(config.text_config.layer_types[3], "full_attention");
        assert_eq!(config.text_config.rope_theta(), 1_000_000.0);
        assert_eq!(config.text_config.partial_rotary_factor(), 0.25);
    }

    #[test]
    fn normalized_config_supplies_hybrid_layer_pattern() {
        let cfg = Config {
            text_config: TextConfig {
                vocab_size: 16,
                hidden_size: 32,
                intermediate_size: 64,
                num_hidden_layers: 8,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                hidden_act: Activation::Silu,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: false,
                attention_bias: false,
                attention_dropout: 0.0,
                head_dim: 8,
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 8,
                linear_value_head_dim: 8,
                linear_num_key_heads: 2,
                linear_num_value_heads: 4,
                layer_types: Vec::new(),
                rope_parameters: None,
            },
        }
        .normalized();

        assert_eq!(
            cfg.text_config.layer_types,
            vec![
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
        );
    }

    #[test]
    fn metal_chunk_size_scales_with_sequence_length() {
        assert_eq!(recommended_metal_linear_chunk_size(512), 16);
        assert_eq!(recommended_metal_linear_chunk_size(2048), 24);
        assert_eq!(recommended_metal_linear_chunk_size(8192), 24);
    }

    #[test]
    fn hip_chunk_size_scales_with_sequence_length() {
        assert_eq!(recommended_hip_linear_chunk_size(1), 4);
        assert_eq!(recommended_hip_linear_chunk_size(4), 4);
        assert_eq!(recommended_hip_linear_chunk_size(5), 8);
        assert_eq!(recommended_hip_linear_chunk_size(8), 8);
        assert_eq!(recommended_hip_linear_chunk_size(9), 16);
        assert_eq!(recommended_hip_linear_chunk_size(16), 16);
        assert_eq!(recommended_hip_linear_chunk_size(17), 32);
        assert_eq!(recommended_hip_linear_chunk_size(32), 32);
        assert_eq!(recommended_hip_linear_chunk_size(33), 64);
        assert_eq!(recommended_hip_linear_chunk_size(8192), 64);
    }

    #[test]
    fn parses_delta_scan_modes() {
        assert_eq!(
            parse_delta_net_scan_mode("flat3d"),
            Some(DeltaNetScanMode::Flat3d)
        );
        assert_eq!(
            parse_delta_net_scan_mode("hoisted-decays"),
            Some(DeltaNetScanMode::HoistedDecays)
        );
        assert_eq!(
            parse_delta_net_scan_mode("prebatched-local"),
            Some(DeltaNetScanMode::PrebatchedLocal)
        );
        assert_eq!(
            parse_delta_net_scan_mode("torch-like"),
            Some(DeltaNetScanMode::TorchLike)
        );
        assert_eq!(parse_delta_net_scan_mode("unknown"), None);
    }

    #[test]
    fn recommended_delta_scan_mode_uses_prebatched_local_for_long_metal_contexts() {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        let short = recommended_delta_net_execution_policy(&device, 512, 32);
        let long = recommended_delta_net_execution_policy(&device, 2048, 86);
        match device.location() {
            DeviceLocation::Metal { .. } => {
                assert_eq!(short.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!short.use_flattened_solve);
                assert_eq!(long.scan_mode, DeltaNetScanMode::PrebatchedLocal);
                assert!(long.use_flattened_solve);
            }
            _ => {
                assert_eq!(short.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!short.use_flattened_solve);
                assert_eq!(long.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!long.use_flattened_solve);
            }
        }
    }
}
