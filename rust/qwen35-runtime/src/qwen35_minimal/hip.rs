use std::ffi::{c_int, c_uint, c_void};

use candle_core::{DType, Error, Result};

pub fn dtype_code(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::F16 => Ok(0),
        DType::F32 => Ok(1),
        DType::BF16 => Ok(2),
        other => Err(Error::Hip(
            format!("unsupported Qwen3.5 minimal HIP dtype {other:?}").into(),
        )),
    }
}

pub fn index_dtype_code(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::U8 => Ok(0),
        DType::U32 => Ok(1),
        DType::I64 => Ok(2),
        other => Err(Error::Hip(
            format!("unsupported Qwen3.5 minimal HIP index dtype {other:?}").into(),
        )),
    }
}

pub fn hip_error(op: &str, status: c_int) -> Error {
    Error::Hip(format!("{op} failed with HIP status {status}").into())
}

#[cfg(feature = "qwen35-minimal-hip")]
fn with_device<T>(device_ordinal: usize, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let device_ordinal = c_int::try_from(device_ordinal).map_err(|_| {
        Error::Hip(format!("hip device ordinal {device_ordinal} does not fit c_int").into())
    })?;
    let mut previous_device = 0;
    let mut restore_device = None;
    let status = unsafe { hipGetDevice(&mut previous_device) };
    if status != 0 {
        return Err(hip_error("hipGetDevice", status));
    }
    if previous_device != device_ordinal {
        let status = unsafe { hipSetDevice(device_ordinal) };
        if status != 0 {
            return Err(hip_error("hipSetDevice", status));
        }
        restore_device = Some(previous_device);
    }
    let result = f();
    if let Some(previous_device) = restore_device {
        let status = unsafe { hipSetDevice(previous_device) };
        if status != 0 {
            return Err(hip_error("hipSetDevice", status));
        }
    }
    result
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn register_host_mapping_for_device(
    device_ordinal: usize,
    ptr: *const c_void,
    len_bytes: usize,
) -> Result<*const c_void> {
    if ptr.is_null() || len_bytes == 0 {
        return Err(Error::Hip(
            "hip host registration requires a non-null pointer and non-zero length".into(),
        ));
    }
    with_device(device_ordinal, || {
        let status =
            unsafe { hipHostRegister(ptr as *mut c_void, len_bytes, HIP_HOST_REGISTER_MAPPED) };
        if status != 0 {
            return Err(hip_error("hipHostRegister", status));
        }
        let mut device_ptr = std::ptr::null_mut();
        let status = unsafe { hipHostGetDevicePointer(&mut device_ptr, ptr as *mut c_void, 0) };
        if status != 0 {
            unsafe {
                let _ = hipHostUnregister(ptr as *mut c_void);
            }
            return Err(hip_error("hipHostGetDevicePointer", status));
        }
        Ok(device_ptr as *const c_void)
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn unregister_host_mapping(ptr: *const c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = hipHostUnregister(ptr as *mut c_void);
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn alloc_device_bytes(device_ordinal: usize, len_bytes: usize) -> Result<*mut c_void> {
    if len_bytes == 0 {
        return Err(Error::Hip("hip device allocation requires non-zero length".into()));
    }
    with_device(device_ordinal, || {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { hipMalloc(&mut ptr, len_bytes) };
        if status != 0 {
            return Err(hip_error("hipMalloc", status));
        }
        Ok(ptr)
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn free_device_bytes(device_ordinal: usize, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let _ = with_device(device_ordinal, || {
        let status = unsafe { hipFree(ptr) };
        if status != 0 {
            return Err(hip_error("hipFree", status));
        }
        Ok(())
    });
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn copy_host_to_device(
    device_ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    len_bytes: usize,
) -> Result<()> {
    if dst.is_null() || src.is_null() || len_bytes == 0 {
        return Err(Error::Hip("hipMemcpy H2D requires valid pointers and non-zero length".into()));
    }
    with_device(device_ordinal, || {
        let status = unsafe { hipMemcpy(dst, src, len_bytes, HIP_MEMCPY_HOST_TO_DEVICE) };
        if status != 0 {
            return Err(hip_error("hipMemcpyHostToDevice", status));
        }
        Ok(())
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub fn copy_device_to_host(
    device_ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    len_bytes: usize,
) -> Result<()> {
    if dst.is_null() || src.is_null() || len_bytes == 0 {
        return Err(Error::Hip("hipMemcpy D2H requires valid pointers and non-zero length".into()));
    }
    with_device(device_ordinal, || {
        let status = unsafe { hipMemcpy(dst, src, len_bytes, HIP_MEMCPY_DEVICE_TO_HOST) };
        if status != 0 {
            return Err(hip_error("hipMemcpyDeviceToHost", status));
        }
        Ok(())
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
const HIP_HOST_REGISTER_MAPPED: c_uint = 0x2;
#[cfg(feature = "qwen35-minimal-hip")]
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
#[cfg(feature = "qwen35-minimal-hip")]
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[cfg(feature = "qwen35-minimal-hip")]
#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipHostRegister(host_ptr: *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn hipHostUnregister(host_ptr: *mut c_void) -> c_int;
    fn hipHostGetDevicePointer(
        dev_ptr: *mut *mut c_void,
        host_ptr: *mut c_void,
        flags: c_uint,
    ) -> c_int;
    fn hipGetDevice(device: *mut c_int) -> c_int;
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
}

pub mod ffi {
    use super::*;

    unsafe extern "C" {
        pub fn dotcache_qwen35_hip_full_attention_prefill(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            q_heads: usize,
            kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
            num_kv_groups: usize,
            scale: f32,
            seqlen_offset: usize,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_prefill_conv_pack(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            conv_dim: usize,
            total_len: usize,
            seq_len: usize,
            kernel_size: usize,
            mixed_qkv: *const c_void,
            weights: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_stateful_conv(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            conv_dim: usize,
            seq_len: usize,
            state_len: usize,
            kernel_size: usize,
            mixed_qkv: *const c_void,
            prev_state: *const c_void,
            weights: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_stateful_conv_value_decay(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            conv_dim: usize,
            seq_len: usize,
            state_len: usize,
            kernel_size: usize,
            num_heads: usize,
            mixed_qkv: *const c_void,
            prev_state: *const c_void,
            weights: *const c_void,
            a: *const c_void,
            dt_bias: *const c_void,
            a_log_exp: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_stateful_conv_value_decay_with_state(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            conv_dim: usize,
            seq_len: usize,
            state_len: usize,
            kernel_size: usize,
            num_heads: usize,
            mixed_qkv: *const c_void,
            prev_state: *const c_void,
            weights: *const c_void,
            a: *const c_void,
            dt_bias: *const c_void,
            a_log_exp: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_decode_prepare(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            num_v_heads: usize,
            head_k_dim: usize,
            head_v_dim: usize,
            state_len: usize,
            kernel_size: usize,
            head_repeat: usize,
            mixed_qkv: *const c_void,
            prev_conv_state: *const c_void,
            weights: *const c_void,
            a_beta_raw: *const c_void,
            dt_bias: *const c_void,
            a_log_exp: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_linear_decode_apply(
            device_ordinal: usize,
            batch_size: usize,
            num_v_heads: usize,
            head_k_dim: usize,
            head_v_dim: usize,
            packed: *const c_void,
            initial_state: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_recurrent_prefill(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            seq_len: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_chunk_single_prefill(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_chunk_step(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            prev_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_chunk_scan_raw(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_state_scan(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            packed_scan: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_chunk_fused(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            prev_state: *const c_void,
            packed_chunk: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_full_scan(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            weighted_key_scan: *const c_void,
            k_cumdecay_scan: *const c_void,
            q_state_scan: *const c_void,
            local_attn_scan: *const c_void,
            state_decay_scan: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_full_scan_pack(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            query_scan: *const c_void,
            key_scan: *const c_void,
            exp_g_scan: *const c_void,
            k_cumdecay_scan: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_local_attn_scan(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            query_scan: *const c_void,
            key_scan: *const c_void,
            exp_g_scan: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_base_attn_scan(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            k_beta_scan: *const c_void,
            key_scan: *const c_void,
            exp_g_scan: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_attn_solve_scan(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            base_attn_scan: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_attn_solve_from_inputs(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            k_beta_scan: *const c_void,
            key_scan: *const c_void,
            exp_g_scan: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_delta_full_scan_packed(
            dtype: c_int,
            device_ordinal: usize,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            packed_scan: *const c_void,
            local_attn_scan: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_l2norm(
            dtype: c_int,
            device_ordinal: usize,
            n_rows: usize,
            n_cols: usize,
            eps: f32,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_value_decay(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            num_heads: usize,
            a: *const c_void,
            dt_bias: *const c_void,
            a_log_exp: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_rms_norm(
            dtype: c_int,
            device_ordinal: usize,
            n_rows: usize,
            n_cols: usize,
            eps: f32,
            add_unit_offset: c_int,
            xs: *const c_void,
            weight: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_rms_norm_gated(
            dtype: c_int,
            device_ordinal: usize,
            n_rows: usize,
            n_cols: usize,
            eps: f32,
            hidden: *const c_void,
            gate: *const c_void,
            weight: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_swiglu_mul(
            dtype: c_int,
            device_ordinal: usize,
            elem_count: usize,
            gate: *const c_void,
            up: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_embedding_lookup(
            dtype: c_int,
            index_dtype: c_int,
            device_ordinal: usize,
            token_count: usize,
            vocab_size: usize,
            hidden_size: usize,
            embeddings: *const c_void,
            indexes: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_output_projection_lookup(
            dtype: c_int,
            device_ordinal: usize,
            rows: usize,
            hidden_size: usize,
            vocab_size: usize,
            hidden: *const c_void,
            weights: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_causal_mask(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            tgt_len: usize,
            seqlen_offset: usize,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_cumsum_last_dim(
            dtype: c_int,
            device_ordinal: usize,
            rows: usize,
            cols: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_exp(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_recip(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_sigmoid(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_log(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_unary_view(
            op: c_int,
            dtype: c_int,
            device_ordinal: usize,
            rank: c_int,
            total_elems: usize,
            scalar: f32,
            xs: *const c_void,
            in_strides: *const c_int,
            out_dims: *const c_int,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_cast_view(
            input_dtype: c_int,
            output_dtype: c_int,
            device_ordinal: usize,
            rank: c_int,
            total_elems: usize,
            xs: *const c_void,
            in_strides: *const c_int,
            out_dims: *const c_int,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_cast(
            input_dtype: c_int,
            output_dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_binary_broadcast(
            op: c_int,
            dtype: c_int,
            device_ordinal: usize,
            rank: c_int,
            total_elems: usize,
            lhs: *const c_void,
            rhs: *const c_void,
            lhs_strides: *const c_int,
            rhs_strides: *const c_int,
            out_dims: *const c_int,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_batched_matmul(
            dtype: c_int,
            device_ordinal: usize,
            batch_rank: c_int,
            batch_elems: usize,
            m: c_int,
            n: c_int,
            k: c_int,
            lhs_batch_dims: *const c_int,
            rhs_batch_dims: *const c_int,
            out_batch_dims: *const c_int,
            lhs: *const c_void,
            rhs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_batched_matmul_view(
            dtype: c_int,
            device_ordinal: usize,
            batch_rank: c_int,
            batch_elems: usize,
            m: c_int,
            n: c_int,
            k: c_int,
            lhs_batch_strides: *const c_int,
            rhs_batch_strides: *const c_int,
            out_batch_dims: *const c_int,
            lhs_row_stride: c_int,
            lhs_k_stride: c_int,
            rhs_k_stride: c_int,
            rhs_col_stride: c_int,
            lhs: *const c_void,
            rhs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_mul_scalar(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            scalar: f32,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_reduce_keepdim(
            dtype: c_int,
            device_ordinal: usize,
            outer: usize,
            reduce: usize,
            inner: usize,
            sum: c_int,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_reduce_keepdim_view(
            dtype: c_int,
            device_ordinal: usize,
            rank: c_int,
            reduce_dim: c_int,
            reduce_len: usize,
            total_out_elems: usize,
            sum: c_int,
            xs: *const c_void,
            in_strides: *const c_int,
            out_dims: *const c_int,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_add_scalar(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            scalar: f32,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn dotcache_qwen35_hip_sqrt(
            dtype: c_int,
            device_ordinal: usize,
            total_elems: usize,
            xs: *const c_void,
            out: *mut c_void,
        ) -> c_int;
    }
}
