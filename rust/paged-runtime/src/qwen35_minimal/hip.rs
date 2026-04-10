use std::ffi::{c_int, c_void};

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

pub fn hip_error(op: &str, status: c_int) -> Error {
    Error::Hip(format!("{op} failed with HIP status {status}").into())
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
    }
}
