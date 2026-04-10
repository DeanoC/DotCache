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
    }
}
