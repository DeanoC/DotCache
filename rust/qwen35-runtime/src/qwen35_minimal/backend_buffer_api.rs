use candle_core::{D, DType, Device, Result, Tensor};

use super::{backend_ops, ops};
use super::model::ImmutableEmbedding;
use super::types::StateBuffer;
use crate::backends;

fn repeat_heads_impl(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}

fn repeat_kv_impl(xs: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(xs.clone());
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![xs; repeats];
    Tensor::cat(&repeated, 2)?.reshape((b_sz, kv_heads * repeats, seq_len, head_dim))
}

pub(super) trait Qwen35BackendBufferApi: Sync {
    fn tensor_to_buffer(&self, xs: Tensor) -> Result<StateBuffer>;
    fn zeros_state(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer>;
    fn copy_state_into_scratch(
        &self,
        src: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer>;
    fn zeros_tensor(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor>;
    fn reshape_tensor_to_buffer(&self, xs: &Tensor, dims: &[usize]) -> Result<StateBuffer>;
    fn narrow_tensor_to_buffer(
        &self,
        xs: &Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<StateBuffer>;
    fn prepare_depthwise_conv_input(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<(Tensor, Option<StateBuffer>)>;
    fn update_depthwise_conv_state(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<Option<StateBuffer>>;
    fn concat_last_dim(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer>;
    fn pack_delta_state_scan(
        &self,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        state_decay_feature: &Tensor,
    ) -> Result<StateBuffer>;
    fn pack_delta_chunk_fused(
        &self,
        weighted_key: &Tensor,
        k_cumdecay: &Tensor,
        q_state: &Tensor,
        state_decay: &Tensor,
    ) -> Result<StateBuffer>;
    fn unpack_linear_decode_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        value_dim: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<(Tensor, StateBuffer)>;
    fn unpack_linear_prefill_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        conv_dim: usize,
        num_v_heads: usize,
        state_len: usize,
    ) -> Result<(Tensor, Tensor, StateBuffer)>;
    fn embedding_lookup(&self, embeddings: &Tensor, indexes: &Tensor) -> Result<StateBuffer>;
    fn immutable_embedding_lookup(
        &self,
        embedding: &ImmutableEmbedding,
        input_ids: &Tensor,
    ) -> Result<Tensor>;
    fn output_projection_tensor(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &Tensor,
    ) -> Result<Tensor>;
    fn output_projection(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
    ) -> Result<StateBuffer>;
    fn output_projection_into_scratch(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer>;
    fn linear_forward(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<StateBuffer>;
    fn linear_forward_into_scratch(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer>;
    #[allow(clippy::too_many_arguments)]
    fn prepare_full_attention_inputs(
        &self,
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
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer)>;
    #[allow(clippy::too_many_arguments)]
    fn prepare_full_attention_inputs_into_scratch(
        &self,
        q_and_gate: &StateBuffer,
        k_proj: &StateBuffer,
        v_proj: &StateBuffer,
        gate_scratch: &StateBuffer,
        query_scratch: &StateBuffer,
        key_scratch: &StateBuffer,
        value_scratch: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_norm_weight: &Tensor,
        q_norm_eps: f64,
        k_norm_weight: &Tensor,
        k_norm_eps: f64,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer, StateBuffer)>;
    #[allow(clippy::too_many_arguments)]
    fn prepare_linear_attention_inputs(
        &self,
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
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    fn rms_norm(
        &self,
        xs: &StateBuffer,
        weight: &Tensor,
        eps: f64,
        add_unit_offset: bool,
    ) -> Result<StateBuffer>;
    fn rms_norm_gated(
        &self,
        hidden_states: &StateBuffer,
        gate: &StateBuffer,
        weight: &Tensor,
        eps: f64,
    ) -> Result<StateBuffer>;
    fn swiglu_mul(&self, gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer>;
    fn l2norm(&self, xs: &StateBuffer, eps: f64) -> Result<StateBuffer>;
    fn cumsum_last_dim(&self, xs: &StateBuffer) -> Result<StateBuffer>;
    fn value_decay(
        &self,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
    ) -> Result<StateBuffer>;
    fn add(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer>;
    fn slice_last_token(&self, xs: &StateBuffer) -> Result<StateBuffer>;
    fn causal_mask(
        &self,
        device: &Device,
        dtype: DType,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor>;
    fn full_attention_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
    ) -> Result<StateBuffer>;
    fn full_attention_decode(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
    ) -> Result<StateBuffer>;
    fn wrap_kv_cache(
        &self,
        key_states: Tensor,
        value_states: Tensor,
    ) -> Result<(StateBuffer, StateBuffer)>;
    #[allow(clippy::too_many_arguments)]
    fn prepare_full_attention_output(
        &self,
        attn_output: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
    ) -> Result<StateBuffer>;
    fn prepare_full_attention_output_buffer(
        &self,
        attn_output: &StateBuffer,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
    ) -> Result<StateBuffer>;
    fn append_full_attention_kv(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor)>;
    fn append_full_attention_kv_buffers(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)>;
    fn prepare_full_attention_kernel_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)>;
    fn prepare_full_attention_kernel_inputs_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(Tensor, Tensor, Tensor)>;
    fn prepare_full_attention_kernel_input_buffers_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)>;
    fn materialize_full_attention_dense_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        num_kv_groups: usize,
    ) -> Result<(Tensor, Tensor, Tensor)>;
    fn dense_full_attention_fallback(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn dense_full_attention_fallback_buffer(
        &self,
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
    ) -> Result<StateBuffer>;
    fn linear_prefill_conv(
        &self,
        mixed_qkv: &Tensor,
        weights: &Tensor,
        seq_len: usize,
        kernel_size: usize,
    ) -> Result<Tensor>;
    fn linear_stateful_conv(
        &self,
        mixed_qkv: &Tensor,
        prev_state: &Tensor,
        weights: &Tensor,
        kernel_size: usize,
    ) -> Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn linear_decode_step(
        &self,
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
    ) -> Result<StateBuffer>;
    fn linear_stateful_conv_value_decay_with_state(
        &self,
        mixed_qkv: &StateBuffer,
        prev_state: &Tensor,
        weights: &Tensor,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
        kernel_size: usize,
    ) -> Result<StateBuffer>;
    fn delta_recurrent_prefill(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_chunk_single_prefill(
        &self,
        initial_state: &StateBuffer,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        beta: &Tensor,
        g: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_chunk_scan_raw(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn unpack_scan_fused_output_and_state(
        &self,
        fused: &StateBuffer,
        total_sequence_length: usize,
        output_sequence_length: usize,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
        k_head_dim: usize,
        output_dtype: DType,
    ) -> Result<(StateBuffer, StateBuffer)>;
    fn state_scan_chunk(&self, state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer>;
    fn state_scan_next_chunk(
        &self,
        state_scan: &StateBuffer,
        next_chunk_idx: usize,
    ) -> Result<StateBuffer>;
    fn unpack_chunk_fused(
        &self,
        fused: &StateBuffer,
        chunk_size: usize,
        k_head_dim: usize,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)>;
    fn delta_base_attn_scan(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_attn_solve_from_inputs(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_attn_solve_scan(&self, base_attn_scan: &StateBuffer) -> Result<StateBuffer>;
    fn delta_local_attn_scan(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_full_scan_pack(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_full_scan_packed(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        local_attn_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer>;
    #[allow(clippy::too_many_arguments)]
    fn delta_full_scan(
        &self,
        initial_state: &StateBuffer,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        q_state_scan: &Tensor,
        local_attn_scan: &StateBuffer,
        state_decay_scan: &Tensor,
        value: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_state_scan(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_chunk_fused(
        &self,
        prev_state: &StateBuffer,
        packed_chunk: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer>;
    fn delta_chunk_recurrent_read(
        &self,
        prev_state: &StateBuffer,
        k_cumdecay_chunk: &Tensor,
        q_state_chunk: &Tensor,
        value_chunk: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)>;
    fn mix_chunk_attention(
        &self,
        attn: &Tensor,
        attn_inter: &StateBuffer,
        value_chunk: &StateBuffer,
    ) -> Result<StateBuffer>;
    fn delta_state_update(
        &self,
        prev_state_scaled: &Tensor,
        weighted_key: &Tensor,
        value: &StateBuffer,
        use_kernel: bool,
    ) -> Result<StateBuffer>;
}

struct GenericBackendBufferApi;
struct HipBackendBufferApi;

impl Qwen35BackendBufferApi for GenericBackendBufferApi {
    fn tensor_to_buffer(&self, xs: Tensor) -> Result<StateBuffer> {
        if xs.device().is_cuda() {
            backends::cuda::tensor_to_buffer(xs)
        } else if xs.device().is_metal() {
            backends::metal::tensor_to_buffer(xs)
        } else {
            backends::cpu::tensor_to_buffer(xs)
        }
    }
    fn zeros_state(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
        if device.is_cuda() {
            backends::cuda::zeros_state(device, dtype, dims)
        } else if device.is_metal() {
            backends::metal::zeros_state(device, dtype, dims)
        } else {
            backends::cpu::zeros_state(device, dtype, dims)
        }
    }
    fn copy_state_into_scratch(
        &self,
        src: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        if src.dtype() != scratch.dtype() {
            candle_core::bail!(
                "scratch dtype mismatch: src={:?} scratch={:?}",
                src.dtype(),
                scratch.dtype(),
            );
        }
        if src.tensor().dims() != scratch.tensor().dims() {
            candle_core::bail!(
                "scratch shape mismatch: src={:?} scratch={:?}",
                src.tensor().dims(),
                scratch.tensor().dims(),
            );
        }
        Ok(src.clone())
    }
    fn zeros_tensor(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
        if device.is_cuda() {
            backends::cuda::zeros_tensor(device, dtype, dims)
        } else if device.is_metal() {
            backends::metal::zeros_tensor(device, dtype, dims)
        } else {
            backends::cpu::zeros_tensor(device, dtype, dims)
        }
    }
    fn reshape_tensor_to_buffer(&self, xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
        if xs.device().is_cuda() {
            backends::cuda::reshape_tensor_to_buffer(xs, dims)
        } else if xs.device().is_metal() {
            backends::metal::reshape_tensor_to_buffer(xs, dims)
        } else {
            backends::cpu::reshape_tensor_to_buffer(xs, dims)
        }
    }
    fn narrow_tensor_to_buffer(
        &self,
        xs: &Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<StateBuffer> {
        if xs.device().is_cuda() {
            backends::cuda::narrow_tensor_to_buffer(xs, dim, start, len)
        } else if xs.device().is_metal() {
            backends::metal::narrow_tensor_to_buffer(xs, dim, start, len)
        } else {
            backends::cpu::narrow_tensor_to_buffer(xs, dim, start, len)
        }
    }
    fn prepare_depthwise_conv_input(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<(Tensor, Option<StateBuffer>)> {
        if mixed_qkv.device().is_cuda() {
            backends::cuda::prepare_depthwise_conv_input(prev_state, mixed_qkv, kernel_size)
        } else if mixed_qkv.device().is_metal() {
            backends::metal::prepare_depthwise_conv_input(prev_state, mixed_qkv, kernel_size)
        } else {
            backends::cpu::prepare_depthwise_conv_input(prev_state, mixed_qkv, kernel_size)
        }
    }
    fn update_depthwise_conv_state(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<Option<StateBuffer>> {
        if mixed_qkv.device().is_cuda() {
            backends::cuda::update_depthwise_conv_state(prev_state, mixed_qkv, kernel_size)
        } else if mixed_qkv.device().is_metal() {
            backends::metal::update_depthwise_conv_state(prev_state, mixed_qkv, kernel_size)
        } else {
            backends::cpu::update_depthwise_conv_state(prev_state, mixed_qkv, kernel_size)
        }
    }
    fn concat_last_dim(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
        if lhs.device().is_cuda() {
            backends::cuda::concat_last_dim(lhs, rhs)
        } else if lhs.device().is_metal() {
            backends::metal::concat_last_dim(lhs, rhs)
        } else {
            backends::cpu::concat_last_dim(lhs, rhs)
        }
    }
    fn pack_delta_state_scan(
        &self,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        state_decay_feature: &Tensor,
    ) -> Result<StateBuffer> {
        if weighted_key_scan.device().is_cuda() {
            backends::cuda::pack_delta_state_scan(
                weighted_key_scan,
                k_cumdecay_scan,
                state_decay_feature,
            )
        } else if weighted_key_scan.device().is_metal() {
            backends::metal::pack_delta_state_scan(
                weighted_key_scan,
                k_cumdecay_scan,
                state_decay_feature,
            )
        } else {
            backends::cpu::pack_delta_state_scan(
                weighted_key_scan,
                k_cumdecay_scan,
                state_decay_feature,
            )
        }
    }
    fn pack_delta_chunk_fused(
        &self,
        weighted_key: &Tensor,
        k_cumdecay: &Tensor,
        q_state: &Tensor,
        state_decay: &Tensor,
    ) -> Result<StateBuffer> {
        if weighted_key.device().is_cuda() {
            backends::cuda::pack_delta_chunk_fused(weighted_key, k_cumdecay, q_state, state_decay)
        } else if weighted_key.device().is_metal() {
            backends::metal::pack_delta_chunk_fused(weighted_key, k_cumdecay, q_state, state_decay)
        } else {
            backends::cpu::pack_delta_chunk_fused(weighted_key, k_cumdecay, q_state, state_decay)
        }
    }
    fn unpack_linear_decode_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        value_dim: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<(Tensor, StateBuffer)> {
        if fused.device().is_cuda() {
            backends::cuda::unpack_linear_decode_output(
                fused,
                batch_size,
                seq_len,
                value_dim,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
        } else if fused.device().is_metal() {
            backends::metal::unpack_linear_decode_output(
                fused,
                batch_size,
                seq_len,
                value_dim,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
        } else {
            backends::cpu::unpack_linear_decode_output(
                fused,
                batch_size,
                seq_len,
                value_dim,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
        }
    }
    fn unpack_linear_prefill_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        conv_dim: usize,
        num_v_heads: usize,
        state_len: usize,
    ) -> Result<(Tensor, Tensor, StateBuffer)> {
        if fused.device().is_cuda() {
            backends::cuda::unpack_linear_prefill_output(
                fused,
                batch_size,
                seq_len,
                conv_dim,
                num_v_heads,
                state_len,
            )
        } else if fused.device().is_metal() {
            backends::metal::unpack_linear_prefill_output(
                fused,
                batch_size,
                seq_len,
                conv_dim,
                num_v_heads,
                state_len,
            )
        } else {
            backends::cpu::unpack_linear_prefill_output(
                fused,
                batch_size,
                seq_len,
                conv_dim,
                num_v_heads,
                state_len,
            )
        }
    }
    fn embedding_lookup(&self, embeddings: &Tensor, indexes: &Tensor) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::embedding_lookup(embeddings, indexes)?)
    }
    fn immutable_embedding_lookup(
        &self,
        embedding: &ImmutableEmbedding,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        backend_ops::immutable_embedding_lookup(embedding, input_ids)
    }
    fn output_projection_tensor(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        backend_ops::output_projection(embedding, hidden_states)
    }
    fn output_projection(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
    ) -> Result<StateBuffer> {
        backend_ops::output_projection_buffer(embedding, hidden_states)
    }
    fn output_projection_into_scratch(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        let output = self.output_projection(embedding, hidden_states)?;
        self.copy_state_into_scratch(&output, scratch)
    }
    fn linear_forward(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::linear_forward(x.tensor(), weight, bias)?)
    }
    fn linear_forward_into_scratch(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        let output = self.linear_forward(x, weight, bias)?;
        self.copy_state_into_scratch(&output, scratch)
    }
    fn prepare_full_attention_inputs(
        &self,
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
        let q_and_gate = q_and_gate
            .tensor()
            .reshape((b_sz, q_len, num_heads, head_dim * 2))?;
        let query_states = self
            .rms_norm(
                &self.narrow_tensor_to_buffer(&q_and_gate, q_and_gate.rank() - 1, 0, head_dim)?,
                q_norm_weight,
                q_norm_eps,
                true,
            )?
            .tensor()
            .transpose(1, 2)?;
        let gate = q_and_gate
            .narrow(D::Minus1, head_dim, head_dim)?
            .reshape((b_sz, q_len, num_heads * head_dim))?;
        let key_states = self
            .rms_norm(
                &self.reshape_tensor_to_buffer(
                    k_proj.tensor(),
                    &[b_sz, q_len, num_kv_heads, head_dim],
                )?,
                k_norm_weight,
                k_norm_eps,
                true,
            )?
            .tensor()
            .transpose(1, 2)?;
        let value_states = v_proj
            .tensor()
            .reshape((b_sz, q_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        Ok((
            StateBuffer::from_tensor(query_states)?,
            StateBuffer::from_tensor(gate)?,
            StateBuffer::from_tensor(key_states)?,
            StateBuffer::from_tensor(value_states)?,
        ))
    }
    fn prepare_full_attention_inputs_into_scratch(
        &self,
        q_and_gate: &StateBuffer,
        k_proj: &StateBuffer,
        v_proj: &StateBuffer,
        gate_scratch: &StateBuffer,
        query_scratch: &StateBuffer,
        key_scratch: &StateBuffer,
        value_scratch: &StateBuffer,
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
        let (query_states, gate, key_states, value_states) = self.prepare_full_attention_inputs(
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
            self.copy_state_into_scratch(&query_states, query_scratch)?,
            self.copy_state_into_scratch(&gate, gate_scratch)?,
            self.copy_state_into_scratch(&key_states, key_scratch)?,
            self.copy_state_into_scratch(&value_states, value_scratch)?,
        ))
    }
    fn prepare_linear_attention_inputs(
        &self,
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
        let query = mixed_qkv.narrow(D::Minus1, 0, key_dim)?.reshape((
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        ))?;
        let key = mixed_qkv
            .narrow(D::Minus1, key_dim, key_dim)?
            .reshape((batch_size, seq_len, num_k_heads, head_k_dim))?;
        let value = mixed_qkv
            .narrow(D::Minus1, key_dim * 2, value_dim)?
            .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?;

        let query = if query.dtype() == compute_dtype {
            query
        } else {
            query.to_dtype(compute_dtype)?
        };
        let key = if key.dtype() == compute_dtype {
            key
        } else {
            key.to_dtype(compute_dtype)?
        };
        let query = self.l2norm(&StateBuffer::from_tensor(query)?, 1e-6)?.clone_tensor();
        let key = self.l2norm(&StateBuffer::from_tensor(key)?, 1e-6)?.clone_tensor();
        let head_repeat = num_v_heads / num_k_heads;
        let (query, key) = if repeat_kv_heads && head_repeat > 1 {
            (
                repeat_heads_impl(&query, head_repeat)?,
                repeat_heads_impl(&key, head_repeat)?,
            )
        } else {
            (query, key)
        };
        let value = if value.dtype() == compute_dtype {
            value
        } else {
            value.to_dtype(compute_dtype)?
        };
        let beta = ops::sigmoid(beta_raw.tensor())?;
        let beta = if beta.dtype() == compute_dtype {
            beta
        } else {
            beta.to_dtype(compute_dtype)?
        };
        let g = if g.dtype() == compute_dtype {
            g.clone()
        } else {
            g.to_dtype(compute_dtype)?
        };
        Ok((query, key, value, beta, g))
    }
    fn rms_norm(
        &self,
        xs: &StateBuffer,
        weight: &Tensor,
        eps: f64,
        add_unit_offset: bool,
    ) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::rms_norm(
            xs.tensor(),
            weight,
            eps,
            add_unit_offset,
        )?)
    }
    fn rms_norm_gated(
        &self,
        hidden_states: &StateBuffer,
        gate: &StateBuffer,
        weight: &Tensor,
        eps: f64,
    ) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::rms_norm_gated(
            hidden_states.tensor(),
            gate.tensor(),
            weight,
            eps,
        )?)
    }
    fn swiglu_mul(&self, gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::swiglu_mul(gate.tensor(), up.tensor())?)
    }
    fn l2norm(&self, xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::l2norm(xs.tensor(), eps)?)
    }
    fn cumsum_last_dim(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::cumsum_last_dim(xs.tensor())?)
    }
    fn value_decay(
        &self,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
    ) -> Result<StateBuffer> {
        StateBuffer::from_tensor(backend_ops::value_decay(a.tensor(), dt_bias, a_log_exp)?)
    }
    fn add(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
        StateBuffer::from_tensor(lhs.tensor().broadcast_add(rhs.tensor())?)
    }
    fn slice_last_token(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        let (_, seq_len, _) = xs.dims3()?;
        xs.narrow(1, seq_len - 1, 1)
    }
    fn causal_mask(
        &self,
        device: &Device,
        dtype: DType,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        backend_ops::causal_mask(device, dtype, b_size, tgt_len, seqlen_offset)
    }
    fn full_attention_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        backend_ops::full_attention_prefill_buffer(
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )
    }
    fn full_attention_decode(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
        ) -> Result<StateBuffer> {
        backend_ops::full_attention_decode_buffer(
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )
    }
    fn wrap_kv_cache(
        &self,
        key_states: Tensor,
        value_states: Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        Ok((StateBuffer::from_tensor(key_states)?, StateBuffer::from_tensor(value_states)?))
    }
    fn prepare_full_attention_output(
        &self,
        attn_output: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
    ) -> Result<StateBuffer> {
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, attention_size))?
            .to_dtype(hidden_dtype)?;
        let gated = attn_output.broadcast_mul(&ops::sigmoid(gate.tensor())?)?;
        StateBuffer::from_tensor(gated)
    }
    fn prepare_full_attention_output_buffer(
        &self,
        attn_output: &StateBuffer,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
    ) -> Result<StateBuffer> {
        self.prepare_full_attention_output(
            attn_output.tensor(),
            gate,
            b_sz,
            q_len,
            attention_size,
            hidden_dtype,
        )
    }
    fn append_full_attention_kv(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        match (prev_k, prev_v) {
            (Some(prev_k), Some(prev_v)) => {
                let prev_k = prev_k.clone_tensor_as(key_states.dtype())?;
                let prev_v = prev_v.clone_tensor_as(value_states.dtype())?;
                Ok((
                    Tensor::cat(&[&prev_k, key_states], 2)?,
                    Tensor::cat(&[&prev_v, value_states], 2)?,
                ))
            }
            _ => Ok((key_states.clone(), value_states.clone())),
        }
    }
    fn append_full_attention_kv_buffers(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        let (key_states, value_states) =
            self.append_full_attention_kv(prev_k, prev_v, key_states, value_states)?;
        Ok((StateBuffer::from_tensor(key_states)?, StateBuffer::from_tensor(value_states)?))
    }
    fn prepare_full_attention_kernel_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            query_states.contiguous()?,
            key_states.contiguous()?,
            value_states.contiguous()?,
        ))
    }
    fn prepare_full_attention_kernel_inputs_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.prepare_full_attention_kernel_inputs(
            query_states.tensor(),
            key_states.tensor(),
            value_states.tensor(),
        )
    }
    fn prepare_full_attention_kernel_input_buffers_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
        let (query_states, key_states, value_states) =
            self.prepare_full_attention_kernel_inputs_with_buffer_kv(
                query_states,
                key_states,
                value_states,
            )?;
        Ok((
            StateBuffer::from_tensor(query_states)?,
            StateBuffer::from_tensor(key_states)?,
            StateBuffer::from_tensor(value_states)?,
        ))
    }
    fn materialize_full_attention_dense_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        num_kv_groups: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let key_states = repeat_kv_impl(key_states, num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv_impl(value_states, num_kv_groups)?.contiguous()?;
        Ok((
            query_states.to_dtype(DType::F32)?,
            key_states.to_dtype(DType::F32)?,
            value_states.to_dtype(DType::F32)?,
        ))
    }
    fn dense_full_attention_fallback(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
        let mut attn_weights = (query_states_f.matmul(&key_states_t)? * scale)?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?;
        }
        let attn_weights = ops::softmax_last_dim(&attn_weights)?;
        attn_weights.matmul(value_states_f)
    }
    fn dense_full_attention_fallback_buffer(
        &self,
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
        let attn_output = self.dense_full_attention_fallback(
            query_states_f,
            key_states_f,
            value_states_f,
            attention_mask,
            scale,
        )?;
        self.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            attention_size,
            hidden_dtype,
        )
    }
    fn linear_prefill_conv(
        &self,
        mixed_qkv: &Tensor,
        weights: &Tensor,
        seq_len: usize,
        kernel_size: usize,
    ) -> Result<Tensor> {
        backend_ops::linear_prefill_conv(mixed_qkv, weights, seq_len, kernel_size)
    }
    fn linear_stateful_conv(
        &self,
        mixed_qkv: &Tensor,
        prev_state: &Tensor,
        weights: &Tensor,
        kernel_size: usize,
    ) -> Result<Tensor> {
        backend_ops::linear_stateful_conv(mixed_qkv, prev_state, weights, kernel_size)
    }
    fn linear_decode_step(
        &self,
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
        backend_ops::linear_decode_step_buffer(
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
        )
    }
    fn linear_stateful_conv_value_decay_with_state(
        &self,
        mixed_qkv: &StateBuffer,
        prev_state: &Tensor,
        weights: &Tensor,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
        kernel_size: usize,
        ) -> Result<StateBuffer> {
        backend_ops::linear_stateful_conv_value_decay_with_state_buffer(
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            kernel_size,
        )
    }
    fn delta_recurrent_prefill(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_recurrent_prefill_buffer(
            initial_state,
            query_scan,
            key_scan,
            value_scan,
            beta_scan,
            g_scan,
        )
    }
    fn delta_chunk_single_prefill(
        &self,
        initial_state: &StateBuffer,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        beta: &Tensor,
        g: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_chunk_single_prefill_buffer(initial_state, query, key, value, beta, g)
    }
    fn delta_chunk_scan_raw(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_chunk_scan_raw_buffer(
            initial_state,
            query_scan,
            key_scan,
            value_scan,
            beta_scan,
            g_scan,
        )
    }
    fn unpack_scan_fused_output_and_state(
        &self,
        fused: &StateBuffer,
        total_sequence_length: usize,
        output_sequence_length: usize,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
        k_head_dim: usize,
        output_dtype: DType,
    ) -> Result<(StateBuffer, StateBuffer)> {
        let output_scan = fused.tensor().narrow(1, 0, total_sequence_length)?.reshape((
            batch_size,
            num_heads,
            total_sequence_length,
            v_head_dim,
        ))?;
        let output = output_scan
            .narrow(2, 0, output_sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(output_dtype)?;
        let recurrent_state = fused
            .tensor()
            .narrow(1, total_sequence_length, k_head_dim)?
            .reshape((batch_size * num_heads, k_head_dim, v_head_dim))?
            .contiguous()?;
        Ok((
            StateBuffer::from_tensor(output)?,
            StateBuffer::from_tensor(recurrent_state)?,
        ))
    }
    fn state_scan_chunk(&self, state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
        use candle_core::IndexOp;
        StateBuffer::from_tensor(state_scan.tensor().i((.., chunk_idx, .., ..))?)
    }
    fn state_scan_next_chunk(
        &self,
        state_scan: &StateBuffer,
        next_chunk_idx: usize,
    ) -> Result<StateBuffer> {
        use candle_core::IndexOp;
        StateBuffer::from_tensor(state_scan.tensor().i((.., next_chunk_idx, .., ..))?.contiguous()?)
    }
    fn unpack_chunk_fused(
        &self,
        fused: &StateBuffer,
        chunk_size: usize,
        k_head_dim: usize,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
        Ok((
            StateBuffer::from_tensor(fused.tensor().narrow(1, 0, chunk_size)?)?,
            StateBuffer::from_tensor(fused.tensor().narrow(1, chunk_size, chunk_size)?)?,
            StateBuffer::from_tensor(fused.tensor().narrow(1, 2 * chunk_size, k_head_dim)?)?,
        ))
    }
    fn delta_base_attn_scan(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_base_attn_scan_buffer(k_beta_scan, key_scan, exp_g_scan)
    }
    fn delta_attn_solve_from_inputs(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_attn_solve_from_inputs_buffer(k_beta_scan, key_scan, exp_g_scan)
    }
    fn delta_attn_solve_scan(&self, base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
        backend_ops::delta_attn_solve_scan_buffer(base_attn_scan)
    }
    fn delta_local_attn_scan(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_local_attn_scan_buffer(query_scan, key_scan, exp_g_scan)
    }
    fn delta_full_scan_pack(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_full_scan_pack_buffer(query_scan, key_scan, exp_g_scan, k_cumdecay_scan)
    }
    fn delta_full_scan_packed(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        local_attn_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_full_scan_packed_buffer(initial_state, packed_scan, local_attn_scan, value)
    }
    fn delta_full_scan(
        &self,
        initial_state: &StateBuffer,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        q_state_scan: &Tensor,
        local_attn_scan: &StateBuffer,
        state_decay_scan: &Tensor,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_full_scan_buffer(
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
        )
    }
    fn delta_state_scan(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_state_scan_buffer(initial_state, packed_scan, value)
    }
    fn delta_chunk_fused(
        &self,
        prev_state: &StateBuffer,
        packed_chunk: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backend_ops::delta_chunk_fused_buffer(prev_state, packed_chunk, value)
    }
    fn delta_chunk_recurrent_read(
        &self,
        prev_state: &StateBuffer,
        k_cumdecay_chunk: &Tensor,
        q_state_chunk: &Tensor,
        value_chunk: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        backend_ops::delta_chunk_recurrent_read(prev_state, k_cumdecay_chunk, q_state_chunk, value_chunk)
    }
    fn mix_chunk_attention(
        &self,
        attn: &Tensor,
        attn_inter: &StateBuffer,
        value_chunk: &StateBuffer,
    ) -> Result<StateBuffer> {
        backend_ops::mix_chunk_attention(attn, attn_inter, value_chunk)
    }
    fn delta_state_update(
        &self,
        prev_state_scaled: &Tensor,
        weighted_key: &Tensor,
        value: &StateBuffer,
        use_kernel: bool,
    ) -> Result<StateBuffer> {
        backend_ops::delta_state_update_buffer(prev_state_scaled, weighted_key, value, use_kernel)
    }
}

impl Qwen35BackendBufferApi for HipBackendBufferApi {
    fn tensor_to_buffer(&self, xs: Tensor) -> Result<StateBuffer> {
        backends::hip::tensor_to_buffer(xs)
    }
    fn zeros_state(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<StateBuffer> {
        backends::hip::zeros_state(device, dtype, dims)
    }
    fn copy_state_into_scratch(
        &self,
        src: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        backends::hip::copy_state_into_scratch(src, scratch)
    }
    fn zeros_tensor(&self, device: &Device, dtype: DType, dims: &[usize]) -> Result<Tensor> {
        backends::hip::zeros_tensor(device, dtype, dims)
    }
    fn reshape_tensor_to_buffer(&self, xs: &Tensor, dims: &[usize]) -> Result<StateBuffer> {
        backends::hip::reshape_tensor_to_buffer(xs, dims)
    }
    fn narrow_tensor_to_buffer(
        &self,
        xs: &Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<StateBuffer> {
        backends::hip::narrow_tensor_to_buffer(xs, dim, start, len)
    }
    fn prepare_depthwise_conv_input(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<(Tensor, Option<StateBuffer>)> {
        backends::hip::prepare_depthwise_conv_input(prev_state, mixed_qkv, kernel_size)
    }
    fn update_depthwise_conv_state(
        &self,
        prev_state: Option<&StateBuffer>,
        mixed_qkv: &Tensor,
        kernel_size: usize,
    ) -> Result<Option<StateBuffer>> {
        backends::hip::update_depthwise_conv_state(prev_state, mixed_qkv, kernel_size)
    }
    fn concat_last_dim(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::concat_last_dim(lhs, rhs)
    }
    fn pack_delta_state_scan(
        &self,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        state_decay_feature: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::pack_delta_state_scan(weighted_key_scan, k_cumdecay_scan, state_decay_feature)
    }
    fn pack_delta_chunk_fused(
        &self,
        weighted_key: &Tensor,
        k_cumdecay: &Tensor,
        q_state: &Tensor,
        state_decay: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::pack_delta_chunk_fused(weighted_key, k_cumdecay, q_state, state_decay)
    }
    fn unpack_linear_decode_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        value_dim: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<(Tensor, StateBuffer)> {
        backends::hip::unpack_linear_decode_output(
            fused,
            batch_size,
            seq_len,
            value_dim,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )
    }
    fn unpack_linear_prefill_output(
        &self,
        fused: &StateBuffer,
        batch_size: usize,
        seq_len: usize,
        conv_dim: usize,
        num_v_heads: usize,
        state_len: usize,
    ) -> Result<(Tensor, Tensor, StateBuffer)> {
        backends::hip::unpack_linear_prefill_output(
            fused,
            batch_size,
            seq_len,
            conv_dim,
            num_v_heads,
            state_len,
        )
    }
    fn embedding_lookup(&self, embeddings: &Tensor, indexes: &Tensor) -> Result<StateBuffer> {
        backends::hip::embedding_lookup(embeddings, indexes)
    }
    fn immutable_embedding_lookup(
        &self,
        embedding: &ImmutableEmbedding,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        backends::hip::immutable_embedding_lookup(embedding, input_ids)
    }
    fn output_projection_tensor(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        backends::hip::output_projection_tensor(embedding, hidden_states)
    }
    fn output_projection(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
    ) -> Result<StateBuffer> {
        backends::hip::output_projection(embedding, hidden_states)
    }
    fn output_projection_into_scratch(
        &self,
        embedding: &ImmutableEmbedding,
        hidden_states: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        backends::hip::output_projection_into_scratch(embedding, hidden_states, scratch)
    }
    fn linear_forward(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<StateBuffer> {
        backends::hip::linear_forward(x, weight, bias)
    }
    fn linear_forward_into_scratch(
        &self,
        x: &StateBuffer,
        weight: &Tensor,
        bias: Option<&Tensor>,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        backends::hip::linear_forward_into_scratch(x, weight, bias, scratch)
    }
    fn prepare_full_attention_inputs(
        &self,
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
        backends::hip::prepare_full_attention_inputs(
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
        )
    }
    fn prepare_full_attention_inputs_into_scratch(
        &self,
        q_and_gate: &StateBuffer,
        k_proj: &StateBuffer,
        v_proj: &StateBuffer,
        gate_scratch: &StateBuffer,
        query_scratch: &StateBuffer,
        key_scratch: &StateBuffer,
        value_scratch: &StateBuffer,
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
        backends::hip::prepare_full_attention_inputs_into_scratch(
            q_and_gate,
            k_proj,
            v_proj,
            gate_scratch,
            query_scratch,
            key_scratch,
            value_scratch,
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
    fn prepare_linear_attention_inputs(
        &self,
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
        backends::hip::prepare_linear_attention_inputs(
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
        )
    }
    fn rms_norm(
        &self,
        xs: &StateBuffer,
        weight: &Tensor,
        eps: f64,
        add_unit_offset: bool,
    ) -> Result<StateBuffer> {
        backends::hip::rms_norm(xs, weight, eps, add_unit_offset)
    }
    fn rms_norm_gated(
        &self,
        hidden_states: &StateBuffer,
        gate: &StateBuffer,
        weight: &Tensor,
        eps: f64,
    ) -> Result<StateBuffer> {
        backends::hip::rms_norm_gated(hidden_states, gate, weight, eps)
    }
    fn swiglu_mul(&self, gate: &StateBuffer, up: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::swiglu_mul(gate, up)
    }
    fn l2norm(&self, xs: &StateBuffer, eps: f64) -> Result<StateBuffer> {
        backends::hip::l2norm(xs, eps)
    }
    fn cumsum_last_dim(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::cumsum_last_dim(xs)
    }
    fn value_decay(
        &self,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::value_decay(a, dt_bias, a_log_exp)
    }
    fn add(&self, lhs: &StateBuffer, rhs: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::add(lhs, rhs)
    }
    fn slice_last_token(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::slice_last_token(xs)
    }
    fn causal_mask(
        &self,
        device: &Device,
        dtype: DType,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        backends::hip::causal_mask(device, dtype, b_size, tgt_len, seqlen_offset)
    }
    fn full_attention_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        backends::hip::full_attention_prefill(
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )
    }
    fn full_attention_decode(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
        ) -> Result<StateBuffer> {
        backends::hip::full_attention_decode(
            query,
            key,
            value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )
    }
    fn wrap_kv_cache(
        &self,
        key_states: Tensor,
        value_states: Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        backends::hip::wrap_kv_cache(key_states, value_states)
    }
    fn prepare_full_attention_output(
        &self,
        attn_output: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
        ) -> Result<StateBuffer> {
        backends::hip::prepare_full_attention_output(
            attn_output,
            gate,
            b_sz,
            q_len,
            attention_size,
            hidden_dtype,
        )
    }
    fn prepare_full_attention_output_buffer(
        &self,
        attn_output: &StateBuffer,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        attention_size: usize,
        hidden_dtype: DType,
    ) -> Result<StateBuffer> {
        backends::hip::prepare_full_attention_output_buffer(
            attn_output,
            gate,
            b_sz,
            q_len,
            attention_size,
            hidden_dtype,
        )
    }
    fn append_full_attention_kv(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        backends::hip::append_full_attention_kv(prev_k, prev_v, key_states, value_states)
    }
    fn append_full_attention_kv_buffers(
        &self,
        prev_k: Option<&StateBuffer>,
        prev_v: Option<&StateBuffer>,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        backends::hip::append_full_attention_kv_buffers(prev_k, prev_v, key_states, value_states)
    }
    fn prepare_full_attention_kernel_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        backends::hip::prepare_full_attention_kernel_inputs(query_states, key_states, value_states)
    }
    fn prepare_full_attention_kernel_inputs_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        backends::hip::prepare_full_attention_kernel_inputs_with_buffer_kv(
            query_states,
            key_states,
            value_states,
        )
    }
    fn prepare_full_attention_kernel_input_buffers_with_buffer_kv(
        &self,
        query_states: &StateBuffer,
        key_states: &StateBuffer,
        value_states: &StateBuffer,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
        backends::hip::prepare_full_attention_kernel_input_buffers_with_buffer_kv(
            query_states,
            key_states,
            value_states,
        )
    }
    fn materialize_full_attention_dense_inputs(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        num_kv_groups: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        backends::hip::materialize_full_attention_dense_inputs(
            query_states,
            key_states,
            value_states,
            num_kv_groups,
        )
    }
    fn dense_full_attention_fallback(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        backends::hip::dense_full_attention_fallback(
            query_states_f,
            key_states_f,
            value_states_f,
            attention_mask,
            scale,
        )
    }
    fn dense_full_attention_fallback_buffer(
        &self,
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
        backends::hip::dense_full_attention_fallback_buffer(
            query_states_f,
            key_states_f,
            value_states_f,
            attention_mask,
            scale,
            gate,
            b_sz,
            q_len,
            attention_size,
            hidden_dtype,
        )
    }
    fn linear_prefill_conv(
        &self,
        mixed_qkv: &Tensor,
        weights: &Tensor,
        seq_len: usize,
        kernel_size: usize,
    ) -> Result<Tensor> {
        backends::hip::linear_prefill_conv(mixed_qkv, weights, seq_len, kernel_size)
    }
    fn linear_stateful_conv(
        &self,
        mixed_qkv: &Tensor,
        prev_state: &Tensor,
        weights: &Tensor,
        kernel_size: usize,
    ) -> Result<Tensor> {
        backends::hip::linear_stateful_conv(mixed_qkv, prev_state, weights, kernel_size)
    }
    fn linear_decode_step(
        &self,
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
        backends::hip::linear_decode_step(
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
        )
    }
    fn linear_stateful_conv_value_decay_with_state(
        &self,
        mixed_qkv: &StateBuffer,
        prev_state: &Tensor,
        weights: &Tensor,
        a: &StateBuffer,
        dt_bias: &Tensor,
        a_log_exp: &Tensor,
        kernel_size: usize,
    ) -> Result<StateBuffer> {
        backends::hip::linear_stateful_conv_value_decay_with_state(
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            kernel_size,
        )
    }
    fn delta_recurrent_prefill(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_recurrent_prefill(
            initial_state,
            query_scan,
            key_scan,
            value_scan,
            beta_scan,
            g_scan,
        )
    }
    fn delta_chunk_single_prefill(
        &self,
        initial_state: &StateBuffer,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        beta: &Tensor,
        g: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_chunk_single_prefill(initial_state, query, key, value, beta, g)
    }
    fn delta_chunk_scan_raw(
        &self,
        initial_state: &StateBuffer,
        query_scan: &Tensor,
        key_scan: &Tensor,
        value_scan: &Tensor,
        beta_scan: &Tensor,
        g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_chunk_scan_raw(
            initial_state,
            query_scan,
            key_scan,
            value_scan,
            beta_scan,
            g_scan,
        )
    }
    fn unpack_scan_fused_output_and_state(
        &self,
        fused: &StateBuffer,
        total_sequence_length: usize,
        output_sequence_length: usize,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
        k_head_dim: usize,
        output_dtype: DType,
    ) -> Result<(StateBuffer, StateBuffer)> {
        backends::hip::unpack_scan_fused_output_and_state(
            fused,
            total_sequence_length,
            output_sequence_length,
            batch_size,
            num_heads,
            v_head_dim,
            k_head_dim,
            output_dtype,
        )
    }
    fn state_scan_chunk(&self, state_scan: &StateBuffer, chunk_idx: usize) -> Result<StateBuffer> {
        backends::hip::state_scan_chunk(state_scan, chunk_idx)
    }
    fn state_scan_next_chunk(
        &self,
        state_scan: &StateBuffer,
        next_chunk_idx: usize,
    ) -> Result<StateBuffer> {
        backends::hip::state_scan_next_chunk(state_scan, next_chunk_idx)
    }
    fn unpack_chunk_fused(
        &self,
        fused: &StateBuffer,
        chunk_size: usize,
        k_head_dim: usize,
    ) -> Result<(StateBuffer, StateBuffer, StateBuffer)> {
        backends::hip::unpack_chunk_fused(fused, chunk_size, k_head_dim)
    }
    fn delta_base_attn_scan(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_base_attn_scan(k_beta_scan, key_scan, exp_g_scan)
    }
    fn delta_attn_solve_from_inputs(
        &self,
        k_beta_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_attn_solve_from_inputs(k_beta_scan, key_scan, exp_g_scan)
    }
    fn delta_attn_solve_scan(&self, base_attn_scan: &StateBuffer) -> Result<StateBuffer> {
        backends::hip::delta_attn_solve_scan(base_attn_scan)
    }
    fn delta_local_attn_scan(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_local_attn_scan(query_scan, key_scan, exp_g_scan)
    }
    fn delta_full_scan_pack(
        &self,
        query_scan: &Tensor,
        key_scan: &Tensor,
        exp_g_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_full_scan_pack(query_scan, key_scan, exp_g_scan, k_cumdecay_scan)
    }
    fn delta_full_scan_packed(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        local_attn_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_full_scan_packed(initial_state, packed_scan, local_attn_scan, value)
    }
    fn delta_full_scan(
        &self,
        initial_state: &StateBuffer,
        weighted_key_scan: &Tensor,
        k_cumdecay_scan: &Tensor,
        q_state_scan: &Tensor,
        local_attn_scan: &StateBuffer,
        state_decay_scan: &Tensor,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_full_scan(
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
        )
    }
    fn delta_state_scan(
        &self,
        initial_state: &StateBuffer,
        packed_scan: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_state_scan(initial_state, packed_scan, value)
    }
    fn delta_chunk_fused(
        &self,
        prev_state: &StateBuffer,
        packed_chunk: &StateBuffer,
        value: &Tensor,
    ) -> Result<StateBuffer> {
        backends::hip::delta_chunk_fused(prev_state, packed_chunk, value)
    }
    fn delta_chunk_recurrent_read(
        &self,
        prev_state: &StateBuffer,
        k_cumdecay_chunk: &Tensor,
        q_state_chunk: &Tensor,
        value_chunk: &Tensor,
    ) -> Result<(StateBuffer, StateBuffer)> {
        backends::hip::delta_chunk_recurrent_read(prev_state, k_cumdecay_chunk, q_state_chunk, value_chunk)
    }
    fn mix_chunk_attention(
        &self,
        attn: &Tensor,
        attn_inter: &StateBuffer,
        value_chunk: &StateBuffer,
    ) -> Result<StateBuffer> {
        backends::hip::mix_chunk_attention(attn, attn_inter, value_chunk)
    }
    fn delta_state_update(
        &self,
        prev_state_scaled: &Tensor,
        weighted_key: &Tensor,
        value: &StateBuffer,
        use_kernel: bool,
    ) -> Result<StateBuffer> {
        backends::hip::delta_state_update(prev_state_scaled, weighted_key, value, use_kernel)
    }
}

static GENERIC_BACKEND_BUFFER_API: GenericBackendBufferApi = GenericBackendBufferApi;
static HIP_BACKEND_BUFFER_API: HipBackendBufferApi = HipBackendBufferApi;

pub(super) fn for_device(device: &Device) -> &'static dyn Qwen35BackendBufferApi {
    if device.is_hip() {
        &HIP_BACKEND_BUFFER_API
    } else {
        &GENERIC_BACKEND_BUFFER_API
    }
}
