use super::activation::Activation;
use super::frontend::max_abs_delta;
use crate::{BufferMutability, BufferViewDesc, ScalarType};
use candle_core as candle;
use candle_core::{DType, Device, Result, Tensor};

fn default_attention_bias() -> bool {
    false
}

fn default_attention_dropout() -> f64 {
    0.0
}

fn default_head_dim() -> usize {
    256
}

fn default_linear_conv_kernel_dim() -> usize {
    4
}

fn default_linear_key_head_dim() -> usize {
    128
}

fn default_linear_value_head_dim() -> usize {
    128
}

fn default_linear_num_key_heads() -> usize {
    16
}

fn default_linear_num_value_heads() -> usize {
    32
}

fn default_partial_rotary_factor() -> f64 {
    0.25
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

impl TextConfig {
    pub fn normalized(mut self) -> Self {
        if self.layer_types.is_empty() {
            self.layer_types = (0..self.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        self
    }

    pub(crate) fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|params| params.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    pub(crate) fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|params| params.partial_rotary_factor)
            .unwrap_or_else(default_partial_rotary_factor)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

impl Config {
    pub fn normalized(mut self) -> Self {
        self.text_config = self.text_config.normalized();
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeProfile {
    pub qkv_projection_millis: f64,
    pub kv_append_write_millis: f64,
    pub layout_prepare_millis: f64,
    pub attention_score_millis: f64,
    pub attention_softmax_millis: f64,
    pub attention_mix_millis: f64,
    pub output_projection_millis: f64,
    pub full_attention_mask_prepare_millis: f64,
    pub full_attention_input_layout_millis: f64,
    pub full_attention_kv_materialize_millis: f64,
    pub full_attention_output_collect_millis: f64,
    pub full_attention_output_reshape_millis: f64,
    pub full_attention_gate_millis: f64,
    pub full_attention_kernel_execute_millis: f64,
    pub scheduler_planning_millis: f64,
    pub transfer_millis: f64,
    pub linear_attention_millis: f64,
    pub full_attention_millis: f64,
    pub mlp_millis: f64,
    pub linear_conv_millis: f64,
    pub linear_chunk_prepare_millis: f64,
    pub linear_chunk_prepare_k_beta_millis: f64,
    pub linear_chunk_prepare_g_millis: f64,
    pub linear_chunk_prepare_cache_millis: f64,
    pub linear_chunk_prepare_base_attn_millis: f64,
    pub linear_chunk_prepare_base_attn_decay_mask_millis: f64,
    pub linear_chunk_prepare_base_attn_key_t_millis: f64,
    pub linear_chunk_prepare_base_attn_flatten_millis: f64,
    pub linear_chunk_prepare_base_attn_matmul_millis: f64,
    pub linear_chunk_prepare_base_attn_post_millis: f64,
    pub linear_chunk_solve_millis: f64,
    pub linear_chunk_scan_millis: f64,
    pub linear_chunk_index_millis: f64,
    pub linear_chunk_local_attn_millis: f64,
    pub linear_chunk_recurrent_read_millis: f64,
    pub linear_chunk_state_update_millis: f64,
    pub linear_recurrent_loop_millis: f64,
    pub linear_full_kernel_pack_millis: f64,
    pub linear_full_kernel_execute_millis: f64,
    pub linear_full_kernel_unpack_millis: f64,
}

impl RuntimeProfile {
    pub fn add_assign(&mut self, other: &Self) {
        self.qkv_projection_millis += other.qkv_projection_millis;
        self.kv_append_write_millis += other.kv_append_write_millis;
        self.layout_prepare_millis += other.layout_prepare_millis;
        self.attention_score_millis += other.attention_score_millis;
        self.attention_softmax_millis += other.attention_softmax_millis;
        self.attention_mix_millis += other.attention_mix_millis;
        self.output_projection_millis += other.output_projection_millis;
        self.full_attention_mask_prepare_millis += other.full_attention_mask_prepare_millis;
        self.full_attention_input_layout_millis += other.full_attention_input_layout_millis;
        self.full_attention_kv_materialize_millis += other.full_attention_kv_materialize_millis;
        self.full_attention_output_collect_millis += other.full_attention_output_collect_millis;
        self.full_attention_output_reshape_millis += other.full_attention_output_reshape_millis;
        self.full_attention_gate_millis += other.full_attention_gate_millis;
        self.full_attention_kernel_execute_millis += other.full_attention_kernel_execute_millis;
        self.scheduler_planning_millis += other.scheduler_planning_millis;
        self.transfer_millis += other.transfer_millis;
        self.linear_attention_millis += other.linear_attention_millis;
        self.full_attention_millis += other.full_attention_millis;
        self.mlp_millis += other.mlp_millis;
        self.linear_conv_millis += other.linear_conv_millis;
        self.linear_chunk_prepare_millis += other.linear_chunk_prepare_millis;
        self.linear_chunk_prepare_k_beta_millis += other.linear_chunk_prepare_k_beta_millis;
        self.linear_chunk_prepare_g_millis += other.linear_chunk_prepare_g_millis;
        self.linear_chunk_prepare_cache_millis += other.linear_chunk_prepare_cache_millis;
        self.linear_chunk_prepare_base_attn_millis += other.linear_chunk_prepare_base_attn_millis;
        self.linear_chunk_prepare_base_attn_decay_mask_millis +=
            other.linear_chunk_prepare_base_attn_decay_mask_millis;
        self.linear_chunk_prepare_base_attn_key_t_millis +=
            other.linear_chunk_prepare_base_attn_key_t_millis;
        self.linear_chunk_prepare_base_attn_flatten_millis +=
            other.linear_chunk_prepare_base_attn_flatten_millis;
        self.linear_chunk_prepare_base_attn_matmul_millis +=
            other.linear_chunk_prepare_base_attn_matmul_millis;
        self.linear_chunk_prepare_base_attn_post_millis +=
            other.linear_chunk_prepare_base_attn_post_millis;
        self.linear_chunk_solve_millis += other.linear_chunk_solve_millis;
        self.linear_chunk_scan_millis += other.linear_chunk_scan_millis;
        self.linear_chunk_index_millis += other.linear_chunk_index_millis;
        self.linear_chunk_local_attn_millis += other.linear_chunk_local_attn_millis;
        self.linear_chunk_recurrent_read_millis += other.linear_chunk_recurrent_read_millis;
        self.linear_chunk_state_update_millis += other.linear_chunk_state_update_millis;
        self.linear_recurrent_loop_millis += other.linear_recurrent_loop_millis;
        self.linear_full_kernel_pack_millis += other.linear_full_kernel_pack_millis;
        self.linear_full_kernel_execute_millis += other.linear_full_kernel_execute_millis;
        self.linear_full_kernel_unpack_millis += other.linear_full_kernel_unpack_millis;
    }

    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            qkv_projection_millis: self.qkv_projection_millis * factor,
            kv_append_write_millis: self.kv_append_write_millis * factor,
            layout_prepare_millis: self.layout_prepare_millis * factor,
            attention_score_millis: self.attention_score_millis * factor,
            attention_softmax_millis: self.attention_softmax_millis * factor,
            attention_mix_millis: self.attention_mix_millis * factor,
            output_projection_millis: self.output_projection_millis * factor,
            full_attention_mask_prepare_millis: self.full_attention_mask_prepare_millis * factor,
            full_attention_input_layout_millis: self.full_attention_input_layout_millis * factor,
            full_attention_kv_materialize_millis: self.full_attention_kv_materialize_millis * factor,
            full_attention_output_collect_millis: self.full_attention_output_collect_millis * factor,
            full_attention_output_reshape_millis: self.full_attention_output_reshape_millis * factor,
            full_attention_gate_millis: self.full_attention_gate_millis * factor,
            full_attention_kernel_execute_millis: self.full_attention_kernel_execute_millis * factor,
            scheduler_planning_millis: self.scheduler_planning_millis * factor,
            transfer_millis: self.transfer_millis * factor,
            linear_attention_millis: self.linear_attention_millis * factor,
            full_attention_millis: self.full_attention_millis * factor,
            mlp_millis: self.mlp_millis * factor,
            linear_conv_millis: self.linear_conv_millis * factor,
            linear_chunk_prepare_millis: self.linear_chunk_prepare_millis * factor,
            linear_chunk_prepare_k_beta_millis: self.linear_chunk_prepare_k_beta_millis * factor,
            linear_chunk_prepare_g_millis: self.linear_chunk_prepare_g_millis * factor,
            linear_chunk_prepare_cache_millis: self.linear_chunk_prepare_cache_millis * factor,
            linear_chunk_prepare_base_attn_millis: self.linear_chunk_prepare_base_attn_millis * factor,
            linear_chunk_prepare_base_attn_decay_mask_millis: self
                .linear_chunk_prepare_base_attn_decay_mask_millis
                * factor,
            linear_chunk_prepare_base_attn_key_t_millis: self
                .linear_chunk_prepare_base_attn_key_t_millis
                * factor,
            linear_chunk_prepare_base_attn_flatten_millis: self
                .linear_chunk_prepare_base_attn_flatten_millis
                * factor,
            linear_chunk_prepare_base_attn_matmul_millis: self
                .linear_chunk_prepare_base_attn_matmul_millis
                * factor,
            linear_chunk_prepare_base_attn_post_millis: self
                .linear_chunk_prepare_base_attn_post_millis
                * factor,
            linear_chunk_solve_millis: self.linear_chunk_solve_millis * factor,
            linear_chunk_scan_millis: self.linear_chunk_scan_millis * factor,
            linear_chunk_index_millis: self.linear_chunk_index_millis * factor,
            linear_chunk_local_attn_millis: self.linear_chunk_local_attn_millis * factor,
            linear_chunk_recurrent_read_millis: self.linear_chunk_recurrent_read_millis * factor,
            linear_chunk_state_update_millis: self.linear_chunk_state_update_millis * factor,
            linear_recurrent_loop_millis: self.linear_recurrent_loop_millis * factor,
            linear_full_kernel_pack_millis: self.linear_full_kernel_pack_millis * factor,
            linear_full_kernel_execute_millis: self.linear_full_kernel_execute_millis * factor,
            linear_full_kernel_unpack_millis: self.linear_full_kernel_unpack_millis * factor,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttentionBenchResult {
    pub layer_id: usize,
    pub sequence_length: usize,
    pub repeats: usize,
    pub mean_total_millis: f64,
    pub best_total_millis: f64,
    pub iteration_total_millis: Vec<f64>,
    pub mean_profile: RuntimeProfile,
    pub best_profile: RuntimeProfile,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionLayerSpec {
    pub layer_id: usize,
    pub conv_dim: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub state_len: usize,
    pub kernel_size: usize,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionTrace {
    pub layer_id: usize,
    pub sequence_length: usize,
    pub layer_output: StateBuffer,
    pub recurrent_state: StateBuffer,
    pub profile: RuntimeProfile,
}

#[derive(Debug, Clone)]
pub struct DecoderLayerTrace {
    pub layer_id: usize,
    pub sequence_length: usize,
    pub input_layernorm_output: StateBuffer,
    pub linear_projection_trace: Option<LinearAttentionProjectionTrace>,
    pub linear_core_trace: Option<LinearAttentionCoreTrace>,
    pub token_mixer_output: StateBuffer,
    pub post_attention_layernorm_output: StateBuffer,
    pub mlp_output: StateBuffer,
    pub layer_output: StateBuffer,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionProjectionTrace {
    pub qkv_output: StateBuffer,
    pub z_output: StateBuffer,
    pub b_output: StateBuffer,
    pub a_output: StateBuffer,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionCoreTrace {
    pub conv_weight_squeezed: StateBuffer,
    pub post_conv_mixed_qkv: StateBuffer,
    pub explicit_post_conv_mixed_qkv: StateBuffer,
    pub explicit_post_conv_reversed_taps_mixed_qkv: StateBuffer,
    pub post_conv_value_focus_head: StateBuffer,
    pub explicit_post_conv_value_focus_head: StateBuffer,
    pub explicit_post_conv_reversed_taps_value_focus_head: StateBuffer,
    pub prepared_value_focus_head: StateBuffer,
    pub pre_gated_norm_output: StateBuffer,
    pub pre_gated_norm_mean_square: StateBuffer,
    pub pre_gated_norm_rsqrt: StateBuffer,
    pub gated_norm_gate_input: StateBuffer,
    pub gated_norm_weight: StateBuffer,
    pub gated_norm_weighted_hidden: StateBuffer,
    pub gated_norm_weighted_hidden_fallback: StateBuffer,
    pub gated_norm_silu_gate: StateBuffer,
    pub post_gated_norm_output: StateBuffer,
}

pub struct ExternalFullAttentionOutput {
    pub attn_output: Tensor,
    pub profile: RuntimeProfile,
}

pub trait ExternalFullAttention {
    fn forward(
        &mut self,
        layer_id: usize,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        num_kv_groups: usize,
        head_dim: usize,
        seqlen_offset: usize,
    ) -> Result<ExternalFullAttentionOutput>;
}

#[derive(Debug, Clone, Default)]
pub struct FullAttentionCacheState {
    pub kv_cache: Option<(StateBuffer, StateBuffer)>,
}

#[derive(Debug, Clone, Default)]
pub struct LinearAttentionCacheState {
    pub conv_state: Option<StateBuffer>,
    pub recurrent_state: Option<StateBuffer>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeFullAttentionCacheState {
    pub key: Option<BufferViewDesc>,
    pub value: Option<BufferViewDesc>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeLinearAttentionCacheState {
    pub conv_state: Option<BufferViewDesc>,
    pub recurrent_state: Option<BufferViewDesc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeLayerCacheState {
    Linear(NativeLinearAttentionCacheState),
    Full(NativeFullAttentionCacheState),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeCacheState {
    pub layers: Vec<NativeLayerCacheState>,
}

#[derive(Debug, Clone)]
pub enum LayerCacheState {
    Linear(LinearAttentionCacheState),
    Full(FullAttentionCacheState),
}

#[derive(Debug, Clone, Default)]
pub struct CacheState {
    pub layers: Vec<LayerCacheState>,
}

fn scalar_type_for_dtype(dtype: DType) -> Result<ScalarType> {
    match dtype {
        DType::U8 => Ok(ScalarType::U8),
        DType::U32 => Ok(ScalarType::U32),
        DType::I16 => Ok(ScalarType::I16),
        DType::I32 => Ok(ScalarType::I32),
        DType::I64 => Ok(ScalarType::I64),
        DType::BF16 => Ok(ScalarType::BF16),
        DType::F16 => Ok(ScalarType::F16),
        DType::F32 => Ok(ScalarType::F32),
        other => candle::bail!("unsupported dtype for buffer view descriptor: {other:?}"),
    }
}

fn tensor_buffer_view_desc(tensor: &Tensor) -> Result<BufferViewDesc> {
    Ok(BufferViewDesc {
        scalar_type: scalar_type_for_dtype(tensor.dtype())?,
        shape: tensor.dims().to_vec(),
        byte_offset: 0,
        byte_len: (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64,
        mutability: BufferMutability::Mutable,
    })
}

#[derive(Debug, Clone)]
pub struct StateBuffer {
    tensor: Tensor,
    desc: BufferViewDesc,
}

impl StateBuffer {
    pub(crate) fn from_tensor(tensor: Tensor) -> Result<Self> {
        Ok(Self {
            desc: tensor_buffer_view_desc(&tensor)?,
            tensor,
        })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn device(&self) -> &Device {
        self.tensor.device()
    }

    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.tensor.dims3()
    }

    pub fn clone_tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    pub(crate) fn clone_tensor_as(&self, dtype: DType) -> Result<Tensor> {
        if self.tensor.dtype() == dtype {
            Ok(self.tensor.clone())
        } else {
            Ok(self.tensor.to_dtype(dtype)?)
        }
    }

    pub fn desc(&self) -> &BufferViewDesc {
        &self.desc
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        Self::from_tensor(self.tensor.narrow(dim, start, len)?)
    }

    pub fn contiguous(&self) -> Result<Self> {
        Self::from_tensor(self.tensor.contiguous()?)
    }
}

impl CacheState {
    pub fn layer_max_abs_deltas(&self, other: &Self) -> Result<Vec<String>> {
        if self.layers.len() != other.layers.len() {
            candle::bail!(
                "cache delta layer count mismatch: {} vs {}",
                self.layers.len(),
                other.layers.len()
            );
        }
        let mut lines = Vec::with_capacity(self.layers.len());
        for (layer_id, (lhs, rhs)) in self.layers.iter().zip(other.layers.iter()).enumerate() {
            match (lhs, rhs) {
                (LayerCacheState::Linear(lhs), LayerCacheState::Linear(rhs)) => {
                    let conv_delta = match (&lhs.conv_state, &rhs.conv_state) {
                        (Some(lhs), Some(rhs)) => {
                            Some(max_abs_delta(lhs.tensor(), rhs.tensor())?)
                        }
                        (None, None) => None,
                        _ => {
                            lines.push(format!(
                                "layer={layer_id} kind=linear conv_state_presence_mismatch"
                            ));
                            None
                        }
                    };
                    let recurrent_delta = match (&lhs.recurrent_state, &rhs.recurrent_state) {
                        (Some(lhs), Some(rhs)) => {
                            Some(max_abs_delta(lhs.tensor(), rhs.tensor())?)
                        }
                        (None, None) => None,
                        _ => {
                            lines.push(format!(
                                "layer={layer_id} kind=linear recurrent_state_presence_mismatch"
                            ));
                            None
                        }
                    };
                    if conv_delta.is_some() || recurrent_delta.is_some() {
                        lines.push(format!(
                            "layer={layer_id} kind=linear conv_state_max_abs_delta={:.6} recurrent_state_max_abs_delta={:.6}",
                            conv_delta.unwrap_or(0.0),
                            recurrent_delta.unwrap_or(0.0)
                        ));
                    }
                }
                (LayerCacheState::Full(lhs), LayerCacheState::Full(rhs)) => {
                    let (key_delta, value_delta) = match (&lhs.kv_cache, &rhs.kv_cache) {
                        (Some((lhs_key, lhs_value)), Some((rhs_key, rhs_value))) => (
                            Some(max_abs_delta(lhs_key.tensor(), rhs_key.tensor())?),
                            Some(max_abs_delta(lhs_value.tensor(), rhs_value.tensor())?),
                        ),
                        (None, None) => (None, None),
                        _ => {
                            lines.push(format!(
                                "layer={layer_id} kind=full kv_presence_mismatch"
                            ));
                            (None, None)
                        }
                    };
                    if key_delta.is_some() || value_delta.is_some() {
                        lines.push(format!(
                            "layer={layer_id} kind=full key_max_abs_delta={:.6} value_max_abs_delta={:.6}",
                            key_delta.unwrap_or(0.0),
                            value_delta.unwrap_or(0.0)
                        ));
                    }
                }
                (LayerCacheState::Linear(_), LayerCacheState::Full(_))
                | (LayerCacheState::Full(_), LayerCacheState::Linear(_)) => {
                    lines.push(format!("layer={layer_id} kind_mismatch"));
                }
            }
        }
        Ok(lines)
    }

    pub fn sequence_length(&self) -> usize {
        for layer in &self.layers {
            if let LayerCacheState::Full(FullAttentionCacheState {
                kv_cache: Some((key, _)),
            }) = layer
            {
                if let Ok((_, _, seq_len, _)) = key.tensor().dims4() {
                    return seq_len;
                }
            }
        }
        0
    }

    pub fn describe(&self) -> Result<NativeCacheState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let native = match layer {
                LayerCacheState::Linear(state) => NativeLayerCacheState::Linear(
                    NativeLinearAttentionCacheState {
                        conv_state: state.conv_state.as_ref().map(|buffer| buffer.desc().clone()),
                        recurrent_state: state
                            .recurrent_state
                            .as_ref()
                            .map(|buffer| buffer.desc().clone()),
                    },
                ),
                LayerCacheState::Full(state) => {
                    let (key, value) = match &state.kv_cache {
                        Some((key, value)) => {
                            (Some(key.desc().clone()), Some(value.desc().clone()))
                        }
                        None => (None, None),
                    };
                    NativeLayerCacheState::Full(NativeFullAttentionCacheState { key, value })
                }
            };
            layers.push(native);
        }
        Ok(NativeCacheState { layers })
    }
}
