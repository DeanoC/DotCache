#![allow(unexpected_cfgs)]

use super::activation::Activation;
use super::backend_buffer_api;
use super::backend_buffer_api::Qwen35BackendBufferApi;
use super::backend_ops;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
#[cfg(feature = "qwen35-minimal-hip")]
use super::hip;
use super::ops;
use super::prepared::PreparedTensorSource;
use super::rotary;
#[cfg(any(feature = "hf", test))]
use super::with_tracing::{linear_b, linear_no_bias};
use super::with_tracing::Linear;
use crate::backends;
use candle::{DType, Device, DeviceLocation, IndexOp, Module, Result, Tensor, D};
use candle_core as candle;
use candle_core::backend::{BackendDevice, BackendStorage};
use crate::{
    BufferMutability, BufferViewDesc, ImmutableBufferView, ImmutableWeightHandle,
    PreparedQwen35DirectMetadata, ScalarType, TargetSpec,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn elapsed_millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn hip_output_bytes_to_cpu_storage(dtype: DType, output: Vec<u8>) -> Result<candle::CpuStorage> {
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

fn hip_tensor_from_host_bytes<S: Into<candle::Shape>>(
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

fn trace_hip_wrapper_fallback(op: &str, tensor: &Tensor) {
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

fn repeat_kv(xs: Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(xs);
    }
    let (b_sz, kv_heads, seq_len, head_dim) = xs.dims4()?;
    let repeated = vec![&xs; repeats];
    Tensor::cat(&repeated, 2)?.reshape((b_sz, kv_heads * repeats, seq_len, head_dim))
}

fn prepared_linear_no_bias(source: &PreparedTensorSource) -> Result<Linear> {
    Ok(Linear::new(source.get("weight")?, None))
}

fn build_prepared_linear_source_no_bias(
    source: &PreparedTensorSource,
    in_dim: usize,
    out_dim: usize,
    immutable_requested: bool,
) -> Result<LinearSource> {
    let eager = || prepared_linear_no_bias(source).map(LinearSource::Materialized);
    if !immutable_requested || !source.device().is_hip() {
        return eager();
    }
    let Some(handle) = source.get_immutable("weight")? else {
        return eager();
    };
    let shape = handle.shape().to_vec();
    if shape.len() != 2 || shape[0] != out_dim || shape[1] != in_dim {
        return eager();
    }
    if !matches!(
        handle.layout(),
        crate::model_package::TensorLayout::StandardContiguous
    ) {
        return eager();
    }
    Ok(LinearSource::Deferred(DeferredLinear::new(
        handle,
        source.device().clone(),
    )))
}

fn prepared_linear_b(source: &PreparedTensorSource, bias: bool) -> Result<Linear> {
    let weight = source.get("weight")?;
    let bias = if bias { Some(source.get("bias")?) } else { None };
    Ok(Linear::new(weight, bias))
}

fn prepared_embedding(source: &PreparedTensorSource, hidden_size: usize) -> Result<Embedding> {
    Ok(Embedding::new(source.get("weight")?, hidden_size))
}

#[derive(Debug, Clone)]
struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub(super) fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }
}

impl Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        values.reshape(final_dims)
    }
}

impl Embedding {
    fn forward_buffer(&self, indexes: &Tensor) -> Result<StateBuffer> {
        backend_buffer_api::for_device(indexes.device()).embedding_lookup(&self.embeddings, indexes)
    }
}

#[cfg(any(feature = "hf", test))]
fn embedding(in_size: usize, out_size: usize, vb: WeightBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        candle_nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

fn prepared_dtype_to_candle(dtype: crate::model_package::PreparedDType) -> Option<DType> {
    match dtype {
        crate::model_package::PreparedDType::F16 => Some(DType::F16),
        crate::model_package::PreparedDType::BF16 => Some(DType::BF16),
        crate::model_package::PreparedDType::F32 => Some(DType::F32),
        _ => None,
    }
}

fn build_prepared_embedding_source(
    source: &PreparedTensorSource,
    hidden_size: usize,
    immutable_requested: bool,
) -> Result<(EmbeddingSource, bool, Option<String>)> {
    let eager = || prepared_embedding(source, hidden_size).map(EmbeddingSource::Materialized);
    if !immutable_requested {
        return Ok((eager()?, false, None));
    }
    if !source.device().is_hip() {
        return Ok((eager()?, false, Some("backend-unsupported".to_string())));
    }

    let Some(handle) = source.get_immutable("weight")? else {
        return Ok((
            eager()?,
            false,
            Some("immutable-handle-unavailable".to_string()),
        ));
    };
    let shape = handle.shape().to_vec();
    if shape.len() != 2 || shape[1] != hidden_size {
        return Ok((
            eager()?,
            false,
            Some("immutable-embedding-shape-mismatch".to_string()),
        ));
    }
    if !matches!(
        handle.layout(),
        crate::model_package::TensorLayout::StandardContiguous
    ) {
        return Ok((
            eager()?,
            false,
            Some("immutable-embedding-layout-unsupported".to_string()),
        ));
    }
    let Some(dtype) = prepared_dtype_to_candle(handle.dtype()) else {
        return Ok((
            eager()?,
            false,
            Some("immutable-embedding-dtype-unsupported".to_string()),
        ));
    };

    Ok((
        EmbeddingSource::Immutable(ImmutableEmbedding::new(
            handle,
            EmbeddingMeta {
                vocab_size: shape[0],
                hidden_size,
                dtype,
                device: source.device().clone(),
            },
        )),
        true,
        None,
    ))
}

fn immutable_embedding_enabled() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_IMMUTABLE_EMBED").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn deferred_linear_auto_enabled(cfg: &TextConfig) -> bool {
    let total_mlp_weight_bytes = (cfg.num_hidden_layers as u128)
        * 3
        * (cfg.hidden_size as u128)
        * (cfg.intermediate_size as u128)
        * 2;
    total_mlp_weight_bytes >= (1u128 << 30)
}

fn immutable_linear_enabled(cfg: &TextConfig) -> bool {
    match std::env::var("DOTCACHE_QWEN35_IMMUTABLE_LINEAR") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            "auto" => deferred_linear_auto_enabled(cfg),
            _ => deferred_linear_auto_enabled(cfg),
        },
        Err(_) => deferred_linear_auto_enabled(cfg),
    }
}

fn deferred_in_proj_qkv_enabled() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_DEFERRED_IN_PROJ_QKV").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[derive(Debug, Clone)]
pub(super) struct EmbeddingMeta {
    vocab_size: usize,
    hidden_size: usize,
    dtype: DType,
    device: Device,
}

#[derive(Debug, Clone)]
pub(super) struct NativeImmutableWeight {
    handle: ImmutableWeightHandle,
    target: TargetSpec,
    view_desc: BufferViewDesc,
}

impl NativeImmutableWeight {
    fn new(handle: ImmutableWeightHandle) -> Self {
        let target = handle.target_spec();
        let view_desc = handle.buffer_view_desc();
        Self {
            handle,
            target,
            view_desc,
        }
    }

    fn immutable_buffer_view(&self) -> ImmutableBufferView<'_> {
        self.handle.immutable_buffer_view()
    }

    fn materialize(&self, device: &Device) -> crate::Result<Tensor> {
        self.handle.materialize(device).map_err(Into::into)
    }
}

#[derive(Debug)]
struct RegisteredHostWeight {
    host_ptr: usize,
    device_ptr: usize,
}

impl RegisteredHostWeight {
    fn device_ptr(&self) -> *const std::ffi::c_void {
        self.device_ptr as *const std::ffi::c_void
    }
}

impl Drop for RegisteredHostWeight {
    fn drop(&mut self) {
        #[cfg(feature = "qwen35-minimal-hip")]
        hip::unregister_host_mapping(self.host_ptr as *const std::ffi::c_void);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImmutableEmbedding {
    weight: NativeImmutableWeight,
    meta: EmbeddingMeta,
    state: Arc<Mutex<ImmutableEmbeddingState>>,
}

#[derive(Debug)]
enum ImmutableEmbeddingState {
    Uninitialized,
    Registered(RegisteredHostWeight),
    Fallback(Embedding),
}

impl ImmutableEmbedding {
    fn new(handle: ImmutableWeightHandle, meta: EmbeddingMeta) -> Self {
        Self {
            weight: NativeImmutableWeight::new(handle),
            meta,
            state: Arc::new(Mutex::new(ImmutableEmbeddingState::Uninitialized)),
        }
    }

    fn current_mode(&self) -> &'static str {
        let state = self.state.lock().expect("immutable embedding state poisoned");
        match &*state {
            ImmutableEmbeddingState::Uninitialized | ImmutableEmbeddingState::Registered(_) => {
                "immutable"
            }
            ImmutableEmbeddingState::Fallback(_) => "fallback",
        }
    }

    pub(super) fn device(&self) -> &Device {
        &self.meta.device
    }

    pub(crate) fn dtype(&self) -> DType {
        self.meta.dtype
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.meta.vocab_size
    }

    fn materialized_embedding(&self) -> Result<Embedding> {
        Ok(Embedding::new(
            self.weight
                .materialize(&self.meta.device)
                .map_err(|err| candle::Error::Msg(err.to_string()))?,
            self.meta.hidden_size,
        ))
    }

    fn ensure_fallback_embedding(&self) -> Result<Embedding> {
        let mut state = self.state.lock().expect("immutable embedding state poisoned");
        if let ImmutableEmbeddingState::Fallback(embedding) = &*state {
            return Ok(embedding.clone());
        }
        let embedding = self.materialized_embedding()?;
        *state = ImmutableEmbeddingState::Fallback(embedding.clone());
        Ok(embedding)
    }

    pub(super) fn fallback_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let fallback = self.ensure_fallback_embedding()?;
        immutable_embedding_forward_fallback(&fallback, input_ids)
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    pub(crate) fn registered_device_ptr(
        &self,
        device_ordinal: usize,
    ) -> Result<*const std::ffi::c_void> {
        let mut state = self.state.lock().expect("immutable embedding state poisoned");
        match &*state {
            ImmutableEmbeddingState::Registered(weight) => return Ok(weight.device_ptr()),
            ImmutableEmbeddingState::Fallback(_) => {
                candle::bail!("immutable embedding already fell back to eager storage")
            }
            ImmutableEmbeddingState::Uninitialized => {}
        }
        let buffer = self.weight.immutable_buffer_view();
        debug_assert_eq!(buffer.target, self.weight.target);
        debug_assert_eq!(buffer.desc, self.weight.view_desc);
        let host_ptr = buffer.bytes.as_ptr() as *const std::ffi::c_void;
        let byte_len = buffer.bytes.len();
        let device_ptr = hip::register_host_mapping_for_device(device_ordinal, host_ptr, byte_len)?;
        *state = ImmutableEmbeddingState::Registered(RegisteredHostWeight {
            host_ptr: host_ptr as usize,
            device_ptr: device_ptr as usize,
        });
        match &*state {
            ImmutableEmbeddingState::Registered(weight) => Ok(weight.device_ptr()),
            _ => unreachable!("registered immutable embedding state replaced unexpectedly"),
        }
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "qwen35-minimal-hip")]
        if self.meta.device.is_hip() && input_ids.device().is_hip() {
            match backend_buffer_api::for_device(input_ids.device())
                .immutable_embedding_lookup(self, input_ids)
            {
                Ok(output) => return Ok(output),
                Err(_) => {
                    let fallback = self.ensure_fallback_embedding()?;
                    return immutable_embedding_forward_fallback(&fallback, input_ids);
                }
            }
        }
        let fallback = self.ensure_fallback_embedding()?;
        immutable_embedding_forward_fallback(&fallback, input_ids)
    }
}

fn immutable_embedding_forward_fallback(
    embedding: &Embedding,
    input_ids: &Tensor,
) -> Result<Tensor> {
    backend_buffer_api::for_device(input_ids.device())
        .embedding_lookup(embedding.embeddings(), input_ids)
        .map(|buffer| buffer.clone_tensor())
}

#[derive(Debug, Clone)]
enum EmbeddingSource {
    Materialized(Embedding),
    Immutable(ImmutableEmbedding),
}

impl EmbeddingSource {
    fn embeddings(&self) -> Option<&Tensor> {
        match self {
            Self::Materialized(embedding) => Some(embedding.embeddings()),
            Self::Immutable(_) => None,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Materialized(embedding) => embedding.embeddings().dtype(),
            Self::Immutable(embedding) => embedding.meta.dtype,
        }
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(embedding) => immutable_embedding_forward_fallback(embedding, input_ids),
            Self::Immutable(embedding) => embedding.forward(input_ids),
        }
    }

    fn forward_buffer(&self, input_ids: &Tensor) -> Result<StateBuffer> {
        match self {
            Self::Materialized(embedding) => embedding.forward_buffer(input_ids),
            Self::Immutable(embedding) => {
                let backend = backend_buffer_api::for_device(input_ids.device());
                backend.tensor_to_buffer(backend.immutable_embedding_lookup(embedding, input_ids)?)
            }
        }
    }

    fn runtime_mode(&self) -> &'static str {
        match self {
            Self::Materialized(_) => "eager",
            Self::Immutable(embedding) => embedding.current_mode(),
        }
    }
}

#[derive(Debug, Clone)]
enum OutputProjectionSource {
    Materialized(Linear),
    TiedImmutable(ImmutableEmbedding),
}

impl OutputProjectionSource {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(linear) => linear.forward(hidden_states),
            Self::TiedImmutable(embedding) => backend_buffer_api::for_device(hidden_states.device())
                .output_projection_tensor(embedding, hidden_states),
        }
    }

    fn forward_buffer(&self, hidden_states: &StateBuffer) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => linear.forward_buffer(hidden_states),
            Self::TiedImmutable(embedding) => {
                backend_buffer_api::for_device(hidden_states.device())
                    .output_projection(embedding, hidden_states)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DeferredLinear {
    weight: NativeImmutableWeight,
    device: Device,
    state: Arc<Mutex<Option<Linear>>>,
}

impl DeferredLinear {
    fn new(handle: ImmutableWeightHandle, device: Device) -> Self {
        Self {
            weight: NativeImmutableWeight::new(handle),
            device,
            state: Arc::new(Mutex::new(None)),
        }
    }

    fn ensure_materialized(&self) -> Result<Linear> {
        let mut state = self.state.lock().expect("deferred linear state poisoned");
        if let Some(linear) = &*state {
            return Ok(linear.clone());
        }
        let linear = Linear::new(
            self.weight
                .materialize(&self.device)
                .map_err(|err| candle::Error::Msg(err.to_string()))?,
            None,
        );
        *state = Some(linear.clone());
        Ok(linear)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.ensure_materialized()?.forward(xs)
    }
}

#[derive(Debug, Clone)]
enum LinearSource {
    Materialized(Linear),
    Deferred(DeferredLinear),
}

impl LinearSource {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(linear) => linear.forward(xs),
            Self::Deferred(linear) => linear.forward(xs),
        }
    }

    fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => linear.forward_buffer(xs),
            Self::Deferred(linear) => linear.ensure_materialized()?.forward_buffer(xs),
        }
    }

    fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred(_))
    }
}

fn profile_sync_enabled(device: &Device) -> bool {
    matches!(
        device.location(),
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. }
    ) && matches!(
        std::env::var("CANDLE_QWEN35_PROFILE_SYNC").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn profile_start(device: &Device) -> Result<Instant> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(Instant::now())
}

fn profile_elapsed(start: Instant, device: &Device) -> Result<f64> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(elapsed_millis(start))
}

fn debug_full_prefill_kernel_compare_enabled() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_DEBUG_HIP_PREFILL_KERNEL").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if lhs.len() != rhs.len() {
        candle::bail!(
            "max_abs_delta shape mismatch lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        );
    }
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max))
}

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

    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|params| params.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    fn partial_rotary_factor(&self) -> f64 {
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
            full_attention_kv_materialize_millis: self.full_attention_kv_materialize_millis
                * factor,
            full_attention_output_collect_millis: self.full_attention_output_collect_millis
                * factor,
            full_attention_output_reshape_millis: self.full_attention_output_reshape_millis
                * factor,
            full_attention_gate_millis: self.full_attention_gate_millis * factor,
            full_attention_kernel_execute_millis: self.full_attention_kernel_execute_millis
                * factor,
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
            linear_chunk_prepare_base_attn_millis: self.linear_chunk_prepare_base_attn_millis
                * factor,
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

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        let rotary_dim = ((cfg.head_dim as f64) * cfg.partial_rotary_factor()).round() as usize;
        let rotary_dim = rotary_dim.max(2).min(cfg.head_dim);
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        if self.rotary_dim >= head_dim {
            let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
            let q_embed = if q.device().is_hip() {
                backends::hip::rope(q, &cos, &sin)?
            } else {
                rotary::rope(&q.contiguous()?, &cos, &sin)?
            };
            let k_embed = if k.device().is_hip() {
                backends::hip::rope(k, &cos, &sin)?
            } else {
                rotary::rope(&k.contiguous()?, &cos, &sin)?
            };
            return Ok((q_embed, k_embed));
        }

        let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
        let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_rot = if q.device().is_hip() {
            backends::hip::rope(&q_rot, &cos, &sin)?
        } else {
            rotary::rope(&q_rot.contiguous()?, &cos, &sin)?
        };
        let k_rot = if k.device().is_hip() {
            backends::hip::rope(&k_rot, &cos, &sin)?
        } else {
            rotary::rope(&k_rot.contiguous()?, &cos, &sin)?
        };
        Ok((
            Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?,
            Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?,
        ))
    }

    fn apply_buffer(
        &self,
        q: &StateBuffer,
        k: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, StateBuffer)> {
        let backend = backend_buffer_api::for_device(q.device());
        let (_, _, seq_len, head_dim) = q.tensor().dims4()?;
        if self.rotary_dim >= head_dim {
            let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
            let q_embed = if q.device().is_hip() {
                backends::hip::rope_buffer(q, &cos, &sin)?
            } else {
                backend.tensor_to_buffer(rotary::rope(&q.tensor().contiguous()?, &cos, &sin)?)?
            };
            let k_embed = if k.device().is_hip() {
                backends::hip::rope_buffer(k, &cos, &sin)?
            } else {
                backend.tensor_to_buffer(rotary::rope(&k.tensor().contiguous()?, &cos, &sin)?)?
            };
            return Ok((q_embed, k_embed));
        }

        let q_rot = q.narrow(q.tensor().rank() - 1, 0, self.rotary_dim)?;
        let q_pass = q.narrow(
            q.tensor().rank() - 1,
            self.rotary_dim,
            head_dim - self.rotary_dim,
        )?;
        let k_rot = k.narrow(k.tensor().rank() - 1, 0, self.rotary_dim)?;
        let k_pass = k.narrow(
            k.tensor().rank() - 1,
            self.rotary_dim,
            head_dim - self.rotary_dim,
        )?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_rot = if q.device().is_hip() {
            backends::hip::rope_buffer(&q_rot, &cos, &sin)?
        } else {
            backend.tensor_to_buffer(rotary::rope(&q_rot.tensor().contiguous()?, &cos, &sin)?)?
        };
        let k_rot = if k.device().is_hip() {
            backends::hip::rope_buffer(&k_rot, &cos, &sin)?
        } else {
            backend.tensor_to_buffer(rotary::rope(&k_rot.tensor().contiguous()?, &cos, &sin)?)?
        };
        Ok((
            backend.concat_last_dim(&q_rot, &q_pass)?,
            backend.concat_last_dim(&k_rot, &k_pass)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

#[derive(Debug, Clone, Copy)]
struct HipRmsNorm {
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    add_unit_offset: bool,
}

impl candle::CustomOp2 for HipRmsNorm {
    fn name(&self) -> &'static str {
        "dotcache-hip-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("dotcache-hip-rms-norm has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        xs: &candle::HipStorage,
        xs_layout: &candle::Layout,
        weight: &candle::HipStorage,
        weight_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(xs_layout.is_contiguous() && weight_layout.is_contiguous()) {
            candle::bail!("dotcache-hip-rms-norm requires contiguous inputs")
        }
        if xs.dtype() != weight.dtype() {
            candle::bail!(
                "dotcache-hip-rms-norm requires matching dtypes, got xs={:?} weight={:?}",
                xs.dtype(),
                weight.dtype()
            )
        }

        let xs_dims = xs_layout.shape().dims();
        let n_cols = *xs_dims.last().ok_or_else(|| {
            candle::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into())
        })?;
        let n_rows = xs_layout.shape().elem_count() / n_cols;
        let weight_dim = weight_layout.shape().elem_count();
        if n_rows != self.n_rows || n_cols != self.n_cols || weight_dim != self.n_cols {
            candle::bail!(
                "dotcache-hip-rms-norm shape mismatch xs={:?} weight={:?} expected_rows={} expected_cols={}",
                xs_layout.shape().dims(),
                weight_layout.shape().dims(),
                self.n_rows,
                self.n_cols
            )
        }

        let device = xs.device().clone();
        let out_shape = xs_layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(xs.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_rms_norm(
                hip::dtype_code(xs.dtype())?,
                device.ordinal(),
                self.n_rows,
                self.n_cols,
                self.eps,
                if self.add_unit_offset { 1 } else { 0 },
                xs.raw_device_ptr_with_offset(xs_layout.start_offset())? as *const c_void,
                weight.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(xs.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_rms_norm_host_buffer(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let weight = weight.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !weight.device().same_device(xs.device()) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    if xs.dtype() != weight.dtype() {
        return Ok(None);
    }
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (Storage::Hip(xs_storage), Storage::Hip(weight_storage)) = (&*xs_storage, &*weight_storage) else {
        return Ok(None);
    };
    if !(xs_layout.is_contiguous() && weight_layout.is_contiguous()) {
        return Ok(None);
    }
    let shape = xs_layout.shape().dims().to_vec();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into()))?;
    let n_rows = xs_layout.shape().elem_count() / n_cols;
    if weight_layout.shape().elem_count() != n_cols {
        return Ok(None);
    }
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
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
            xs_storage.raw_device_ptr_with_offset(xs_layout.start_offset())? as *const c_void,
            weight_storage.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_rms_norm_host_buffer(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
    add_unit_offset: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, weight, eps, add_unit_offset);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipRmsNormGated {
    n_rows: usize,
    n_cols: usize,
    eps: f32,
}

impl candle::CustomOp3 for HipRmsNormGated {
    fn name(&self) -> &'static str {
        "dotcache-hip-rms-norm-gated"
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
        candle::bail!("dotcache-hip-rms-norm-gated has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        hidden: &candle::HipStorage,
        hidden_layout: &candle::Layout,
        gate: &candle::HipStorage,
        gate_layout: &candle::Layout,
        weight: &candle::HipStorage,
        weight_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(hidden_layout.is_contiguous()
            && gate_layout.is_contiguous()
            && weight_layout.is_contiguous())
        {
            candle::bail!("dotcache-hip-rms-norm-gated requires contiguous inputs")
        }
        if hidden.dtype() != gate.dtype() || hidden.dtype() != weight.dtype() {
            candle::bail!(
                "dotcache-hip-rms-norm-gated requires matching dtypes, got hidden={:?} gate={:?} weight={:?}",
                hidden.dtype(),
                gate.dtype(),
                weight.dtype()
            )
        }

        let hidden_dims = hidden_layout.shape().dims();
        let n_cols = *hidden_dims.last().ok_or_else(|| {
            candle::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
        })?;
        let n_rows = hidden_layout.shape().elem_count() / n_cols;
        let gate_elems = gate_layout.shape().elem_count();
        let weight_elems = weight_layout.shape().elem_count();
        if n_rows != self.n_rows
            || n_cols != self.n_cols
            || gate_elems != hidden_layout.shape().elem_count()
            || weight_elems != self.n_cols
        {
            candle::bail!(
                "dotcache-hip-rms-norm-gated shape mismatch hidden={:?} gate={:?} weight={:?} expected_rows={} expected_cols={}",
                hidden_layout.shape().dims(),
                gate_layout.shape().dims(),
                weight_layout.shape().dims(),
                self.n_rows,
                self.n_cols
            )
        }

        let device = hidden.device().clone();
        let out_shape = hidden_layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(hidden.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_rms_norm_gated(
                hip::dtype_code(hidden.dtype())?,
                device.ordinal(),
                self.n_rows,
                self.n_cols,
                self.eps,
                hidden.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
                gate.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
                weight.raw_device_ptr_with_offset(weight_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(hidden.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_rms_norm_gated_host_buffer(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let hidden = hidden.contiguous()?;
    let gate = gate.contiguous()?;
    let weight = weight.contiguous()?;
    let ordinal = match hidden.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(gate.device().same_device(hidden.device()) && weight.device().same_device(hidden.device())) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(hidden.dtype()) else {
        return Ok(None);
    };
    if hidden.dtype() != gate.dtype() || hidden.dtype() != weight.dtype() {
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
    if !(hidden_layout.is_contiguous() && gate_layout.is_contiguous() && weight_layout.is_contiguous()) {
        return Ok(None);
    }
    let shape = hidden_layout.shape().dims().to_vec();
    let n_cols = *shape.last().ok_or_else(|| {
        candle::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
    })?;
    let n_rows = hidden_layout.shape().elem_count() / n_cols;
    if gate_layout.shape().elem_count() != hidden_layout.shape().elem_count()
        || weight_layout.shape().elem_count() != n_cols
    {
        return Ok(None);
    }
    let mut out =
        vec![0u8; shape.iter().product::<usize>().saturating_mul(hidden.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
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
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-rms-norm-gated-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_rms_norm_gated_host_buffer(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (hidden, gate, weight, eps);
    Ok(None)
}

pub(crate) fn hip_rms_norm(xs: &Tensor, weight: &Tensor, eps: f64, add_unit_offset: bool) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    let weight = weight.contiguous()?;
    let weight = if weight.dtype() == xs.dtype() {
        weight
    } else {
        weight.to_dtype(xs.dtype())?
    };
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_rms_norm_host_buffer(&xs, &weight, eps, add_unit_offset)? {
        return hip_tensor_from_host_bytes(xs.device(), xs.dtype(), shape, output);
    }
    let xs_dims = xs.dims();
    let n_cols = *xs_dims.last().ok_or_else(|| {
        candle::Error::Msg("dotcache-hip-rms-norm requires non-empty shape".into())
    })?;
    let n_rows = xs.elem_count() / n_cols;
    trace_hip_wrapper_fallback("hip_rms_norm", &xs);
    xs.apply_op2_no_bwd(
        &weight,
        &HipRmsNorm {
            n_rows,
            n_cols,
            eps: eps as f32,
            add_unit_offset,
        },
    )
}

pub(crate) fn hip_rms_norm_gated(
    hidden_states: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let hidden_states = hidden_states.contiguous()?;
    let gate = gate.contiguous()?;
    let gate = if gate.dtype() == hidden_states.dtype() {
        gate
    } else {
        gate.to_dtype(hidden_states.dtype())?
    };
    let weight = weight.contiguous()?;
    let weight = if weight.dtype() == hidden_states.dtype() {
        weight
    } else {
        weight.to_dtype(hidden_states.dtype())?
    };
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_rms_norm_gated_host_buffer(
        &hidden_states,
        &gate,
        &weight,
        eps,
    )? {
        return hip_tensor_from_host_bytes(hidden_states.device(), hidden_states.dtype(), shape, output);
    }
    let hidden_dims = hidden_states.dims();
    let n_cols = *hidden_dims.last().ok_or_else(|| {
        candle::Error::Msg("dotcache-hip-rms-norm-gated requires non-empty shape".into())
    })?;
    let n_rows = hidden_states.elem_count() / n_cols;
    trace_hip_wrapper_fallback("hip_rms_norm_gated", &hidden_states);
    hidden_states.apply_op3_no_bwd(
        &gate,
        &weight,
        &HipRmsNormGated {
            n_rows,
            n_cols,
            eps: eps as f32,
        },
    )
}

impl Qwen35RmsNorm {
    #[cfg(any(feature = "hf", test))]
    fn new(dim: usize, eps: f64, vb: WeightBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn from_prepared(eps: f64, source: &PreparedTensorSource) -> Result<Self> {
        Ok(Self {
            weight: source.get("weight")?,
            eps,
        })
    }

    fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        backend_buffer_api::for_device(xs.device()).rms_norm(xs, &self.weight, self.eps, true)
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }

    fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        backend_ops::rms_norm(xs, &self.weight, self.eps, true)
    }
}

#[derive(Debug, Clone)]
struct Qwen35RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNormGated {
    #[cfg(any(feature = "hf", test))]
    fn new(dim: usize, eps: f64, vb: WeightBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn from_prepared(eps: f64, source: &PreparedTensorSource) -> Result<Self> {
        Ok(Self {
            weight: source.get("weight")?,
            eps,
        })
    }

    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        backend_ops::rms_norm_gated(hidden_states, gate, &self.weight, self.eps)
    }

    fn forward_buffer(&self, hidden_states: &StateBuffer, gate: &StateBuffer) -> Result<StateBuffer> {
        backend_buffer_api::for_device(hidden_states.device())
            .rms_norm_gated(hidden_states, gate, &self.weight, self.eps)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: LinearSource,
    up_proj: LinearSource,
    down_proj: LinearSource,
    act_fn: Activation,
}

impl Mlp {
    #[cfg(any(feature = "hf", test))]
    fn new(cfg: &TextConfig, vb: WeightBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: LinearSource::Materialized(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?),
            up_proj: LinearSource::Materialized(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?),
            down_proj: LinearSource::Materialized(linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?),
            act_fn: cfg.hidden_act,
        })
    }

    fn from_prepared(cfg: &TextConfig, source: &PreparedTensorSource) -> Result<Self> {
        let immutable_requested = immutable_linear_enabled(cfg);
        Ok(Self {
            gate_proj: build_prepared_linear_source_no_bias(
                &source.pp("gate_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                immutable_requested,
            )?,
            up_proj: build_prepared_linear_source_no_bias(
                &source.pp("up_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                immutable_requested,
            )?,
            down_proj: build_prepared_linear_source_no_bias(
                &source.pp("down_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                immutable_requested,
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn deferred_linear_count(&self) -> usize {
        usize::from(self.gate_proj.is_deferred())
            + usize::from(self.up_proj.is_deferred())
            + usize::from(self.down_proj.is_deferred())
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = if matches!(self.act_fn, Activation::Silu) {
            backend_ops::swiglu_mul(&gate, &up)?
        } else {
            (gate.apply(&self.act_fn)? * up)?
        };
        self.down_proj.forward(&hidden)
    }
}

impl Mlp {
    fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        let backend = backend_buffer_api::for_device(xs.device());
        let gate = self.gate_proj.forward_buffer(xs)?;
        let up = self.up_proj.forward_buffer(xs)?;
        let hidden = if matches!(self.act_fn, Activation::Silu) {
            backend.swiglu_mul(&gate, &up)?
        } else {
            backend.tensor_to_buffer((gate.tensor().apply(&self.act_fn)? * up.tensor())?)?
        };
        self.down_proj.forward_buffer(&hidden)
    }
}

fn repeat_heads(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}

#[cfg(test)]
fn l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    backend_ops::l2norm(xs, eps)
}

#[derive(Debug, Clone, Copy)]
struct HipSwigluMul;

impl candle::CustomOp2 for HipSwigluMul {
    fn name(&self) -> &'static str {
        "hip-swiglu-mul"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-swiglu-mul has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        gate: &candle::HipStorage,
        gate_layout: &candle::Layout,
        up: &candle::HipStorage,
        up_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(gate_layout.is_contiguous() && up_layout.is_contiguous()) {
            candle::bail!("hip-swiglu-mul requires contiguous inputs")
        }
        if gate_layout.shape() != up_layout.shape() {
            candle::bail!(
                "hip-swiglu-mul shape mismatch: gate={:?} up={:?}",
                gate_layout.shape().dims(),
                up_layout.shape().dims()
            )
        }
        if gate.dtype() != up.dtype() {
            candle::bail!(
                "hip-swiglu-mul requires matching dtypes, got gate={:?} up={:?}",
                gate.dtype(),
                up.dtype()
            )
        }

        let device = gate.device().clone();
        let storage_dtype = gate.dtype();
        let elem_count = gate_layout.shape().elem_count();
        let out_shape = gate_layout.shape().clone();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_swiglu_mul(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                elem_count,
                gate.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
                up.raw_device_ptr_with_offset(up_layout.start_offset())? as *const c_void,
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

pub(crate) fn hip_swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_swiglu_mul_host_buffer(gate, up)? {
        return hip_tensor_from_host_bytes(gate.device(), gate.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("hip_swiglu_mul", gate);
    gate.apply_op2_no_bwd(up, &HipSwigluMul)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_swiglu_mul_host_buffer(
    gate: &Tensor,
    up: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let ordinal = match gate.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !up.device().same_device(gate.device()) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(gate.dtype()) else {
        return Ok(None);
    };
    if gate.dtype() != up.dtype() {
        return Ok(None);
    }
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (up_storage, up_layout) = up.storage_and_layout();
    let (Storage::Hip(gate_storage), Storage::Hip(up_storage)) = (&*gate_storage, &*up_storage) else {
        return Ok(None);
    };
    if !(gate_layout.is_contiguous() && up_layout.is_contiguous()) {
        return Ok(None);
    }
    if gate_layout.shape() != up_layout.shape() {
        return Ok(None);
    }
    let shape = gate_layout.shape().dims().to_vec();
    let elem_count = gate_layout.shape().elem_count();
    let mut out =
        vec![0u8; elem_count.saturating_mul(gate.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_swiglu_mul(
            dtype_code,
            ordinal,
            elem_count,
            gate_storage.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
            up_storage.raw_device_ptr_with_offset(up_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-swiglu-mul-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_swiglu_mul_host_buffer(
    gate: &Tensor,
    up: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (gate, up);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipEmbeddingLookup {
    vocab_size: usize,
    hidden_size: usize,
}

impl candle::CustomOp2 for HipEmbeddingLookup {
    fn name(&self) -> &'static str {
        "hip-embedding-lookup"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-embedding-lookup has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        embeddings: &candle::HipStorage,
        embeddings_layout: &candle::Layout,
        indexes: &candle::HipStorage,
        indexes_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(embeddings_layout.is_contiguous() && indexes_layout.is_contiguous()) {
            candle::bail!("hip-embedding-lookup requires contiguous inputs")
        }
        let dims = embeddings_layout.shape().dims();
        if dims.len() != 2 {
            candle::bail!(
                "hip-embedding-lookup expected [vocab, hidden] embeddings, got {:?}",
                dims
            )
        }
        if dims[0] != self.vocab_size || dims[1] != self.hidden_size {
            candle::bail!(
                "hip-embedding-lookup embedding shape mismatch got {:?} expected [{}, {}]",
                dims,
                self.vocab_size,
                self.hidden_size
            )
        }

        let mut out_dims = indexes_layout.shape().dims().to_vec();
        out_dims.push(self.hidden_size);
        let out_shape = candle::Shape::from(out_dims);
        let device = embeddings.device().clone();
        let token_count = indexes_layout.shape().elem_count();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(embeddings.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_embedding_lookup(
                hip::dtype_code(embeddings.dtype())?,
                hip::index_dtype_code(indexes.dtype())?,
                device.ordinal(),
                token_count,
                self.vocab_size,
                self.hidden_size,
                embeddings.raw_device_ptr_with_offset(embeddings_layout.start_offset())?
                    as *const c_void,
                indexes.raw_device_ptr_with_offset(indexes_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(embeddings.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_embedding_lookup(embeddings: &Tensor, indexes: &Tensor) -> Result<Tensor> {
    let embeddings = embeddings.contiguous()?;
    let indexes = indexes.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_embedding_lookup_host_buffer(&embeddings, &indexes)? {
        return hip_tensor_from_host_bytes(embeddings.device(), embeddings.dtype(), shape, output);
    }
    let (vocab_size, hidden_size) = embeddings.dims2()?;
    trace_hip_wrapper_fallback("hip_embedding_lookup", &embeddings);
    embeddings.apply_op2_no_bwd(
        &indexes,
        &HipEmbeddingLookup {
            vocab_size,
            hidden_size,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_embedding_lookup_host_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let embeddings = embeddings.contiguous()?;
    let indexes = indexes.contiguous()?;
    let ordinal = match embeddings.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !indexes.device().same_device(embeddings.device()) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(embeddings.dtype()) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.dtype()) else {
        return Ok(None);
    };
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
    let (vocab_size, hidden_size) = embeddings_layout.shape().dims2()?;
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(hidden_size);
    let mut out =
        vec![0u8; token_count.saturating_mul(hidden_size).saturating_mul(embeddings.dtype().size_in_bytes())];
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
            embeddings_storage
                .raw_device_ptr_with_offset(embeddings_layout.start_offset())? as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-embedding-lookup-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_embedding_lookup_host_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embeddings, indexes);
    Ok(None)
}

#[derive(Debug, Clone)]
struct HipImmutableEmbeddingLookup {
    embedding: ImmutableEmbedding,
}

impl candle::CustomOp1 for HipImmutableEmbeddingLookup {
    fn name(&self) -> &'static str {
        "hip-immutable-embedding-lookup"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-immutable-embedding-lookup has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        indexes: &candle::HipStorage,
        indexes_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !indexes_layout.is_contiguous() {
            candle::bail!("hip-immutable-embedding-lookup requires contiguous indexes")
        }
        let device = indexes.device().clone();
        let device_ptr = self.embedding.registered_device_ptr(device.ordinal())?;
        let token_count = indexes_layout.shape().elem_count();
        let mut out_dims = indexes_layout.shape().dims().to_vec();
        out_dims.push(self.embedding.meta.hidden_size);
        let out_shape = candle::Shape::from(out_dims);
        let elem_count = out_shape.elem_count();
        let mut output =
            vec![0u8; elem_count.saturating_mul(self.embedding.meta.dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_embedding_lookup(
                hip::dtype_code(self.embedding.meta.dtype)?,
                hip::index_dtype_code(indexes.dtype())?,
                device.ordinal(),
                token_count,
                self.embedding.meta.vocab_size,
                self.embedding.meta.hidden_size,
                device_ptr as *const c_void,
                indexes.raw_device_ptr_with_offset(indexes_layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(self.embedding.meta.dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_immutable_embedding_lookup(embedding: &ImmutableEmbedding, indexes: &Tensor) -> Result<Tensor> {
    let indexes = indexes.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_immutable_embedding_lookup_host_buffer(embedding, &indexes)? {
        return hip_tensor_from_host_bytes(indexes.device(), embedding.meta.dtype, shape, output);
    }
    trace_hip_wrapper_fallback("hip_immutable_embedding_lookup", &indexes);
    indexes.apply_op1_no_bwd(&HipImmutableEmbeddingLookup {
        embedding: embedding.clone(),
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_immutable_embedding_lookup_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let indexes = indexes.contiguous()?;
    let ordinal = match indexes.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(embedding.meta.dtype) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.dtype()) else {
        return Ok(None);
    };
    let (indexes_storage, indexes_layout) = indexes.storage_and_layout();
    let Storage::Hip(indexes_storage) = &*indexes_storage else {
        return Ok(None);
    };
    if !indexes_layout.is_contiguous() {
        return Ok(None);
    }
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(embedding.meta.hidden_size);
    let mut out = vec![
        0u8;
        token_count
            .saturating_mul(embedding.meta.hidden_size)
            .saturating_mul(embedding.meta.dtype.size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let embedding_ptr = embedding.registered_device_ptr(ordinal)?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            embedding.meta.vocab_size,
            embedding.meta.hidden_size,
            embedding_ptr as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "hip-immutable-embedding-lookup-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_immutable_embedding_lookup_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embedding, indexes);
    Ok(None)
}

#[derive(Debug, Clone)]
struct HipImmutableOutputProjection {
    embedding: ImmutableEmbedding,
}

impl candle::CustomOp1 for HipImmutableOutputProjection {
    fn name(&self) -> &'static str {
        "hip-immutable-output-projection"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-immutable-output-projection has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        hidden: &candle::HipStorage,
        hidden_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !hidden_layout.is_contiguous() {
            candle::bail!("hip-immutable-output-projection requires contiguous hidden states")
        }
        let dims = hidden_layout.shape().dims();
        let hidden_size = *dims.last().ok_or_else(|| candle::Error::Msg("hidden state rank must be >= 1".to_string()))?;
        if hidden_size != self.embedding.meta.hidden_size {
            candle::bail!(
                "hip-immutable-output-projection hidden size mismatch got {} expected {}",
                hidden_size,
                self.embedding.meta.hidden_size
            )
        }
        let rows = hidden_layout.shape().elem_count() / hidden_size;
        let device = hidden.device().clone();
        let weight_ptr = self.embedding.registered_device_ptr(device.ordinal())?;
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().expect("validated non-empty dims") = self.embedding.meta.vocab_size;
        let out_shape = candle::Shape::from(out_dims);
        let elem_count = out_shape.elem_count();
        let mut output =
            vec![0u8; elem_count.saturating_mul(self.embedding.meta.dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
                hip::dtype_code(self.embedding.meta.dtype)?,
                device.ordinal(),
                rows,
                self.embedding.meta.hidden_size,
                self.embedding.meta.vocab_size,
                hidden.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
                weight_ptr,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(self.embedding.meta.dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn immutable_output_projection(embedding: &ImmutableEmbedding, hidden_states: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if embedding.meta.device.is_hip() && hidden_states.device().is_hip() {
        let hidden_states = hidden_states.contiguous()?;
        if let Some((output, shape)) = immutable_output_projection_host_buffer(embedding, &hidden_states)? {
            return hip_tensor_from_host_bytes(hidden_states.device(), embedding.meta.dtype, shape, output);
        }
        trace_hip_wrapper_fallback("immutable_output_projection", &hidden_states);
        return hidden_states.apply_op1_no_bwd(&HipImmutableOutputProjection {
            embedding: embedding.clone(),
        });
    }

    let fallback = embedding.ensure_fallback_embedding()?;
    let weight = fallback.embeddings().t()?;
    hidden_states.matmul(&weight)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn immutable_output_projection_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    if !(embedding.device().is_hip() && hidden_states.device().is_hip()) {
        return Ok(None);
    }
    let hidden_states = hidden_states.contiguous()?;
    let ordinal = match hidden_states.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (hidden_storage, hidden_layout) = hidden_states.storage_and_layout();
    let Storage::Hip(hidden_storage) = &*hidden_storage else {
        return Ok(None);
    };
    if !hidden_layout.is_contiguous() {
        return Ok(None);
    }
    let dims = hidden_layout.shape().dims();
    let hidden_size = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("hidden state rank must be >= 1".to_string()))?;
    if hidden_size != embedding.meta.hidden_size {
        return Ok(None);
    }
    let rows = hidden_layout.shape().elem_count() / hidden_size;
    let mut shape = dims.to_vec();
    *shape.last_mut().expect("validated non-empty dims") = embedding.meta.vocab_size;
    let mut out = vec![
        0u8;
        shape.iter().product::<usize>().saturating_mul(embedding.dtype().size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let weight_ptr = embedding.registered_device_ptr(ordinal)?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
            hip::dtype_code(embedding.dtype())?,
            ordinal,
            rows,
            embedding.meta.hidden_size,
            embedding.meta.vocab_size,
            hidden_storage.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
            weight_ptr,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "hip-immutable-output-projection-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn immutable_output_projection_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embedding, hidden_states);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipCausalMask {
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
}

impl candle::CustomOp1 for HipCausalMask {
    fn name(&self) -> &'static str {
        "hip-causal-mask"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-causal-mask has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        let device = storage.device().clone();
        let kv_len = self.tgt_len + self.seqlen_offset;
        let out_shape = candle::Shape::from((self.batch_size, 1usize, self.tgt_len, kv_len));
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_causal_mask(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.batch_size,
                self.tgt_len,
                self.seqlen_offset,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_causal_mask(device: &Device, dtype: DType, batch_size: usize, tgt_len: usize, seqlen_offset: usize) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        hip_causal_mask_host_buffer(device, dtype, batch_size, tgt_len, seqlen_offset)?
    {
        return hip_tensor_from_host_bytes(device, dtype, shape, output);
    }
    let seed = Tensor::zeros(1usize, dtype, device)?;
    trace_hip_wrapper_fallback("hip_causal_mask", &seed);
    seed.apply_op1_no_bwd(&HipCausalMask {
        batch_size,
        tgt_len,
        seqlen_offset,
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_causal_mask_host_buffer(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let ordinal = match device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(dtype) else {
        return Ok(None);
    };
    let kv_len = tgt_len + seqlen_offset;
    let shape = vec![batch_size, 1, tgt_len, kv_len];
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const std::ffi::c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_causal_mask(
            dtype_code,
            ordinal,
            batch_size,
            tgt_len,
            seqlen_offset,
            device_ptr as *mut std::ffi::c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-causal-mask-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_causal_mask_host_buffer(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (device, dtype, batch_size, tgt_len, seqlen_offset);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipCumsumLastDim {
    rows: usize,
    cols: usize,
}

impl candle::CustomOp1 for HipCumsumLastDim {
    fn name(&self) -> &'static str {
        "hip-cumsum-last-dim"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-cumsum-last-dim has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !layout.is_contiguous() {
            candle::bail!("hip-cumsum-last-dim requires contiguous input")
        }
        let dims = layout.shape().dims();
        let cols = *dims.last().ok_or_else(|| {
            candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
        })?;
        let rows = layout.shape().elem_count() / cols;
        if rows != self.rows || cols != self.cols {
            candle::bail!(
                "hip-cumsum-last-dim shape mismatch input={:?} expected_rows={} expected_cols={}",
                dims,
                self.rows,
                self.cols
            )
        }

        let device = storage.device().clone();
        let out_shape = layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.rows,
                self.cols,
                storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_cumsum_last_dim(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_cumsum_last_dim_host_buffer(&xs)? {
        return hip_tensor_from_host_bytes(xs.device(), xs.dtype(), shape, output);
    }
    let dims = xs.dims();
    let cols = *dims.last().ok_or_else(|| {
        candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = xs.elem_count() / cols;
    trace_hip_wrapper_fallback("hip_cumsum_last_dim", &xs);
    xs.apply_op1_no_bwd(&HipCumsumLastDim { rows, cols })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_cumsum_last_dim_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let cols = *shape.last().ok_or_else(|| {
        candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = layout.shape().elem_count() / cols;
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
            dtype_code,
            ordinal,
            rows,
            cols,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-cumsum-last-dim-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_cumsum_last_dim_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_exp_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_exp(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-exp-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_exp_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_recip_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_recip(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-recip-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_recip_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_sigmoid_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sigmoid(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-sigmoid-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_sigmoid_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_log_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_log(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-log-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_log_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_cast_host_buffer(
    xs: &Tensor,
    output_dtype: DType,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(input_dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let Ok(output_dtype_code) = hip::dtype_code(output_dtype) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(output_dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cast(
            input_dtype_code,
            output_dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-cast-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_cast_host_buffer(
    xs: &Tensor,
    output_dtype: DType,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, output_dtype);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn hip_binary_broadcast_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: i32,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    const MAX_RANK: usize = 8;

    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    let ordinal = match lhs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    match rhs.device().location() {
        DeviceLocation::Hip { gpu_id } if gpu_id == ordinal => {}
        _ => return Ok(None),
    }
    if lhs.dtype() != rhs.dtype() {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(lhs.dtype()) else {
        return Ok(None);
    };
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let rank = lhs_dims.len().max(rhs_dims.len());
    if rank == 0 || rank > MAX_RANK {
        return Ok(None);
    }

    let mut out_dims = [1i32; MAX_RANK];
    let mut lhs_strides = [0i32; MAX_RANK];
    let mut rhs_strides = [0i32; MAX_RANK];

    let mut lhs_contig = vec![0usize; lhs_dims.len()];
    let mut rhs_contig = vec![0usize; rhs_dims.len()];
    let mut stride = 1usize;
    for (i, dim) in lhs_dims.iter().enumerate().rev() {
        lhs_contig[i] = stride;
        stride = stride.saturating_mul(*dim);
    }
    stride = 1usize;
    for (i, dim) in rhs_dims.iter().enumerate().rev() {
        rhs_contig[i] = stride;
        stride = stride.saturating_mul(*dim);
    }

    let lhs_pad = rank - lhs_dims.len();
    let rhs_pad = rank - rhs_dims.len();
    let mut total_elems = 1usize;
    for dim in 0..rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_dims[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_dims[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_dims[dim] = i32::try_from(out_dim)
            .map_err(|_| candle::Error::Msg("broadcast dim overflow".into()))?;
        total_elems = total_elems.saturating_mul(out_dim);
        lhs_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
            0
        } else {
            i32::try_from(lhs_contig[dim - lhs_pad])
                .map_err(|_| candle::Error::Msg("lhs stride overflow".into()))?
        };
        rhs_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
            0
        } else {
            i32::try_from(rhs_contig[dim - rhs_pad])
                .map_err(|_| candle::Error::Msg("rhs stride overflow".into()))?
        };
    }

    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let Storage::Hip(lhs_storage) = &*lhs_storage else {
        return Ok(None);
    };
    let Storage::Hip(rhs_storage) = &*rhs_storage else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }

    let shape: Vec<usize> = out_dims[..rank].iter().map(|&d| d as usize).collect();
    let mut out = vec![0u8; total_elems.saturating_mul(lhs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_binary_broadcast(
            op,
            dtype_code,
            ordinal,
            i32::try_from(rank).map_err(|_| candle::Error::Msg("rank overflow".into()))?,
            total_elems,
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            lhs_strides.as_ptr(),
            rhs_strides.as_ptr(),
            out_dims.as_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-binary-broadcast-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn hip_binary_broadcast_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: i32,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (lhs, rhs, op);
    Ok(None)
}

pub(crate) fn hip_broadcast_add_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 0)
}

pub(crate) fn hip_broadcast_sub_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 1)
}

pub(crate) fn hip_broadcast_mul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 2)
}

pub(crate) fn hip_broadcast_div_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 3)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_matmul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    const MAX_BATCH_RANK: usize = 8;

    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    let ordinal = match lhs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    match rhs.device().location() {
        DeviceLocation::Hip { gpu_id } if gpu_id == ordinal => {}
        _ => return Ok(None),
    }
    if lhs.dtype() != rhs.dtype() {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(lhs.dtype()) else {
        return Ok(None);
    };

    let lhs_shape = lhs.dims();
    let rhs_shape = rhs.dims();
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
    if batch_rank > MAX_BATCH_RANK {
        return Ok(None);
    }
    let mut out_batch_dims = [1i32; MAX_BATCH_RANK];
    let mut lhs_batch_dims = [1i32; MAX_BATCH_RANK];
    let mut rhs_batch_dims = [1i32; MAX_BATCH_RANK];
    let lhs_pad = batch_rank.saturating_sub(lhs_batch.len());
    let rhs_pad = batch_rank.saturating_sub(rhs_batch.len());
    let mut batch_elems = 1usize;
    for dim in 0..batch_rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_batch[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_batch[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_batch_dims[dim] = i32::try_from(out_dim)
            .map_err(|_| candle::Error::Msg("matmul batch dim overflow".into()))?;
        lhs_batch_dims[dim] = i32::try_from(lhs_dim)
            .map_err(|_| candle::Error::Msg("matmul lhs batch dim overflow".into()))?;
        rhs_batch_dims[dim] = i32::try_from(rhs_dim)
            .map_err(|_| candle::Error::Msg("matmul rhs batch dim overflow".into()))?;
        batch_elems = batch_elems.saturating_mul(out_dim);
    }

    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let Storage::Hip(lhs_storage) = &*lhs_storage else {
        return Ok(None);
    };
    let Storage::Hip(rhs_storage) = &*rhs_storage else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }

    let mut out_shape = lhs_batch
        .iter()
        .zip(rhs_batch.iter())
        .map(|(a, b)| (*a).max(*b))
        .collect::<Vec<_>>();
    if out_shape.len() != batch_rank {
        out_shape = out_batch_dims[..batch_rank].iter().map(|&d| d as usize).collect();
    }
    if lhs_rank >= 2 {
        out_shape.push(m);
    }
    out_shape.push(n);

    let total_elems = batch_elems
        .saturating_mul(m)
        .saturating_mul(n);
    let mut out = vec![0u8; total_elems.saturating_mul(lhs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_batched_matmul(
            dtype_code,
            ordinal,
            i32::try_from(batch_rank).map_err(|_| candle::Error::Msg("matmul batch rank overflow".into()))?,
            batch_elems,
            i32::try_from(m).map_err(|_| candle::Error::Msg("matmul m overflow".into()))?,
            i32::try_from(n).map_err(|_| candle::Error::Msg("matmul n overflow".into()))?,
            i32::try_from(lhs_k).map_err(|_| candle::Error::Msg("matmul k overflow".into()))?,
            lhs_batch_dims.as_ptr(),
            rhs_batch_dims.as_ptr(),
            out_batch_dims.as_ptr(),
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-matmul-host-buffer", status));
    }
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_matmul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (lhs, rhs);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_mul_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_mul_scalar(
            dtype_code,
            ordinal,
            total_elems,
            scalar as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-mul-scalar-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_mul_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, scalar);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn hip_reduce_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
    sum: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    if dim >= shape.len() {
        return Ok(None);
    }
    let outer = shape[..dim].iter().product::<usize>().max(1);
    let reduce = shape[dim];
    let inner = shape[dim + 1..].iter().product::<usize>().max(1);
    let mut out_shape = shape.clone();
    out_shape[dim] = 1;
    let mut out = vec![0u8; out_shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_reduce_keepdim(
            dtype_code,
            ordinal,
            outer,
            reduce,
            inner,
            if sum { 1 } else { 0 },
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-reduce-keepdim-host-buffer", status));
    }
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn hip_reduce_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
    sum: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, dim, sum);
    Ok(None)
}

pub(crate) fn hip_sum_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_reduce_keepdim_host_buffer(xs, dim, true)
}

pub(crate) fn hip_max_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_reduce_keepdim_host_buffer(xs, dim, false)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_add_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_add_scalar(
            dtype_code,
            ordinal,
            total_elems,
            scalar as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-add-scalar-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_add_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, scalar);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_sqrt_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sqrt(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-sqrt-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_sqrt_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipL2Norm {
    n_rows: usize,
    n_cols: usize,
    eps: f32,
}

impl candle::CustomOp1 for HipL2Norm {
    fn name(&self) -> &'static str {
        "dotcache-hip-l2norm"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("dotcache-hip-l2norm has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !layout.is_contiguous() {
            candle::bail!("dotcache-hip-l2norm requires contiguous input")
        }
        let dims = layout.shape().dims();
        let n_cols = *dims.last().ok_or_else(|| {
            candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into())
        })?;
        let n_rows = layout.shape().elem_count() / n_cols;
        if n_rows != self.n_rows || n_cols != self.n_cols {
            candle::bail!(
                "dotcache-hip-l2norm shape mismatch input={:?} expected_rows={} expected_cols={}",
                layout.shape().dims(),
                self.n_rows,
                self.n_cols
            )
        }

        let device = storage.device().clone();
        let out_shape = layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_l2norm(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.n_rows,
                self.n_cols,
                self.eps,
                storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_l2norm_host_buffer(&xs, eps)? {
        return hip_tensor_from_host_bytes(xs.device(), xs.dtype(), shape, output);
    }
    let dims = xs.dims();
    let n_cols = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = xs.elem_count() / n_cols;
    trace_hip_wrapper_fallback("hip_l2norm", &xs);
    xs.apply_op1_no_bwd(&HipL2Norm {
        n_rows,
        n_cols,
        eps: eps as f32,
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_l2norm_host_buffer(
    xs: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
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
        .ok_or_else(|| candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = layout.shape().elem_count() / n_cols;
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_l2norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-l2norm-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_l2norm_host_buffer(
    xs: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, eps);
    Ok(None)
}

pub(super) fn softplus(xs: &Tensor) -> Result<Tensor> {
    ((xs.exp()? + 1.0)?).log()
}

#[derive(Debug, Clone, Copy)]
struct HipValueDecay {
    total_elems: usize,
    num_heads: usize,
}

impl candle::CustomOp3 for HipValueDecay {
    fn name(&self) -> &'static str {
        "dotcache-hip-value-decay"
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
        candle::bail!("dotcache-hip-value-decay has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        a: &candle::HipStorage,
        a_layout: &candle::Layout,
        dt_bias: &candle::HipStorage,
        dt_bias_layout: &candle::Layout,
        a_log_exp: &candle::HipStorage,
        a_log_exp_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(a_layout.is_contiguous()
            && dt_bias_layout.is_contiguous()
            && a_log_exp_layout.is_contiguous())
        {
            candle::bail!("dotcache-hip-value-decay requires contiguous inputs")
        }
        if a.dtype() != dt_bias.dtype() || a.dtype() != a_log_exp.dtype() {
            candle::bail!(
                "dotcache-hip-value-decay requires matching dtypes, got a={:?} dt_bias={:?} a_log_exp={:?}",
                a.dtype(),
                dt_bias.dtype(),
                a_log_exp.dtype()
            )
        }

        let a_elems = a_layout.shape().elem_count();
        let dt_bias_elems = dt_bias_layout.shape().elem_count();
        let a_log_exp_elems = a_log_exp_layout.shape().elem_count();
        if a_elems != self.total_elems
            || dt_bias_elems != self.num_heads
            || a_log_exp_elems != self.num_heads
        {
            candle::bail!(
                "dotcache-hip-value-decay shape mismatch a={:?} dt_bias={:?} a_log_exp={:?} expected_total={} expected_heads={}",
                a_layout.shape().dims(),
                dt_bias_layout.shape().dims(),
                a_log_exp_layout.shape().dims(),
                self.total_elems,
                self.num_heads
            )
        }

        let device = a.device().clone();
        let out_shape = a_layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(a.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_value_decay(
                hip::dtype_code(a.dtype())?,
                device.ordinal(),
                self.total_elems,
                self.num_heads,
                a.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
                dt_bias.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
                a_log_exp.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                    as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(a.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_value_decay(a: &Tensor, dt_bias: &Tensor, a_log_exp: &Tensor) -> Result<Tensor> {
    let a = a.contiguous()?;
    let target_dtype = a.dtype();
    let dt_bias = dt_bias.contiguous()?;
    let dt_bias = if dt_bias.dtype() == target_dtype {
        dt_bias
    } else {
        dt_bias.to_dtype(target_dtype)?
    };
    let a_log_exp = a_log_exp.contiguous()?;
    let a_log_exp = if a_log_exp.dtype() == target_dtype {
        a_log_exp
    } else {
        a_log_exp.to_dtype(target_dtype)?
    };
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_value_decay_host_buffer(&a, &dt_bias, &a_log_exp)? {
        return hip_tensor_from_host_bytes(a.device(), a.dtype(), shape, output);
    }
    let total_elems = a.elem_count();
    let num_heads = dt_bias.elem_count();
    trace_hip_wrapper_fallback("hip_value_decay", &a);
    a.apply_op3_no_bwd(
        &dt_bias,
        &a_log_exp,
        &HipValueDecay {
            total_elems,
            num_heads,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_value_decay_host_buffer(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let a = a.contiguous()?;
    let target_dtype = a.dtype();
    let dt_bias = dt_bias.contiguous()?;
    let dt_bias = if dt_bias.dtype() == target_dtype {
        dt_bias
    } else {
        dt_bias.to_dtype(target_dtype)?
    };
    let a_log_exp = a_log_exp.contiguous()?;
    let a_log_exp = if a_log_exp.dtype() == target_dtype {
        a_log_exp
    } else {
        a_log_exp.to_dtype(target_dtype)?
    };
    let ordinal = match a.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(dt_bias.device().same_device(a.device()) && a_log_exp.device().same_device(a.device())) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(target_dtype) else {
        return Ok(None);
    };
    let (a_storage, a_layout) = a.storage_and_layout();
    let (dt_bias_storage, dt_bias_layout) = dt_bias.storage_and_layout();
    let (a_log_exp_storage, a_log_exp_layout) = a_log_exp.storage_and_layout();
    let (Storage::Hip(a_storage), Storage::Hip(dt_bias_storage), Storage::Hip(a_log_exp_storage)) =
        (&*a_storage, &*dt_bias_storage, &*a_log_exp_storage)
    else {
        return Ok(None);
    };
    if !(a_layout.is_contiguous() && dt_bias_layout.is_contiguous() && a_log_exp_layout.is_contiguous()) {
        return Ok(None);
    }
    let total_elems = a_layout.shape().elem_count();
    let num_heads = dt_bias_layout.shape().elem_count();
    if a_log_exp_layout.shape().elem_count() != num_heads {
        return Ok(None);
    }
    let shape = a_layout.shape().dims().to_vec();
    let mut out = vec![0u8; total_elems.saturating_mul(target_dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_value_decay(
            dtype_code,
            ordinal,
            total_elems,
            num_heads,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-value-decay-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_value_decay_host_buffer(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (a, dt_bias, a_log_exp);
    Ok(None)
}

fn linear_attention_compute_dtype(device: &Device, input_dtype: DType) -> DType {
    match (device.location(), input_dtype) {
        (DeviceLocation::Metal { .. }, DType::F16 | DType::BF16) => input_dtype,
        _ => DType::F32,
    }
}

fn recommended_metal_linear_chunk_size(sequence_length: usize) -> usize {
    match sequence_length {
        0..=1024 => 16,
        _ => 24,
    }
}

fn recommended_hip_linear_chunk_size(sequence_length: usize) -> usize {
    match sequence_length {
        0..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

fn use_hip_short_linear_chunks() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_HIP_SHORT_LINEAR_CHUNKS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn debug_linear_chunk_choice(sequence_length: usize, chunk_size: usize) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if std::env::var("CANDLE_QWEN35_DEBUG_CHUNK").is_ok() && !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "qwen3.5 linear chunk choice: sequence_length={} chunk_size={}",
            sequence_length, chunk_size
        );
    }
}

fn linear_attention_chunk_size(device: &Device, sequence_length: usize) -> usize {
    if let Ok(raw_value) = std::env::var("CANDLE_QWEN35_LINEAR_CHUNK_SIZE") {
        if let Ok(parsed) = raw_value.trim().parse::<usize>() {
            if parsed > 0 {
                debug_linear_chunk_choice(sequence_length, parsed);
                return parsed;
            }
        }
    }
    let chunk_size = match device.location() {
        DeviceLocation::Metal { .. } => recommended_metal_linear_chunk_size(sequence_length),
        DeviceLocation::Hip { .. } if use_hip_short_linear_chunks() => {
            recommended_hip_linear_chunk_size(sequence_length)
        }
        _ => 64,
    };
    debug_linear_chunk_choice(sequence_length, chunk_size);
    chunk_size
}

fn use_delta_state_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= min_sequence
        && matches!(
            std::env::var("CANDLE_QWEN35_DELTA_STATE_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

fn use_delta_state_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

fn use_delta_chunk_fused_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

fn use_delta_full_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Hip { .. } => match std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL") {
            Ok(raw) => !matches!(
                raw.trim(),
                "0" | "false" | "FALSE" | "no" | "NO"
            ),
            Err(_) => true,
        },
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

fn use_hip_exact_multi_chunk_full_scan_prefill(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    num_chunks: usize,
    chunk_size: usize,
) -> bool {
    if !matches!(device.location(), DeviceLocation::Hip { .. }) {
        return false;
    }
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && num_chunks > 1
        && num_chunks <= 4
        && sequence_length > chunk_size
        && chunk_size <= 64)
    {
        return false;
    }

    match std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL") {
        Ok(raw) => !matches!(raw.trim(), "0" | "false" | "FALSE" | "no" | "NO"),
        Err(_) => true,
    }
}

fn use_delta_recurrent_prefill_kernel(device: &Device, sequence_length: usize) -> bool {
    sequence_length >= 4096
        && match device.location() {
            DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => matches!(
                std::env::var("CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            ),
            _ => false,
        }
}

fn use_delta_chunk_step_kernel(
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
            match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL") {
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
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
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

fn use_delta_chunk_windowed_kernel(
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

fn use_linear_prefill_packed_kernel(device: &Device, sequence_length: usize) -> bool {
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

fn use_hip_short_linear_prefill_recurrent(device: &Device, sequence_length: usize) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && sequence_length > 1
        && sequence_length <= linear_attention_chunk_size(device, sequence_length)
        && matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_SHORT_LINEAR_PREFILL_RECURRENT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

fn use_hip_combined_linear_prefill(device: &Device, sequence_length: usize) -> bool {
    matches!(device.location(), DeviceLocation::Hip { .. })
        && sequence_length > 1
        && !matches!(
            std::env::var("DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_PREFILL").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
}

fn use_hip_combined_linear_decode(device: &Device, sequence_length: usize) -> bool {
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

fn use_hip_chunk_single_prefill_kernel(
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

fn use_hip_multi_chunk_scan_prefill_kernel(
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

fn use_full_attention_prefill_megakernel(
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

fn use_full_attention_decode_megakernel(
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

fn use_delta_chunk_scan_kernel(
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

fn delta_chunk_step_raw(
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
fn delta_chunk_step_raw_host_buffer(
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

fn delta_chunk_step_windowed_raw(
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
fn delta_chunk_step_windowed_raw_host_buffer(
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
enum DeltaNetScanMode {
    Flat3d,
    HoistedDecays,
    PrebatchedLocal,
    TorchLike,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeltaNetExecutionPolicy {
    scan_mode: DeltaNetScanMode,
    use_flattened_solve: bool,
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

fn delta_net_execution_policy(
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

fn parse_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn full_attention_blockwise_tiles(
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

fn full_attention_sdpa_q_block(device: &Device, q_len: usize) -> Option<usize> {
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

fn use_full_attention_torchlike_eager(device: &Device) -> bool {
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(
            std::env::var("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        )
}

fn delta_net_compute_dtype(scan_mode: DeltaNetScanMode, initial_dtype: DType) -> DType {
    match scan_mode {
        DeltaNetScanMode::TorchLike => DType::F32,
        _ => initial_dtype,
    }
}

#[derive(Debug, Clone)]
struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attention_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(StateBuffer, StateBuffer)>,
}

impl FullAttention {
    fn cache_state(&self) -> FullAttentionCacheState {
        FullAttentionCacheState {
            kv_cache: self.kv_cache.clone(),
        }
    }

    fn restore_cache_state(&mut self, state: &FullAttentionCacheState) {
        self.kv_cache = state.kv_cache.clone();
    }

    fn causal_block_mask(
        device: &Device,
        q_start: usize,
        q_len: usize,
        k_start: usize,
        k_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let query_base = seqlen_offset + q_start;
        let mut mask = Vec::with_capacity(q_len * k_len);
        for q_idx in 0..q_len {
            let q_abs = query_base + q_idx;
            for k_idx in 0..k_len {
                let k_abs = k_start + k_idx;
                mask.push(if k_abs > q_abs {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        Tensor::from_slice(&mask, (q_len, k_len), device)?.reshape((1, 1, q_len, k_len))
    }

    fn blockwise_attention_profiled(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        scale: f64,
        seqlen_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states_f.device();
        let (b_sz, q_heads, q_len, _head_dim) = query_states_f.dims4()?;
        let (_, _, kv_len, value_dim) = value_states_f.dims4()?;
        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let q_block = query_states_f.narrow(2, q_start, q_block_len)?;
            let mut running_max =
                Tensor::full(f32::NEG_INFINITY, (b_sz, q_heads, q_block_len, 1), device)?;
            let mut running_sum =
                Tensor::zeros((b_sz, q_heads, q_block_len, 1), DType::F32, device)?;
            let mut running_acc =
                Tensor::zeros((b_sz, q_heads, q_block_len, value_dim), DType::F32, device)?;

            for k_start in (0..kv_len).step_by(k_block_size) {
                let k_block_len = (kv_len - k_start).min(k_block_size);
                let q_abs_min = seqlen_offset + q_start;
                let q_abs_max = q_abs_min + q_block_len - 1;
                let k_abs_min = k_start;
                let k_abs_max = k_start + k_block_len - 1;
                if k_abs_min > q_abs_max {
                    break;
                }

                let score_start = profile_start(device)?;
                let k_block = key_states_f.narrow(2, k_start, k_block_len)?;
                let mut scores = (q_block.matmul(&k_block.transpose(2, 3)?)? * scale)?;
                let needs_partial_mask = !(k_abs_max <= q_abs_min);
                if needs_partial_mask {
                    let mask = Self::causal_block_mask(
                        device,
                        q_start,
                        q_block_len,
                        k_start,
                        k_block_len,
                        seqlen_offset,
                    )?;
                    scores = scores.broadcast_add(&mask)?;
                }
                profile.attention_score_millis += profile_elapsed(score_start, device)?;

                let softmax_start = profile_start(device)?;
                let block_max = scores.max_keepdim(D::Minus1)?;
                let new_max = running_max.maximum(&block_max)?;
                let prev_scale = running_max.broadcast_sub(&new_max)?.exp()?;
                let exp_scores = scores.broadcast_sub(&new_max)?.exp()?;
                let new_sum = running_sum
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.sum_keepdim(D::Minus1)?)?;
                profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

                let mix_start = profile_start(device)?;
                let v_block = value_states_f.narrow(2, k_start, k_block_len)?;
                let new_acc = running_acc
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.matmul(&v_block)?)?;
                running_max = new_max;
                running_sum = new_sum;
                running_acc = new_acc;
                profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            }

            q_outputs.push(running_acc.broadcast_div(&running_sum)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn blockwise_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f64,
        seqlen_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.blockwise_attention_profiled(
            query_states_f,
            key_states_f,
            value_states_f,
            scale,
            seqlen_offset,
            q_block_size,
            k_block_size,
            profile,
        )?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn sdpa_chunked_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        scale: f32,
        seqlen_offset: usize,
        q_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let (b_sz, q_heads, q_len, _) = query_states.dims4()?;
        let (_, kv_heads, kv_len, _) = key_states.dims4()?;
        let base_head_chunk = parse_usize_env("CANDLE_QWEN35_FULL_SDPA_Q_HEADS")
            .unwrap_or(q_heads)
            .min(q_heads)
            .max(self.num_kv_groups);
        let q_head_chunk =
            ((base_head_chunk / self.num_kv_groups).max(1) * self.num_kv_groups).min(q_heads);
        let kv_head_chunk = q_head_chunk / self.num_kv_groups;
        if kv_head_chunk == 0 || kv_head_chunk > kv_heads {
            candle::bail!("invalid sdpa q_head chunk for grouped attention")
        }

        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let mask_base =
                Self::causal_block_mask(device, q_start, q_block_len, 0, kv_len, seqlen_offset)?
                    .to_dtype(query_states.dtype())?;
            let mut head_outputs = Vec::with_capacity(q_heads.div_ceil(q_head_chunk));
            for q_head_start in (0..q_heads).step_by(q_head_chunk) {
                let q_head_len = (q_heads - q_head_start).min(q_head_chunk);
                let kv_head_start = q_head_start / self.num_kv_groups;
                let kv_head_len = q_head_len / self.num_kv_groups;
                let q_chunk = query_states
                    .narrow(1, q_head_start, q_head_len)?
                    .narrow(2, q_start, q_block_len)?
                    .contiguous()?;
                let k_chunk = key_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let v_chunk = value_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let mask = mask_base.broadcast_as((b_sz, q_head_len, q_block_len, kv_len))?;
                let fused_start = profile_start(device)?;
                let output =
                    ops::sdpa(&q_chunk, &k_chunk, &v_chunk, Some(&mask), false, scale, 1.0)?;
                let fused_elapsed = profile_elapsed(fused_start, device)?;
                profile.attention_mix_millis += fused_elapsed;
                head_outputs.push(output);
            }
            q_outputs.push(Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn sdpa_chunked_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f32,
        seqlen_offset: usize,
        q_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.sdpa_chunked_attention_profiled(
            query_states,
            key_states,
            value_states,
            scale,
            seqlen_offset,
            q_block_size,
            profile,
        )?
        .to_dtype(DType::F32)?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn grouped_torchlike_eager_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let compute_dtype = query_states.dtype();
        let (_, q_heads, _, _) = query_states.dims4()?;
        let (_, kv_heads, _, _) = key_states.dims4()?;
        let mask_start = profile_start(device)?;
        let mask = attention_mask
            .map(|mask| mask.to_dtype(compute_dtype))
            .transpose()?;
        profile.full_attention_mask_prepare_millis += profile_elapsed(mask_start, device)?;
        let mut outputs = Vec::with_capacity(kv_heads);

        for kv_head_idx in 0..kv_heads {
            let q_head_start = kv_head_idx * self.num_kv_groups;
            let q_head_len = (q_heads - q_head_start).min(self.num_kv_groups);
            if q_head_len == 0 {
                break;
            }
            let query_layout_start = profile_start(device)?;
            let q_chunk = query_states
                .narrow(1, q_head_start, q_head_len)?
                .contiguous()?;
            profile.full_attention_input_layout_millis +=
                profile_elapsed(query_layout_start, device)?;
            let kv_len = key_states.dim(2)?;
            let value_dim = value_states.dim(3)?;
            let kv_materialize_start = profile_start(device)?;
            let k_chunk = key_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                self.head_dim,
            ))?;
            let v_chunk = value_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                value_dim,
            ))?;
            let key_states_t = k_chunk.transpose(2, 3)?.contiguous()?;
            profile.full_attention_kv_materialize_millis +=
                profile_elapsed(kv_materialize_start, device)?;

            let score_start = profile_start(device)?;
            let mut attn_weights = (q_chunk.matmul(&key_states_t)? * scale)?;
            if let Some(mask) = &mask {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }
            profile.attention_score_millis += profile_elapsed(score_start, device)?;

            let softmax_start = profile_start(device)?;
            let attn_weights = ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
                .to_dtype(compute_dtype)?;
            profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

            let mix_start = profile_start(device)?;
            let attn_output = attn_weights.matmul(&v_chunk)?;
            profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            outputs.push(attn_output);
        }

        let collect_start = profile_start(device)?;
        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?;
        profile.full_attention_output_collect_millis += profile_elapsed(collect_start, device)?;
        Ok(output)
    }

    fn grouped_torchlike_eager_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        attention_mask: Option<&Tensor>,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        scale: f64,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let attn_output = self.grouped_torchlike_eager_attention_profiled(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scale,
            profile,
        )?
        .to_dtype(DType::F32)?;
        backend.prepare_full_attention_output(
            &attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    fn external_attention_prepared_profiled(
        &self,
        backend: &dyn Qwen35BackendBufferApi,
        handler: &mut dyn ExternalFullAttention,
        layer_id: usize,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        gate: &StateBuffer,
        b_sz: usize,
        q_len: usize,
        output_dtype: DType,
        seqlen_offset: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<StateBuffer> {
        let device = query_states.device();
        let external_started = profile_start(device)?;
        let external = handler.forward(
            layer_id,
            query_states,
            key_states,
            value_states,
            self.num_kv_groups,
            self.head_dim,
            seqlen_offset,
        )?;
        let external_elapsed = profile_elapsed(external_started, device)?;
        profile.add_assign(&external.profile);
        if external.profile.full_attention_millis == 0.0 {
            profile.full_attention_millis += external_elapsed;
        }
        backend.prepare_full_attention_output(
            &external.attn_output,
            gate,
            b_sz,
            q_len,
            self.attention_size,
            output_dtype,
        )
    }

    #[cfg(any(feature = "hf", test))]
    fn new(cfg: &TextConfig, rotary_emb: Arc<RotaryEmbedding>, vb: WeightBuilder) -> Result<Self> {
        let q_proj = linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim * 2,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attention_size: cfg.num_attention_heads * cfg.head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn from_prepared(
        cfg: &TextConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        source: &PreparedTensorSource,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: prepared_linear_b(&source.pp("q_proj"), cfg.attention_bias)?,
            k_proj: prepared_linear_b(&source.pp("k_proj"), cfg.attention_bias)?,
            v_proj: prepared_linear_b(&source.pp("v_proj"), cfg.attention_bias)?,
            o_proj: prepared_linear_b(&source.pp("o_proj"), cfg.attention_bias)?,
            q_norm: Qwen35RmsNorm::from_prepared(cfg.rms_norm_eps, &source.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::from_prepared(cfg.rms_norm_eps, &source.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attention_size: cfg.num_attention_heads * cfg.head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward_profiled_with_external(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_id: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;
        let qkv_start = profile_start(device)?;
        let q_and_gate = backend.tensor_to_buffer(self.q_proj.forward(xs)?)?;
        let k_proj = backend.tensor_to_buffer(self.k_proj.forward(xs)?)?;
        let v_proj = backend.tensor_to_buffer(self.v_proj.forward(xs)?)?;
        let (query_states, gate, key_states, value_states) = backend.prepare_full_attention_inputs(
            &q_and_gate,
            &k_proj,
            &v_proj,
            b_sz,
            q_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.weight(),
            self.k_norm.eps(),
        )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&query_states, &key_states, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let kv_append_start = profile_start(device)?;
        let appended_kv = if external_full_attention.is_none() {
            Some(backend.append_full_attention_kv_buffers(
                self.kv_cache.as_ref().map(|(k, _)| k),
                self.kv_cache.as_ref().map(|(_, v)| v),
                key_states.tensor(),
                value_states.tensor(),
            )?)
        } else {
            None
        };
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) = if let Some((key_states, value_states)) =
            appended_kv.as_ref()
        {
            backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                &query_states,
                key_states,
                value_states,
            )?
        } else {
            backend.prepare_full_attention_kernel_inputs(
                query_states.tensor(),
                key_states.tensor(),
                value_states.tensor(),
            )?
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let kv_len = key_states.dim(2)?;
        let prepared_attn_output = if let Some(handler) = external_full_attention.as_deref_mut() {
            self.external_attention_prepared_profiled(
                backend,
                handler,
                layer_id,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.dtype(),
                seqlen_offset,
                &mut profile,
            )?
        } else if use_full_attention_decode_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend
                .full_attention_decode(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.dtype(),
            )?
        } else if use_full_attention_prefill_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend
                .full_attention_prefill(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            if matches!(device.location(), DeviceLocation::Hip { .. })
                && debug_full_prefill_kernel_compare_enabled()
            {
                let key_states_ref =
                    repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
                let value_states_ref =
                    repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
                let query_states_f = query_states.to_dtype(DType::F32)?;
                let key_states_f = key_states_ref.to_dtype(DType::F32)?;
                let value_states_f = value_states_ref.to_dtype(DType::F32)?;
                let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
                let mut attn_weights = (query_states_f.matmul(&key_states_t)? * scale)?;
                if let Some(mask) = attention_mask {
                    attn_weights = attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?;
                }
                let attn_weights = ops::softmax_last_dim(&attn_weights)?;
                let fallback = attn_weights.matmul(&value_states_f)?;
                eprintln!(
                    "hip full-prefill layer={} q_len={} kv_len={} dtype={:?} max_delta={:.6}",
                    layer_id,
                    q_len,
                    kv_len,
                    query_states.dtype(),
                    max_abs_delta(&output.tensor(), &fallback)?,
                );
            }
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.dtype(),
            )?
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.dtype(),
                scale,
                &mut profile,
            )?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.dtype(),
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.dtype(),
                scale,
                &mut profile,
            )?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let (query_states_f, key_states_f, value_states_f) = backend
                .materialize_full_attention_dense_inputs(
                    &query_states,
                    &key_states,
                    &value_states,
                    self.num_kv_groups,
                )?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_prepared_profiled(
                    backend,
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    &gate,
                    b_sz,
                    q_len,
                    xs.dtype(),
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    attention_mask,
                    scale,
                    &gate,
                    b_sz,
                    q_len,
                    self.attention_size,
                    xs.dtype(),
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            }
        };

        let output_reshape_start = profile_start(device)?;
        let attn_output = prepared_attn_output;
        profile.full_attention_output_reshape_millis +=
            profile_elapsed(output_reshape_start, device)?;
        if external_full_attention.is_none() {
            self.kv_cache = appended_kv;
        } else {
            self.kv_cache = None;
        }
        let gate_start = profile_start(device)?;
        profile.full_attention_gate_millis += profile_elapsed(gate_start, device)?;
        let output_start = profile_start(device)?;
        let output = self.o_proj.forward_buffer(&attn_output)?.clone_tensor_as(xs.dtype())?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    fn forward_profiled_with_external_buffer(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_id: usize,
        external_full_attention: &mut Option<&mut dyn ExternalFullAttention>,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = xs.device();
        let backend = backend_buffer_api::for_device(device);
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv_start = profile_start(device)?;
        let q_and_gate = self.q_proj.forward_buffer(xs)?;
        let k_proj = self.k_proj.forward_buffer(xs)?;
        let v_proj = self.v_proj.forward_buffer(xs)?;
        let (query_states, gate, key_states, value_states) = backend.prepare_full_attention_inputs(
            &q_and_gate,
            &k_proj,
            &v_proj,
            b_sz,
            q_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.weight(),
            self.k_norm.eps(),
        )?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply_buffer(&query_states, &key_states, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let backend = backend_buffer_api::for_device(device);
        let kv_append_start = profile_start(device)?;
        let appended_kv = if external_full_attention.is_none() {
            Some(backend.append_full_attention_kv_buffers(
                self.kv_cache.as_ref().map(|(k, _)| k),
                self.kv_cache.as_ref().map(|(_, v)| v),
                key_states.tensor(),
                value_states.tensor(),
            )?)
        } else {
            None
        };
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let (query_states, key_states, value_states) = if let Some((ref key_states, ref value_states)) = appended_kv {
            backend.prepare_full_attention_kernel_inputs_with_buffer_kv(
                &query_states,
                key_states,
                value_states,
            )?
        } else {
            backend.prepare_full_attention_kernel_inputs(
                query_states.tensor(),
                key_states.tensor(),
                value_states.tensor(),
            )?
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let kv_len = key_states.dim(2)?;
        let prepared_attn_output = if let Some(handler) = external_full_attention.as_deref_mut() {
            self.external_attention_prepared_profiled(
                backend,
                handler,
                layer_id,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                seqlen_offset,
                &mut profile,
            )?
        } else if use_full_attention_decode_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend.full_attention_decode(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.tensor().dtype(),
            )?
        } else if use_full_attention_prefill_megakernel(
            device,
            q_len,
            kv_len,
            self.head_dim,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = backend.full_attention_prefill(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            backend.prepare_full_attention_output_buffer(
                &output,
                &gate,
                b_sz,
                q_len,
                self.attention_size,
                xs.tensor().dtype(),
            )?
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_prepared_profiled(
                backend,
                &query_states,
                &key_states,
                &value_states,
                &gate,
                attention_mask,
                b_sz,
                q_len,
                xs.tensor().dtype(),
                scale,
                &mut profile,
            )?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let key_states = repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
            let value_states = repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;

            let query_states_f = query_states.to_dtype(DType::F32)?;
            let key_states_f = key_states.to_dtype(DType::F32)?;
            let value_states_f = value_states.to_dtype(DType::F32)?;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_prepared_profiled(
                    backend,
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    &gate,
                    b_sz,
                    q_len,
                    xs.tensor().dtype(),
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let score_start = profile_start(device)?;
                let attn_output = backend.dense_full_attention_fallback_buffer(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    attention_mask,
                    scale,
                    &gate,
                    b_sz,
                    q_len,
                    self.attention_size,
                    xs.tensor().dtype(),
                )?;
                profile.attention_score_millis += profile_elapsed(score_start, device)?;
                attn_output
            }
        };

        let output_reshape_start = profile_start(device)?;
        let attn_output = prepared_attn_output;
        profile.full_attention_output_reshape_millis +=
            profile_elapsed(output_reshape_start, device)?;
        if external_full_attention.is_none() {
            self.kv_cache = appended_kv;
        } else {
            self.kv_cache = None;
        }
        let gate_start = profile_start(device)?;
        profile.full_attention_gate_millis += profile_elapsed(gate_start, device)?;
        let output_start = profile_start(device)?;
        let output = self.o_proj.forward_buffer(&attn_output)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    fn forward_profiled(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let mut no_external = None;
        self.forward_profiled_with_external(
            xs,
            attention_mask,
            seqlen_offset,
            usize::MAX,
            &mut no_external,
        )
    }

    fn forward_profiled_buffer(
        &mut self,
        xs: &StateBuffer,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let mut no_external = None;
        self.forward_profiled_with_external_buffer(
            xs,
            attention_mask,
            seqlen_offset,
            usize::MAX,
            &mut no_external,
        )
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_profiled(xs, attention_mask, seqlen_offset)
            .map(|(output, _)| output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: LinearSource,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d_raw_weight: Option<Tensor>,
    conv1d_weight_squeezed: Option<Tensor>,
    dt_bias_raw: Option<Tensor>,
    a_log_raw: Option<Tensor>,
    dt_bias_prepared: Option<Tensor>,
    a_log_exp_prepared: Option<Tensor>,
    norm: Qwen35RmsNormGated,
    out_proj: Linear,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_state: Option<StateBuffer>,
    recurrent_state: Option<StateBuffer>,
    chunk_cache: Option<LinearChunkCache>,
    value_cache: Option<LinearValueCache>,
}

#[derive(Debug, Clone)]
struct LinearChunkCache {
    chunk_size: usize,
    dtype: DType,
    device_location: DeviceLocation,
    lower: Tensor,
    eye: Tensor,
    strict_lower: Tensor,
    lower_2d: Tensor,
}

#[derive(Debug, Clone)]
struct LinearValueCache {
    dtype: DType,
    device_location: DeviceLocation,
    dt_bias: Tensor,
    a_log_exp: Tensor,
}

impl GatedDeltaNet {
    fn finalize_linear_output_buffer(
        &self,
        hidden_dtype: DType,
        batch_size: usize,
        seq_len: usize,
        z: &StateBuffer,
        core_attn_out: &Tensor,
    ) -> Result<StateBuffer> {
        let backend = backend_buffer_api::for_device(z.device());
        let core_attn_out = self
            .norm
            .forward_buffer(
                &backend.reshape_tensor_to_buffer(
                    core_attn_out,
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
                &backend.reshape_tensor_to_buffer(
                    z.tensor(),
                    &[batch_size * seq_len * self.num_v_heads, self.head_v_dim],
                )?,
            )?
            .tensor()
            .reshape((batch_size, seq_len, self.value_dim))?;
        let core_attn_out = if core_attn_out.dtype() == hidden_dtype {
            core_attn_out
        } else {
            core_attn_out.to_dtype(hidden_dtype)?
        };
        self.out_proj.forward_buffer(&backend.reshape_tensor_to_buffer(
            &core_attn_out,
            &[batch_size, seq_len, self.value_dim],
        )?)
    }

    fn cache_state(&self) -> LinearAttentionCacheState {
        LinearAttentionCacheState {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }

    fn restore_cache_state(&mut self, state: &LinearAttentionCacheState) {
        self.conv_state = state.conv_state.clone();
        self.recurrent_state = state.recurrent_state.clone();
    }

    fn deferred_linear_count(&self) -> usize {
        usize::from(self.in_proj_qkv.is_deferred())
    }

    #[cfg(any(feature = "hf", test))]
    fn new(cfg: &TextConfig, vb: WeightBuilder) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d_raw_weight =
            vb.pp("conv1d")
                .get((conv_dim, 1, cfg.linear_conv_kernel_dim), "weight")?;
        Ok(Self {
            in_proj_qkv: LinearSource::Materialized(linear_no_bias(
                cfg.hidden_size,
                conv_dim,
                vb.pp("in_proj_qkv"),
            )?),
            in_proj_z: linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?,
            in_proj_a: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?,
            conv1d_raw_weight: Some(conv1d_raw_weight),
            conv1d_weight_squeezed: None,
            dt_bias_raw: Some(vb.get(cfg.linear_num_value_heads, "dt_bias")?),
            a_log_raw: Some(vb.get(cfg.linear_num_value_heads, "A_log")?),
            dt_bias_prepared: None,
            a_log_exp_prepared: None,
            norm: Qwen35RmsNormGated::new(
                cfg.linear_value_head_dim,
                cfg.rms_norm_eps,
                vb.pp("norm"),
            )?,
            out_proj: linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: None,
            recurrent_state: None,
            chunk_cache: None,
            value_cache: None,
        })
    }

    fn from_prepared(cfg: &TextConfig, source: &PreparedTensorSource) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let dt_bias_prepared = source
            .contains_tensor("dt_bias.__dotcache_head_bias_reshaped")
            .then(|| source.get("dt_bias.__dotcache_head_bias_reshaped"))
            .transpose()?;
        let a_log_exp_prepared = source
            .contains_tensor("A_log.__dotcache_head_exp_reshaped")
            .then(|| source.get("A_log.__dotcache_head_exp_reshaped"))
            .transpose()?;
        let conv1d_weight_squeezed = source
            .contains_tensor("conv1d.weight.__dotcache_depthwise_squeezed")
            .then(|| source.get("conv1d.weight.__dotcache_depthwise_squeezed"))
            .transpose()?;
        Ok(Self {
            in_proj_qkv: build_prepared_linear_source_no_bias(
                &source.pp("in_proj_qkv"),
                cfg.hidden_size,
                key_dim * 2 + value_dim,
                deferred_in_proj_qkv_enabled(),
            )?,
            in_proj_z: prepared_linear_no_bias(&source.pp("in_proj_z"))?,
            in_proj_b: prepared_linear_no_bias(&source.pp("in_proj_b"))?,
            in_proj_a: prepared_linear_no_bias(&source.pp("in_proj_a"))?,
            conv1d_raw_weight: if conv1d_weight_squeezed.is_some() {
                None
            } else {
                Some(source.get("conv1d.weight")?)
            },
            conv1d_weight_squeezed,
            dt_bias_raw: if dt_bias_prepared.is_some() {
                None
            } else {
                Some(source.get("dt_bias")?)
            },
            a_log_raw: if a_log_exp_prepared.is_some() {
                None
            } else {
                Some(source.get("A_log")?)
            },
            dt_bias_prepared,
            a_log_exp_prepared,
            norm: Qwen35RmsNormGated::from_prepared(cfg.rms_norm_eps, &source.pp("norm"))?,
            out_proj: prepared_linear_no_bias(&source.pp("out_proj"))?,
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: None,
            recurrent_state: None,
            chunk_cache: None,
            value_cache: None,
        })
    }

    fn value_cache(&mut self, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let device_location = device.location();
        let rebuild = self
            .value_cache
            .as_ref()
            .map(|cache| cache.dtype != dtype || cache.device_location != device_location)
            .unwrap_or(true);
        if rebuild {
            let dt_bias_base = if let Some(dt_bias) = &self.dt_bias_prepared {
                dt_bias.clone()
            } else {
                let dt_bias = self.dt_bias_raw.as_ref().ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both prepared and raw dt_bias tensor".into(),
                    )
                })?;
                if dt_bias.dtype() == dtype {
                    dt_bias.clone()
                } else {
                    dt_bias.to_dtype(dtype)?
                }
            };
            let dt_bias = if dt_bias_base.rank() == 3 {
                if dt_bias_base.dtype() == dtype {
                    dt_bias_base
                } else {
                    dt_bias_base.to_dtype(dtype)?
                }
            } else {
                let dt_bias = if dt_bias_base.dtype() == dtype {
                    dt_bias_base
                } else {
                    dt_bias_base.to_dtype(dtype)?
                };
                dt_bias.reshape((1, 1, self.num_v_heads))?
            };
            let a_log_exp = if let Some(prepared) = &self.a_log_exp_prepared {
                if prepared.dtype() == dtype {
                    prepared.clone()
                } else {
                    prepared.to_dtype(dtype)?
                }
            } else {
                let a_log = self.a_log_raw.as_ref().ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both prepared and raw A_log tensor".into(),
                    )
                })?;
                let a_log = if a_log.dtype() == dtype {
                    a_log.clone()
                } else {
                    a_log.to_dtype(dtype)?
                };
                a_log.exp()?.reshape((1, 1, self.num_v_heads))?
            };
            self.value_cache = Some(LinearValueCache {
                dtype,
                device_location,
                dt_bias,
                a_log_exp,
            });
        }
        let cache = self
            .value_cache
            .as_ref()
            .expect("linear value cache must be initialized");
        Ok((cache.dt_bias.clone(), cache.a_log_exp.clone()))
    }

    fn conv1d_weight_squeezed(&self) -> Result<Tensor> {
        if let Some(weight) = &self.conv1d_weight_squeezed {
            Ok(weight.clone())
        } else {
            self.conv1d_raw_weight
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "native qwen35 load missing both squeezed and raw conv1d weight".into(),
                    )
                })?
                .squeeze(1)
        }
    }

    fn chunk_cache(
        &mut self,
        device: &Device,
        dtype: DType,
        chunk_size: usize,
    ) -> Result<LinearChunkCache> {
        let device_location = device.location();
        if let Some(cache) = &self.chunk_cache {
            if cache.chunk_size == chunk_size
                && cache.dtype == dtype
                && cache.device_location == device_location
            {
                return Ok(cache.clone());
            }
        }

        let lower =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let eye =
            Tensor::eye(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let strict_lower = lower.broadcast_sub(&eye)?;
        let lower_2d =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, chunk_size, chunk_size))?;
        let cache = LinearChunkCache {
            chunk_size,
            dtype,
            device_location,
            lower,
            eye,
            strict_lower,
            lower_2d,
        };
        self.chunk_cache = Some(cache.clone());
        Ok(cache)
    }

    fn chunk_gated_delta_rule_torch_like(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        _sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = 64usize;
        let compute_dtype = DType::F32;
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }
        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let prepare_start = profile_start(device)?;
        let batch_heads = batch_size * num_heads;
        let k_beta_start = profile_start(device)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let query = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let k_beta = k_beta.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let v_beta = v_beta.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        profile.linear_chunk_prepare_k_beta_millis += profile_elapsed(k_beta_start, device)?;
        let g_start = profile_start(device)?;
        let backend = backend_buffer_api::for_device(g.device());
        let g = {
            let g = g.reshape((batch_heads, num_chunks, chunk_size))?;
            if g.device().is_hip() {
                backend
                    .cumsum_last_dim(&backend.tensor_to_buffer(g.clone())?)?
                    .clone_tensor()
            } else {
                g.cumsum(D::Minus1)?
            }
        };
        profile.linear_chunk_prepare_g_millis += profile_elapsed(g_start, device)?;
        let cache_start = profile_start(device)?;
        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower_2d = cache.lower_2d.reshape((1, chunk_size, chunk_size))?;
        let eye_2d = Tensor::eye(chunk_size, compute_dtype, query.device())?
            .reshape((1, chunk_size, chunk_size))?;
        let strict_lower_2d = lower_2d.broadcast_sub(&eye_2d)?;
        profile.linear_chunk_prepare_cache_millis += profile_elapsed(cache_start, device)?;
        let base_attn_start = profile_start(device)?;
        let decay_deltas = g
            .unsqueeze(3)?
            .broadcast_sub(&g.unsqueeze(2)?)?
            .broadcast_mul(&lower_2d)?;
        let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower_2d)?;
        let exp_g = g.exp()?;
        profile.linear_chunk_prepare_base_attn_millis += profile_elapsed(base_attn_start, device)?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let solve_batch = batch_heads * num_chunks;
        let key_t = key.transpose(3, 2)?.contiguous()?;
        let base_attn = k_beta
            .matmul(&key_t)?
            .broadcast_mul(&decay_mask)?
            .neg()?
            .broadcast_mul(&strict_lower_2d)?
            .reshape((solve_batch, chunk_size, chunk_size))?;
        let mut rows = Vec::with_capacity(chunk_size);
        rows.push(Tensor::zeros(
            (solve_batch, 1, chunk_size),
            compute_dtype,
            query.device(),
        )?);
        for i in 1..chunk_size {
            let row = base_attn
                .narrow(1, i, 1)?
                .narrow(2, 0, i)?
                .reshape((solve_batch, i))?;
            let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
            let correction = row
                .unsqueeze(1)?
                .broadcast_mul(&sub)?
                .sum(1)?
                .reshape((solve_batch, i))?;
            let row = row.broadcast_add(&correction)?;
            let row =
                row.pad_with_zeros(1, 0, chunk_size - i)?
                    .reshape((solve_batch, 1, chunk_size))?;
            rows.push(row);
        }
        let attn = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
            .broadcast_add(&eye_2d)?
            .reshape((batch_heads, num_chunks, chunk_size, chunk_size))?;
        let solved_value = attn.matmul(&v_beta)?;
        let weighted_k = k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?;
        let attn_flat = attn
            .reshape((solve_batch, chunk_size, chunk_size))?
            .contiguous()?;
        let weighted_k_flat = weighted_k
            .reshape((solve_batch, chunk_size, k_head_dim))?
            .contiguous()?;
        let k_cumdecay = attn_flat.matmul(&weighted_k_flat)?.reshape((
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
        ))?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let scan_start = profile_start(device)?;
        let mut last_recurrent_state = backend_buffer_api::for_device(query.device()).zeros_tensor(
            query.device(),
            compute_dtype,
            &[batch_heads, k_head_dim, v_head_dim],
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let q_i = query.i((.., chunk_idx, .., ..))?;
            let k_i = key.i((.., chunk_idx, .., ..))?;
            let v_i = solved_value.i((.., chunk_idx, .., ..))?;
            let g_i = g.i((.., chunk_idx, ..))?;
            let decay_i = decay_mask.i((.., chunk_idx, .., ..))?;

            let recurrent_read_start = profile_start(device)?;
            let v_prime = k_cumdecay
                .i((.., chunk_idx, .., ..))?
                .matmul(&last_recurrent_state)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.exp()?.unsqueeze(D::Minus1)?)?
                .matmul(&last_recurrent_state)?;
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let v_new = v_i.broadcast_sub(&v_prime)?;

            let local_attn_start = profile_start(device)?;
            let k_i_t = k_i.transpose(2, 1)?.contiguous()?;
            let local_attn = q_i
                .matmul(&k_i_t)?
                .broadcast_mul(&decay_i)?
                .broadcast_mul(&lower_2d)?;
            let local_out = local_attn.matmul(&v_new)?;
            outputs.push(attn_inter.broadcast_add(&local_out)?.unsqueeze(1)?);
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let state_update_start = profile_start(device)?;
            let g_last = g_i.i((.., chunk_size - 1))?;
            let state_decay = g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let chunk_decay = g_last
                .unsqueeze(D::Minus1)?
                .broadcast_sub(&g_i)?
                .exp()?
                .unsqueeze(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(&state_decay)?
                .broadcast_add(
                    &k_i.broadcast_mul(&chunk_decay)?
                        .transpose(2, 1)?
                        .matmul(&v_new)?,
                )?;
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state, profile))
    }

    fn apply_mask_to_padding_states(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match attention_mask {
            Some(mask) if mask.dim(1)? > 1 && mask.dim(0)? > 1 => hidden_states
                .broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(hidden_states.dtype())?),
            None => Ok(hidden_states.clone()),
            Some(_) => Ok(hidden_states.clone()),
        }
    }

    fn prepare_depthwise_conv_input(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let backend = backend_buffer_api::for_device(mixed_qkv.device());
        let (mixed_qkv, next_state) =
            backend.prepare_depthwise_conv_input(self.conv_state.as_ref(), mixed_qkv, self.conv_kernel_size)?;
        self.conv_state = next_state;
        Ok(mixed_qkv)
    }

    fn update_depthwise_conv_state_from_raw(&mut self, mixed_qkv: &Tensor) -> Result<()> {
        let backend = backend_buffer_api::for_device(mixed_qkv.device());
        self.conv_state =
            backend.update_depthwise_conv_state(self.conv_state.as_ref(), mixed_qkv, self.conv_kernel_size)?;
        Ok(())
    }

    fn depthwise_conv_from_state(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let kernel = self.conv_kernel_size;
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?;
        let weights = self.conv1d_weight_squeezed()?;
        let mut output: Option<Tensor> = None;
        for tap in 0..kernel {
            let xs = mixed_qkv.narrow(2, tap, seq_len)?;
            let w = weights.i((.., tap))?.reshape((1, self.conv_dim(), 1))?;
            let contrib = xs.broadcast_mul(&w)?;
            output = Some(match output {
                Some(acc) => acc.broadcast_add(&contrib)?,
                None => contrib,
            });
        }
        output
            .expect("depthwise conv produced at least one tap")
            .silu()
    }

    fn run_depthwise_conv(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        if mixed_qkv.device().is_hip() {
            return self
                .run_depthwise_conv_packed_prefill(mixed_qkv)?
                .transpose(1, 2);
        }
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_update(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        if mixed_qkv.device().is_hip() {
            return self
                .run_depthwise_conv_materialized_pack(mixed_qkv)?
                .transpose(1, 2);
        }
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_materialized_pack(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?.contiguous()?;
        let weights = self.conv1d_weight_squeezed()?.contiguous()?;
        backend_buffer_api::for_device(mixed_qkv.device())
            .linear_prefill_conv(&mixed_qkv, &weights, seq_len, self.conv_kernel_size)
    }

    fn run_depthwise_conv_packed_prefill(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let weights = self.conv1d_weight_squeezed()?.contiguous()?;
        if mixed_qkv.device().is_hip() {
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(mixed_qkv.dtype())?,
                None => backend_buffer_api::for_device(mixed_qkv.device()).zeros_tensor(
                    mixed_qkv.device(),
                    mixed_qkv.dtype(),
                    &[mixed_qkv.dim(0)?, mixed_qkv.dim(1)?, state_len],
                )?,
            };
            let output = backend_buffer_api::for_device(mixed_qkv.device()).linear_stateful_conv(
                &mixed_qkv.contiguous()?,
                &prev_state,
                &weights,
                self.conv_kernel_size,
            )?;
            self.update_depthwise_conv_state_from_raw(mixed_qkv)?;
            return Ok(output);
        }

        self.run_depthwise_conv_materialized_pack(mixed_qkv)
    }

    fn conv_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim
    }

    fn recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let compute_dtype = initial_dtype;
        let query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let mut recurrent_state = match initial_state {
            Some(state) => state.to_dtype(compute_dtype)?,
            None => backend_buffer_api::for_device(query.device()).zeros_tensor(
                query.device(),
                compute_dtype,
                &[batch_size, num_heads, k_head_dim, v_head_dim],
            )?,
        };

        if device.is_hip() && k_head_dim <= 256 {
            let batch_heads = batch_size * num_heads;
            let pack_start = profile_start(device)?;
            let initial_state = backend_buffer_api::for_device(device).reshape_tensor_to_buffer(
                &recurrent_state.contiguous()?,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let query_scan = query
                .reshape((batch_heads, seq_len, k_head_dim))?
                .contiguous()?;
            let key_scan = key
                .reshape((batch_heads, seq_len, k_head_dim))?
                .contiguous()?;
            let value_scan = value
                .reshape((batch_heads, seq_len, v_head_dim))?
                .contiguous()?;
            let beta_scan = beta.reshape((batch_heads, seq_len))?.contiguous()?;
            let g_scan = g.reshape((batch_heads, seq_len))?.contiguous()?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_recurrent_prefill(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let output = fused
                .tensor()
                .narrow(1, 0, seq_len)?
                .reshape((batch_size, num_heads, seq_len, v_head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            let recurrent_state = fused
                .tensor()
                .narrow(1, seq_len, k_head_dim)?
                .reshape((batch_size, num_heads, k_head_dim, v_head_dim))?
                .contiguous()?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_recurrent_loop_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, recurrent_state, profile));
        }

        let mut outputs = Vec::with_capacity(seq_len);
        let loop_start = profile_start(device)?;
        for step in 0..seq_len {
            let q_t = query.i((.., .., step, ..))?.contiguous()?;
            let k_t = key.i((.., .., step, ..))?.contiguous()?;
            let v_t = value.i((.., .., step, ..))?.contiguous()?;
            let beta_t = beta.i((.., .., step))?.unsqueeze(D::Minus1)?;
            let g_t = g
                .i((.., .., step))?
                .exp()?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?;

            recurrent_state = recurrent_state.broadcast_mul(&g_t)?;
            let kv_mem = recurrent_state
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            let delta = (v_t.broadcast_sub(&kv_mem)?).broadcast_mul(&beta_t)?;
            recurrent_state = recurrent_state.broadcast_add(
                &k_t.unsqueeze(D::Minus1)?
                    .broadcast_mul(&delta.unsqueeze(2)?)?,
            )?;
            let out_t = recurrent_state
                .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            outputs.push(out_t.unsqueeze(2)?);
        }
        profile.linear_recurrent_loop_millis += profile_elapsed(loop_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, recurrent_state, profile))
    }

    fn chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = linear_attention_chunk_size(query.device(), sequence_length);
        let estimated_pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let estimated_num_chunks = (sequence_length + estimated_pad_size) / chunk_size;
        let scan_policy =
            delta_net_execution_policy(query.device(), sequence_length, estimated_num_chunks);
        let scan_mode = scan_policy.scan_mode;
        if scan_mode == DeltaNetScanMode::TorchLike {
            return self.chunk_gated_delta_rule_torch_like(
                query,
                key,
                value,
                g,
                beta,
                sequence_length,
            );
        }
        let compute_dtype = delta_net_compute_dtype(scan_mode, initial_dtype);
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;

        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }

        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        if use_delta_recurrent_prefill_kernel(query.device(), sequence_length) {
            let batch_heads = batch_size * num_heads;
            let pack_start = profile_start(device)?;
            let query_scan = query
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let key_scan = key
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let value_scan = value
                .reshape((batch_heads, total_sequence_length, v_head_dim))?
                .contiguous()?;
            let beta_scan = beta
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let g_scan = g
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let backend = backend_buffer_api::for_device(device);
            let kernel_start = profile_start(device)?;
            let fused = backend.delta_recurrent_prefill(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), recurrent_state.clone_tensor(), profile));
        }

        let query = query.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let value = value.reshape((batch_size, num_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta = beta.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let g_raw = g.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let batch_heads = batch_size * num_heads;
        let query_scan = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key_scan = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let value_scan = value.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta_scan = beta.reshape((batch_heads, num_chunks, chunk_size))?;
        let g_scan = g_raw.reshape((batch_heads, num_chunks, chunk_size))?;

        if use_hip_chunk_single_prefill_kernel(
            query.device(),
            sequence_length,
            num_chunks,
            chunk_size,
        ) {
            let pack_start = profile_start(device)?;
            let query_i = query_scan.i((.., 0, .., ..))?.contiguous()?;
            let key_i = key_scan.i((.., 0, .., ..))?.contiguous()?;
            let value_i = value_scan.i((.., 0, .., ..))?.contiguous()?;
            let beta_i = beta_scan.i((.., 0, ..))?.contiguous()?;
            let g_i = g_scan.i((.., 0, ..))?.contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_chunk_single_prefill(
                &initial_state,
                &query_i,
                &key_i,
                &value_i,
                &beta_i,
                &g_i,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        if use_delta_chunk_scan_kernel(query.device(), scan_mode, sequence_length, chunk_size)
            || use_hip_multi_chunk_scan_prefill_kernel(
                query.device(),
                sequence_length,
                num_chunks,
                chunk_size,
            )
        {
            let pack_start = profile_start(device)?;
            let query_scan = query_scan.contiguous()?;
            let key_scan = key_scan.contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let beta_scan = beta_scan.contiguous()?;
            let g_scan = g_scan.contiguous()?;
            let initial_state = backend_buffer_api::for_device(query.device()).zeros_state(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let backend = backend_buffer_api::for_device(device);
            let fused = backend.delta_chunk_scan_raw(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                &fused,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        if use_delta_chunk_step_kernel(query.device(), scan_mode, sequence_length, chunk_size) {
            if use_delta_chunk_windowed_kernel(
                query.device(),
                scan_mode,
                sequence_length,
                chunk_size,
            ) {
                let pack_start = profile_start(device)?;
                let initial_state = backend_buffer_api::for_device(query.device()).zeros_tensor(
                    query.device(),
                    compute_dtype,
                    &[batch_heads, k_head_dim, v_head_dim],
                )?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let scan_start = profile_start(device)?;
                let (output, last_recurrent_state) = if let Some((bytes, shape)) =
                    delta_chunk_step_windowed_raw_host_buffer(
                        &initial_state,
                        &query_scan,
                        &key_scan,
                        &value_scan,
                        &beta_scan,
                        &g_scan,
                    )?
                {
                    let fused = crate::backends::hip::state_buffer_from_host_bytes(
                        bytes,
                        shape,
                        compute_dtype,
                        query.device(),
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(scan_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let backend = backend_buffer_api::for_device(device);
                    let (output, last_recurrent_state) =
                        backend.unpack_scan_fused_output_and_state(
                            &fused,
                            total_sequence_length,
                            sequence_length,
                            batch_size,
                            num_heads,
                            v_head_dim,
                            k_head_dim,
                            initial_dtype,
                        )?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                    (output.clone_tensor(), last_recurrent_state.clone_tensor())
                } else {
                    let fused = delta_chunk_step_windowed_raw(
                        &initial_state,
                        &query_scan,
                        &key_scan,
                        &value_scan,
                        &beta_scan,
                        &g_scan,
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(scan_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let output = fused
                        .narrow(1, 0, total_sequence_length)?
                        .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                        .narrow(2, 0, sequence_length)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(initial_dtype)?;
                    let last_recurrent_state = fused
                        .narrow(1, total_sequence_length, k_head_dim)?
                        .reshape((batch_heads, k_head_dim, v_head_dim))?
                        .contiguous()?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                    (output, last_recurrent_state)
                };
                profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                    + profile.linear_full_kernel_execute_millis
                    + profile.linear_full_kernel_unpack_millis;
                profile.linear_attention_millis += profile_elapsed(total_start, device)?;
                return Ok((output, last_recurrent_state, profile));
            }

            let mut last_recurrent_state =
                backend_buffer_api::for_device(query.device()).zeros_tensor(
                query.device(),
                compute_dtype,
                &[batch_heads, k_head_dim, v_head_dim],
            )?;
            let mut outputs = Vec::with_capacity(num_chunks);
            let scan_start = profile_start(device)?;
            for chunk_idx in 0..num_chunks {
                let pack_start = profile_start(device)?;
                let q_i = query_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let k_i = key_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let v_i = value_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let beta_i = beta_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let g_i = g_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let prev_state_i = last_recurrent_state.contiguous()?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let kernel_start = profile_start(device)?;
                if let Some((bytes, shape)) = delta_chunk_step_raw_host_buffer(
                    &prev_state_i,
                    &q_i,
                    &k_i,
                    &v_i,
                    &beta_i,
                    &g_i,
                )? {
                    let fused = crate::backends::hip::state_buffer_from_host_bytes(
                        bytes,
                        shape,
                        compute_dtype,
                        query.device(),
                    )?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(kernel_start, device)?;

                    let unpack_start = profile_start(device)?;
                    let (output_i, recurrent_state) =
                        crate::backends::hip::unpack_delta_chunk_step_output(
                            &fused,
                            chunk_size,
                            k_head_dim,
                        )?;
                    outputs.push(output_i.clone_tensor().unsqueeze(1)?);
                    last_recurrent_state = recurrent_state.clone_tensor();
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                } else {
                    let fused =
                        delta_chunk_step_raw(&prev_state_i, &q_i, &k_i, &v_i, &beta_i, &g_i)?;
                    profile.linear_full_kernel_execute_millis +=
                        profile_elapsed(kernel_start, device)?;

                    let unpack_start = profile_start(device)?;
                    outputs.push(fused.narrow(1, 0, chunk_size)?.unsqueeze(1)?);
                    last_recurrent_state = fused
                        .narrow(1, chunk_size, k_head_dim)?
                        .reshape((batch_heads, k_head_dim, v_head_dim))?
                        .contiguous()?;
                    let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                    profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                    profile.transfer_millis += unpack_elapsed;
                }
            }
            profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;
            let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, last_recurrent_state, profile));
        }
        let backend = backend_buffer_api::for_device(device);
        let prepare_start = profile_start(device)?;
        let k_beta_start = profile_start(device)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        profile.linear_chunk_prepare_k_beta_millis += profile_elapsed(k_beta_start, device)?;
        let g_start = profile_start(device)?;
        let g = backend
            .cumsum_last_dim(&backend.tensor_to_buffer(g_raw.clone())?)?
            .clone_tensor();
        let exp_g = g.exp()?;
        let exp_g_scan = exp_g.reshape((batch_heads, num_chunks, chunk_size))?;
        profile.linear_chunk_prepare_g_millis += profile_elapsed(g_start, device)?;

        let cache_start = profile_start(device)?;
        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower = cache.lower;
        let eye = cache.eye;
        let strict_lower = cache.strict_lower;
        let use_state_kernel = use_delta_state_kernel(query.device(), scan_mode, sequence_length);
        let use_state_scan_kernel =
            use_delta_state_scan_kernel(query.device(), scan_mode, sequence_length);
        let use_chunk_fused_kernel =
            use_delta_chunk_fused_kernel(query.device(), scan_mode, sequence_length);
        let use_full_scan_kernel = use_delta_full_scan_kernel(query.device(), scan_mode, sequence_length)
            || use_hip_exact_multi_chunk_full_scan_prefill(
                query.device(),
                scan_mode,
                sequence_length,
                num_chunks,
                chunk_size,
            );
        let hip_full_scan_fast_path = query.device().is_hip() && use_full_scan_kernel;
        let hip_base_attn_fast_path = query.device().is_hip() && !hip_full_scan_fast_path;
        profile.linear_chunk_prepare_cache_millis += profile_elapsed(cache_start, device)?;

        let solve_batch = batch_size * num_heads * num_chunks;
        let base_attn_start = profile_start(device)?;
        let (base_attn, decay_scan) = if hip_full_scan_fast_path {
            (None, None)
        } else if hip_base_attn_fast_path {
            let decay_mask_start = profile_start(device)?;
            let decay_deltas = g
                .unsqueeze(4)?
                .broadcast_sub(&g.unsqueeze(3)?)?
                .broadcast_mul(&lower)?;
            let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower)?;
            profile.linear_chunk_prepare_base_attn_decay_mask_millis +=
                profile_elapsed(decay_mask_start, device)?;

            let post_start = profile_start(device)?;
            let base_attn = backend.delta_base_attn_scan(
                &k_beta
                    .reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?
                    .contiguous()?,
                &key_scan.contiguous()?,
                &exp_g_scan.contiguous()?,
            )?;
            profile.linear_chunk_prepare_base_attn_post_millis +=
                profile_elapsed(post_start, device)?;
            (
                Some(base_attn),
                Some(decay_mask.reshape((batch_heads, num_chunks, chunk_size, chunk_size))?),
            )
        } else {
            let decay_mask_start = profile_start(device)?;
            let decay_deltas = g
                .unsqueeze(4)?
                .broadcast_sub(&g.unsqueeze(3)?)?
                .broadcast_mul(&lower)?;
            let decay_mask = decay_deltas.exp()?.broadcast_mul(&lower)?;
            profile.linear_chunk_prepare_base_attn_decay_mask_millis +=
                profile_elapsed(decay_mask_start, device)?;

            let key_t_start = profile_start(device)?;
            let key_t = key.transpose(4, 3)?.contiguous()?;
            profile.linear_chunk_prepare_base_attn_key_t_millis +=
                profile_elapsed(key_t_start, device)?;

            let flatten_start = profile_start(device)?;
            let k_beta_flat = k_beta
                .reshape((solve_batch, chunk_size, k_head_dim))?
                .contiguous()?;
            let key_t_flat = key_t
                .reshape((solve_batch, k_head_dim, chunk_size))?
                .contiguous()?;
            let decay_mask_flat = decay_mask.reshape((solve_batch, chunk_size, chunk_size))?;
            profile.linear_chunk_prepare_base_attn_flatten_millis +=
                profile_elapsed(flatten_start, device)?;

            let matmul_start = profile_start(device)?;
            let raw_attn = k_beta_flat.matmul(&key_t_flat)?;
            profile.linear_chunk_prepare_base_attn_matmul_millis +=
                profile_elapsed(matmul_start, device)?;

            let post_start = profile_start(device)?;
            let result = (
                Some(backend.tensor_to_buffer(
                    raw_attn
                        .broadcast_mul(&decay_mask_flat)?
                        .neg()?
                        .broadcast_mul(&strict_lower.reshape((1, chunk_size, chunk_size))?)?
                        .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?,
                )?),
                Some(decay_mask.reshape((batch_heads, num_chunks, chunk_size, chunk_size))?),
            );
            profile.linear_chunk_prepare_base_attn_post_millis +=
                profile_elapsed(post_start, device)?;
            result
        };
        profile.linear_chunk_prepare_base_attn_millis += profile_elapsed(base_attn_start, device)?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let attn = if hip_full_scan_fast_path {
            backend.delta_attn_solve_from_inputs(
                &k_beta
                    .reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?
                    .contiguous()?,
                &key_scan.contiguous()?,
                &exp_g_scan.contiguous()?,
            )?
            .clone_tensor()
            .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
        } else if hip_base_attn_fast_path {
            let base_attn = base_attn.as_ref().expect("base_attn exists on HIP base-attn path");
            backend.delta_attn_solve_scan(&backend.reshape_tensor_to_buffer(
                &base_attn.clone_tensor().contiguous()?,
                &[batch_heads, num_chunks, chunk_size, chunk_size],
            )?)?
            .clone_tensor()
            .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
        } else if scan_policy.use_flattened_solve {
            let solve_batch = batch_size * num_heads * num_chunks;
            let base_attn = base_attn.as_ref().expect("base_attn exists on non-HIP solve path");
            let base_attn_flat = base_attn
                .tensor()
                .reshape((solve_batch, chunk_size, chunk_size))?;
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (solve_batch, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn_flat
                    .narrow(1, i, 1)?
                    .narrow(2, 0, i)?
                    .reshape((solve_batch, i))?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
                let correction = row
                    .unsqueeze(1)?
                    .broadcast_mul(&sub)?
                    .sum(1)?
                    .reshape((solve_batch, i))?;
                let row = row.broadcast_add(&correction)?;
                let row = row.pad_with_zeros(1, 0, chunk_size - i)?.reshape((
                    solve_batch,
                    1,
                    chunk_size,
                ))?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
                .broadcast_add(&eye)?
        } else {
            let base_attn = base_attn.as_ref().expect("base_attn exists on non-HIP solve path");
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (batch_size, num_heads, num_chunks, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn.tensor().narrow(3, i, 1)?.narrow(4, 0, i)?.squeeze(3)?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 3)?.narrow(4, 0, i)?;
                let correction = row.unsqueeze(4)?.broadcast_mul(&sub)?.sum(3)?;
                let row = (row + correction)?;
                let row = row.pad_with_zeros(3, 0, chunk_size - i)?.unsqueeze(3)?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 3)?.broadcast_add(&eye)?
        };
        let weighted_k = k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?;
        let attn_flat = attn
            .reshape((solve_batch, chunk_size, chunk_size))?
            .contiguous()?;
        let weighted_k_flat = weighted_k
            .reshape((solve_batch, chunk_size, k_head_dim))?
            .contiguous()?;
        let k_cumdecay = attn_flat
            .matmul(&weighted_k_flat)?
            .reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let lower_2d = cache.lower_2d;
        let k_cumdecay_scan =
            k_cumdecay.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let lower_2d_scan = lower_2d.reshape((1, 1, chunk_size, chunk_size))?;
        let local_attn_scan = match scan_mode {
            DeltaNetScanMode::PrebatchedLocal => Some({
                if hip_full_scan_fast_path {
                    backend.delta_local_attn_scan(
                        &query_scan.contiguous()?,
                        &key_scan.contiguous()?,
                        &exp_g_scan.contiguous()?,
                    )?
                } else {
                    let key_scan_t = key_scan.transpose(3, 2)?.contiguous()?;
                    backend.tensor_to_buffer(
                        query_scan
                            .matmul(&key_scan_t)?
                            .broadcast_mul(decay_scan.as_ref().ok_or_else(|| {
                                candle::Error::Msg(
                                    "prebatched-local attention requires decay scan".into(),
                                )
                            })?)?
                            .broadcast_mul(&lower_2d_scan)?,
                    )?
                }
            }),
            _ => None,
        };
        let needs_q_state_scan = !hip_full_scan_fast_path;
        let needs_hoisted_decays =
            !hip_full_scan_fast_path
                && matches!(
                    scan_mode,
                    DeltaNetScanMode::HoistedDecays | DeltaNetScanMode::PrebatchedLocal
                );
        let q_state_scan = if needs_q_state_scan {
            Some(query_scan.broadcast_mul(&exp_g_scan.unsqueeze(D::Minus1)?)?)
        } else {
            None
        };
        let (state_decay_scan, chunk_decay_scan) = if needs_hoisted_decays {
            let exp_g_last_scan = exp_g_scan.i((.., .., chunk_size - 1))?;
            (
                Some(exp_g_last_scan.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?),
                Some(
                    exp_g_last_scan
                        .unsqueeze(D::Minus1)?
                        .broadcast_div(&exp_g_scan)?
                        .unsqueeze(D::Minus1)?,
                ),
            )
        } else {
            (None, None)
        };
        let lower_2d = lower_2d.reshape((1, chunk_size, chunk_size))?;
        let mut last_recurrent_state = backend.zeros_state(
            query.device(),
            compute_dtype,
            &[batch_heads, k_head_dim, v_head_dim],
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        let full_scan = if use_full_scan_kernel {
            let full_pack_start = profile_start(device)?;
            let local_attn_scan = local_attn_scan
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg("delta-full-scan requires prebatched local attention".into())
                })?
                .contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let full_scan = if query.device().is_hip() {
                let packed_scan = backend.delta_full_scan_pack(
                    &query_scan.contiguous()?,
                    &key_scan.contiguous()?,
                    &exp_g_scan.contiguous()?,
                    &k_cumdecay_scan.contiguous()?,
                )?;
                let full_pack_elapsed = profile_elapsed(full_pack_start, device)?;
                profile.linear_full_kernel_pack_millis += full_pack_elapsed;
                profile.transfer_millis += full_pack_elapsed;
                let full_kernel_start = profile_start(device)?;
                let full_scan = backend.delta_full_scan_packed(
                    &last_recurrent_state,
                    &packed_scan,
                    &local_attn_scan,
                    &value_scan,
                )?;
                profile.linear_full_kernel_execute_millis +=
                    profile_elapsed(full_kernel_start, device)?;
                Some(full_scan)
            } else {
                let state_decay_scan = state_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted state decay".into())
                    })?
                    .squeeze(3)?
                    .squeeze(2)?
                    .contiguous()?;
                let weighted_key_scan = key_scan
                    .broadcast_mul(chunk_decay_scan.as_ref().ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted chunk decay".into())
                    })?)?
                    .contiguous()?;
                let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
                let q_state_scan = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-full-scan requires hoisted q-state".into())
                    })?
                    .contiguous()?;
                let full_pack_elapsed = profile_elapsed(full_pack_start, device)?;
                profile.linear_full_kernel_pack_millis += full_pack_elapsed;
                profile.transfer_millis += full_pack_elapsed;
                let full_kernel_start = profile_start(device)?;
                let full_scan = backend.delta_full_scan(
                    &last_recurrent_state,
                    &weighted_key_scan,
                    &k_cumdecay_scan,
                    &q_state_scan,
                    &local_attn_scan,
                    &state_decay_scan,
                    &value_scan,
                )?;
                profile.linear_full_kernel_execute_millis +=
                    profile_elapsed(full_kernel_start, device)?;
                Some(full_scan)
            };
            full_scan
        } else {
            None
        };
        let state_scan = if use_state_scan_kernel {
            let state_decay_scan = state_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted state decay".into())
            })?;
            let chunk_decay_scan = chunk_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted chunk decay".into())
            })?;
            let weighted_key_scan = key_scan.broadcast_mul(chunk_decay_scan)?;
            let state_decay_feature =
                state_decay_scan.broadcast_as((batch_heads, num_chunks, chunk_size, 1))?;
            let packed_scan = backend.pack_delta_state_scan(
                &weighted_key_scan,
                &k_cumdecay_scan,
                &state_decay_feature,
            )?;
            Some(backend.delta_state_scan(
                &last_recurrent_state,
                &packed_scan,
                &value_scan.contiguous()?,
            )?)
        } else {
            None
        };
        if let Some(full_scan) = &full_scan {
            let full_unpack_start = profile_start(device)?;
            let (output, last_recurrent_state) = backend.unpack_scan_fused_output_and_state(
                full_scan,
                total_sequence_length,
                sequence_length,
                batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                initial_dtype,
            )?;
            let full_unpack_elapsed = profile_elapsed(full_unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += full_unpack_elapsed;
            profile.transfer_millis += full_unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output.clone_tensor(), last_recurrent_state.clone_tensor(), profile));
        }

        let scan_start = profile_start(device)?;
        for chunk_idx in 0..num_chunks {
            let index_start = profile_start(device)?;
            let q_i = query_scan.i((.., chunk_idx, .., ..))?;
            let k_i = key_scan.i((.., chunk_idx, .., ..))?;
            let v_i = value_scan.i((.., chunk_idx, .., ..))?;
            let g_i = g_scan.i((.., chunk_idx, ..))?;
            let prev_state_i = if let Some(state_scan) = &state_scan {
                backend.state_scan_chunk(state_scan, chunk_idx)?
            } else {
                last_recurrent_state.clone()
            };
            profile.linear_chunk_index_millis += profile_elapsed(index_start, device)?;

            let local_attn_start = profile_start(device)?;
            let attn = if let Some(local_attn_scan) = &local_attn_scan {
                local_attn_scan.tensor().i((.., chunk_idx, .., ..))?
            } else {
                let decay_i = decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta chunk fallback requires decay scan".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let k_i_t = k_i.transpose(2, 1)?.contiguous()?;
                q_i.matmul(&k_i_t)?
                    .broadcast_mul(&decay_i)?
                    .broadcast_mul(&lower_2d)?
            };
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let recurrent_read_start = profile_start(device)?;
            let (v_new, attn_inter, fused_next_state) = if use_chunk_fused_kernel
                && state_scan.is_none()
            {
                let weighted_key = chunk_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted chunk decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_mul(&k_i)?;
                let q_state = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted q-state".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let state_decay = state_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted state decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_as((batch_heads, chunk_size, 1))?;
                let packed_chunk = backend.pack_delta_chunk_fused(
                    &weighted_key,
                    &k_cumdecay_scan.i((.., chunk_idx, .., ..))?,
                    &q_state,
                    &state_decay,
                )?;
                let prev_state_i = prev_state_i.contiguous()?;
                let fused = backend.delta_chunk_fused(
                    &prev_state_i,
                    &packed_chunk,
                    &v_i.contiguous()?,
                )?;
                let (v_new, attn_inter, fused_next_state) = backend.unpack_chunk_fused(
                    &fused,
                    chunk_size,
                    k_head_dim,
                )?;
                (v_new, attn_inter, Some(fused_next_state))
            } else {
                let q_state_i = q_state_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk loop requires hoisted q-state".into())
                    })?
                    .i((.., chunk_idx, .., ..))?;
                let (v_new, attn_inter) = backend.delta_chunk_recurrent_read(
                    &prev_state_i,
                    &k_cumdecay_scan.i((.., chunk_idx, .., ..))?,
                    &q_state_i,
                    &v_i,
                )?;
                (v_new, attn_inter, None)
            };
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let local_mix_start = profile_start(device)?;
            outputs.push(
                backend
                    .mix_chunk_attention(&attn, &attn_inter, &v_new)?
                    .clone_tensor()
                    .unsqueeze(1)?,
            );
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_mix_start, device)?;

            let state_update_start = profile_start(device)?;
            let (state_decay, chunk_decay) =
                if let (Some(state_decay_scan), Some(chunk_decay_scan)) =
                    (&state_decay_scan, &chunk_decay_scan)
                {
                    (
                        state_decay_scan.i((.., chunk_idx, .., ..))?,
                        chunk_decay_scan.i((.., chunk_idx, .., ..))?,
                    )
                } else {
                    let g_last = g_i.i((.., chunk_size - 1))?;
                    (
                        g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?,
                        g_last
                            .unsqueeze(D::Minus1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )
                };
            if let Some(fused_next_state) = fused_next_state {
                last_recurrent_state = fused_next_state.contiguous()?;
            } else if let Some(state_scan) = &state_scan {
                last_recurrent_state = backend.state_scan_next_chunk(state_scan, chunk_idx + 1)?;
            } else {
                let prev_state_scaled = last_recurrent_state
                    .tensor()
                    .broadcast_mul(&state_decay)?
                    .contiguous()?;
                let weighted_key = k_i.broadcast_mul(&chunk_decay)?.contiguous()?;
                last_recurrent_state = backend.delta_state_update(
                    &prev_state_scaled,
                    &weighted_key,
                    &v_new,
                    use_state_kernel,
                )?;
            }
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state.clone_tensor(), profile))
    }

    fn forward_profiled_with_state(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let backend = backend_buffer_api::for_device(device);
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let mixed_qkv =
            backend.tensor_to_buffer(self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?)?;
        let z = backend.reshape_tensor_to_buffer(
            &self.in_proj_z.forward(&hidden_states)?,
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = backend.tensor_to_buffer(self.in_proj_b.forward(&hidden_states)?)?;
        let a = backend.tensor_to_buffer(self.in_proj_a.forward(&hidden_states)?)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;
        let (output, recurrent_state, linear_profile) = self.forward_profiled_with_state_projected(
            hidden_states.dtype(),
            batch_size,
            seq_len,
            &mixed_qkv,
            &z,
            &beta_raw,
            &a,
            compute_dtype,
        )?;
        profile.add_assign(&linear_profile);
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output.clone_tensor(), recurrent_state.clone_tensor(), profile))
    }

    fn forward_profiled_with_state_projected(
        &mut self,
        hidden_dtype: DType,
        batch_size: usize,
        seq_len: usize,
        mixed_qkv: &StateBuffer,
        z: &StateBuffer,
        beta_raw: &StateBuffer,
        a: &StateBuffer,
        compute_dtype: DType,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = mixed_qkv.device();
        let backend = backend_buffer_api::for_device(device);
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();

        if use_hip_combined_linear_decode(device, seq_len) {
            let kv_append_start = profile_start(device)?;
            let target_dtype = mixed_qkv.tensor().dtype();
            let weights = self.conv1d_weight_squeezed()?.contiguous()?;
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_conv_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(target_dtype)?,
                None => backend
                    .zeros_tensor(
                        mixed_qkv.device(),
                        target_dtype,
                        &[mixed_qkv.tensor().dim(0)?, mixed_qkv.tensor().dim(1)?, state_len],
                    )?,
            };
            let a = if a.tensor().dtype() == target_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(target_dtype)?
            };
            let beta_raw = if beta_raw.tensor().dtype() == target_dtype {
                beta_raw.clone_tensor()
            } else {
                beta_raw.tensor().to_dtype(target_dtype)?
            };
            let a_beta_raw = backend.concat_last_dim(
                &backend.tensor_to_buffer(a)?,
                &backend.tensor_to_buffer(beta_raw)?,
            )?;
            let (dt_bias, a_log_exp) = self.value_cache(device, target_dtype)?;
            let initial_state = match &self.recurrent_state {
                Some(state) => {
                    let state = state.clone_tensor();
                    let state = if state.rank() == 3 {
                        state.reshape((batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim))?
                    } else {
                        state
                    };
                    if state.dtype() == DType::F32 {
                        state
                    } else {
                        state.to_dtype(DType::F32)?
                    }
                }
                None => backend.zeros_tensor(
                    device,
                    DType::F32,
                    &[batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim],
                )?,
            };
            let head_repeat = self.num_v_heads / self.num_k_heads;
            let fused = backend.linear_decode_step(
                &mixed_qkv.contiguous()?,
                &prev_conv_state,
                &weights,
                &a_beta_raw,
                &dt_bias,
                &a_log_exp,
                &initial_state,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                head_repeat,
            )?;
            self.update_depthwise_conv_state_from_raw(mixed_qkv.tensor())?;
            let (core_attn_out, recurrent_state) = backend.unpack_linear_decode_output(
                &fused,
                batch_size,
                seq_len,
                self.value_dim,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )?;
            let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
            profile.linear_conv_millis += kv_append_elapsed;
            profile.kv_append_write_millis += kv_append_elapsed;
            profile.linear_recurrent_loop_millis += kv_append_elapsed;

            let output_start = profile_start(device)?;
            let output = self.finalize_linear_output_buffer(
                hidden_dtype,
                batch_size,
                seq_len,
                z,
                &core_attn_out,
            )?;
            profile.output_projection_millis += profile_elapsed(output_start, device)?;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, recurrent_state, profile));
        }

        let kv_append_start = profile_start(device)?;
        let (mixed_qkv, g) = if use_hip_combined_linear_prefill(device, seq_len) {
            let target_dtype = mixed_qkv.tensor().dtype();
            let a_tensor = if a.tensor().dtype() == target_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(target_dtype)?
            };
            let a = backend.tensor_to_buffer(a_tensor)?;
            let (dt_bias, a_log_exp) = self.value_cache(device, target_dtype)?;
            let weights = self.conv1d_weight_squeezed()?.contiguous()?;
            let state_len = self.conv_kernel_size.saturating_sub(1);
            let prev_state = match &self.conv_state {
                Some(prev_state) => prev_state.clone_tensor_as(target_dtype)?,
                None => backend
                    .zeros_tensor(
                        mixed_qkv.device(),
                        target_dtype,
                        &[mixed_qkv.tensor().dim(0)?, mixed_qkv.tensor().dim(1)?, state_len],
                    )?,
            };
            let fused = backend.linear_stateful_conv_value_decay_with_state(
                &mixed_qkv.contiguous()?,
                &prev_state,
                &weights,
                &a,
                &dt_bias,
                &a_log_exp,
                self.conv_kernel_size,
            )?;
            let conv_dim = self.conv_dim();
            let (mixed_qkv, g, conv_state) = backend.unpack_linear_prefill_output(
                &fused,
                batch_size,
                seq_len,
                conv_dim,
                self.num_v_heads,
                state_len,
            )?;
            self.conv_state = Some(conv_state);
            let g = if g.dtype() == compute_dtype {
                g
            } else {
                g.to_dtype(compute_dtype)?
            };
            (mixed_qkv, g)
        } else {
            let mixed_qkv = if seq_len == 1 {
                self.run_depthwise_conv_update(mixed_qkv.tensor())?
                    .transpose(1, 2)?
            } else if use_linear_prefill_packed_kernel(device, seq_len) {
                self.run_depthwise_conv_packed_prefill(mixed_qkv.tensor())?
            } else {
                self.run_depthwise_conv(mixed_qkv.tensor())?.transpose(1, 2)?
            };
            let a = if a.tensor().dtype() == compute_dtype {
                a.clone_tensor()
            } else {
                a.tensor().to_dtype(compute_dtype)?
            };
            let (dt_bias, a_log_exp) = self.value_cache(device, compute_dtype)?;
            let g = backend
                .value_decay(&backend.tensor_to_buffer(a.clone())?, &dt_bias, &a_log_exp)?
                .clone_tensor();
            (mixed_qkv, g)
        };
        let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
        profile.linear_conv_millis += kv_append_elapsed;
        profile.kv_append_write_millis += kv_append_elapsed;

        let layout_start = profile_start(device)?;
        let use_short_recurrent_prefill = use_hip_short_linear_prefill_recurrent(device, seq_len);
        let (query, key, value, beta, g) = backend.prepare_linear_attention_inputs(
            &mixed_qkv,
            beta_raw,
            &g,
            batch_size,
            seq_len,
            self.key_dim,
            self.value_dim,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            compute_dtype,
            seq_len == 1 || use_short_recurrent_prefill,
        )?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let (core_attn_out, recurrent_state, linear_profile) =
            if seq_len == 1 && self.recurrent_state.is_some() {
                self.recurrent_gated_delta_rule(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    self.recurrent_state.as_ref().map(StateBuffer::tensor),
                )?
            } else if seq_len == 1 {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else if use_short_recurrent_prefill {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else {
                self.chunk_gated_delta_rule(&query, &key, &value, &g, &beta, seq_len)?
            };
        profile.add_assign(&linear_profile);

        let output_start = profile_start(device)?;
        let output = self.finalize_linear_output_buffer(
            hidden_dtype,
            batch_size,
            seq_len,
            z,
            &core_attn_out,
        )?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output, backend.tensor_to_buffer(recurrent_state)?, profile))
    }

    fn forward_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let (output, recurrent_state, profile) =
            self.forward_profiled_with_state(hidden_states, attention_mask)?;
        let backend = backend_buffer_api::for_device(hidden_states.device());
        self.recurrent_state = Some(backend.tensor_to_buffer(recurrent_state)?);
        Ok((output, profile))
    }

    fn forward_profiled_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let (output, recurrent_state, profile) =
            self.forward_profiled_with_state_buffer(hidden_states, attention_mask)?;
        self.recurrent_state = Some(recurrent_state);
        Ok((output, profile))
    }

    fn trace_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        self.forward_profiled_with_state(hidden_states, attention_mask)
    }

    fn trace_profiled_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        self.forward_profiled_with_state_buffer(hidden_states, attention_mask)
    }

    fn forward_profiled_with_state_buffer(
        &mut self,
        hidden_states: &StateBuffer,
        attention_mask: Option<&Tensor>,
    ) -> Result<(StateBuffer, StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.tensor().dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = backend_buffer_api::for_device(device).tensor_to_buffer(
            self.apply_mask_to_padding_states(hidden_states.tensor(), attention_mask)?,
        )?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let backend = backend_buffer_api::for_device(device);
        let mixed_qkv = backend.tensor_to_buffer(
            self.in_proj_qkv
                .forward_buffer(&hidden_states)?
                .tensor()
                .transpose(1, 2)?,
        )?;
        let z = backend.reshape_tensor_to_buffer(
            self.in_proj_z.forward_buffer(&hidden_states)?.tensor(),
            &[batch_size, seq_len, self.num_v_heads, self.head_v_dim],
        )?;
        let beta_raw = self.in_proj_b.forward_buffer(&hidden_states)?;
        let a = self.in_proj_a.forward_buffer(&hidden_states)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let (output, recurrent_state, linear_profile) =
            self.forward_profiled_with_state_projected(
                hidden_states.tensor().dtype(),
                batch_size,
                seq_len,
                &mixed_qkv,
                &z,
                &beta_raw,
                &a,
                compute_dtype,
            )?;
        profile.add_assign(&linear_profile);
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output, recurrent_state, profile))
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_profiled(hidden_states, attention_mask)
            .map(|(output, _)| output)
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
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

    pub(crate) fn decode_hidden_state_profiled_direct_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        hidden_state_t: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
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
        let (_, seq_len, _) = hidden_state_t.dims3()?;
        if seq_len != 1 {
            candle::bail!(
                "direct-hip-v1 decode expects a single-token hidden state, got seq_len={seq_len}"
            );
        }
        let mut profile = RuntimeProfile::default();
        let mut xs = hidden_state_t.clone();
        for (layer_idx, (layer, layer_meta)) in self
            .layers
            .iter_mut()
            .zip(metadata.layers.iter())
            .enumerate()
        {
            if layer_meta.layer_idx != layer_idx {
                candle::bail!(
                    "direct-hip-v1 decode metadata index mismatch at layer {}: got {}",
                    layer_idx,
                    layer_meta.layer_idx
                );
            }
            if layer.layer_type() != layer_meta.layer_type {
                candle::bail!(
                    "direct-hip-v1 decode layer type mismatch at layer {}: model={} metadata={}",
                    layer_idx,
                    layer.layer_type(),
                    layer_meta.layer_type
                );
            }
            let (next_xs, layer_profile) = layer.forward_profiled(&xs, None, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward_buffer(&xs)?, profile))
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

    pub(crate) fn decode_hidden_state_profiled_direct_hip_v1(
        &mut self,
        metadata: &PreparedQwen35DirectMetadata,
        hidden_state_t: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_state_t.device();
        let backend = backend_buffer_api::for_device(device);
        let (hidden_states, mut profile) = self
            .language_model
            .decode_hidden_state_profiled_direct_hip_v1(
                metadata,
                hidden_state_t,
                seqlen_offset,
            )?;
        let output_start = profile_start(device)?;
        let logits = backend.slice_last_token(&hidden_states)?;
        let logits = self.lm_head.forward_buffer(&logits)?;
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
