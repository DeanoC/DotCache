use super::activation::Activation;
use super::backend_buffer_api;
use super::backend_ops;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
#[cfg(feature = "qwen35-minimal-hip")]
use super::hip;
use super::model::{
    elapsed_millis, hip_output_bytes_to_cpu_storage, hip_tensor_from_host_bytes,
    trace_hip_wrapper_fallback,
};
use super::prepared::PreparedTensorSource;
use super::rotary;
use super::types::{StateBuffer, TextConfig};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::linear_no_bias;
use super::with_tracing::Linear;
use crate::backends;
use crate::{BufferViewDesc, ImmutableBufferView, ImmutableWeightHandle, TargetSpec};
use candle::{DType, Device, DeviceLocation, Module, Result, Tensor, D};
use candle_core as candle;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(super) fn prepared_linear_no_bias(source: &PreparedTensorSource) -> Result<Linear> {
    Ok(Linear::new(source.get("weight")?, None))
}

pub(super) fn build_prepared_linear_source_no_bias(
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

pub(super) fn prepared_linear_b(source: &PreparedTensorSource, bias: bool) -> Result<Linear> {
    let weight = source.get("weight")?;
    let bias = if bias { Some(source.get("bias")?) } else { None };
    Ok(Linear::new(weight, bias))
}

fn prepared_embedding(source: &PreparedTensorSource, hidden_size: usize) -> Result<Embedding> {
    Ok(Embedding::new(source.get("weight")?, hidden_size))
}

#[derive(Debug, Clone)]
pub(super) struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub(super) fn new(embeddings: Tensor, hidden_size: usize) -> Self {
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
    pub(super) fn forward_buffer(&self, indexes: &Tensor) -> Result<StateBuffer> {
        backend_buffer_api::for_device(indexes.device()).embedding_lookup(&self.embeddings, indexes)
    }
}

#[cfg(any(feature = "hf", test))]
pub(super) fn embedding(in_size: usize, out_size: usize, vb: WeightBuilder) -> Result<Embedding> {
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

pub(super) fn build_prepared_embedding_source(
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

pub(super) fn immutable_embedding_enabled() -> bool {
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

pub(super) fn immutable_linear_enabled(cfg: &TextConfig) -> bool {
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

pub(super) fn deferred_in_proj_qkv_enabled() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_DEFERRED_IN_PROJ_QKV").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[derive(Debug, Clone)]
pub(crate) struct EmbeddingMeta {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

#[derive(Debug, Clone)]
pub(super) struct NativeImmutableWeight {
    handle: ImmutableWeightHandle,
    target: TargetSpec,
    view_desc: BufferViewDesc,
}

impl NativeImmutableWeight {
    pub(super) fn new(handle: ImmutableWeightHandle) -> Self {
        let target = handle.target_spec();
        let view_desc = handle.buffer_view_desc();
        Self {
            handle,
            target,
            view_desc,
        }
    }

    pub(super) fn immutable_buffer_view(&self) -> ImmutableBufferView<'_> {
        self.handle.immutable_buffer_view()
    }

    pub(super) fn materialize(&self, device: &Device) -> crate::Result<Tensor> {
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
    pub(crate) meta: EmbeddingMeta,
    state: Arc<Mutex<ImmutableEmbeddingState>>,
}

#[derive(Debug)]
enum ImmutableEmbeddingState {
    Uninitialized,
    Registered(RegisteredHostWeight),
    Fallback(Embedding),
}

impl ImmutableEmbedding {
    pub(super) fn new(handle: ImmutableWeightHandle, meta: EmbeddingMeta) -> Self {
        Self {
            weight: NativeImmutableWeight::new(handle),
            meta,
            state: Arc::new(Mutex::new(ImmutableEmbeddingState::Uninitialized)),
        }
    }

    pub(super) fn current_mode(&self) -> &'static str {
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

    pub(super) fn ensure_fallback_embedding(&self) -> Result<Embedding> {
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

    pub(super) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
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
pub(super) enum EmbeddingSource {
    Materialized(Embedding),
    Immutable(ImmutableEmbedding),
}

impl EmbeddingSource {
    pub(super) fn embeddings(&self) -> Option<&Tensor> {
        match self {
            Self::Materialized(embedding) => Some(embedding.embeddings()),
            Self::Immutable(_) => None,
        }
    }

    pub(super) fn dtype(&self) -> DType {
        match self {
            Self::Materialized(embedding) => embedding.embeddings().dtype(),
            Self::Immutable(embedding) => embedding.meta.dtype,
        }
    }

    pub(super) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(embedding) => immutable_embedding_forward_fallback(embedding, input_ids),
            Self::Immutable(embedding) => embedding.forward(input_ids),
        }
    }

    pub(super) fn forward_buffer(&self, input_ids: &Tensor) -> Result<StateBuffer> {
        match self {
            Self::Materialized(embedding) => embedding.forward_buffer(input_ids),
            Self::Immutable(embedding) => {
                let backend = backend_buffer_api::for_device(input_ids.device());
                backend.tensor_to_buffer(backend.immutable_embedding_lookup(embedding, input_ids)?)
            }
        }
    }

    pub(super) fn runtime_mode(&self) -> &'static str {
        match self {
            Self::Materialized(_) => "eager",
            Self::Immutable(embedding) => embedding.current_mode(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum OutputProjectionSource {
    Materialized(Linear),
    TiedImmutable(ImmutableEmbedding),
}

impl OutputProjectionSource {
    pub(super) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(linear) => linear.forward(hidden_states),
            Self::TiedImmutable(embedding) => backend_buffer_api::for_device(hidden_states.device())
                .output_projection_tensor(embedding, hidden_states),
        }
    }

    pub(super) fn forward_buffer(&self, hidden_states: &StateBuffer) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => linear.forward_buffer(hidden_states),
            Self::TiedImmutable(embedding) => {
                backend_buffer_api::for_device(hidden_states.device())
                    .output_projection(embedding, hidden_states)
            }
        }
    }

    pub(super) fn forward_buffer_into_scratch(
        &self,
        hidden_states: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => {
                let backend = backend_buffer_api::for_device(hidden_states.device());
                let output = linear.forward_buffer(hidden_states)?;
                backend.copy_state_into_scratch(&output, scratch)
            }
            Self::TiedImmutable(embedding) => backend_buffer_api::for_device(hidden_states.device())
                .output_projection_into_scratch(embedding, hidden_states, scratch),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct DeferredLinear {
    weight: NativeImmutableWeight,
    device: Device,
    state: Arc<Mutex<Option<Linear>>>,
}

impl DeferredLinear {
    pub(super) fn new(handle: ImmutableWeightHandle, device: Device) -> Self {
        Self {
            weight: NativeImmutableWeight::new(handle),
            device,
            state: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn ensure_materialized(&self) -> Result<Linear> {
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

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.ensure_materialized()?.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub(super) enum LinearSource {
    Materialized(Linear),
    Deferred(DeferredLinear),
}

impl LinearSource {
    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Materialized(linear) => linear.forward(xs),
            Self::Deferred(linear) => linear.forward(xs),
        }
    }

    pub(super) fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => linear.forward_buffer(xs),
            Self::Deferred(linear) => linear.ensure_materialized()?.forward_buffer(xs),
        }
    }

    pub(super) fn forward_buffer_into_scratch(
        &self,
        xs: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        match self {
            Self::Materialized(linear) => linear.forward_buffer_into_scratch(xs, scratch),
            Self::Deferred(linear) => linear
                .ensure_materialized()?
                .forward_buffer_into_scratch(xs, scratch),
        }
    }

    pub(super) fn is_deferred(&self) -> bool {
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

pub(super) fn profile_start(device: &Device) -> Result<Instant> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(Instant::now())
}

pub(super) fn profile_elapsed(start: Instant, device: &Device) -> Result<f64> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(elapsed_millis(start))
}

pub(super) fn debug_full_prefill_kernel_compare_enabled() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_DEBUG_HIP_PREFILL_KERNEL").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(super) fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
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


#[derive(Debug, Clone)]
pub(super) struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    pub(super) fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
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

    pub(super) fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
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

    pub(super) fn apply_buffer(
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
pub(super) struct Qwen35RmsNorm {
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
    pub(super) fn new(dim: usize, eps: f64, vb: WeightBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    pub(super) fn from_prepared(eps: f64, source: &PreparedTensorSource) -> Result<Self> {
        Ok(Self {
            weight: source.get("weight")?,
            eps,
        })
    }

    pub(super) fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
        backend_buffer_api::for_device(xs.device()).rms_norm(xs, &self.weight, self.eps, true)
    }

    pub(super) fn trace_buffer(&self, xs: &StateBuffer) -> Result<super::types::RmsNormTrace> {
        let output = self.forward_buffer(xs)?;
        let xs_tensor = xs.tensor();
        let hidden = xs_tensor.to_dtype(DType::F32)?.contiguous()?;
        let mean_square = hidden.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = mean_square.affine(1.0, self.eps)?.sqrt()?.recip()?;
        let weight = self.weight.to_dtype(DType::F32)?.contiguous()?;
        let effective_weight = (&weight + 1.0)?.contiguous()?;
        let weighted_hidden = hidden
            .broadcast_mul(&rsqrt)?
            .broadcast_mul(&effective_weight.reshape((1, 1, effective_weight.dim(0)?))?)?;
        Ok(super::types::RmsNormTrace {
            input_hidden: backend_buffer_api::for_device(xs.device()).tensor_to_buffer(hidden)?,
            mean_square: backend_buffer_api::for_device(xs.device()).tensor_to_buffer(mean_square)?,
            rsqrt: backend_buffer_api::for_device(xs.device()).tensor_to_buffer(rsqrt)?,
            weight: backend_buffer_api::for_device(xs.device()).tensor_to_buffer(weight)?,
            weighted_hidden: backend_buffer_api::for_device(xs.device())
                .tensor_to_buffer(weighted_hidden.clone())?,
            output,
        })
    }

    pub(super) fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub(super) fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        backend_ops::rms_norm(xs, &self.weight, self.eps, true)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Qwen35RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNormGated {
    #[cfg(any(feature = "hf", test))]
    pub(super) fn new(dim: usize, eps: f64, vb: WeightBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    pub(super) fn from_prepared(eps: f64, source: &PreparedTensorSource) -> Result<Self> {
        Ok(Self {
            weight: source.get("weight")?,
            eps,
        })
    }

    pub(super) fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        backend_ops::rms_norm_gated(hidden_states, gate, &self.weight, self.eps)
    }

    pub(super) fn forward_buffer(
        &self,
        hidden_states: &StateBuffer,
        gate: &StateBuffer,
    ) -> Result<StateBuffer> {
        backend_buffer_api::for_device(hidden_states.device())
            .rms_norm_gated(hidden_states, gate, &self.weight, self.eps)
    }

    pub(super) fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub(super) fn eps(&self) -> f64 {
        self.eps
    }
}

#[derive(Debug, Clone)]
pub(super) struct Mlp {
    gate_proj: LinearSource,
    up_proj: LinearSource,
    down_proj: LinearSource,
    act_fn: Activation,
}

impl Mlp {
    #[cfg(any(feature = "hf", test))]
    pub(super) fn new(cfg: &TextConfig, vb: WeightBuilder) -> Result<Self> {
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

    pub(super) fn from_prepared(cfg: &TextConfig, source: &PreparedTensorSource) -> Result<Self> {
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

    pub(super) fn deferred_linear_count(&self) -> usize {
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
    pub(super) fn forward_buffer(&self, xs: &StateBuffer) -> Result<StateBuffer> {
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

    pub(super) fn trace_buffer(&self, xs: &StateBuffer) -> Result<super::types::MlpTrace> {
        let backend = backend_buffer_api::for_device(xs.device());
        let gate_proj_output = self.gate_proj.forward_buffer(xs)?;
        let up_proj_output = self.up_proj.forward_buffer(xs)?;
        let activated_hidden = if matches!(self.act_fn, Activation::Silu) {
            backend.swiglu_mul(&gate_proj_output, &up_proj_output)?
        } else {
            backend.tensor_to_buffer(
                (gate_proj_output.tensor().apply(&self.act_fn)? * up_proj_output.tensor())?,
            )?
        };
        let down_proj_output = self.down_proj.forward_buffer(&activated_hidden)?;
        Ok(super::types::MlpTrace {
            gate_proj_output,
            up_proj_output,
            activated_hidden,
            down_proj_output,
        })
    }
}

pub(super) fn repeat_heads(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}
