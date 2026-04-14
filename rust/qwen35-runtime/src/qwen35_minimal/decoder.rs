use super::backend_buffer_api;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
use super::direct_decoder;
use super::frontend::{
    build_prepared_embedding_source, immutable_embedding_enabled, immutable_linear_enabled,
    prepared_linear_no_bias, profile_elapsed, profile_start, EmbeddingSource, Mlp,
    OutputProjectionSource, Qwen35RmsNorm, RotaryEmbedding,
};
#[cfg(any(feature = "hf", test))]
use super::frontend::embedding;
use super::full_attention::FullAttention;
use super::linear_attention::GatedDeltaNet;
use super::prepared::PreparedTensorSource;
use super::types::{
    CacheState, Config, ExternalFullAttention, LayerCacheState, LinearAttentionBenchResult,
    DecoderLayerTrace, LinearAttentionLayerSpec, LinearAttentionTrace, MlpTrace, RmsNormTrace,
    RuntimeProfile, StateBuffer, TextConfig,
};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::linear_no_bias;
use super::with_tracing::Linear;
use crate::PreparedQwen35DirectMetadata;
use candle::{DType, Device, DeviceLocation, Result, Tensor};
use candle_core as candle;
#[cfg(feature = "qwen35-minimal-hip")]
use std::ffi::c_void;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) enum LayerKind {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    pub(super) layer_type: String,
    pub(super) token_mixer: LayerKind,
    pub(super) mlp: Mlp,
    pub(super) input_layernorm: Qwen35RmsNorm,
    pub(super) post_attention_layernorm: Qwen35RmsNorm,
}

/// Cached device buffers for megakernel scratch, avoiding per-call hipMalloc/hipFree.
/// Protected by a mutex for safety but only accessed from the decode hot path.
#[cfg(feature = "qwen35-minimal-hip")]
pub(super) mod megakernel_scratch {
    use std::ffi::c_void;
    use std::sync::Mutex;

    static SCRATCH: Mutex<Option<ScratchState>> = Mutex::new(None);

    struct ScratchState {
        device_ordinal: usize,
        counter_ptr: *mut c_void,
        proj_table_ptr: *mut c_void,
        proj_table_capacity: usize,
        mlp_scratch_ptr: *mut c_void,
        mlp_scratch_capacity: usize,
    }
    unsafe impl Send for ScratchState {}

    pub(crate) fn get_counter(ordinal: usize) -> candle_core::Result<*mut c_void> {
        let mut guard = SCRATCH.lock().unwrap();
        let state = guard.get_or_insert_with(|| {
            let counter = super::super::hip::alloc_device_bytes(ordinal, 4).expect("alloc counter");
            ScratchState {
                device_ordinal: ordinal,
                counter_ptr: counter,
                proj_table_ptr: std::ptr::null_mut(),
                proj_table_capacity: 0,
                mlp_scratch_ptr: std::ptr::null_mut(),
                mlp_scratch_capacity: 0,
            }
        });
        Ok(state.counter_ptr)
    }

    pub(crate) fn get_proj_table(ordinal: usize, bytes: usize) -> candle_core::Result<*mut c_void> {
        let mut guard = SCRATCH.lock().unwrap();
        let state = guard.get_or_insert_with(|| {
            let counter = super::super::hip::alloc_device_bytes(ordinal, 4).expect("alloc counter");
            ScratchState {
                device_ordinal: ordinal,
                counter_ptr: counter,
                proj_table_ptr: std::ptr::null_mut(),
                proj_table_capacity: 0,
                mlp_scratch_ptr: std::ptr::null_mut(),
                mlp_scratch_capacity: 0,
            }
        });
        if state.proj_table_capacity < bytes {
            if !state.proj_table_ptr.is_null() {
                super::super::hip::free_device_bytes(state.device_ordinal, state.proj_table_ptr);
            }
            state.proj_table_ptr = super::super::hip::alloc_device_bytes(ordinal, bytes)?;
            state.proj_table_capacity = bytes;
        }
        Ok(state.proj_table_ptr)
    }

    pub(crate) fn get_mlp_scratch(ordinal: usize, bytes: usize) -> candle_core::Result<*mut c_void> {
        let mut guard = SCRATCH.lock().unwrap();
        let state = guard.get_or_insert_with(|| {
            let counter = super::super::hip::alloc_device_bytes(ordinal, 4).expect("alloc counter");
            ScratchState {
                device_ordinal: ordinal,
                counter_ptr: counter,
                proj_table_ptr: std::ptr::null_mut(),
                proj_table_capacity: 0,
                mlp_scratch_ptr: std::ptr::null_mut(),
                mlp_scratch_capacity: 0,
            }
        });
        if state.mlp_scratch_capacity < bytes {
            if !state.mlp_scratch_ptr.is_null() {
                super::super::hip::free_device_bytes(state.device_ordinal, state.mlp_scratch_ptr);
            }
            state.mlp_scratch_ptr = super::super::hip::alloc_device_bytes(ordinal, bytes)?;
            state.mlp_scratch_capacity = bytes;
        }
        Ok(state.mlp_scratch_ptr)
    }
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

    /// Perform residual addition in F32 to avoid BF16 rounding accumulation.
    /// Upcasts both operands to F32, adds, then casts back to the hidden dtype.
    fn f32_residual_add(
        residual: &StateBuffer,
        xs: &StateBuffer,
        hidden_dtype: DType,
    ) -> Result<StateBuffer> {
        if hidden_dtype == DType::F32 {
            let backend = backend_buffer_api::for_device(residual.device());
            return backend.add(residual, xs);
        }
        let sum = residual
            .tensor()
            .to_dtype(DType::F32)?
            .broadcast_add(&xs.tensor().to_dtype(DType::F32)?)?
            .to_dtype(hidden_dtype)?;
        StateBuffer::from_tensor(sum)
    }

    fn use_attn_megakernel(device: &Device, xs: &StateBuffer) -> bool {
        if !matches!(device.location(), DeviceLocation::Hip { .. }) {
            return false;
        }
        let Ok((_, seq_len, _)) = xs.dims3() else {
            return false;
        };
        if seq_len != 1 {
            return false;
        }
        matches!(
            std::env::var("CANDLE_QWEN35_ATTN_MEGAKERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    fn use_mlp_megakernel(device: &Device, xs: &StateBuffer) -> bool {
        if !matches!(device.location(), DeviceLocation::Hip { .. }) {
            return false;
        }
        let Ok((_, seq_len, _)) = xs.dims3() else {
            return false;
        };
        if seq_len != 1 {
            return false;
        }
        matches!(
            std::env::var("CANDLE_QWEN35_MLP_MEGAKERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn mlp_megakernel_forward(
        &self,
        pre_norm_hidden: &StateBuffer,
        norm: &Qwen35RmsNorm,
    ) -> Result<StateBuffer> {
        use candle::Storage;
        use std::ffi::c_void;

        let hidden = pre_norm_hidden.contiguous()?;
        let (b_sz, seq_len, hidden_dim) = hidden.dims3()?;
        if b_sz != 1 || seq_len != 1 {
            candle::bail!("mlp_megakernel requires batch=1 seq=1, got b={b_sz} s={seq_len}");
        }
        let ordinal = match hidden.device().location() {
            DeviceLocation::Hip { gpu_id } => gpu_id,
            _ => candle::bail!("mlp_megakernel requires HIP device"),
        };
        let dtype_code = super::hip::dtype_code(hidden.tensor().dtype())
            .map_err(|e| candle::Error::Msg(format!("mlp megakernel dtype: {e}")))?;

        let (gate_w, up_w, down_w) = self.mlp.weight_tensors()?;
        let norm_w = norm.weight().contiguous()?;
        let gate_w = gate_w.contiguous()?;
        let up_w = up_w.contiguous()?;
        let down_w = down_w.contiguous()?;

        let intermediate_size = gate_w.dim(0)?;

        // Use cached scratch buffers (allocated once, reused across calls)
        let scratch_bytes = intermediate_size * 2 * std::mem::size_of::<f32>();
        let scratch_ptr = megakernel_scratch::get_mlp_scratch(ordinal, scratch_bytes)?;
        let counter_ptr = megakernel_scratch::get_counter(ordinal)?;

        // Allocate output tensor on device — stays on GPU
        let out_tensor = Tensor::zeros(
            (1, 1, hidden_dim), hidden.tensor().dtype(), hidden.device(),
        )?;
        let (out_storage, out_layout) = out_tensor.storage_and_layout();
        let Storage::Hip(out_s) = &*out_storage else {
            super::hip::free_device_bytes(ordinal, scratch_ptr);
            super::hip::free_device_bytes(ordinal, counter_ptr);
            candle::bail!("mlp megakernel: output tensor not on HIP");
        };
        let out_device_ptr =
            out_s.raw_device_ptr_with_offset(out_layout.start_offset())? as *mut c_void;

        // Extract device pointers
        let (h_storage, h_layout) = hidden.tensor().storage_and_layout();
        let (nw_storage, nw_layout) = norm_w.storage_and_layout();
        let (gw_storage, gw_layout) = gate_w.storage_and_layout();
        let (uw_storage, uw_layout) = up_w.storage_and_layout();
        let (dw_storage, dw_layout) = down_w.storage_and_layout();

        let (Storage::Hip(h_s), Storage::Hip(nw_s), Storage::Hip(gw_s), Storage::Hip(uw_s), Storage::Hip(dw_s)) =
            (&*h_storage, &*nw_storage, &*gw_storage, &*uw_storage, &*dw_storage)
        else {
            candle::bail!("mlp megakernel: all tensors must be on HIP");
        };

        let status = unsafe {
            super::hip::ffi::dotcache_qwen35_hip_mlp_decode_megakernel(
                dtype_code,
                ordinal,
                hidden_dim,
                intermediate_size,
                norm.eps() as f32,
                h_s.raw_device_ptr_with_offset(h_layout.start_offset())? as *const c_void,
                nw_s.raw_device_ptr_with_offset(nw_layout.start_offset())? as *const c_void,
                gw_s.raw_device_ptr_with_offset(gw_layout.start_offset())? as *const c_void,
                uw_s.raw_device_ptr_with_offset(uw_layout.start_offset())? as *const c_void,
                dw_s.raw_device_ptr_with_offset(dw_layout.start_offset())? as *const c_void,
                scratch_ptr as *mut c_void,
                out_device_ptr,
                counter_ptr as *mut c_void,
            )
        };

        // Drop storage refs before using out_tensor
        drop(out_storage);
        drop(h_storage);
        drop(nw_storage);
        drop(gw_storage);
        drop(uw_storage);
        drop(dw_storage);
        // scratch_ptr and counter_ptr are cached — not freed here

        if status != 0 {
            candle::bail!("mlp-decode-megakernel: HIP kernel failed with status {status}");
        }

        Ok(StateBuffer::from_tensor(out_tensor)?)
    }

    #[cfg(not(feature = "qwen35-minimal-hip"))]
    fn mlp_megakernel_forward(
        &self,
        _pre_norm_hidden: &StateBuffer,
        _norm: &Qwen35RmsNorm,
    ) -> Result<StateBuffer> {
        candle::bail!("mlp megakernel requires qwen35-minimal-hip feature")
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
        let hidden_dtype = xs.tensor().dtype();
        let mut profile = RuntimeProfile::default();
        let residual = xs.clone();
        let xs = match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => {
                if Self::use_attn_megakernel(device, xs) {
                    if let Some((qkv, z, b, a)) =
                        linear_attn.fused_norm_projections(xs, &self.input_layernorm)?
                    {
                        let (xs, lp) = linear_attn.forward_from_projections_buffer(
                            &qkv, &z, &b, &a, attention_mask,
                        )?;
                        profile.add_assign(&lp);
                        xs
                    } else {
                        let xs_norm = self.input_layernorm.forward_buffer(xs)?;
                        let (xs, lp) = linear_attn.forward_profiled_buffer(&xs_norm, attention_mask)?;
                        profile.add_assign(&lp);
                        xs
                    }
                } else {
                    let xs_norm = self.input_layernorm.forward_buffer(xs)?;
                    let (xs, layer_profile) =
                        linear_attn.forward_profiled_buffer(&xs_norm, attention_mask)?;
                    profile.add_assign(&layer_profile);
                    xs
                }
            }
            LayerKind::Full(self_attn) => {
                if Self::use_attn_megakernel(device, xs) {
                    if let Some((q, k, v)) =
                        self_attn.fused_norm_qkv_projections(xs, &self.input_layernorm)?
                    {
                        let (xs, layer_profile) = self_attn.forward_from_projections_buffer(
                            &q, &k, &v,
                            attention_mask, seqlen_offset, layer_id,
                            external_full_attention, hidden_dtype,
                        )?;
                        profile.add_assign(&layer_profile);
                        xs
                    } else {
                        let xs_norm = self.input_layernorm.forward_buffer(xs)?;
                        let (xs, lp) = self_attn.forward_profiled_with_external_buffer(
                            &xs_norm, attention_mask, seqlen_offset, layer_id,
                            external_full_attention,
                        )?;
                        profile.add_assign(&lp);
                        xs
                    }
                } else {
                    let xs_norm = self.input_layernorm.forward_buffer(xs)?;
                    let (xs, layer_profile) = self_attn.forward_profiled_with_external_buffer(
                        &xs_norm, attention_mask, seqlen_offset, layer_id,
                        external_full_attention,
                    )?;
                    profile.add_assign(&layer_profile);
                    xs
                }
            }
        };
        let xs = Self::f32_residual_add(&residual, &xs, hidden_dtype)?;
        let residual = xs.clone();
        let mlp_start = profile_start(device)?;
        let xs = if Self::use_mlp_megakernel(device, &xs) {
            self.mlp_megakernel_forward(&xs, &self.post_attention_layernorm)?
        } else {
            let normed = self.post_attention_layernorm.forward_buffer(&xs)?;
            self.mlp.forward_buffer(&normed)?
        };
        profile.mlp_millis += profile_elapsed(mlp_start, device)?;
        Ok((Self::f32_residual_add(&residual, &xs, hidden_dtype)?, profile))
    }

    pub(super) fn forward_profiled(
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

    pub(super) fn cache_state(&self) -> LayerCacheState {
        match &self.token_mixer {
            LayerKind::Linear(linear_attn) => LayerCacheState::Linear(linear_attn.cache_state()),
            LayerKind::Full(self_attn) => LayerCacheState::Full(self_attn.cache_state()),
        }
    }

    pub(super) fn restore_cache_state(&mut self, state: &LayerCacheState) -> Result<()> {
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

/// Cached device-side allocations for the persistent decode kernel.
/// Allocated lazily on first use, reused across tokens to avoid hipMalloc/hipFree overhead.
#[cfg(feature = "qwen35-minimal-hip")]
struct PersistentDecodeCache {
    ordinal: usize,
    desc_ptr: *mut c_void,
    desc_capacity: usize,
    workspace_ptr: *mut c_void,
    /// Combined: counters[4×u32] + barrier_counter[u32] + barrier_flag[u32] = 24 bytes
    sync_ptr: *mut c_void,
}

#[cfg(feature = "qwen35-minimal-hip")]
impl PersistentDecodeCache {
    fn counters_ptr(&self) -> *mut c_void { self.sync_ptr }
    fn barrier_counter_ptr(&self) -> *mut c_void {
        unsafe { (self.sync_ptr as *mut u8).add(16) as *mut c_void }
    }
    fn barrier_flag_ptr(&self) -> *mut c_void {
        unsafe { (self.sync_ptr as *mut u8).add(20) as *mut c_void }
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
impl Drop for PersistentDecodeCache {
    fn drop(&mut self) {
        super::hip::free_device_bytes(self.ordinal, self.desc_ptr);
        super::hip::free_device_bytes(self.ordinal, self.workspace_ptr);
        super::hip::free_device_bytes(self.ordinal, self.sync_ptr);
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
impl Clone for PersistentDecodeCache {
    fn clone(&self) -> Self {
        // Don't clone device pointers — the clone gets a fresh cache.
        Self { ordinal: 0, desc_ptr: std::ptr::null_mut(), desc_capacity: 0,
               workspace_ptr: std::ptr::null_mut(), sync_ptr: std::ptr::null_mut() }
    }
}

#[cfg(feature = "qwen35-minimal-hip")]
impl std::fmt::Debug for PersistentDecodeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PersistentDecodeCache(ordinal={})", self.ordinal)
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: EmbeddingSource,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) norm: Qwen35RmsNorm,
    device: Device,
    dtype: DType,
    immutable_embedding_requested: bool,
    immutable_embedding_active: bool,
    immutable_embedding_fallback_reason: Option<String>,
    immutable_linear_requested: bool,
    deferred_linear_count: usize,
    #[cfg(feature = "qwen35-minimal-hip")]
    persistent_decode_cache: Option<PersistentDecodeCache>,
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
            #[cfg(feature = "qwen35-minimal-hip")]
            persistent_decode_cache: None,
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
            #[cfg(feature = "qwen35-minimal-hip")]
            persistent_decode_cache: None,
        })
    }

    pub(super) fn prepare_causal_attention_mask(
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

    pub fn trace_decoder_layer_output(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        if target_layer >= self.layers.len() {
            candle::bail!(
                "decoder trace target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
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
        let mask = if target.layer_type() == "full_attention" {
            attention_mask.as_ref()
        } else {
            None
        };
        let (next_xs, _) = target.forward_profiled(&xs, mask, seqlen_offset)?;
        self.clear_kv_cache();
        Ok(next_xs)
    }

    pub fn trace_decoder_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<DecoderLayerTrace> {
        if target_layer >= self.layers.len() {
            candle::bail!(
                "decoder trace target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
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
        let mask = if target.layer_type() == "full_attention" {
            attention_mask.as_ref()
        } else {
            None
        };
        let input_layernorm_trace = Some(target.input_layernorm.trace_buffer(&xs)?);
        let input_layernorm_output = input_layernorm_trace
            .as_ref()
            .expect("just created input rmsnorm trace")
            .output
            .clone();
        let (linear_projection_trace, linear_core_trace, full_attention_trace, token_mixer_output, _) =
            match &mut target.token_mixer {
            LayerKind::Linear(linear_attn) => {
                let projection_trace =
                    linear_attn.trace_projection_components_buffer(&input_layernorm_output)?;
                let (core_trace, token_mixer_output, _, profile) =
                    linear_attn.trace_core_components_buffer(&input_layernorm_output, mask)?;
                (
                    Some(projection_trace),
                    Some(core_trace),
                    None,
                    token_mixer_output,
                    profile,
                )
            }
            LayerKind::Full(self_attn) => self_attn
                .trace_components_buffer(&input_layernorm_output, mask, seqlen_offset)
                .map(|(trace, output, profile)| (None, None, Some(trace), output, profile))?,
        };
        let backend = backend_buffer_api::for_device(xs.device());
        let attention_residual = backend.add(&xs, &token_mixer_output)?;
        let post_attention_layernorm_trace =
            Some(target.post_attention_layernorm.trace_buffer(&attention_residual)?);
        let post_attention_layernorm_output = post_attention_layernorm_trace
            .as_ref()
            .expect("just created post-attention rmsnorm trace")
            .output
            .clone();
        let mlp_trace = Some(target.mlp.trace_buffer(&post_attention_layernorm_output)?);
        let mlp_output = mlp_trace
            .as_ref()
            .expect("just created mlp trace")
            .down_proj_output
            .clone();
        let layer_output = backend.add(&attention_residual, &mlp_output)?;
        self.clear_kv_cache();
        Ok(DecoderLayerTrace {
            layer_id: target_layer,
            sequence_length: seq_len,
            input_layernorm_trace,
            input_layernorm_output,
            linear_projection_trace,
            linear_core_trace,
            full_attention_trace,
            token_mixer_output,
            post_attention_layernorm_trace,
            post_attention_layernorm_output,
            mlp_trace,
            mlp_output,
            layer_output,
        })
    }

    pub fn trace_decoder_layer_with_cache(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        cache_state: &CacheState,
    ) -> Result<DecoderLayerTrace> {
        let saved_cache = self.cache_state();
        self.restore_cache_state(cache_state)?;
        let trace = self.trace_decoder_layer_without_clearing_cache(
            input_ids,
            target_layer,
            seqlen_offset,
        );
        self.restore_cache_state(&saved_cache)?;
        trace
    }

    fn trace_decoder_layer_without_clearing_cache(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<DecoderLayerTrace> {
        if target_layer >= self.layers.len() {
            candle::bail!(
                "decoder trace target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
            );
        }

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
        let mask = if target.layer_type() == "full_attention" {
            attention_mask.as_ref()
        } else {
            None
        };
        let input_layernorm_trace = Some(target.input_layernorm.trace_buffer(&xs)?);
        let input_layernorm_output = input_layernorm_trace
            .as_ref()
            .expect("just created input rmsnorm trace")
            .output
            .clone();
        let (linear_projection_trace, linear_core_trace, full_attention_trace, token_mixer_output, _) =
            match &mut target.token_mixer {
                LayerKind::Linear(linear_attn) => {
                    let projection_trace =
                        linear_attn.trace_projection_components_buffer(&input_layernorm_output)?;
                    let (core_trace, token_mixer_output, _, profile) =
                        linear_attn.trace_core_components_buffer(&input_layernorm_output, mask)?;
                    (
                        Some(projection_trace),
                        Some(core_trace),
                        None,
                        token_mixer_output,
                        profile,
                    )
                }
                LayerKind::Full(self_attn) => self_attn
                    .trace_components_buffer(&input_layernorm_output, mask, seqlen_offset)
                    .map(|(trace, output, profile)| (None, None, Some(trace), output, profile))?,
            };
        let backend = backend_buffer_api::for_device(xs.device());
        let attention_residual = backend.add(&xs, &token_mixer_output)?;
        let post_attention_layernorm_trace =
            Some(target.post_attention_layernorm.trace_buffer(&attention_residual)?);
        let post_attention_layernorm_output = post_attention_layernorm_trace
            .as_ref()
            .expect("just created post-attention rmsnorm trace")
            .output
            .clone();
        let mlp_trace = Some(target.mlp.trace_buffer(&post_attention_layernorm_output)?);
        let mlp_output = mlp_trace
            .as_ref()
            .expect("just created mlp trace")
            .down_proj_output
            .clone();
        let layer_output = backend.add(&attention_residual, &mlp_output)?;
        Ok(DecoderLayerTrace {
            layer_id: target_layer,
            sequence_length: seq_len,
            input_layernorm_trace,
            input_layernorm_output,
            linear_projection_trace,
            linear_core_trace,
            full_attention_trace,
            token_mixer_output,
            post_attention_layernorm_trace,
            post_attention_layernorm_output,
            mlp_trace,
            mlp_output,
            layer_output,
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

    #[cfg(feature = "qwen35-minimal-hip")]
    fn use_persistent_decode(&self, device: &Device, seq_len: usize) -> bool {
        if seq_len != 1 { return false; }
        if !matches!(device.location(), DeviceLocation::Hip { .. }) { return false; }
        matches!(
            std::env::var("CANDLE_QWEN35_PERSISTENT_DECODE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    /// Extract a raw HIP device pointer from a contiguous tensor.
    #[cfg(feature = "qwen35-minimal-hip")]
    fn tensor_device_ptr(t: &Tensor) -> Result<*const c_void> {
        use candle::Storage;
        let t = t.contiguous()?;
        let (storage, layout) = t.storage_and_layout();
        let Storage::Hip(hip_s) = &*storage else {
            candle::bail!("tensor_device_ptr: not on HIP");
        };
        Ok(hip_s.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void)
    }

    /// Build DecodeLayerDesc array from model layers (called once per forward pass).
    #[cfg(feature = "qwen35-minimal-hip")]
    fn build_layer_descs(&mut self, seqlen_offset: usize) -> Result<Vec<super::DecodeLayerDesc>> {
        use std::ptr;

        let mut descs = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let mut d = super::DecodeLayerDesc::default();

            // Common fields
            d.input_norm_eps = layer.input_layernorm.eps() as f32;
            d.post_attn_norm_eps = layer.post_attention_layernorm.eps() as f32;
            d.input_norm_w = Self::tensor_device_ptr(layer.input_layernorm.weight())? as *const _;
            d.post_attn_norm_w = Self::tensor_device_ptr(layer.post_attention_layernorm.weight())? as *const _;

            // MLP
            let (gate_w, up_w, down_w) = layer.mlp.weight_tensors()?;
            d.gate_proj_w = Self::tensor_device_ptr(&gate_w)?;
            d.up_proj_w = Self::tensor_device_ptr(&up_w)?;
            d.down_proj_w = Self::tensor_device_ptr(&down_w)?;
            d.intermediate_size = gate_w.dim(0)? as i32;

            match &mut layer.token_mixer {
                LayerKind::Linear(la) => {
                    d.layer_type = 0;
                    la.fill_persistent_desc(&mut d)?;
                }
                LayerKind::Full(fa) => {
                    d.layer_type = 1;
                    fa.fill_persistent_desc(&mut d, seqlen_offset)?;
                }
            }
            descs.push(d);
        }
        Ok(descs)
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();

        // Try persistent decode kernel for single-token decode
        #[cfg(feature = "qwen35-minimal-hip")]
        if self.use_persistent_decode(device, seq_len) {
            let xs = self.hidden_states_from_input_ids(input_ids)?;
            if let Ok(result) = self.persistent_decode_forward(&xs, seqlen_offset) {
                let normed = self.norm.forward_buffer(&result)?;
                return Ok((normed.clone_tensor(), profile));
            }
        }

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

    #[cfg(feature = "qwen35-minimal-hip")]
    fn persistent_decode_forward(
        &mut self,
        hidden: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        let hidden = hidden.contiguous()?;
        let (_, _, hidden_dim) = hidden.dims3()?;
        let ordinal = match hidden.device().location() {
            DeviceLocation::Hip { gpu_id } => gpu_id,
            _ => candle::bail!("persistent decode requires HIP"),
        };
        let dtype_code = super::hip::dtype_code(hidden.tensor().dtype())
            .map_err(|e| candle::Error::Msg(format!("{e}")))?;

        let num_layers = self.layers.len();
        let intermediate_size = self.layers[0].mlp.weight_tensors()?.0.dim(0)?;

        // Extract RoPE cos/sin device pointers from first full attention layer
        let (cos_ptr, sin_ptr, rotary_dim) = {
            let mut found = None;
            for layer in &self.layers {
                if let LayerKind::Full(fa) = &layer.token_mixer {
                    let re = fa.rotary_emb();
                    let cos_p = Self::tensor_device_ptr(re.cos())?;
                    let sin_p = Self::tensor_device_ptr(re.sin())?;
                    found = Some((cos_p, sin_p, re.rotary_dim()));
                    break;
                }
            }
            found.unwrap_or((std::ptr::null(), std::ptr::null(), 0))
        };

        // Build layer descriptors (must be rebuilt each token — kv_len changes)
        let descs = self.build_layer_descs(seqlen_offset)?;
        let desc_bytes = descs.len() * std::mem::size_of::<super::DecodeLayerDesc>();

        // Lazily initialize cached device allocations (workspace, sync buffers)
        let cache = if let Some(ref mut c) = self.persistent_decode_cache {
            c
        } else {
            let workspace_floats = hidden_dim + hidden_dim + intermediate_size * 2
                + hidden_dim + hidden_dim + 8224 + 2048;
            let workspace_bytes = workspace_floats * std::mem::size_of::<f32>();
            let workspace_ptr = super::hip::alloc_device_bytes(ordinal, workspace_bytes)?;
            // counters[4×u32=16B] + barrier_counter[u32=4B] + barrier_flag[u32=4B] = 24B
            let sync_ptr = super::hip::alloc_device_bytes(ordinal, 24)?;
            // Zero the sync region (barrier_counter=0, barrier_flag=0 for first launch)
            super::hip::memset_device_bytes(ordinal, sync_ptr, 0, 24)?;
            let desc_ptr = super::hip::alloc_device_bytes(ordinal, desc_bytes)?;
            self.persistent_decode_cache = Some(PersistentDecodeCache {
                ordinal,
                desc_ptr,
                desc_capacity: desc_bytes,
                workspace_ptr,
                sync_ptr,
            });
            self.persistent_decode_cache.as_mut().unwrap()
        };

        // Validate descriptors: check for null weight pointers
        for (i, d) in descs.iter().enumerate() {
            if d.input_norm_w.is_null() || d.gate_proj_w.is_null() || d.down_proj_w.is_null() {
                candle::bail!("persistent decode: layer {i} has null common weight pointers");
            }
            if d.layer_type == 0 {
                // Linear attention
                if d.qkv_proj_w.is_null() || d.conv1d_w.is_null() || d.linear_out_proj_w.is_null() {
                    candle::bail!("persistent decode: linear layer {i} has null projection weights");
                }
                if d.conv_state.is_null() || d.recurrent_state.is_null() {
                    candle::bail!("persistent decode: linear layer {i} has null state (conv={} rec={})",
                        !d.conv_state.is_null(), !d.recurrent_state.is_null());
                }
            } else {
                // Full attention
                if d.q_proj_w.is_null() || d.o_proj_w.is_null() {
                    candle::bail!("persistent decode: full attn layer {i} has null projection weights");
                }
                if d.kv_cache_k.is_null() || d.kv_cache_v.is_null() {
                    candle::bail!("persistent decode: full attn layer {i} has null KV cache");
                }
            }
        }

        // Re-upload descriptors (kv_len changes every token)
        if desc_bytes > cache.desc_capacity {
            super::hip::free_device_bytes(cache.ordinal, cache.desc_ptr);
            cache.desc_ptr = super::hip::alloc_device_bytes(ordinal, desc_bytes)?;
            cache.desc_capacity = desc_bytes;
        }
        super::hip::copy_host_to_device(
            ordinal, cache.desc_ptr, descs.as_ptr() as *const c_void, desc_bytes,
        )?;

        // Create output tensor (kernel writes BF16 in-place)
        let output = Tensor::zeros((1, 1, hidden_dim), hidden.tensor().dtype(), hidden.device())?;

        // Copy input hidden to output (kernel reads/writes same buffer)
        {
            use candle::Storage;
            let (h_s, h_l) = hidden.tensor().storage_and_layout();
            let (o_s, o_l) = output.storage_and_layout();
            let (Storage::Hip(h_hip), Storage::Hip(o_hip)) = (&*h_s, &*o_s) else {
                candle::bail!("persistent decode: tensors not on HIP");
            };
            let bytes = hidden_dim * hidden.tensor().dtype().size_in_bytes();
            super::hip::copy_device_to_device(
                ordinal,
                o_hip.raw_device_ptr_with_offset(o_l.start_offset())? as *mut c_void,
                h_hip.raw_device_ptr_with_offset(h_l.start_offset())? as *const c_void,
                bytes,
            )?;
            let hidden_io_ptr = o_hip.raw_device_ptr_with_offset(o_l.start_offset())? as *mut c_void;
            drop(h_s); drop(o_s);

            let status = unsafe {
                super::hip::ffi::dotcache_qwen35_hip_persistent_decode(
                    dtype_code,
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    cache.desc_ptr as *const c_void,
                    hidden_io_ptr,
                    cache.workspace_ptr as *mut c_void,
                    cache.counters_ptr() as *mut c_void,
                    cache.barrier_counter_ptr() as *mut c_void,
                    cache.barrier_flag_ptr() as *mut c_void,
                    cos_ptr,
                    sin_ptr,
                    rotary_dim,
                )
            };

            if status != 0 {
                candle::bail!("persistent decode kernel failed with status {status}");
            }
        }

        StateBuffer::from_tensor(output)
    }

    pub fn forward_hidden_states_profiled(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        let device = hidden_states.device();
        let (b_size, seq_len, _) = hidden_states.dims3()?;
        let mut profile = RuntimeProfile::default();

        // Try persistent decode kernel for single-token decode
        // Requires seqlen_offset > 0 (need initialized KV/conv/recurrent state from prefill)
        #[cfg(feature = "qwen35-minimal-hip")]
        if seqlen_offset > 0 && self.use_persistent_decode(device, seq_len) {
            if let Ok(result) = self.persistent_decode_forward(hidden_states, seqlen_offset) {
                let normed = self.norm.forward_buffer(&result)?;
                return Ok((normed, profile));
            }
        }

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

    pub fn trace_final_norm_from_hidden_states(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<RmsNormTrace> {
        let (b_size, seq_len, _) = hidden_states.dims3()?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        let mut xs = hidden_states.clone();
        for layer in self.layers.iter_mut() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, _) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            xs = next_xs;
        }
        self.norm.trace_buffer(&xs)
    }

    pub fn trace_layer_mlp_from_buffer(
        &self,
        layer_id: usize,
        input: &StateBuffer,
    ) -> Result<super::types::MlpTrace> {
        let layer = self.layers.get(layer_id).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "trace_layer_mlp_from_buffer: layer {} out of range for {} layers",
                layer_id,
                self.layers.len()
            ))
        })?;
        layer.mlp.trace_buffer(input)
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
    pub(super) language_model: TextModel,
    pub(super) lm_head: OutputProjectionSource,
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
        direct_decoder::model_forward_hidden_states_profiled_direct_hip_v1(
            self,
            metadata,
            hidden_states,
            seqlen_offset,
        )
    }

    pub(crate) fn validate_direct_hip_metadata(
        &self,
        metadata: &PreparedQwen35DirectMetadata,
    ) -> Result<()> {
        direct_decoder::model_validate_direct_hip_metadata(self, metadata)
    }

    pub(crate) fn finalize_direct_decode_logits_hip_v1(
        &mut self,
        hidden_states: &StateBuffer,
        logits_scratch: &StateBuffer,
    ) -> Result<(StateBuffer, RuntimeProfile)> {
        direct_decoder::model_finalize_direct_decode_logits_hip_v1(
            self,
            hidden_states,
            logits_scratch,
        )
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

    pub fn forward_text_hidden_states(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        self.language_model
            .forward_hidden_states_profiled(hidden_states, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn trace_text_final_norm_from_hidden_states(
        &mut self,
        hidden_states: &StateBuffer,
        seqlen_offset: usize,
    ) -> Result<RmsNormTrace> {
        self.language_model
            .trace_final_norm_from_hidden_states(hidden_states, seqlen_offset)
    }

    pub fn project_final_hidden_to_logits(
        &mut self,
        final_hidden_states: &StateBuffer,
    ) -> Result<StateBuffer> {
        let backend = backend_buffer_api::for_device(final_hidden_states.device());
        let last_token = backend.slice_last_token(final_hidden_states)?;
        self.lm_head.forward_buffer(&last_token)
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

    pub fn trace_decoder_layer_output(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<StateBuffer> {
        self.language_model
            .trace_decoder_layer_output(input_ids, target_layer, seqlen_offset)
    }

    pub fn trace_decoder_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<DecoderLayerTrace> {
        self.language_model
            .trace_decoder_layer(input_ids, target_layer, seqlen_offset)
    }

    pub fn trace_decoder_layer_with_cache(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        cache_state: &CacheState,
    ) -> Result<DecoderLayerTrace> {
        self.language_model.trace_decoder_layer_with_cache(
            input_ids,
            target_layer,
            seqlen_offset,
            cache_state,
        )
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

    pub fn trace_layer_mlp_from_buffer(
        &self,
        layer_id: usize,
        input: &StateBuffer,
    ) -> Result<MlpTrace> {
        self.language_model.trace_layer_mlp_from_buffer(layer_id, input)
    }
}
