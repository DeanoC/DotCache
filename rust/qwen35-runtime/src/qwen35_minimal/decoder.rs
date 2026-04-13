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
    DecoderLayerTrace, LinearAttentionLayerSpec, LinearAttentionTrace, RuntimeProfile,
    StateBuffer, TextConfig,
};
#[cfg(any(feature = "hf", test))]
use super::with_tracing::linear_no_bias;
use super::with_tracing::Linear;
use crate::PreparedQwen35DirectMetadata;
use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
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
        let mlp_output = target.mlp.forward_buffer(&post_attention_layernorm_output)?;
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
        direct_decoder::text_model_forward_hidden_states_profiled_direct_hip_v1(
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
        direct_decoder::validate_text_model_direct_hip_metadata(self, metadata)
    }

    pub(crate) fn finalize_direct_decode_hidden_hip_v1(
        &mut self,
        xs: &StateBuffer,
    ) -> Result<StateBuffer> {
        direct_decoder::text_model_finalize_direct_decode_hidden_hip_v1(self, xs)
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
