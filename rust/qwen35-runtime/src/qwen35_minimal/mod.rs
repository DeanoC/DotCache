mod activation;
mod backend_buffer_api;
pub(crate) mod backend_ops;
#[cfg(any(feature = "hf", test))]
mod builder;
mod hip;
pub(crate) mod model;
mod ops;
mod prepared;
mod rotary;
mod with_tracing;

pub use model::{
    CacheState as MinimalQwen35KvCache, Config as MinimalQwen35Config,
    LinearAttentionBenchResult as MinimalQwen35LinearAttentionBenchResult,
    LinearAttentionLayerSpec as MinimalQwen35LinearAttentionLayerSpec, ModelForCausalLM,
    NativeCacheState as MinimalQwen35NativeCacheState,
    NativeFullAttentionCacheState as MinimalQwen35NativeFullAttentionCacheState,
    NativeLayerCacheState as MinimalQwen35NativeLayerCacheState,
    NativeLinearAttentionCacheState as MinimalQwen35NativeLinearAttentionCacheState,
    LinearAttentionTrace as MinimalQwen35LinearAttentionTrace,
    StateBuffer as MinimalQwen35StateBuffer,
    TextConfig as MinimalQwen35TextConfig,
};

use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::{ModelPackage, PreparedPackageSummary, Result, WeightLoadStats};
#[cfg(feature = "hf")]
use crate::HfHubModelSource;
#[cfg(any(feature = "hf", test))]
use builder::WeightBuilder;
use prepared::PreparedTensorSource;

#[derive(Debug, Clone)]
pub struct MinimalQwen35Weights {
    pub model_id: String,
    pub revision: String,
    pub tokenizer_path: PathBuf,
    pub package_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MinimalQwen35LoadTrace {
    pub package_resolve_millis: Option<f64>,
    pub config_parse_millis: f64,
    pub model_build_millis: f64,
    pub total_load_millis: f64,
    pub package_stats: Option<PreparedPackageSummary>,
    pub weight_load_stats: Option<WeightLoadStats>,
    pub immutable_embedding_requested: bool,
    pub immutable_embedding_active: bool,
    pub immutable_embedding_fallback_reason: Option<String>,
    pub immutable_embedding_runtime_mode: String,
    pub immutable_linear_requested: bool,
    pub deferred_linear_count: usize,
}

#[derive(Debug)]
pub struct MinimalQwen35Runner {
    pub config: MinimalQwen35Config,
    pub weights: MinimalQwen35Weights,
    model: ModelForCausalLM,
    device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimalQwen35LoadMode {
    DirectHf,
    NativeStore,
}

impl MinimalQwen35LoadMode {
    fn from_env() -> Option<Self> {
        let raw = std::env::var("DOTCACHE_QWEN35_LOAD_MODE").ok()?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "direct" | "hf" | "direct-hf" => Some(Self::DirectHf),
            "prepared" | "prepared-candle" | "native" | "native-store" => Some(Self::NativeStore),
            _ => None,
        }
    }
}

impl MinimalQwen35Runner {
    fn hidden_states_on_runner_device(
        &self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<Tensor> {
        let hidden_states = hidden_states.tensor();
        if hidden_states.device().same_device(&self.device) {
            Ok(hidden_states.clone())
        } else {
            Ok(hidden_states.to_device(&self.device)?)
        }
    }

    pub fn load_from_hf_direct_f16(model_id: &str, device: &Device) -> Result<Self> {
        #[cfg(not(feature = "hf"))]
        {
            let _ = (model_id, device);
            return Err(crate::RuntimeError::External {
                context: "qwen35-runtime",
                message: "direct HF loading requires the `hf` feature".to_string(),
            });
        }
        #[cfg(feature = "hf")]
        {
        let source = HfHubModelSource::new()?;
        let artifacts = source.snapshot(model_id)?;
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(&artifacts.config_path)?)?;
        let vb = unsafe {
            WeightBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F16, device)?
        };
        let model = ModelForCausalLM::new(&config, vb)?;
        Ok(Self {
            config,
            weights: MinimalQwen35Weights {
                model_id: artifacts.model_id,
                revision: artifacts.revision,
                tokenizer_path: artifacts.tokenizer_path,
                package_root: PathBuf::new(),
            },
            model,
            device: device.clone(),
        })
        }
    }

    pub fn load_native_for_device(model_id: &str, device: &Device) -> Result<Self> {
        let package = Arc::new(
            ModelPackage::resolve_or_build_qwen35_minimal(model_id, device).map_err(|err| {
                crate::RuntimeError::External {
                    context: "model-store",
                    message: err.to_string(),
                }
            })?,
        );
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(package.config_path())?)?;
        let source = PreparedTensorSource::new(package.clone(), device.clone());
        let model = ModelForCausalLM::from_prepared(&config, source)?;
        Ok(Self {
            config,
            weights: MinimalQwen35Weights {
                model_id: package.manifest().model_id.clone(),
                revision: package.manifest().revision.clone(),
                tokenizer_path: package.tokenizer_path(),
                package_root: package.root().to_path_buf(),
            },
            model,
            device: device.clone(),
        })
    }

    pub fn load_native_for_device_profiled(
        model_id: &str,
        device: &Device,
    ) -> Result<(Self, MinimalQwen35LoadTrace)> {
        let total_started = Instant::now();
        let package_started = Instant::now();
        let package = Arc::new(
            ModelPackage::resolve_or_build_qwen35_minimal(model_id, device).map_err(|err| {
                crate::RuntimeError::External {
                    context: "model-store",
                    message: err.to_string(),
                }
            })?,
        );
        let package_resolve_millis = package_started.elapsed().as_secs_f64() * 1000.0;
        let package_stats = package.stats().map_err(|err| crate::RuntimeError::External {
            context: "model-store",
            message: err.to_string(),
        })?;
        let config_started = Instant::now();
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(package.config_path())?)?;
        let config_parse_millis = config_started.elapsed().as_secs_f64() * 1000.0;
        let model_started = Instant::now();
        let source = PreparedTensorSource::new_profiled(package.clone(), device.clone());
        let model = ModelForCausalLM::from_prepared(
            &config,
            source.clone(),
        )?;
        let model_build_millis = model_started.elapsed().as_secs_f64() * 1000.0;
        let weight_load_stats = source.load_stats();
        let immutable_embedding_requested = model.immutable_embedding_requested();
        let immutable_embedding_active = model.immutable_embedding_active();
        let immutable_embedding_fallback_reason =
            model.immutable_embedding_fallback_reason().map(str::to_string);
        let immutable_embedding_runtime_mode = model.immutable_embedding_runtime_mode().to_string();
        let immutable_linear_requested = model.immutable_linear_requested();
        let deferred_linear_count = model.deferred_linear_count();
        let total_load_millis = total_started.elapsed().as_secs_f64() * 1000.0;
        Ok((
            Self {
                config,
                weights: MinimalQwen35Weights {
                    model_id: package.manifest().model_id.clone(),
                    revision: package.manifest().revision.clone(),
                    tokenizer_path: package.tokenizer_path(),
                    package_root: package.root().to_path_buf(),
                },
                model,
                device: device.clone(),
            },
            MinimalQwen35LoadTrace {
                package_resolve_millis: Some(package_resolve_millis),
                config_parse_millis,
                model_build_millis,
                total_load_millis,
                package_stats: Some(package_stats),
                weight_load_stats: Some(weight_load_stats),
                immutable_embedding_requested,
                immutable_embedding_active,
                immutable_embedding_fallback_reason,
                immutable_embedding_runtime_mode,
                immutable_linear_requested,
                deferred_linear_count,
            },
        ))
    }

    pub fn load_with_mode(
        model_id: &str,
        device: &Device,
        mode: MinimalQwen35LoadMode,
    ) -> Result<Self> {
        match mode {
            MinimalQwen35LoadMode::DirectHf => Self::load_from_hf_direct_f16(model_id, device),
            MinimalQwen35LoadMode::NativeStore => Self::load_native_for_device(model_id, device),
        }
    }

    pub fn load_for_device(model_id: &str, device: &Device) -> Result<Self> {
        let mode = if let Some(mode) = MinimalQwen35LoadMode::from_env() {
            mode
        } else if matches!(
            std::env::var("DOTCACHE_QWEN35_DISABLE_PREPARED_LOAD").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        ) {
            MinimalQwen35LoadMode::DirectHf
        } else if matches!(
            std::env::var("DOTCACHE_QWEN35_DISABLE_NATIVE_LOAD").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        ) {
            MinimalQwen35LoadMode::DirectHf
        } else {
            MinimalQwen35LoadMode::NativeStore
        };
        Self::load_with_mode(model_id, device, mode)
    }

    pub fn load_from_hf_f16(model_id: &str, device: &Device) -> Result<Self> {
        Self::load_for_device(model_id, device)
    }

    pub fn load_from_hf_0_8b_f16(model_id: &str, device: &Device) -> Result<Self> {
        Self::load_for_device(model_id, device)
    }

    pub fn hidden_states_from_input_ids(
        &self,
        input_ids: &Tensor,
    ) -> Result<MinimalQwen35StateBuffer> {
        let input_ids = if input_ids.device().same_device(&self.device) {
            input_ids.clone()
        } else {
            input_ids.to_device(&self.device)?
        };
        Ok(self.model.hidden_states_from_input_ids(&input_ids)?)
    }

    pub fn prefill_from_hidden_states(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35KvCache)> {
        let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
        let logits = self
            .model
            .forward_hidden_states(&MinimalQwen35StateBuffer::from_tensor(hidden_states)?, 0)?;
        Ok((logits, self.model.cache_state()))
    }

    pub fn decode_from_hidden_state(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<MinimalQwen35StateBuffer> {
        let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
        self.model.restore_cache_state(cache)?;
        let seqlen_offset = cache.sequence_length();
        let logits = self
            .model
            .forward_hidden_states(
                &MinimalQwen35StateBuffer::from_tensor(hidden_state_t)?,
                seqlen_offset,
            )?;
        *cache = self.model.cache_state();
        Ok(logits)
    }

    pub fn linear_attention_layer_ids(&self) -> Vec<usize> {
        self.model.linear_attention_layer_ids()
    }

    pub fn linear_attention_layer_spec(
        &self,
        layer_id: usize,
    ) -> Result<MinimalQwen35LinearAttentionLayerSpec> {
        Ok(self.model.linear_attention_layer_spec(layer_id)?)
    }

    pub fn bench_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        repeats: usize,
    ) -> Result<MinimalQwen35LinearAttentionBenchResult> {
        Ok(self
            .model
            .bench_linear_attention_layer(input_ids, target_layer, seqlen_offset, repeats)?)
    }

    pub fn trace_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<MinimalQwen35LinearAttentionTrace> {
        Ok(self
            .model
            .trace_linear_attention_layer(input_ids, target_layer, seqlen_offset)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    pub fn cache_state(&self) -> MinimalQwen35KvCache {
        self.model.cache_state()
    }

    pub fn restore_cache_state(&mut self, state: &MinimalQwen35KvCache) -> Result<()> {
        Ok(self.model.restore_cache_state(state)?)
    }
}
