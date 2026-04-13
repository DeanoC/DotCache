mod activation;
mod backend_buffer_api;
pub(crate) mod backend_ops;
#[cfg(any(feature = "hf", test))]
mod builder;
mod direct_hip;
mod decoder;
mod frontend;
mod full_attention;
pub(crate) mod hip;
mod hip_wrappers;
mod linear_attention;
mod direct_decoder;
pub(crate) mod model;
mod ops;
mod prepared;
mod rotary;
mod types;
mod with_tracing;

pub use decoder::ModelForCausalLM;
pub use types::{
    CacheState as MinimalQwen35KvCache, Config as MinimalQwen35Config,
    LinearAttentionBenchResult as MinimalQwen35LinearAttentionBenchResult,
    LinearAttentionLayerSpec as MinimalQwen35LinearAttentionLayerSpec,
    LinearAttentionTrace as MinimalQwen35LinearAttentionTrace,
    NativeCacheState as MinimalQwen35NativeCacheState,
    NativeFullAttentionCacheState as MinimalQwen35NativeFullAttentionCacheState,
    NativeLayerCacheState as MinimalQwen35NativeLayerCacheState,
    NativeLinearAttentionCacheState as MinimalQwen35NativeLinearAttentionCacheState,
    RuntimeProfile as MinimalQwen35RuntimeProfile,
    StateBuffer as MinimalQwen35StateBuffer, TextConfig as MinimalQwen35TextConfig,
};

use candle_core::{DType, Device, Tensor};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

#[cfg(feature = "hf")]
use crate::HfHubModelSource;
use crate::{
    ModelPackage, PreparedPackageProfile, PreparedPackageSummary, PreparedQwen35DirectMetadata,
    Result, RuntimeError, TargetSpec, WeightLoadStats,
};
#[cfg(any(feature = "hf", test))]
use builder::WeightBuilder;
use direct_hip::DirectHipQwen35V1Executor;
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
    pub package_profile: Option<PreparedPackageProfile>,
    pub direct_runtime_active: bool,
    pub direct_runtime_profile: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimalQwen35DirectRuntimeProfile {
    HipDirectGfx11V1,
    HipDirectRdna35V1,
}

impl MinimalQwen35DirectRuntimeProfile {
    fn from_package_profile(profile: PreparedPackageProfile) -> Option<Self> {
        match profile {
            PreparedPackageProfile::HipDirectGfx11V1 => Some(Self::HipDirectGfx11V1),
            PreparedPackageProfile::HipDirectRdna35V1 => Some(Self::HipDirectRdna35V1),
            PreparedPackageProfile::StandardPrepared => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HipDirectGfx11V1 => "hip-direct-gfx11-v1",
            Self::HipDirectRdna35V1 => "hip-direct-rdna35-v1",
        }
    }
}

#[derive(Debug)]
pub struct MinimalQwen35DirectRuntime {
    profile: MinimalQwen35DirectRuntimeProfile,
    target: TargetSpec,
    metadata: PreparedQwen35DirectMetadata,
    decode_hidden_ping: MinimalQwen35StateBuffer,
    decode_hidden_pong: MinimalQwen35StateBuffer,
    decode_logits: MinimalQwen35StateBuffer,
    next_hidden_slot_is_ping: bool,
    last_prefill_sequence_length: usize,
    last_decode_sequence_length: usize,
}

impl MinimalQwen35DirectRuntime {
    pub fn profile(&self) -> MinimalQwen35DirectRuntimeProfile {
        self.profile
    }

    pub fn target(&self) -> &TargetSpec {
        &self.target
    }

    pub fn metadata(&self) -> &PreparedQwen35DirectMetadata {
        &self.metadata
    }

    pub fn decode_hidden_ping(&self) -> &MinimalQwen35StateBuffer {
        &self.decode_hidden_ping
    }

    pub fn decode_hidden_pong(&self) -> &MinimalQwen35StateBuffer {
        &self.decode_hidden_pong
    }

    pub fn decode_logits(&self) -> &MinimalQwen35StateBuffer {
        &self.decode_logits
    }

    pub fn last_prefill_sequence_length(&self) -> usize {
        self.last_prefill_sequence_length
    }

    pub fn last_decode_sequence_length(&self) -> usize {
        self.last_decode_sequence_length
    }
}

#[derive(Debug)]
pub struct MinimalQwen35Runner {
    pub config: MinimalQwen35Config,
    pub weights: MinimalQwen35Weights,
    model: ModelForCausalLM,
    device: Device,
    direct_runtime: Option<MinimalQwen35DirectRuntime>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimalQwen35LoadMode {
    DirectHf,
    NativeStore,
    HipDirect,
}

impl MinimalQwen35LoadMode {
    fn from_env() -> Option<Self> {
        let raw = std::env::var("DOTCACHE_QWEN35_LOAD_MODE").ok()?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "direct" | "hf" | "direct-hf" => Some(Self::DirectHf),
            "prepared" | "prepared-candle" | "native" | "native-store" => Some(Self::NativeStore),
            "hip-direct" | "direct-hip" => Some(Self::HipDirect),
            _ => None,
        }
    }
}

impl MinimalQwen35Runner {
    fn direct_hip_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_direct_hip_execution_env<T>(
        enabled: bool,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !enabled {
            return f();
        }
        let _guard = DirectHipExecutionEnvGuard::activate(Self::direct_hip_env_lock());
        f()
    }

    fn build_direct_runtime(
        device: &Device,
        config: &MinimalQwen35Config,
        package: &ModelPackage,
    ) -> Result<MinimalQwen35DirectRuntime> {
        let target = package.target_spec();
        if target.backend != crate::BackendKind::Hip {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct HIP runtime requires a HIP package, got {}:{}",
                    package.manifest().target_backend,
                    package.manifest().target_family
                ),
            });
        }
        let metadata =
            package
                .manifest()
                .qwen35_direct
                .clone()
                .ok_or_else(|| RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: "package is missing qwen35 direct metadata".to_string(),
                })?;
        let profile = MinimalQwen35DirectRuntimeProfile::from_package_profile(
            package.manifest().package_profile,
        )
        .ok_or_else(|| RuntimeError::External {
            context: "qwen35-hip-direct",
            message: "package profile is not a HIP direct profile".to_string(),
        })?;
        let expected_layer_types = config.text_config.clone().normalized().layer_types;
        if expected_layer_types.len() != metadata.layers.len() {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "layer count mismatch between config ({}) and direct package ({})",
                    expected_layer_types.len(),
                    metadata.layers.len()
                ),
            });
        }
        for (expected_idx, (expected, actual)) in expected_layer_types
            .iter()
            .zip(metadata.layers.iter())
            .enumerate()
        {
            if actual.layer_idx != expected_idx || actual.layer_type != *expected {
                return Err(RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!(
                        "direct package layer schedule mismatch at layer {expected_idx}: expected `{expected}`, got `{}`",
                        actual.layer_type
                    ),
                });
            }
        }
        if metadata.num_hidden_layers != 24
            || metadata.linear_attention_layer_ids.len() != 18
            || metadata.full_attention_layer_ids.len() != 6
        {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct HIP runtime currently supports only Qwen3.5-0.8B hybrid schedule (24 layers, 18 linear, 6 full); package has {} / {} / {}",
                    metadata.num_hidden_layers,
                    metadata.linear_attention_layer_ids.len(),
                    metadata.full_attention_layer_ids.len()
                ),
            });
        }
        if metadata.decode_phases.is_empty() {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct HIP runtime requires at least one decode phase".to_string(),
            });
        }
        for binding in metadata.global_tensors.iter() {
            if !package.contains_tensor(&binding.tensor_name) {
                return Err(RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!(
                        "direct HIP package missing global binding `{}` -> `{}`",
                        binding.name, binding.tensor_name
                    ),
                });
            }
        }
        for layer in metadata.layer_bindings.iter() {
            if layer.layer_idx >= metadata.num_hidden_layers {
                return Err(RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!(
                        "direct HIP layer binding index {} is out of range for {} layers",
                        layer.layer_idx, metadata.num_hidden_layers
                    ),
                });
            }
            for binding in layer.tensors.iter() {
                if !package.contains_tensor(&binding.tensor_name) {
                    return Err(RuntimeError::External {
                        context: "qwen35-hip-direct",
                        message: format!(
                            "direct HIP package missing layer {} binding `{}` -> `{}`",
                            layer.layer_idx, binding.name, binding.tensor_name
                        ),
                    });
                }
            }
        }
        let backend = backend_buffer_api::for_device(device);
        let scratch_dtype = DType::BF16;
        let decode_hidden_ping_entry = metadata
            .workspace
            .iter()
            .find(|entry| entry.name == "decode_hidden_ping")
            .ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct HIP metadata missing decode_hidden_ping workspace".to_string(),
            })?;
        let decode_hidden_pong_entry = metadata
            .workspace
            .iter()
            .find(|entry| entry.name == "decode_hidden_pong")
            .ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct HIP metadata missing decode_hidden_pong workspace".to_string(),
            })?;
        let decode_logits_entry = metadata
            .workspace
            .iter()
            .find(|entry| entry.name == "decode_logits")
            .ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct HIP metadata missing decode_logits workspace".to_string(),
            })?;
        let decode_hidden_ping =
            backend.zeros_state(device, scratch_dtype, &decode_hidden_ping_entry.dims)?;
        let decode_hidden_pong =
            backend.zeros_state(device, scratch_dtype, &decode_hidden_pong_entry.dims)?;
        let decode_logits = backend.zeros_state(device, scratch_dtype, &decode_logits_entry.dims)?;
        Ok(MinimalQwen35DirectRuntime {
            profile,
            target,
            metadata,
            decode_hidden_ping,
            decode_hidden_pong,
            decode_logits,
            next_hidden_slot_is_ping: true,
            last_prefill_sequence_length: 0,
            last_decode_sequence_length: 0,
        })
    }

    fn validate_direct_input_ids_v1(&self, input_ids: &Tensor) -> Result<()> {
        let direct = self.direct_runtime.as_ref().ok_or_else(|| RuntimeError::External {
            context: "qwen35-hip-direct",
            message: "direct runtime requested without direct runtime state".to_string(),
        })?;
        let (batch_size, seq_len) = input_ids.dims2()?;
        if batch_size != 1 {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct runtime currently supports only batch size 1, got {batch_size}"
                ),
            });
        }
        if seq_len == 0 {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct runtime requires at least one input token".to_string(),
            });
        }
        if seq_len > direct.metadata.max_position_embeddings {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "direct runtime prompt length {seq_len} exceeds max_position_embeddings {}",
                    direct.metadata.max_position_embeddings
                ),
            });
        }
        Ok(())
    }

    fn validate_direct_hidden_states_v1(
        &self,
        hidden_states: &MinimalQwen35StateBuffer,
        expected_seq_len: Option<usize>,
        phase: &'static str,
    ) -> Result<()> {
        let direct = self.direct_runtime.as_ref().ok_or_else(|| RuntimeError::External {
            context: "qwen35-hip-direct",
            message: format!("direct runtime requested for {phase} without direct runtime state"),
        })?;
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch_size != 1 {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!("{phase} currently supports only batch size 1, got {batch_size}"),
            });
        }
        if let Some(expected_seq_len) = expected_seq_len {
            if seq_len != expected_seq_len {
                return Err(RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!(
                        "{phase} expected sequence length {expected_seq_len}, got {seq_len}"
                    ),
                });
            }
        }
        if hidden_size != direct.metadata.hidden_size {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "{phase} hidden size mismatch: runtime={} input={hidden_size}",
                    direct.metadata.hidden_size
                ),
            });
        }
        Ok(())
    }

    fn validate_direct_decode_v1(
        &self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &MinimalQwen35KvCache,
    ) -> Result<()> {
        self.validate_direct_hidden_states_v1(hidden_state_t, Some(1), "decode")?;
        let direct = self.direct_runtime.as_ref().ok_or_else(|| RuntimeError::External {
            context: "qwen35-hip-direct",
            message: "direct runtime requested without direct runtime state".to_string(),
        })?;
        if cache.layers.len() != direct.metadata.num_hidden_layers {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "decode cache layer count mismatch: runtime={} cache={}",
                    direct.metadata.num_hidden_layers,
                    cache.layers.len()
                ),
            });
        }
        let next_position = cache.sequence_length() + 1;
        if next_position > direct.metadata.max_position_embeddings {
            return Err(RuntimeError::External {
                context: "qwen35-hip-direct",
                message: format!(
                    "decode position {next_position} exceeds max_position_embeddings {}",
                    direct.metadata.max_position_embeddings
                ),
            });
        }
        Ok(())
    }

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
                WeightBuilder::from_mmaped_safetensors(
                    &artifacts.weight_paths,
                    candle_core::DType::F16,
                    device,
                )?
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
                direct_runtime: None,
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
            direct_runtime: None,
        })
    }

    pub fn load_hip_direct_for_device(model_id: &str, device: &Device) -> Result<Self> {
        let package_profile =
            PreparedPackageProfile::qwen35_hip_direct_for_target(&TargetSpec::detect(device))
                .ok_or_else(|| RuntimeError::External {
                    context: "qwen35-hip-direct",
                    message: format!(
                        "direct HIP runtime requires a gfx11 HIP target, got {}",
                        TargetSpec::detect(device).family
                    ),
                })?;
        let package = Arc::new(
            ModelPackage::resolve_or_build_qwen35_minimal_with_profile(
                model_id,
                device,
                package_profile,
            )
            .map_err(|err| crate::RuntimeError::External {
                context: "model-store",
                message: err.to_string(),
            })?,
        );
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(package.config_path())?)?;
        let source = PreparedTensorSource::new(package.clone(), device.clone());
        let model = ModelForCausalLM::from_prepared(&config, source)?;
        let direct_runtime = Some(Self::build_direct_runtime(device, &config, &package)?);
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
            direct_runtime,
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
        let package_stats = package
            .stats()
            .map_err(|err| crate::RuntimeError::External {
                context: "model-store",
                message: err.to_string(),
            })?;
        let config_started = Instant::now();
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(package.config_path())?)?;
        let config_parse_millis = config_started.elapsed().as_secs_f64() * 1000.0;
        let model_started = Instant::now();
        let source = PreparedTensorSource::new_profiled(package.clone(), device.clone());
        let model = ModelForCausalLM::from_prepared(&config, source.clone())?;
        let model_build_millis = model_started.elapsed().as_secs_f64() * 1000.0;
        let weight_load_stats = source.load_stats();
        let immutable_embedding_requested = model.immutable_embedding_requested();
        let immutable_embedding_active = model.immutable_embedding_active();
        let immutable_embedding_fallback_reason = model
            .immutable_embedding_fallback_reason()
            .map(str::to_string);
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
                direct_runtime: None,
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
                package_profile: Some(package.manifest().package_profile),
                direct_runtime_active: false,
                direct_runtime_profile: None,
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
            MinimalQwen35LoadMode::HipDirect => Self::load_hip_direct_for_device(model_id, device),
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

    pub fn direct_runtime(&self) -> Option<&MinimalQwen35DirectRuntime> {
        self.direct_runtime.as_ref()
    }

    pub fn direct_runtime_active(&self) -> bool {
        self.direct_runtime.is_some()
    }

    pub fn load_from_hf_f16(model_id: &str, device: &Device) -> Result<Self> {
        Self::load_for_device(model_id, device)
    }

    pub fn load_from_hf_0_8b_f16(model_id: &str, device: &Device) -> Result<Self> {
        Self::load_for_device(model_id, device)
    }

    fn direct_hidden_states_from_input_ids(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<MinimalQwen35StateBuffer> {
        self.validate_direct_input_ids_v1(input_ids)?;
        Self::with_direct_hip_execution_env(true, || {
            let input_ids = if input_ids.device().same_device(&self.device) {
                input_ids.clone()
            } else {
                input_ids.to_device(&self.device)?
            };
            Ok(self.model.hidden_states_from_input_ids(&input_ids)?)
        })
    }

    fn direct_prefill_from_hidden_states(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35KvCache)> {
        self.validate_direct_hidden_states_v1(hidden_states, None, "prefill")?;
        Self::with_direct_hip_execution_env(true, || {
            let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
            let hidden_states = MinimalQwen35StateBuffer::from_tensor(hidden_states)?;
            let direct = self.direct_runtime.as_mut().ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct runtime requested without direct runtime state".to_string(),
            })?;
            let mut executor = DirectHipQwen35V1Executor::new(&mut self.model, direct);
            let (logits, cache, _profile) = executor.prefill_from_hidden_states(&hidden_states)?;
            Ok((logits, cache))
        })
    }

    fn direct_prefill_from_hidden_states_profiled(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(
        MinimalQwen35StateBuffer,
        MinimalQwen35KvCache,
        MinimalQwen35RuntimeProfile,
    )> {
        self.validate_direct_hidden_states_v1(hidden_states, None, "prefill")?;
        Self::with_direct_hip_execution_env(true, || {
            let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
            let hidden_states = MinimalQwen35StateBuffer::from_tensor(hidden_states)?;
            let direct = self.direct_runtime.as_mut().ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct runtime requested without direct runtime state".to_string(),
            })?;
            let mut executor = DirectHipQwen35V1Executor::new(&mut self.model, direct);
            executor.prefill_from_hidden_states(&hidden_states)
        })
    }

    fn direct_decode_from_hidden_state(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<MinimalQwen35StateBuffer> {
        self.validate_direct_decode_v1(hidden_state_t, cache)?;
        Self::with_direct_hip_execution_env(true, || {
            let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
            let hidden_state_t = MinimalQwen35StateBuffer::from_tensor(hidden_state_t)?;
            let direct = self.direct_runtime.as_mut().ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct runtime requested without direct runtime state".to_string(),
            })?;
            let mut executor = DirectHipQwen35V1Executor::new(&mut self.model, direct);
            let (logits, _profile) = executor.decode_from_hidden_state(&hidden_state_t, cache)?;
            Ok(logits)
        })
    }

    fn direct_decode_from_hidden_state_profiled(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
        self.validate_direct_decode_v1(hidden_state_t, cache)?;
        Self::with_direct_hip_execution_env(true, || {
            let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
            let hidden_state_t = MinimalQwen35StateBuffer::from_tensor(hidden_state_t)?;
            let direct = self.direct_runtime.as_mut().ok_or_else(|| RuntimeError::External {
                context: "qwen35-hip-direct",
                message: "direct runtime requested without direct runtime state".to_string(),
            })?;
            let mut executor = DirectHipQwen35V1Executor::new(&mut self.model, direct);
            executor.decode_from_hidden_state(&hidden_state_t, cache)
        })
    }

    pub fn hidden_states_from_input_ids(
        &self,
        input_ids: &Tensor,
    ) -> Result<MinimalQwen35StateBuffer> {
        Self::with_direct_hip_execution_env(self.direct_runtime.is_some(), || {
            let input_ids = if input_ids.device().same_device(&self.device) {
                input_ids.clone()
            } else {
                input_ids.to_device(&self.device)?
            };
            Ok(self.model.hidden_states_from_input_ids(&input_ids)?)
        })
    }

    pub fn prefill_from_hidden_states(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35KvCache)> {
        if self.direct_runtime.is_some() {
            return self.direct_prefill_from_hidden_states(hidden_states);
        }
        Self::with_direct_hip_execution_env(self.direct_runtime.is_some(), || {
            let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
            let logits = self
                .model
                .forward_hidden_states(&MinimalQwen35StateBuffer::from_tensor(hidden_states)?, 0)?;
            Ok((logits, self.model.cache_state()))
        })
    }

    pub fn prefill_from_hidden_states_profiled(
        &mut self,
        hidden_states: &MinimalQwen35StateBuffer,
    ) -> Result<(
        MinimalQwen35StateBuffer,
        MinimalQwen35KvCache,
        MinimalQwen35RuntimeProfile,
    )> {
        if self.direct_runtime.is_some() {
            return self.direct_prefill_from_hidden_states_profiled(hidden_states);
        }
        Self::with_direct_hip_execution_env(self.direct_runtime.is_some(), || {
            let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
            let hidden_states = MinimalQwen35StateBuffer::from_tensor(hidden_states)?;
            let (logits, profile) = self.model.forward_hidden_states_profiled(&hidden_states, 0)?;
            Ok((logits, self.model.cache_state(), profile))
        })
    }

    pub fn decode_from_hidden_state(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<MinimalQwen35StateBuffer> {
        if self.direct_runtime.is_some() {
            return self.direct_decode_from_hidden_state(hidden_state_t, cache);
        }
        Self::with_direct_hip_execution_env(self.direct_runtime.is_some(), || {
            let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
            self.model.restore_cache_state(cache)?;
            let seqlen_offset = cache.sequence_length();
            let logits = self.model.forward_hidden_states(
                &MinimalQwen35StateBuffer::from_tensor(hidden_state_t)?,
                seqlen_offset,
            )?;
            *cache = self.model.cache_state();
            Ok(logits)
        })
    }

    pub fn decode_from_hidden_state_profiled(
        &mut self,
        hidden_state_t: &MinimalQwen35StateBuffer,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<(MinimalQwen35StateBuffer, MinimalQwen35RuntimeProfile)> {
        if self.direct_runtime.is_some() {
            return self.direct_decode_from_hidden_state_profiled(hidden_state_t, cache);
        }
        Self::with_direct_hip_execution_env(self.direct_runtime.is_some(), || {
            let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
            let hidden_state_t = MinimalQwen35StateBuffer::from_tensor(hidden_state_t)?;
            self.model.restore_cache_state(cache)?;
            let seqlen_offset = cache.sequence_length();
            let (logits, profile) = self
                .model
                .forward_hidden_states_profiled(&hidden_state_t, seqlen_offset)?;
            *cache = self.model.cache_state();
            Ok((logits, profile))
        })
    }

    pub fn hidden_states_from_input_ids_direct(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<MinimalQwen35StateBuffer> {
        if self.direct_runtime.is_some() {
            self.direct_hidden_states_from_input_ids(input_ids)
        } else {
            self.hidden_states_from_input_ids(input_ids)
        }
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
        Ok(self.model.bench_linear_attention_layer(
            input_ids,
            target_layer,
            seqlen_offset,
            repeats,
        )?)
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

struct DirectHipExecutionEnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl DirectHipExecutionEnvGuard {
    fn activate(lock: &'static Mutex<()>) -> Self {
        let guard = lock.lock().unwrap_or_else(|err| err.into_inner());
        let mut saved = Vec::with_capacity(DIRECT_HIP_EXECUTION_ENV.len());
        for (key, value) in DIRECT_HIP_EXECUTION_ENV {
            saved.push((key, std::env::var_os(key)));
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self {
            _lock: guard,
            saved,
        }
    }
}

impl Drop for DirectHipExecutionEnvGuard {
    fn drop(&mut self) {
        for (key, prior) in self.saved.iter().rev() {
            unsafe {
                if let Some(value) = prior {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

const DIRECT_HIP_EXECUTION_ENV: [(&str, &str); 11] = [
    ("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL", "1"),
    ("CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL", "1"),
    ("DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_PREFILL", "1"),
    ("DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_DECODE", "1"),
    ("DOTCACHE_QWEN35_HIP_CHUNK_SINGLE_PREFILL", "1"),
    ("DOTCACHE_QWEN35_HIP_MULTI_CHUNK_SCAN_PREFILL", "1"),
    ("CANDLE_QWEN35_DELTA_SCAN_MODE", "prebatched-local"),
    ("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE", "1"),
    ("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL", "1"),
    ("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL", "1"),
    ("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1"),
];

#[cfg(test)]
mod tests {
    use super::{DirectHipExecutionEnvGuard, DIRECT_HIP_EXECUTION_ENV, MinimalQwen35Runner};

    #[test]
    fn direct_hip_execution_env_guard_sets_and_restores_process_env() {
        const KEY: &str = "CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL";
        unsafe {
            std::env::remove_var(KEY);
            std::env::set_var("CANDLE_QWEN35_DELTA_SCAN_MODE", "torch-like");
        }
        {
            let _guard =
                DirectHipExecutionEnvGuard::activate(MinimalQwen35Runner::direct_hip_env_lock());
            for (key, value) in DIRECT_HIP_EXECUTION_ENV {
                assert_eq!(std::env::var(key).ok().as_deref(), Some(value));
            }
        }
        assert!(std::env::var_os(KEY).is_none());
        assert_eq!(
            std::env::var("CANDLE_QWEN35_DELTA_SCAN_MODE").ok().as_deref(),
            Some("torch-like")
        );
        unsafe {
            std::env::remove_var("CANDLE_QWEN35_DELTA_SCAN_MODE");
        }
    }
}
