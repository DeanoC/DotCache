use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::backend::{
    AttentionPathMode, BackendDescriptor, CandleDeviceSelector, CandlePageBackend, PageBackend,
};
use crate::hf::{HfHubModelSource, HfModelArtifacts};
use crate::instrumented_llama::{InstrumentedLlama, LlamaCache};
use crate::instrumented_qwen2::InstrumentedQwen2;
use crate::instrumented_qwen35::InstrumentedQwen35;
use crate::model::{CausalLm, ModelArchitecture, ModelFamily, RuntimeMode, RuntimeStageMetrics};
use crate::policy::{default_prompt_policy_table, PromptBucketPolicy};
use crate::session::{
    SessionId, SessionMetrics, SessionPrefix, SessionRequestKind, SessionRuntime, SessionState,
};
use crate::virtual_page::{VirtualCacheMetrics, VirtualPagedKvCache};
use crate::{Result, RuntimeError};

fn qwen35_runtime_stage_metrics(
    profile: &candle_transformers::models::qwen3_5::RuntimeProfile,
) -> RuntimeStageMetrics {
    RuntimeStageMetrics {
        qkv_projection_millis: profile.qkv_projection_millis,
        kv_append_write_millis: profile.kv_append_write_millis,
        layout_prepare_millis: profile.layout_prepare_millis,
        attention_score_millis: profile.attention_score_millis,
        attention_softmax_millis: profile.attention_softmax_millis,
        attention_mix_millis: profile.attention_mix_millis,
        output_projection_millis: profile.output_projection_millis,
        full_attention_mask_prepare_millis: profile.full_attention_mask_prepare_millis,
        full_attention_input_layout_millis: profile.full_attention_input_layout_millis,
        full_attention_kv_materialize_millis: profile.full_attention_kv_materialize_millis,
        full_attention_output_collect_millis: profile.full_attention_output_collect_millis,
        full_attention_output_reshape_millis: profile.full_attention_output_reshape_millis,
        full_attention_gate_millis: profile.full_attention_gate_millis,
        full_attention_kernel_execute_millis: profile.full_attention_kernel_execute_millis,
        scheduler_planning_millis: profile.scheduler_planning_millis,
        transfer_millis: profile.transfer_millis,
        linear_attention_millis: profile.linear_attention_millis,
        full_attention_millis: profile.full_attention_millis,
        mlp_millis: profile.mlp_millis,
        ..RuntimeStageMetrics::default()
    }
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestSessionMetrics {
    session_id: SessionId,
    metrics: SessionMetrics,
}

impl RequestSessionMetrics {
    fn new(session_id: SessionId, metrics: SessionMetrics) -> Self {
        Self {
            session_id,
            metrics,
        }
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct RequestMetrics {
    session_ids: Vec<SessionId>,
    kind: SessionRequestKind,
    runtime_mode: RuntimeMode,
    input_token_count: usize,
    cache_delta: VirtualCacheMetrics,
    stage_metrics: RuntimeStageMetrics,
    session_metrics: Vec<RequestSessionMetrics>,
}

impl RequestMetrics {
    fn new(
        session_ids: Vec<SessionId>,
        kind: SessionRequestKind,
        runtime_mode: RuntimeMode,
        input_token_count: usize,
        cache_delta: VirtualCacheMetrics,
        stage_metrics: RuntimeStageMetrics,
        session_metrics: Vec<RequestSessionMetrics>,
    ) -> Self {
        Self {
            session_ids,
            kind,
            runtime_mode,
            input_token_count,
            cache_delta,
            stage_metrics,
            session_metrics,
        }
    }

    pub fn session_ids(&self) -> &[SessionId] {
        self.session_ids.as_slice()
    }

    pub fn kind(&self) -> SessionRequestKind {
        self.kind
    }

    pub fn runtime_mode(&self) -> RuntimeMode {
        self.runtime_mode
    }

    pub fn input_token_count(&self) -> usize {
        self.input_token_count
    }

    pub fn cache_delta(&self) -> &VirtualCacheMetrics {
        &self.cache_delta
    }

    pub fn stage_metrics(&self) -> &RuntimeStageMetrics {
        &self.stage_metrics
    }

    pub fn session_metrics(&self) -> &[RequestSessionMetrics] {
        self.session_metrics.as_slice()
    }
}

#[derive(Debug, Clone)]
struct DenseSessionState {
    #[allow(dead_code)]
    prompt_len: usize,
    token_count: usize,
    next_position: u32,
    metrics: SessionMetrics,
}

impl DenseSessionState {
    fn new(prompt_len: usize) -> Self {
        Self {
            prompt_len,
            token_count: 0,
            next_position: 0,
            metrics: SessionMetrics::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct DenseLlamaSession {
    cache: candle_transformers::models::llama::Cache,
    state: DenseSessionState,
}

#[derive(Debug, Clone)]
struct DenseQwen2Session {
    model: candle_transformers::models::qwen2::ModelForCausalLM,
    state: DenseSessionState,
}

#[derive(Debug, Clone)]
struct DenseQwen35Session {
    model: candle_transformers::models::qwen3_5::ModelForCausalLM,
    state: DenseSessionState,
}

#[derive(Debug)]
enum CandleModelInner {
    LlamaPaged {
        model: InstrumentedLlama,
        config: candle_transformers::models::llama::Config,
        cache: LlamaCache,
        sessions: SessionRuntime,
        session_id: SessionId,
        page_backend: CandlePageBackend,
    },
    Qwen2Paged {
        model: InstrumentedQwen2,
        sessions: SessionRuntime,
        session_id: SessionId,
        page_backend: CandlePageBackend,
    },
    Qwen35Paged {
        model: InstrumentedQwen35,
        sessions: SessionRuntime,
        session_id: SessionId,
        page_backend: CandlePageBackend,
    },
    LlamaDense {
        model: candle_transformers::models::llama::Llama,
        config: candle_transformers::models::llama::Config,
        sessions: Vec<Option<DenseLlamaSession>>,
        session_id: SessionId,
    },
    Qwen2Dense {
        model: candle_transformers::models::qwen2::ModelForCausalLM,
        sessions: Vec<Option<DenseQwen2Session>>,
        session_id: SessionId,
    },
    Qwen35Dense {
        model: candle_transformers::models::qwen3_5::ModelForCausalLM,
        sessions: Vec<Option<DenseQwen35Session>>,
        session_id: SessionId,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct CompatLlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f64,
    #[serde(default = "default_llama_rope_theta")]
    rope_theta: f32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<candle_transformers::models::llama::LlamaEosToks>,
    rope_scaling: Option<candle_transformers::models::llama::Llama3RopeConfig>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

impl CompatLlamaConfig {
    fn into_runtime(self) -> candle_transformers::models::llama::Config {
        candle_transformers::models::llama::Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            use_flash_attn: false,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self
                .max_position_embeddings
                .unwrap_or(candle_transformers::models::llama::DEFAULT_MAX_SEQ_LEN),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompatQwen2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    sliding_window: Option<usize>,
    max_window_layers: usize,
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    use_sliding_window: bool,
    hidden_act: candle_nn::Activation,
}

impl CompatQwen2Config {
    fn into_runtime(self) -> candle_transformers::models::qwen2::Config {
        candle_transformers::models::qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window.unwrap_or(self.max_position_embeddings),
            max_window_layers: self.max_window_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window,
            hidden_act: self.hidden_act,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum CompatEosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl CompatEosTokenId {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![id],
            Self::Multiple(ids) => ids,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompatQwen35Config {
    text_config: candle_transformers::models::qwen3_5::TextConfig,
    eos_token_id: Option<CompatEosTokenId>,
}

fn default_llama_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug)]
pub struct CandleCausalLm {
    architecture: ModelArchitecture,
    artifacts: HfModelArtifacts,
    tokenizer: Tokenizer,
    device: Device,
    device_selector: CandleDeviceSelector,
    dtype: DType,
    tokens_per_page: usize,
    runtime_mode: RuntimeMode,
    inner: CandleModelInner,
    request_log: Vec<RequestMetrics>,
}

impl CandleCausalLm {
    pub const DEFAULT_TOKENS_PER_PAGE: usize = 16;

    pub fn from_hf(
        model_id: &str,
        family: ModelFamily,
        device: CandleDeviceSelector,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_hf_with_runtime_mode(
            model_id,
            family,
            device,
            dtype,
            Self::DEFAULT_TOKENS_PER_PAGE,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_hf_with_paging(
        model_id: &str,
        family: ModelFamily,
        device: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
    ) -> Result<Self> {
        Self::from_hf_with_runtime_mode(
            model_id,
            family,
            device,
            dtype,
            tokens_per_page,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_hf_with_runtime_mode(
        model_id: &str,
        family: ModelFamily,
        device: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
        runtime_mode: RuntimeMode,
    ) -> Result<Self> {
        let source = HfHubModelSource::new()?;
        let artifacts = source.snapshot(model_id)?;
        Self::from_artifacts_with_runtime_mode(
            artifacts,
            family,
            device,
            dtype,
            tokens_per_page,
            runtime_mode,
        )
    }

    pub fn from_artifacts(
        artifacts: HfModelArtifacts,
        family: ModelFamily,
        device: CandleDeviceSelector,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_artifacts_with_runtime_mode(
            artifacts,
            family,
            device,
            dtype,
            Self::DEFAULT_TOKENS_PER_PAGE,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_artifacts_with_paging(
        artifacts: HfModelArtifacts,
        family: ModelFamily,
        device: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
    ) -> Result<Self> {
        Self::from_artifacts_with_runtime_mode(
            artifacts,
            family,
            device,
            dtype,
            tokens_per_page,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_artifacts_with_runtime_mode(
        artifacts: HfModelArtifacts,
        family: ModelFamily,
        device_selector: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
        runtime_mode: RuntimeMode,
    ) -> Result<Self> {
        let device = device_selector.resolve()?;
        let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, dtype, &device)?
        };

        match family {
            ModelFamily::Llama => {
                let config: CompatLlamaConfig =
                    serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
                let runtime_config = config.into_runtime();
                let architecture = ModelArchitecture {
                    model_id: artifacts.model_id.clone(),
                    family,
                    vocab_size: runtime_config.vocab_size,
                    hidden_size: runtime_config.hidden_size,
                    intermediate_size: runtime_config.intermediate_size,
                    num_hidden_layers: runtime_config.num_hidden_layers,
                    num_attention_heads: runtime_config.num_attention_heads,
                    num_key_value_heads: runtime_config.num_key_value_heads,
                    head_dim: runtime_config.hidden_size / runtime_config.num_attention_heads,
                    eos_token_ids: match &runtime_config.eos_token_id {
                        Some(candle_transformers::models::llama::LlamaEosToks::Single(id)) => {
                            vec![*id]
                        }
                        Some(candle_transformers::models::llama::LlamaEosToks::Multiple(ids)) => {
                            ids.clone()
                        }
                        None => Vec::new(),
                    },
                };
                let inner = match runtime_mode {
                    RuntimeMode::DenseControl => {
                        let model = candle_transformers::models::llama::Llama::load(
                            var_builder,
                            &runtime_config,
                        )?;
                        let mut sessions = Vec::new();
                        sessions.push(Some(DenseLlamaSession {
                            cache: candle_transformers::models::llama::Cache::new(
                                true,
                                dtype,
                                &runtime_config,
                                &device,
                            )?,
                            state: DenseSessionState::new(0),
                        }));
                        CandleModelInner::LlamaDense {
                            model,
                            config: runtime_config,
                            sessions,
                            session_id: 0,
                        }
                    }
                    RuntimeMode::PagedControl | RuntimeMode::DotCacheExperimental => {
                        let page_backend = CandlePageBackend::new_with_device(
                            device_selector.clone(),
                            device.clone(),
                        )?;
                        let cache = LlamaCache::new(true, dtype, &runtime_config, &device)?;
                        let model = InstrumentedLlama::load(var_builder, &runtime_config)?;
                        let mut sessions = SessionRuntime::new(
                            runtime_config.num_hidden_layers,
                            runtime_config.num_key_value_heads,
                            tokens_per_page,
                            runtime_config.hidden_size / runtime_config.num_attention_heads,
                        );
                        let session_id = sessions.create_session();
                        CandleModelInner::LlamaPaged {
                            model,
                            config: runtime_config,
                            cache,
                            sessions,
                            session_id,
                            page_backend,
                        }
                    }
                    RuntimeMode::TorchControl => {
                        return Err(RuntimeError::External {
                            context: "candle_model",
                            message: "TorchControl is delegated through the external Python harness and is not a Candle runtime mode".to_string(),
                        });
                    }
                };
                Ok(Self {
                    architecture,
                    artifacts,
                    tokenizer,
                    device,
                    device_selector,
                    dtype,
                    tokens_per_page,
                    runtime_mode,
                    inner,
                    request_log: Vec::new(),
                })
            }
            ModelFamily::Qwen2 => {
                let config: CompatQwen2Config =
                    serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
                let runtime_config = config.into_runtime();
                let architecture = ModelArchitecture {
                    model_id: artifacts.model_id.clone(),
                    family,
                    vocab_size: runtime_config.vocab_size,
                    hidden_size: runtime_config.hidden_size,
                    intermediate_size: runtime_config.intermediate_size,
                    num_hidden_layers: runtime_config.num_hidden_layers,
                    num_attention_heads: runtime_config.num_attention_heads,
                    num_key_value_heads: runtime_config.num_key_value_heads,
                    head_dim: runtime_config.hidden_size / runtime_config.num_attention_heads,
                    eos_token_ids: Vec::new(),
                };
                let inner = match runtime_mode {
                    RuntimeMode::DenseControl => {
                        let model = candle_transformers::models::qwen2::ModelForCausalLM::new(
                            &runtime_config,
                            var_builder,
                        )?;
                        let mut sessions = Vec::new();
                        sessions.push(Some(DenseQwen2Session {
                            model: model.clone(),
                            state: DenseSessionState::new(0),
                        }));
                        CandleModelInner::Qwen2Dense {
                            model,
                            sessions,
                            session_id: 0,
                        }
                    }
                    RuntimeMode::PagedControl | RuntimeMode::DotCacheExperimental => {
                        let page_backend = CandlePageBackend::new_with_device(
                            device_selector.clone(),
                            device.clone(),
                        )?;
                        let model = InstrumentedQwen2::load(var_builder, &runtime_config)?;
                        let mut sessions = SessionRuntime::new(
                            runtime_config.num_hidden_layers,
                            runtime_config.num_key_value_heads,
                            tokens_per_page,
                            runtime_config.hidden_size / runtime_config.num_attention_heads,
                        );
                        let session_id = sessions.create_session();
                        CandleModelInner::Qwen2Paged {
                            model,
                            sessions,
                            session_id,
                            page_backend,
                        }
                    }
                    RuntimeMode::TorchControl => {
                        return Err(RuntimeError::External {
                            context: "candle_model",
                            message: "TorchControl is delegated through the external Python harness and is not a Candle runtime mode".to_string(),
                        });
                    }
                };
                Ok(Self {
                    architecture,
                    artifacts,
                    tokenizer,
                    device,
                    device_selector,
                    dtype,
                    tokens_per_page,
                    runtime_mode,
                    inner,
                    request_log: Vec::new(),
                })
            }
            ModelFamily::Qwen35 => {
                let config: CompatQwen35Config =
                    serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
                let runtime_config = candle_transformers::models::qwen3_5::Config {
                    text_config: config.text_config,
                }
                .normalized();
                let text_config = &runtime_config.text_config;
                let architecture = ModelArchitecture {
                    model_id: artifacts.model_id.clone(),
                    family,
                    vocab_size: text_config.vocab_size,
                    hidden_size: text_config.hidden_size,
                    intermediate_size: text_config.intermediate_size,
                    num_hidden_layers: text_config.num_hidden_layers,
                    num_attention_heads: text_config.num_attention_heads,
                    num_key_value_heads: text_config.num_key_value_heads,
                    head_dim: text_config.head_dim,
                    eos_token_ids: config
                        .eos_token_id
                        .map(CompatEosTokenId::into_vec)
                        .unwrap_or_default(),
                };
                let inner = match runtime_mode {
                    RuntimeMode::DenseControl => {
                        let model = candle_transformers::models::qwen3_5::ModelForCausalLM::new(
                            &runtime_config,
                            var_builder,
                        )?;
                        let mut sessions = Vec::new();
                        sessions.push(Some(DenseQwen35Session {
                            model: model.clone(),
                            state: DenseSessionState::new(0),
                        }));
                        CandleModelInner::Qwen35Dense {
                            model,
                            sessions,
                            session_id: 0,
                        }
                    }
                    RuntimeMode::PagedControl | RuntimeMode::DotCacheExperimental => {
                        let page_backend = CandlePageBackend::new_with_device(
                            device_selector.clone(),
                            device.clone(),
                        )?;
                        let model = InstrumentedQwen35::load(var_builder, &runtime_config)?;
                        let mut sessions = SessionRuntime::new(
                            text_config.num_hidden_layers,
                            text_config.num_key_value_heads,
                            tokens_per_page,
                            text_config.head_dim,
                        );
                        let session_id = sessions.create_session();
                        CandleModelInner::Qwen35Paged {
                            model,
                            sessions,
                            session_id,
                            page_backend,
                        }
                    }
                    RuntimeMode::TorchControl => {
                        return Err(RuntimeError::External {
                            context: "candle_model",
                            message: "TorchControl is delegated through the external Python harness and is not a Candle runtime mode".to_string(),
                        });
                    }
                };
                Ok(Self {
                    architecture,
                    artifacts,
                    tokenizer,
                    device,
                    device_selector,
                    dtype,
                    tokens_per_page,
                    runtime_mode,
                    inner,
                    request_log: Vec::new(),
                })
            }
        }
    }

    pub fn from_local_paths(
        model_id: impl Into<String>,
        family: ModelFamily,
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weight_paths: Vec<PathBuf>,
        device: CandleDeviceSelector,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_local_paths_with_runtime_mode(
            model_id,
            family,
            config_path,
            tokenizer_path,
            weight_paths,
            device,
            dtype,
            Self::DEFAULT_TOKENS_PER_PAGE,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_local_paths_with_paging(
        model_id: impl Into<String>,
        family: ModelFamily,
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weight_paths: Vec<PathBuf>,
        device: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
    ) -> Result<Self> {
        Self::from_local_paths_with_runtime_mode(
            model_id,
            family,
            config_path,
            tokenizer_path,
            weight_paths,
            device,
            dtype,
            tokens_per_page,
            RuntimeMode::PagedControl,
        )
    }

    pub fn from_local_paths_with_runtime_mode(
        model_id: impl Into<String>,
        family: ModelFamily,
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weight_paths: Vec<PathBuf>,
        device: CandleDeviceSelector,
        dtype: DType,
        tokens_per_page: usize,
        runtime_mode: RuntimeMode,
    ) -> Result<Self> {
        Self::from_artifacts_with_runtime_mode(
            HfModelArtifacts {
                model_id: model_id.into(),
                revision: "local".to_string(),
                config_path,
                tokenizer_path,
                weight_paths,
            },
            family,
            device,
            dtype,
            tokens_per_page,
            runtime_mode,
        )
    }

    fn create_dense_llama_session(
        config: &candle_transformers::models::llama::Config,
        dtype: DType,
        device: &Device,
        prompt_len: usize,
    ) -> Result<DenseLlamaSession> {
        Ok(DenseLlamaSession {
            cache: candle_transformers::models::llama::Cache::new(true, dtype, config, device)?,
            state: DenseSessionState::new(prompt_len),
        })
    }

    fn create_dense_qwen2_session(
        template: &candle_transformers::models::qwen2::ModelForCausalLM,
        prompt_len: usize,
    ) -> DenseQwen2Session {
        DenseQwen2Session {
            model: template.clone(),
            state: DenseSessionState::new(prompt_len),
        }
    }

    fn create_dense_qwen35_session(
        template: &candle_transformers::models::qwen3_5::ModelForCausalLM,
        prompt_len: usize,
    ) -> DenseQwen35Session {
        DenseQwen35Session {
            model: template.clone(),
            state: DenseSessionState::new(prompt_len),
        }
    }

    pub fn runtime_mode(&self) -> RuntimeMode {
        self.runtime_mode
    }

    pub fn set_runtime_mode(&mut self, runtime_mode: RuntimeMode) -> Result<()> {
        if self.runtime_mode == runtime_mode {
            return Ok(());
        }
        let rebuilt = Self::from_artifacts_with_runtime_mode(
            self.artifacts.clone(),
            self.architecture.family,
            self.device_selector.clone(),
            self.dtype,
            self.tokens_per_page,
            runtime_mode,
        )?;
        *self = rebuilt;
        Ok(())
    }

    fn paged_runtime_only_error(action: &str) -> RuntimeError {
        RuntimeError::External {
            context: "candle_model",
            message: format!("{action} is only available for paged runtime modes"),
        }
    }

    fn dense_session_ref<'a, T>(sessions: &'a [Option<T>], session_id: SessionId) -> Result<&'a T> {
        let session_count = sessions.len();
        sessions
            .get(session_id)
            .and_then(|session| session.as_ref())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })
    }

    fn dense_session_mut<'a, T>(
        sessions: &'a mut [Option<T>],
        session_id: SessionId,
    ) -> Result<&'a mut T> {
        let session_count = sessions.len();
        sessions
            .get_mut(session_id)
            .and_then(|session| session.as_mut())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })
    }

    fn first_live_dense_session_id<T>(sessions: &[Option<T>]) -> Option<SessionId> {
        sessions
            .iter()
            .enumerate()
            .find_map(|(session_id, session)| session.as_ref().map(|_| session_id))
    }

    fn qwen35_paged_direct_full_attention_enabled(
        token_count: usize,
        batched_decode: bool,
        attention_path: AttentionPathMode,
    ) -> bool {
        if attention_path != AttentionPathMode::Fused {
            return false;
        }
        let env_force_enable = matches!(
            std::env::var("DOTCACHE_QWEN35_PAGED_DIRECT_FULL_ATTN").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );
        let env_force_disable = matches!(
            std::env::var("DOTCACHE_QWEN35_PAGED_DIRECT_FULL_ATTN").as_deref(),
            Ok("0" | "false" | "FALSE" | "no" | "NO")
        );
        if env_force_disable {
            return false;
        }
        let allow_prefill = matches!(
            std::env::var("DOTCACHE_QWEN35_PAGED_DIRECT_FULL_ATTN_PREFILL").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );
        if allow_prefill {
            return true;
        }
        if env_force_enable {
            return token_count == 1;
        }
        let _ = batched_decode;
        token_count == 1
    }

    fn qwen35_restore_full_attention_pages(
        sessions: &SessionRuntime,
        session_id: SessionId,
        full_layer_ids: &[usize],
        dtype: DType,
        device: &Device,
        mut cache_state: candle_transformers::models::qwen3_5::CacheState,
        kv_head_count: usize,
        head_dim: usize,
    ) -> Result<candle_transformers::models::qwen3_5::CacheState> {
        for &layer_id in full_layer_ids {
            let mut key_rows = Vec::new();
            let mut value_rows = Vec::new();
            let mut token_count = None::<usize>;
            for kv_head in 0..kv_head_count {
                let page_ids = sessions.resolve_physical_page_ids(session_id, layer_id, kv_head)?;
                let mut key_head = Vec::new();
                let mut value_head = Vec::new();
                for page_id in page_ids {
                    let page = sessions.cache().physical().store().page(page_id)?;
                    for token_idx in 0..page.token_len() {
                        key_head.extend(page.key_row(token_idx).iter().map(|v| v.to_f32()));
                        value_head.extend(page.value_row(token_idx).iter().map(|v| v.to_f32()));
                    }
                }
                let head_tokens = key_head.len() / head_dim;
                if let Some(existing) = token_count {
                    if existing != head_tokens {
                        return Err(RuntimeError::DimensionMismatch {
                            context: "qwen35 full-attention paged cache token count",
                            expected: existing,
                            got: head_tokens,
                        });
                    }
                } else {
                    token_count = Some(head_tokens);
                }
                key_rows.push(key_head);
                value_rows.push(value_head);
            }

            let token_count = token_count.unwrap_or(0);
            let kv_cache = if token_count == 0 {
                None
            } else {
                let mut keys = Vec::with_capacity(kv_head_count * token_count * head_dim);
                let mut values = Vec::with_capacity(kv_head_count * token_count * head_dim);
                for kv_head in 0..kv_head_count {
                    keys.extend_from_slice(&key_rows[kv_head]);
                    values.extend_from_slice(&value_rows[kv_head]);
                }
                let key =
                    Tensor::from_slice(&keys, (1, kv_head_count, token_count, head_dim), device)?
                        .to_dtype(dtype)?;
                let value =
                    Tensor::from_slice(&values, (1, kv_head_count, token_count, head_dim), device)?
                        .to_dtype(dtype)?;
                Some((key, value))
            };

            match cache_state.layers.get_mut(layer_id) {
                Some(candle_transformers::models::qwen3_5::LayerCacheState::Full(layer_state)) => {
                    layer_state.kv_cache = kv_cache;
                }
                Some(_) => {
                    return Err(RuntimeError::External {
                        context: "candle_model",
                        message: format!(
                            "qwen35 layer {layer_id} was expected to be full-attention when restoring page-backed cache"
                        ),
                    });
                }
                None => {
                    return Err(RuntimeError::InvalidLayer {
                        layer: layer_id,
                        layer_count: cache_state.layers.len(),
                    });
                }
            }
        }
        Ok(cache_state)
    }

    fn qwen35_store_full_attention_pages(
        sessions: &mut SessionRuntime,
        session_id: SessionId,
        full_layer_ids: &[usize],
        cache_state: &mut candle_transformers::models::qwen3_5::CacheState,
        start_position: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<()> {
        if token_count == 0 {
            return Ok(());
        }
        for &layer_id in full_layer_ids {
            let layer_count = cache_state.layers.len();
            let layer_state =
                cache_state
                    .layers
                    .get_mut(layer_id)
                    .ok_or(RuntimeError::InvalidLayer {
                        layer: layer_id,
                        layer_count,
                    })?;
            match layer_state {
                candle_transformers::models::qwen3_5::LayerCacheState::Full(layer_state) => {
                    if let Some((key, value)) = layer_state.kv_cache.as_ref() {
                        let (_, kv_head_count, total_tokens, _) = key.dims4()?;
                        if total_tokens < start_position + token_count {
                            return Err(RuntimeError::DimensionMismatch {
                                context: "qwen35 full-attention cache append window",
                                expected: start_position + token_count,
                                got: total_tokens,
                            });
                        }
                        let key = key.narrow(2, start_position, token_count)?.contiguous()?;
                        let value = value.narrow(2, start_position, token_count)?.contiguous()?;
                        let key_values =
                            key.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let value_values = value
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        for token_idx in 0..token_count {
                            let absolute_pos = (start_position + token_idx) as u32;
                            for kv_head in 0..kv_head_count {
                                let row_offset = (kv_head * token_count + token_idx) * head_dim;
                                let row_end = row_offset + head_dim;
                                sessions.append_kv_row_at(
                                    session_id,
                                    layer_id,
                                    kv_head,
                                    absolute_pos,
                                    &key_values[row_offset..row_end],
                                    &value_values[row_offset..row_end],
                                )?;
                            }
                        }
                    }
                    layer_state.kv_cache = None;
                }
                candle_transformers::models::qwen3_5::LayerCacheState::Linear(_) => {}
            }
        }
        Ok(())
    }

    pub fn paged_cache(&self) -> Option<&VirtualPagedKvCache> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => Some(sessions.cache()),
            CandleModelInner::Qwen2Paged { sessions, .. }
            | CandleModelInner::Qwen35Paged { sessions, .. } => Some(sessions.cache()),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => None,
        }
    }

    pub fn paged_cache_mut(&mut self) -> Option<&mut VirtualPagedKvCache> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => Some(sessions.cache_mut()),
            CandleModelInner::Qwen2Paged { sessions, .. }
            | CandleModelInner::Qwen35Paged { sessions, .. } => Some(sessions.cache_mut()),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => None,
        }
    }

    pub fn tokens_per_page(&self) -> usize {
        self.tokens_per_page
    }

    pub fn backend_descriptor(&self) -> BackendDescriptor {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. }
            | CandleModelInner::Qwen2Paged { page_backend, .. }
            | CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.descriptor(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => BackendDescriptor {
                name: "candle_dense",
                device: self.device_selector.backend_device(),
                supports_prepare_cache: false,
                supports_virtual_pages: false,
                supports_device_resident_pages: false,
            },
        }
    }

    pub fn attention_path(&self) -> AttentionPathMode {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. }
            | CandleModelInner::Qwen2Paged { page_backend, .. }
            | CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.attention_path(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => AttentionPathMode::Fused,
        }
    }

    pub fn set_attention_path(&self, path: AttentionPathMode) {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. }
            | CandleModelInner::Qwen2Paged { page_backend, .. }
            | CandleModelInner::Qwen35Paged { page_backend, .. } => {
                page_backend.set_attention_path(path)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = path;
            }
        }
    }

    pub fn active_session_id(&self) -> Option<SessionId> {
        match &self.inner {
            CandleModelInner::LlamaPaged { session_id, .. }
            | CandleModelInner::Qwen2Paged { session_id, .. }
            | CandleModelInner::Qwen35Paged { session_id, .. }
            | CandleModelInner::LlamaDense { session_id, .. }
            | CandleModelInner::Qwen2Dense { session_id, .. }
            | CandleModelInner::Qwen35Dense { session_id, .. } => Some(*session_id),
        }
    }

    pub fn session_count(&self) -> Option<usize> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. }
            | CandleModelInner::Qwen2Paged { sessions, .. }
            | CandleModelInner::Qwen35Paged { sessions, .. } => Some(sessions.session_count()),
            CandleModelInner::LlamaDense { sessions, .. } => {
                Some(sessions.iter().filter(|session| session.is_some()).count())
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                Some(sessions.iter().filter(|session| session.is_some()).count())
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                Some(sessions.iter().filter(|session| session.is_some()).count())
            }
        }
    }

    pub fn session_state(&self, session_id: SessionId) -> Result<&SessionState> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. }
            | CandleModelInner::Qwen2Paged { sessions, .. }
            | CandleModelInner::Qwen35Paged { sessions, .. } => sessions.session(session_id),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => Err(RuntimeError::External {
                context: "candle_model",
                message: "session_state is only available for paged runtime modes".to_string(),
            }),
        }
    }

    pub fn create_session(&mut self) -> Result<SessionId> {
        self.create_session_with_prompt_len(0)
    }

    pub fn create_session_with_prompt_len(&mut self, prompt_len: usize) -> Result<SessionId> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                Ok(sessions.create_session_with_prompt_len(prompt_len))
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                Ok(sessions.create_session_with_prompt_len(prompt_len))
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                Ok(sessions.create_session_with_prompt_len(prompt_len))
            }
            CandleModelInner::LlamaDense {
                config, sessions, ..
            } => {
                let session_id = sessions.len();
                sessions.push(Some(Self::create_dense_llama_session(
                    config,
                    self.dtype,
                    &self.device,
                    prompt_len,
                )?));
                Ok(session_id)
            }
            CandleModelInner::Qwen2Dense {
                model, sessions, ..
            } => {
                let session_id = sessions.len();
                sessions.push(Some(Self::create_dense_qwen2_session(model, prompt_len)));
                Ok(session_id)
            }
            CandleModelInner::Qwen35Dense {
                model, sessions, ..
            } => {
                let session_id = sessions.len();
                sessions.push(Some(Self::create_dense_qwen35_session(model, prompt_len)));
                Ok(session_id)
            }
        }
    }

    pub fn capture_prefix(&mut self, session_id: SessionId) -> Result<SessionPrefix> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let prefix = sessions.capture_prefix(session_id)?;
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(&prefix)?;
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(prefix)
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let prefix = sessions.capture_prefix(session_id)?;
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(&prefix)?;
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(prefix)
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let prefix = sessions.capture_prefix(session_id)?;
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(&prefix)?;
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(prefix)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                Err(Self::paged_runtime_only_error("capture_prefix"))
            }
        }
    }

    pub fn capture_active_prefix(&mut self) -> Result<SessionPrefix> {
        let session_id = self.active_session_id().ok_or(RuntimeError::External {
            context: "candle_model",
            message: "sessions are not available for this model".to_string(),
        })?;
        self.capture_prefix(session_id)
    }

    pub fn attach_prefix(&mut self, prefix: &SessionPrefix) -> Result<SessionId> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => sessions.attach_prefix(prefix),
            CandleModelInner::Qwen2Paged { sessions, .. } => sessions.attach_prefix(prefix),
            CandleModelInner::Qwen35Paged { sessions, .. } => sessions.attach_prefix(prefix),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = prefix;
                Err(Self::paged_runtime_only_error("attach_prefix"))
            }
        }
    }

    pub fn release_prefix(&mut self, prefix: &SessionPrefix) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_page_ids = sealed_page_ids
                    .into_iter()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_page_ids)?;
                }
                let reclaimed_page_ids = sessions.release_prefix(prefix)?;
                page_backend.release_pages(&reclaimed_page_ids);
                Ok(())
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_page_ids = sealed_page_ids
                    .into_iter()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_page_ids)?;
                }
                let reclaimed_page_ids = sessions.release_prefix(prefix)?;
                page_backend.release_pages(&reclaimed_page_ids);
                Ok(())
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_page_ids = sealed_page_ids
                    .into_iter()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_page_ids)?;
                }
                let reclaimed_page_ids = sessions.release_prefix(prefix)?;
                page_backend.release_pages(&reclaimed_page_ids);
                Ok(())
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = prefix;
                Err(Self::paged_runtime_only_error("release_prefix"))
            }
        }
    }

    pub fn pin_session_pages(&mut self, session_id: SessionId) -> Result<usize> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_count = sealed_page_ids.len();
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(pinned_count)
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_count = sealed_page_ids.len();
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(pinned_count)
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_count = sealed_page_ids.len();
                sessions.cache_mut().pin_physical_pages(&sealed_page_ids);
                page_backend.pin_pages(&sealed_page_ids);
                Ok(pinned_count)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = session_id;
                Err(Self::paged_runtime_only_error("pin_session_pages"))
            }
        }
    }

    pub fn unpin_session_pages(&mut self, session_id: SessionId) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                page_backend.unpin_pages(&sealed_page_ids)
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                page_backend.unpin_pages(&sealed_page_ids)
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                page_backend.unpin_pages(&sealed_page_ids)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = session_id;
                Err(Self::paged_runtime_only_error("unpin_session_pages"))
            }
        }
    }

    pub fn close_session(&mut self, session_id: SessionId) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                session_id: active_session_id,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let reclaimed_page_ids = sessions.close_session(session_id)?;
                page_backend.release_pages(&reclaimed_page_ids);
                if *active_session_id == session_id {
                    *active_session_id = sessions.create_session();
                }
                Ok(())
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                session_id: active_session_id,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let reclaimed_page_ids = sessions.close_session(session_id)?;
                page_backend.release_pages(&reclaimed_page_ids);
                if *active_session_id == session_id {
                    *active_session_id = sessions.create_session();
                }
                Ok(())
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                session_id: active_session_id,
                page_backend,
                ..
            } => {
                let sealed_page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = sealed_page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let reclaimed_page_ids = sessions.close_session(session_id)?;
                page_backend.release_pages(&reclaimed_page_ids);
                if *active_session_id == session_id {
                    *active_session_id = sessions.create_session();
                }
                Ok(())
            }
            CandleModelInner::LlamaDense {
                config,
                sessions,
                session_id: active_session_id,
                ..
            } => {
                let session_count = sessions.len();
                let slot = sessions
                    .get_mut(session_id)
                    .ok_or(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    })?;
                if slot.take().is_none() {
                    return Err(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    });
                }
                if *active_session_id == session_id {
                    if let Some(next_session_id) = Self::first_live_dense_session_id(sessions) {
                        *active_session_id = next_session_id;
                    } else {
                        let next_session_id = sessions.len();
                        sessions.push(Some(Self::create_dense_llama_session(
                            config,
                            self.dtype,
                            &self.device,
                            0,
                        )?));
                        *active_session_id = next_session_id;
                    }
                }
                Ok(())
            }
            CandleModelInner::Qwen2Dense {
                model,
                sessions,
                session_id: active_session_id,
            } => {
                let session_count = sessions.len();
                let slot = sessions
                    .get_mut(session_id)
                    .ok_or(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    })?;
                if slot.take().is_none() {
                    return Err(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    });
                }
                if *active_session_id == session_id {
                    if let Some(next_session_id) = Self::first_live_dense_session_id(sessions) {
                        *active_session_id = next_session_id;
                    } else {
                        let next_session_id = sessions.len();
                        sessions.push(Some(Self::create_dense_qwen2_session(model, 0)));
                        *active_session_id = next_session_id;
                    }
                }
                Ok(())
            }
            CandleModelInner::Qwen35Dense {
                model,
                sessions,
                session_id: active_session_id,
            } => {
                let session_count = sessions.len();
                let slot = sessions
                    .get_mut(session_id)
                    .ok_or(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    })?;
                if slot.take().is_none() {
                    return Err(RuntimeError::InvalidSessionId {
                        session_id,
                        session_count,
                    });
                }
                if *active_session_id == session_id {
                    if let Some(next_session_id) = Self::first_live_dense_session_id(sessions) {
                        *active_session_id = next_session_id;
                    } else {
                        let next_session_id = sessions.len();
                        sessions.push(Some(Self::create_dense_qwen35_session(model, 0)));
                        *active_session_id = next_session_id;
                    }
                }
                Ok(())
            }
        }
    }

    pub fn set_prepare_cache_page_budget(&self, budget: Option<usize>) {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => {
                page_backend.set_prepare_cache_page_budget(budget)
            }
            CandleModelInner::Qwen2Paged { page_backend, .. } => {
                page_backend.set_prepare_cache_page_budget(budget)
            }
            CandleModelInner::Qwen35Paged { page_backend, .. } => {
                page_backend.set_prepare_cache_page_budget(budget)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = budget;
            }
        }
    }

    pub fn set_resident_physical_page_budget(&mut self, budget: Option<usize>) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.cache_mut().set_resident_page_budget(budget)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.cache_mut().set_resident_page_budget(budget)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.cache_mut().set_resident_page_budget(budget)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => Ok(()),
        }
    }

    pub fn resident_physical_page_budget(&self) -> Option<usize> {
        self.paged_cache()
            .and_then(VirtualPagedKvCache::resident_page_budget)
    }

    pub fn set_resident_physical_byte_budget(&mut self, budget: Option<usize>) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.cache_mut().set_resident_byte_budget(budget)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.cache_mut().set_resident_byte_budget(budget)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.cache_mut().set_resident_byte_budget(budget)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => Ok(()),
        }
    }

    pub fn resident_physical_byte_budget(&self) -> Option<usize> {
        self.paged_cache()
            .and_then(VirtualPagedKvCache::resident_byte_budget)
    }

    pub fn set_restore_cooldown_window(&mut self, window: u64) {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.cache_mut().set_restore_cooldown_window(window)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.cache_mut().set_restore_cooldown_window(window)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.cache_mut().set_restore_cooldown_window(window)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = window;
            }
        }
    }

    pub fn restore_cooldown_window(&self) -> Option<u64> {
        self.paged_cache()
            .map(VirtualPagedKvCache::restore_cooldown_window)
    }

    pub fn prompt_token_count(&self, text: &str, add_special_tokens: bool) -> Result<usize> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .len())
    }

    pub fn recommended_prompt_policy_for_token_count(
        &self,
        prompt_token_count: usize,
    ) -> Result<Option<PromptBucketPolicy>> {
        Ok(
            default_prompt_policy_table()?
                .recommended(self.architecture.family, prompt_token_count),
        )
    }

    pub fn recommended_prompt_policy_for_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Option<PromptBucketPolicy>> {
        self.recommended_prompt_policy_for_token_count(
            self.prompt_token_count(text, add_special_tokens)?,
        )
    }

    pub fn apply_prompt_policy(&mut self, policy: &PromptBucketPolicy) -> Result<()> {
        self.set_resident_physical_page_budget(policy.resident_page_budget)?;
        self.set_resident_physical_byte_budget(policy.resident_byte_budget)?;
        if let Some(window) = policy.restore_cooldown_window {
            self.set_restore_cooldown_window(window);
        }
        Ok(())
    }

    pub fn apply_recommended_prompt_policy_for_text(
        &mut self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Option<PromptBucketPolicy>> {
        let policy = self.recommended_prompt_policy_for_text(text, add_special_tokens)?;
        if let Some(policy) = &policy {
            self.apply_prompt_policy(policy)?;
        }
        Ok(policy)
    }

    pub fn cache_metrics(&self) -> Option<&VirtualCacheMetrics> {
        self.paged_cache().map(VirtualPagedKvCache::metrics)
    }

    pub fn reset_cache_metrics(&mut self) {
        if let Some(cache) = self.paged_cache_mut() {
            cache.reset_metrics();
        }
    }

    pub fn request_metrics(&self) -> &[RequestMetrics] {
        self.request_log.as_slice()
    }

    pub fn last_request_metrics(&self) -> Option<&RequestMetrics> {
        self.request_log.last()
    }

    pub fn clear_request_metrics(&mut self) {
        self.request_log.clear();
    }

    pub fn export_request_metrics_jsonl(&self) -> Result<String> {
        let mut lines = Vec::with_capacity(self.request_log.len());
        for entry in &self.request_log {
            lines.push(serde_json::to_string(entry)?);
        }
        Ok(lines.join("\n"))
    }

    pub fn write_request_metrics_jsonl<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let mut jsonl = self.export_request_metrics_jsonl()?;
        if !jsonl.is_empty() {
            jsonl.push('\n');
        }
        std::fs::write(path, jsonl)?;
        Ok(())
    }

    pub fn session_metrics(&self, session_id: SessionId) -> Result<&SessionMetrics> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => sessions.session_metrics(session_id),
            CandleModelInner::Qwen2Paged { sessions, .. } => sessions.session_metrics(session_id),
            CandleModelInner::Qwen35Paged { sessions, .. } => sessions.session_metrics(session_id),
            CandleModelInner::LlamaDense { sessions, .. } => {
                Ok(&Self::dense_session_ref(sessions, session_id)?.state.metrics)
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                Ok(&Self::dense_session_ref(sessions, session_id)?.state.metrics)
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                Ok(&Self::dense_session_ref(sessions, session_id)?.state.metrics)
            }
        }
    }

    pub fn reset_session_metrics(&mut self, session_id: SessionId) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.reset_session_metrics(session_id)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.reset_session_metrics(session_id)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.reset_session_metrics(session_id)
            }
            CandleModelInner::LlamaDense { sessions, .. } => {
                Self::dense_session_mut(sessions, session_id)?.state.metrics =
                    SessionMetrics::default();
                Ok(())
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                Self::dense_session_mut(sessions, session_id)?.state.metrics =
                    SessionMetrics::default();
                Ok(())
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                Self::dense_session_mut(sessions, session_id)?.state.metrics =
                    SessionMetrics::default();
                Ok(())
            }
        }
    }

    pub fn prepare_cache_page_budget(&self) -> Option<usize> {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => {
                page_backend.prepare_cache_page_budget()
            }
            CandleModelInner::Qwen2Paged { page_backend, .. } => {
                page_backend.prepare_cache_page_budget()
            }
            CandleModelInner::Qwen35Paged { page_backend, .. } => {
                page_backend.prepare_cache_page_budget()
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => None,
        }
    }

    pub fn prepared_page_count(&self) -> usize {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => page_backend.prepared_page_count(),
            CandleModelInner::Qwen2Paged { page_backend, .. } => page_backend.prepared_page_count(),
            CandleModelInner::Qwen35Paged { page_backend, .. } => {
                page_backend.prepared_page_count()
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => 0,
        }
    }

    pub fn prepared_cache_hits(&self) -> usize {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => page_backend.cache_hits(),
            CandleModelInner::Qwen2Paged { page_backend, .. } => page_backend.cache_hits(),
            CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.cache_hits(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => 0,
        }
    }

    pub fn prepared_cache_misses(&self) -> usize {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => page_backend.cache_misses(),
            CandleModelInner::Qwen2Paged { page_backend, .. } => page_backend.cache_misses(),
            CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.cache_misses(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => 0,
        }
    }

    pub fn prepared_cache_evictions(&self) -> usize {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => page_backend.cache_evictions(),
            CandleModelInner::Qwen2Paged { page_backend, .. } => page_backend.cache_evictions(),
            CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.cache_evictions(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => 0,
        }
    }

    pub fn prepared_pinned_page_count(&self) -> usize {
        match &self.inner {
            CandleModelInner::LlamaPaged { page_backend, .. } => page_backend.pinned_page_count(),
            CandleModelInner::Qwen2Paged { page_backend, .. } => page_backend.pinned_page_count(),
            CandleModelInner::Qwen35Paged { page_backend, .. } => page_backend.pinned_page_count(),
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => 0,
        }
    }

    pub fn resident_physical_page_count(&self) -> usize {
        self.paged_cache()
            .map(VirtualPagedKvCache::resident_physical_page_count)
            .unwrap_or(0)
    }

    pub fn spilled_physical_page_count(&self) -> usize {
        self.paged_cache()
            .map(VirtualPagedKvCache::spilled_physical_page_count)
            .unwrap_or(0)
    }

    pub fn resident_physical_byte_count(&self) -> usize {
        self.paged_cache()
            .map(VirtualPagedKvCache::resident_physical_byte_count)
            .unwrap_or(0)
    }

    pub fn spilled_physical_byte_count(&self) -> usize {
        self.paged_cache()
            .map(VirtualPagedKvCache::spilled_physical_byte_count)
            .unwrap_or(0)
    }

    pub fn pinned_physical_page_count(&self) -> usize {
        self.paged_cache()
            .map(VirtualPagedKvCache::pinned_physical_page_count)
            .unwrap_or(0)
    }

    pub fn spill_session_pages(&mut self, session_id: SessionId) -> Result<usize> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids(session_id)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = session_id;
                Err(Self::paged_runtime_only_error("spill_session_pages"))
            }
        }
    }

    pub fn spill_prefix(&mut self, prefix: &SessionPrefix) -> Result<usize> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.sealed_physical_page_ids_for_prefix(prefix)?;
                let pinned_physical_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| sessions.cache().is_physical_page_pinned(page_id))
                    .collect::<Vec<_>>();
                let pinned_backend_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| page_backend.is_page_pinned(page_id))
                    .collect::<Vec<_>>();
                if !pinned_physical_page_ids.is_empty() {
                    sessions
                        .cache_mut()
                        .unpin_physical_pages(&pinned_physical_page_ids)?;
                }
                if !pinned_backend_page_ids.is_empty() {
                    page_backend.unpin_pages(&pinned_backend_page_ids)?;
                }
                let spilled = sessions.cache_mut().spill_physical_pages(&page_ids)?;
                Ok(spilled.len())
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = prefix;
                Err(Self::paged_runtime_only_error("spill_prefix"))
            }
        }
    }

    pub fn restore_session_pages(&mut self, session_id: SessionId) -> Result<usize> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restored = sessions.cache_mut().restore_physical_pages(&page_ids)?;
                Ok(restored.len())
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restored = sessions.cache_mut().restore_physical_pages(&page_ids)?;
                Ok(restored.len())
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restored = sessions.cache_mut().restore_physical_pages(&page_ids)?;
                Ok(restored.len())
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = session_id;
                Ok(0)
            }
        }
    }

    pub fn fork_session(&mut self, source_session_id: SessionId) -> Result<SessionId> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.fork_session(source_session_id)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.fork_session(source_session_id)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.fork_session(source_session_id)
            }
            CandleModelInner::LlamaDense { sessions, .. } => {
                let source = Self::dense_session_ref(sessions, source_session_id)?.clone();
                let session_id = sessions.len();
                sessions.push(Some(source));
                Ok(session_id)
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                let source = Self::dense_session_ref(sessions, source_session_id)?.clone();
                let session_id = sessions.len();
                sessions.push(Some(source));
                Ok(session_id)
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                let source = Self::dense_session_ref(sessions, source_session_id)?.clone();
                let session_id = sessions.len();
                sessions.push(Some(source));
                Ok(session_id)
            }
        }
    }

    pub fn fork_active_session(&mut self) -> Result<SessionId> {
        let active_session_id = self.active_session_id().ok_or(RuntimeError::External {
            context: "candle_model",
            message: "sessions are not available for this model".to_string(),
        })?;
        self.fork_session(active_session_id)
    }

    pub fn set_active_session(&mut self, session_id: SessionId) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                sessions.session(session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                sessions.session(session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                sessions.session(session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
            CandleModelInner::LlamaDense {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                Self::dense_session_ref(sessions, session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
            CandleModelInner::Qwen2Dense {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                Self::dense_session_ref(sessions, session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
            CandleModelInner::Qwen35Dense {
                sessions,
                session_id: active_session_id,
                ..
            } => {
                Self::dense_session_ref(sessions, session_id)?;
                *active_session_id = session_id;
                Ok(())
            }
        }
    }

    pub fn session_position(&self, session_id: SessionId) -> Result<u32> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => sessions.current_position(session_id),
            CandleModelInner::Qwen2Paged { sessions, .. } => sessions.current_position(session_id),
            CandleModelInner::Qwen35Paged { sessions, .. } => sessions.current_position(session_id),
            CandleModelInner::LlamaDense { sessions, .. } => {
                Ok(Self::dense_session_ref(sessions, session_id)?
                    .state
                    .next_position)
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                Ok(Self::dense_session_ref(sessions, session_id)?
                    .state
                    .next_position)
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                Ok(Self::dense_session_ref(sessions, session_id)?
                    .state
                    .next_position)
            }
        }
    }

    pub fn resolve_session_physical_page_ids(
        &self,
        session_id: SessionId,
        layer: usize,
        kv_head: usize,
    ) -> Result<Vec<usize>> {
        match &self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.resolve_physical_page_ids(session_id, layer, kv_head)
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.resolve_physical_page_ids(session_id, layer, kv_head)
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.resolve_physical_page_ids(session_id, layer, kv_head)
            }
            CandleModelInner::LlamaDense { .. }
            | CandleModelInner::Qwen2Dense { .. }
            | CandleModelInner::Qwen35Dense { .. } => {
                let _ = (session_id, layer, kv_head);
                Err(Self::paged_runtime_only_error(
                    "resolve_session_physical_page_ids",
                ))
            }
        }
    }

    pub fn forward_next_logits_batch(
        &mut self,
        requests: &[(SessionId, u32)],
    ) -> Result<Vec<(SessionId, Vec<f32>)>> {
        if requests.is_empty() {
            return Err(RuntimeError::EmptyInput {
                context: "forward_next_logits_batch",
            });
        }

        let mut seen = HashSet::with_capacity(requests.len());
        for &(session_id, _) in requests {
            if !seen.insert(session_id) {
                return Err(RuntimeError::External {
                    context: "candle_model",
                    message: format!("duplicate session id {session_id} in batched decode request"),
                });
            }
        }

        let cache_metrics_before = self.cache_metrics_snapshot();
        let session_ids = requests
            .iter()
            .map(|(session_id, _)| *session_id)
            .collect::<Vec<_>>();
        let input_token_counts = vec![1; session_ids.len()];
        let mut stage_metrics = RuntimeStageMetrics::default();

        let logits = match &mut self.inner {
            CandleModelInner::LlamaPaged {
                model,
                cache,
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = session_ids
                    .iter()
                    .map(|&session_id| sessions.physical_page_ids(session_id))
                    .collect::<Result<Vec<_>>>()?;
                let restore_page_ids = page_ids
                    .into_iter()
                    .flatten()
                    .filter(|&page_id| !page_backend.is_page_prepared(page_id))
                    .collect::<Vec<_>>();
                let restore_started = Instant::now();
                let _ = sessions
                    .cache_mut()
                    .restore_physical_pages(&restore_page_ids)?;
                stage_metrics.page_restore_millis += restore_started.elapsed().as_secs_f64() * 1e3;
                sessions
                    .cache_mut()
                    .touch_physical_pages(&restore_page_ids)?;
                let index_positions = session_ids
                    .iter()
                    .map(|&session_id| {
                        sessions
                            .current_position(session_id)
                            .map(|pos| pos as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let input_ids = requests
                    .iter()
                    .map(|(_, token_id)| *token_id)
                    .collect::<Vec<_>>();
                let input = Tensor::from_slice(&input_ids, (requests.len(), 1), &self.device)?;
                let logits = model.forward_decode_batch(
                    &input,
                    &index_positions,
                    cache,
                    sessions,
                    &session_ids,
                    page_backend,
                )?;
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                logits.to_dtype(DType::F32)?.to_vec2::<f32>()?
            }
            CandleModelInner::Qwen2Paged {
                model,
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = session_ids
                    .iter()
                    .map(|&session_id| sessions.physical_page_ids(session_id))
                    .collect::<Result<Vec<_>>>()?;
                let restore_page_ids = page_ids
                    .into_iter()
                    .flatten()
                    .filter(|&page_id| !page_backend.is_page_prepared(page_id))
                    .collect::<Vec<_>>();
                let restore_started = Instant::now();
                let _ = sessions
                    .cache_mut()
                    .restore_physical_pages(&restore_page_ids)?;
                stage_metrics.page_restore_millis += restore_started.elapsed().as_secs_f64() * 1e3;
                sessions
                    .cache_mut()
                    .touch_physical_pages(&restore_page_ids)?;
                let index_positions = session_ids
                    .iter()
                    .map(|&session_id| {
                        sessions
                            .current_position(session_id)
                            .map(|pos| pos as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let input_ids = requests
                    .iter()
                    .map(|(_, token_id)| *token_id)
                    .collect::<Vec<_>>();
                let input = Tensor::from_slice(&input_ids, (requests.len(), 1), &self.device)?;
                let logits = model.forward_decode_batch(
                    &input,
                    &index_positions,
                    sessions,
                    &session_ids,
                    page_backend,
                )?;
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                logits.to_dtype(DType::F32)?.to_vec2::<f32>()?
            }
            CandleModelInner::Qwen35Paged {
                model,
                sessions,
                page_backend,
                ..
            } => {
                let mut outputs = Vec::with_capacity(requests.len());
                for &(session_id, token_id) in requests {
                    let index_pos = sessions.current_position(session_id)? as usize;
                    let page_ids = sessions.physical_page_ids(session_id)?;
                    let restore_started = Instant::now();
                    let _ = sessions.cache_mut().restore_physical_pages(&page_ids)?;
                    stage_metrics.page_restore_millis +=
                        restore_started.elapsed().as_secs_f64() * 1e3;
                    sessions.cache_mut().touch_physical_pages(&page_ids)?;
                    let cache_state = sessions
                        .hybrid_cache_state(session_id)?
                        .cloned()
                        .and_then(|state| match state {
                            crate::session::HybridCacheState::Qwen35(state) => Some(state),
                        })
                        .unwrap_or_else(|| model.empty_cache_state());
                    let hybrid_restore_started = Instant::now();
                    let input = Tensor::from_slice(&[token_id], (1, 1), &self.device)?;
                    let (logits, next_state, profile) =
                        if Self::qwen35_paged_direct_full_attention_enabled(
                            1,
                            true,
                            page_backend.attention_path(),
                        ) {
                            let cache_state = crate::session::HybridCacheState::Qwen35(cache_state);
                            stage_metrics.hybrid_cache_restore_millis +=
                                hybrid_restore_started.elapsed().as_secs_f64() * 1e3;
                            model.forward_profiled_paged_full_attention(
                                &input,
                                index_pos,
                                Some(&cache_state),
                                sessions,
                                session_id,
                                page_backend,
                            )?
                        } else {
                            let full_layer_ids = model.full_attention_layer_ids();
                            let cache_state = Self::qwen35_restore_full_attention_pages(
                                sessions,
                                session_id,
                                &full_layer_ids,
                                self.dtype,
                                &self.device,
                                cache_state,
                                self.architecture.num_key_value_heads,
                                self.architecture.head_dim,
                            )?;
                            stage_metrics.hybrid_cache_restore_millis +=
                                hybrid_restore_started.elapsed().as_secs_f64() * 1e3;
                            let cache_state = crate::session::HybridCacheState::Qwen35(cache_state);
                            model.forward_profiled(&input, index_pos, Some(&cache_state))?
                        };
                    let hybrid_store_started = Instant::now();
                    let next_state = match next_state {
                        crate::session::HybridCacheState::Qwen35(mut state) => {
                            if !Self::qwen35_paged_direct_full_attention_enabled(
                                1,
                                true,
                                page_backend.attention_path(),
                            ) {
                                let full_layer_ids = model.full_attention_layer_ids();
                                Self::qwen35_store_full_attention_pages(
                                    sessions,
                                    session_id,
                                    &full_layer_ids,
                                    &mut state,
                                    index_pos,
                                    1,
                                    self.architecture.head_dim,
                                )?;
                            }
                            crate::session::HybridCacheState::Qwen35(state)
                        }
                    };
                    sessions.set_hybrid_cache_state(session_id, Some(next_state))?;
                    stage_metrics.hybrid_cache_store_millis +=
                        hybrid_store_started.elapsed().as_secs_f64() * 1e3;
                    sessions.commit_positions(session_id, index_pos as u32, 1)?;
                    stage_metrics.add_assign(&qwen35_runtime_stage_metrics(&profile));
                    outputs.push(
                        logits
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                }
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                outputs
            }
            CandleModelInner::LlamaDense {
                model, sessions, ..
            } => {
                let mut outputs = Vec::with_capacity(requests.len());
                for &(session_id, token_id) in requests {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    let index_pos = session.state.next_position as usize;
                    let input = Tensor::from_slice(&[token_id], (1, 1), &self.device)?;
                    let logits = model.forward(&input, index_pos, &mut session.cache)?;
                    session.state.next_position += 1;
                    session.state.token_count += 1;
                    outputs.push(
                        logits
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                }
                outputs
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                let mut outputs = Vec::with_capacity(requests.len());
                for &(session_id, token_id) in requests {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    let index_pos = session.state.next_position as usize;
                    let input = Tensor::from_slice(&[token_id], (1, 1), &self.device)?;
                    let logits = session.model.forward(&input, index_pos)?;
                    session.state.next_position += 1;
                    session.state.token_count += 1;
                    outputs.push(
                        logits
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                }
                outputs
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                let mut outputs = Vec::with_capacity(requests.len());
                for &(session_id, token_id) in requests {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    let index_pos = session.state.next_position as usize;
                    let input = Tensor::from_slice(&[token_id], (1, 1), &self.device)?;
                    let (logits, profile) = session.model.forward_profiled(&input, index_pos)?;
                    session.state.next_position += 1;
                    session.state.token_count += 1;
                    stage_metrics.add_assign(&qwen35_runtime_stage_metrics(&profile));
                    outputs.push(
                        logits
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                }
                outputs
            }
        };
        self.record_request_metrics(
            &session_ids,
            SessionRequestKind::BatchDecode,
            &input_token_counts,
            &cache_metrics_before,
            stage_metrics,
        )?;
        Ok(session_ids.into_iter().zip(logits).collect())
    }

    pub fn prefill_session(
        &mut self,
        session_id: SessionId,
        input_ids: &[u32],
    ) -> Result<Vec<f32>> {
        self.run_session_request(session_id, input_ids, SessionRequestKind::Prefill)
    }

    pub fn prefill_active_session(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let session_id = self.active_session_id().ok_or(RuntimeError::External {
            context: "candle_model",
            message: "sessions are not available for this model".to_string(),
        })?;
        self.prefill_session(session_id, input_ids)
    }

    pub fn prefill_sessions_batch(
        &mut self,
        requests: &[(SessionId, &[u32])],
    ) -> Result<Vec<(SessionId, Vec<f32>)>> {
        if requests.is_empty() {
            return Err(RuntimeError::EmptyInput {
                context: "prefill_sessions_batch",
            });
        }

        let mut seen = HashSet::with_capacity(requests.len());
        let mut logits = Vec::with_capacity(requests.len());
        for &(session_id, input_ids) in requests {
            if !seen.insert(session_id) {
                return Err(RuntimeError::External {
                    context: "candle_model",
                    message: format!(
                        "duplicate session id {session_id} in batched prefill request"
                    ),
                });
            }
            logits.push((session_id, self.prefill_session(session_id, input_ids)?));
        }
        Ok(logits)
    }

    fn cache_metrics_snapshot(&self) -> VirtualCacheMetrics {
        self.paged_cache()
            .map(|cache| cache.metrics().clone())
            .unwrap_or_default()
    }

    fn record_request_metrics(
        &mut self,
        session_ids: &[SessionId],
        kind: SessionRequestKind,
        input_token_counts: &[usize],
        before: &VirtualCacheMetrics,
        stage_metrics: RuntimeStageMetrics,
    ) -> Result<()> {
        let delta = self.cache_metrics_snapshot().delta_since(before);
        let session_metric_snapshots = match &mut self.inner {
            CandleModelInner::LlamaPaged { sessions, .. } => {
                sessions.record_session_request(session_ids, kind, input_token_counts, &delta)?;
                collect_request_session_metrics(sessions, session_ids)?
            }
            CandleModelInner::Qwen2Paged { sessions, .. } => {
                sessions.record_session_request(session_ids, kind, input_token_counts, &delta)?;
                collect_request_session_metrics(sessions, session_ids)?
            }
            CandleModelInner::Qwen35Paged { sessions, .. } => {
                sessions.record_session_request(session_ids, kind, input_token_counts, &delta)?;
                collect_request_session_metrics(sessions, session_ids)?
            }
            CandleModelInner::LlamaDense { sessions, .. } => {
                if session_ids.len() != input_token_counts.len() {
                    return Err(RuntimeError::External {
                        context: "candle_model",
                        message: format!(
                            "session_ids length {} did not match input_token_counts length {}",
                            session_ids.len(),
                            input_token_counts.len()
                        ),
                    });
                }
                let mut snapshots = Vec::with_capacity(session_ids.len());
                for (&session_id, &input_token_count) in
                    session_ids.iter().zip(input_token_counts.iter())
                {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    session
                        .state
                        .metrics
                        .record_request(kind, input_token_count, &delta);
                    snapshots.push(RequestSessionMetrics::new(
                        session_id,
                        session.state.metrics.clone(),
                    ));
                }
                snapshots
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                if session_ids.len() != input_token_counts.len() {
                    return Err(RuntimeError::External {
                        context: "candle_model",
                        message: format!(
                            "session_ids length {} did not match input_token_counts length {}",
                            session_ids.len(),
                            input_token_counts.len()
                        ),
                    });
                }
                let mut snapshots = Vec::with_capacity(session_ids.len());
                for (&session_id, &input_token_count) in
                    session_ids.iter().zip(input_token_counts.iter())
                {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    session
                        .state
                        .metrics
                        .record_request(kind, input_token_count, &delta);
                    snapshots.push(RequestSessionMetrics::new(
                        session_id,
                        session.state.metrics.clone(),
                    ));
                }
                snapshots
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                if session_ids.len() != input_token_counts.len() {
                    return Err(RuntimeError::External {
                        context: "candle_model",
                        message: format!(
                            "session_ids length {} did not match input_token_counts length {}",
                            session_ids.len(),
                            input_token_counts.len()
                        ),
                    });
                }
                let mut snapshots = Vec::with_capacity(session_ids.len());
                for (&session_id, &input_token_count) in
                    session_ids.iter().zip(input_token_counts.iter())
                {
                    let session = Self::dense_session_mut(sessions, session_id)?;
                    session
                        .state
                        .metrics
                        .record_request(kind, input_token_count, &delta);
                    snapshots.push(RequestSessionMetrics::new(
                        session_id,
                        session.state.metrics.clone(),
                    ));
                }
                snapshots
            }
        };
        self.request_log.push(RequestMetrics::new(
            session_ids.to_vec(),
            kind,
            self.runtime_mode,
            input_token_counts.iter().sum(),
            delta,
            stage_metrics,
            session_metric_snapshots,
        ));
        Ok(())
    }

    fn run_session_request(
        &mut self,
        session_id: SessionId,
        input_ids: &[u32],
        kind: SessionRequestKind,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(RuntimeError::EmptyInput {
                context: "session_request",
            });
        }

        let cache_metrics_before = self.cache_metrics_snapshot();
        let input = Tensor::from_slice(input_ids, (1, input_ids.len()), &self.device)?;
        let mut stage_metrics = RuntimeStageMetrics::default();
        let logits = match &mut self.inner {
            CandleModelInner::LlamaPaged {
                model,
                cache,
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restore_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| !page_backend.is_page_prepared(page_id))
                    .collect::<Vec<_>>();
                let restore_started = Instant::now();
                let _ = sessions
                    .cache_mut()
                    .restore_physical_pages(&restore_page_ids)?;
                stage_metrics.page_restore_millis += restore_started.elapsed().as_secs_f64() * 1e3;
                sessions.cache_mut().touch_physical_pages(&page_ids)?;
                let index_pos = sessions.current_position(session_id)? as usize;
                let logits =
                    model.forward(&input, index_pos, cache, sessions, session_id, page_backend)?;
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                logits
            }
            CandleModelInner::Qwen2Paged {
                model,
                sessions,
                page_backend,
                ..
            } => {
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restore_page_ids = page_ids
                    .iter()
                    .copied()
                    .filter(|&page_id| !page_backend.is_page_prepared(page_id))
                    .collect::<Vec<_>>();
                let restore_started = Instant::now();
                let _ = sessions
                    .cache_mut()
                    .restore_physical_pages(&restore_page_ids)?;
                stage_metrics.page_restore_millis += restore_started.elapsed().as_secs_f64() * 1e3;
                sessions.cache_mut().touch_physical_pages(&page_ids)?;
                let index_pos = sessions.current_position(session_id)? as usize;
                let logits =
                    model.forward(&input, index_pos, sessions, session_id, page_backend)?;
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                logits
            }
            CandleModelInner::Qwen35Paged {
                model,
                sessions,
                page_backend,
                ..
            } => {
                let index_pos = sessions.current_position(session_id)? as usize;
                let page_ids = sessions.physical_page_ids(session_id)?;
                let restore_started = Instant::now();
                let _ = sessions.cache_mut().restore_physical_pages(&page_ids)?;
                stage_metrics.page_restore_millis += restore_started.elapsed().as_secs_f64() * 1e3;
                sessions.cache_mut().touch_physical_pages(&page_ids)?;
                let cache_state = sessions
                    .hybrid_cache_state(session_id)?
                    .cloned()
                    .and_then(|state| match state {
                        crate::session::HybridCacheState::Qwen35(state) => Some(state),
                    })
                    .unwrap_or_else(|| model.empty_cache_state());
                let hybrid_restore_started = Instant::now();
                let (logits, next_state, profile) =
                    if Self::qwen35_paged_direct_full_attention_enabled(
                        input_ids.len(),
                        false,
                        page_backend.attention_path(),
                    ) {
                        let cache_state = crate::session::HybridCacheState::Qwen35(cache_state);
                        stage_metrics.hybrid_cache_restore_millis +=
                            hybrid_restore_started.elapsed().as_secs_f64() * 1e3;
                        model.forward_profiled_paged_full_attention(
                            &input,
                            index_pos,
                            Some(&cache_state),
                            sessions,
                            session_id,
                            page_backend,
                        )?
                    } else {
                        let full_layer_ids = model.full_attention_layer_ids();
                        let cache_state = Self::qwen35_restore_full_attention_pages(
                            sessions,
                            session_id,
                            &full_layer_ids,
                            self.dtype,
                            &self.device,
                            cache_state,
                            self.architecture.num_key_value_heads,
                            self.architecture.head_dim,
                        )?;
                        stage_metrics.hybrid_cache_restore_millis +=
                            hybrid_restore_started.elapsed().as_secs_f64() * 1e3;
                        let cache_state = crate::session::HybridCacheState::Qwen35(cache_state);
                        model.forward_profiled(&input, index_pos, Some(&cache_state))?
                    };
                let hybrid_store_started = Instant::now();
                let next_state = match next_state {
                    crate::session::HybridCacheState::Qwen35(mut state) => {
                        if !Self::qwen35_paged_direct_full_attention_enabled(
                            input_ids.len(),
                            false,
                            page_backend.attention_path(),
                        ) {
                            let full_layer_ids = model.full_attention_layer_ids();
                            Self::qwen35_store_full_attention_pages(
                                sessions,
                                session_id,
                                &full_layer_ids,
                                &mut state,
                                index_pos,
                                input_ids.len(),
                                self.architecture.head_dim,
                            )?;
                        }
                        crate::session::HybridCacheState::Qwen35(state)
                    }
                };
                sessions.set_hybrid_cache_state(session_id, Some(next_state))?;
                stage_metrics.hybrid_cache_store_millis +=
                    hybrid_store_started.elapsed().as_secs_f64() * 1e3;
                sessions.commit_positions(session_id, index_pos as u32, input_ids.len())?;
                stage_metrics.add_assign(&qwen35_runtime_stage_metrics(&profile));
                let spill_started = Instant::now();
                let _ = sessions.cache_mut().spill_to_budget()?;
                stage_metrics.page_spill_millis += spill_started.elapsed().as_secs_f64() * 1e3;
                logits
            }
            CandleModelInner::LlamaDense {
                model, sessions, ..
            } => {
                let session = Self::dense_session_mut(sessions, session_id)?;
                let index_pos = session.state.next_position as usize;
                let logits = model.forward(&input, index_pos, &mut session.cache)?;
                session.state.next_position += input_ids.len() as u32;
                session.state.token_count += input_ids.len();
                logits
            }
            CandleModelInner::Qwen2Dense { sessions, .. } => {
                let session = Self::dense_session_mut(sessions, session_id)?;
                let index_pos = session.state.next_position as usize;
                let logits = session.model.forward(&input, index_pos)?;
                session.state.next_position += input_ids.len() as u32;
                session.state.token_count += input_ids.len();
                logits
            }
            CandleModelInner::Qwen35Dense { sessions, .. } => {
                let session = Self::dense_session_mut(sessions, session_id)?;
                let index_pos = session.state.next_position as usize;
                let (logits, profile) = session.model.forward_profiled(&input, index_pos)?;
                session.state.next_position += input_ids.len() as u32;
                session.state.token_count += input_ids.len();
                stage_metrics.add_assign(&qwen35_runtime_stage_metrics(&profile));
                logits
            }
        };
        let logits = logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        self.record_request_metrics(
            &[session_id],
            kind,
            &[input_ids.len()],
            &cache_metrics_before,
            stage_metrics,
        )?;
        Ok(logits)
    }
}

fn collect_request_session_metrics(
    sessions: &SessionRuntime,
    session_ids: &[SessionId],
) -> Result<Vec<RequestSessionMetrics>> {
    session_ids
        .iter()
        .map(|&session_id| {
            Ok(RequestSessionMetrics::new(
                session_id,
                sessions.session_metrics(session_id)?.clone(),
            ))
        })
        .collect()
}

impl CausalLm for CandleCausalLm {
    fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    fn reset(&mut self) -> Result<()> {
        match &mut self.inner {
            CandleModelInner::LlamaPaged {
                config,
                cache,
                sessions,
                session_id,
                page_backend,
                ..
            } => {
                let resident_page_budget = sessions.cache().resident_page_budget();
                let resident_byte_budget = sessions.cache().resident_byte_budget();
                page_backend.reset_page_state();
                *cache = LlamaCache::new(true, self.dtype, config, &self.device)?;
                *sessions = SessionRuntime::new(
                    self.architecture.num_hidden_layers,
                    self.architecture.num_key_value_heads,
                    self.tokens_per_page,
                    self.architecture.head_dim,
                );
                sessions
                    .cache_mut()
                    .set_resident_page_budget(resident_page_budget)?;
                sessions
                    .cache_mut()
                    .set_resident_byte_budget(resident_byte_budget)?;
                *session_id = sessions.create_session();
            }
            CandleModelInner::Qwen2Paged {
                sessions,
                session_id,
                page_backend,
                ..
            } => {
                let resident_page_budget = sessions.cache().resident_page_budget();
                let resident_byte_budget = sessions.cache().resident_byte_budget();
                page_backend.reset_page_state();
                *sessions = SessionRuntime::new(
                    self.architecture.num_hidden_layers,
                    self.architecture.num_key_value_heads,
                    self.tokens_per_page,
                    self.architecture.head_dim,
                );
                sessions
                    .cache_mut()
                    .set_resident_page_budget(resident_page_budget)?;
                sessions
                    .cache_mut()
                    .set_resident_byte_budget(resident_byte_budget)?;
                *session_id = sessions.create_session();
            }
            CandleModelInner::Qwen35Paged {
                sessions,
                session_id,
                page_backend,
                ..
            } => {
                let resident_page_budget = sessions.cache().resident_page_budget();
                let resident_byte_budget = sessions.cache().resident_byte_budget();
                page_backend.reset_page_state();
                *sessions = SessionRuntime::new(
                    self.architecture.num_hidden_layers,
                    self.architecture.num_key_value_heads,
                    self.tokens_per_page,
                    self.architecture.head_dim,
                );
                sessions
                    .cache_mut()
                    .set_resident_page_budget(resident_page_budget)?;
                sessions
                    .cache_mut()
                    .set_resident_byte_budget(resident_byte_budget)?;
                *session_id = sessions.create_session();
            }
            CandleModelInner::LlamaDense {
                config,
                sessions,
                session_id,
                ..
            } => {
                sessions.clear();
                sessions.push(Some(Self::create_dense_llama_session(
                    config,
                    self.dtype,
                    &self.device,
                    0,
                )?));
                *session_id = 0;
            }
            CandleModelInner::Qwen2Dense {
                model,
                sessions,
                session_id,
            } => {
                sessions.clear();
                sessions.push(Some(Self::create_dense_qwen2_session(model, 0)));
                *session_id = 0;
            }
            CandleModelInner::Qwen35Dense {
                model,
                sessions,
                session_id,
            } => {
                sessions.clear();
                sessions.push(Some(Self::create_dense_qwen35_session(model, 0)));
                *session_id = 0;
            }
        }
        self.request_log.clear();
        Ok(())
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .to_vec())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(self.tokenizer.decode(token_ids, skip_special_tokens)?)
    }

    fn forward_next_logits(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let session_id = self.active_session_id().ok_or(RuntimeError::External {
            context: "forward_next_logits",
            message: "sessions are not available for this model".to_string(),
        })?;
        self.run_session_request(
            session_id,
            input_ids,
            if input_ids.len() == 1 {
                SessionRequestKind::Decode
            } else {
                SessionRequestKind::Prefill
            },
        )
    }
}
