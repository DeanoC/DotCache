use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    HfHubModelSource, MinimalQwen35Config, MinimalQwen35KvCache, MinimalQwen35Weights,
    PreparedModelPackage, Result, RuntimeError,
};

use crate::qwen35_minimal::{MinimalQwen35RuntimeProfile, ModelForCausalLM, PreparedTensorSource};

pub const SUPPORTED_MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
pub const BACKEND_IMPL: &str = "qwen35_fast_v1";

const SUPPORTED_HIDDEN_SIZE: usize = 1024;
const SUPPORTED_INTERMEDIATE_SIZE: usize = 3584;
const SUPPORTED_NUM_HIDDEN_LAYERS: usize = 24;
const SUPPORTED_NUM_ATTENTION_HEADS: usize = 8;
const SUPPORTED_NUM_KEY_VALUE_HEADS: usize = 2;
const SUPPORTED_HEAD_DIM: usize = 256;
const SUPPORTED_LINEAR_KEY_HEAD_DIM: usize = 128;
const SUPPORTED_LINEAR_VALUE_HEAD_DIM: usize = 128;
const SUPPORTED_LINEAR_NUM_KEY_HEADS: usize = 16;
const SUPPORTED_LINEAR_NUM_VALUE_HEADS: usize = 16;
const SUPPORTED_LINEAR_CONV_KERNEL_DIM: usize = 4;
const SUPPORTED_VOCAB_SIZE: usize = 248_320;
const SUPPORTED_MAX_POSITION_EMBEDDINGS: usize = 262_144;
const SUPPORTED_LAYER_TYPES: [&str; SUPPORTED_NUM_HIDDEN_LAYERS] = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35FastTopology {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl Qwen35FastTopology {
    pub fn qwen35_0_8b() -> Self {
        Self {
            hidden_size: SUPPORTED_HIDDEN_SIZE,
            intermediate_size: SUPPORTED_INTERMEDIATE_SIZE,
            num_hidden_layers: SUPPORTED_NUM_HIDDEN_LAYERS,
            num_attention_heads: SUPPORTED_NUM_ATTENTION_HEADS,
            num_key_value_heads: SUPPORTED_NUM_KEY_VALUE_HEADS,
            head_dim: SUPPORTED_HEAD_DIM,
            linear_key_head_dim: SUPPORTED_LINEAR_KEY_HEAD_DIM,
            linear_value_head_dim: SUPPORTED_LINEAR_VALUE_HEAD_DIM,
            linear_num_key_heads: SUPPORTED_LINEAR_NUM_KEY_HEADS,
            linear_num_value_heads: SUPPORTED_LINEAR_NUM_VALUE_HEADS,
            linear_conv_kernel_dim: SUPPORTED_LINEAR_CONV_KERNEL_DIM,
            vocab_size: SUPPORTED_VOCAB_SIZE,
            max_position_embeddings: SUPPORTED_MAX_POSITION_EMBEDDINGS,
        }
    }

    pub fn validate_config(config: &MinimalQwen35Config) -> Result<()> {
        let text = &config.text_config;
        let expected = Self::qwen35_0_8b();
        let mismatches = [
            ("hidden_size", text.hidden_size, expected.hidden_size),
            (
                "intermediate_size",
                text.intermediate_size,
                expected.intermediate_size,
            ),
            (
                "num_hidden_layers",
                text.num_hidden_layers,
                expected.num_hidden_layers,
            ),
            (
                "num_attention_heads",
                text.num_attention_heads,
                expected.num_attention_heads,
            ),
            (
                "num_key_value_heads",
                text.num_key_value_heads,
                expected.num_key_value_heads,
            ),
            ("head_dim", text.head_dim, expected.head_dim),
            (
                "linear_key_head_dim",
                text.linear_key_head_dim,
                expected.linear_key_head_dim,
            ),
            (
                "linear_value_head_dim",
                text.linear_value_head_dim,
                expected.linear_value_head_dim,
            ),
            (
                "linear_num_key_heads",
                text.linear_num_key_heads,
                expected.linear_num_key_heads,
            ),
            (
                "linear_num_value_heads",
                text.linear_num_value_heads,
                expected.linear_num_value_heads,
            ),
            (
                "linear_conv_kernel_dim",
                text.linear_conv_kernel_dim,
                expected.linear_conv_kernel_dim,
            ),
            ("vocab_size", text.vocab_size, expected.vocab_size),
            (
                "max_position_embeddings",
                text.max_position_embeddings,
                expected.max_position_embeddings,
            ),
        ];
        if let Some((field, got, expected)) = mismatches
            .into_iter()
            .find(|(_, got, expected)| got != expected)
        {
            return Err(RuntimeError::External {
                context: "qwen35_fast",
                message: format!(
                    "unsupported Qwen3.5-0.8B config field {field}: expected {expected}, got {got}"
                ),
            });
        }

        if text.layer_types.len() != SUPPORTED_LAYER_TYPES.len() {
            return Err(RuntimeError::External {
                context: "qwen35_fast",
                message: format!(
                    "unsupported Qwen3.5-0.8B layer_types length: expected {}, got {}",
                    SUPPORTED_LAYER_TYPES.len(),
                    text.layer_types.len()
                ),
            });
        }
        if let Some((idx, got, expected)) = text
            .layer_types
            .iter()
            .zip(SUPPORTED_LAYER_TYPES.iter())
            .enumerate()
            .find(|(_, (got, expected))| got.as_str() != **expected)
            .map(|(idx, (got, expected))| (idx, got, *expected))
        {
            return Err(RuntimeError::External {
                context: "qwen35_fast",
                message: format!(
                    "unsupported Qwen3.5-0.8B layer_types[{idx}]: expected {expected}, got {got}"
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Qwen35FastRunner {
    pub config: MinimalQwen35Config,
    pub topology: Qwen35FastTopology,
    pub weights: MinimalQwen35Weights,
    pub model: ModelForCausalLM,
    pub device: Device,
}

impl Qwen35FastRunner {
    fn argmax_last_token(logits: &Tensor) -> Result<u32> {
        let (_, seq_len, _vocab_size) = logits.dims3()?;
        let last_token = logits.narrow(1, seq_len.saturating_sub(1), 1)?;
        Ok(last_token
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1::<u32>()?[0])
    }

    fn hidden_states_on_runner_device(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if hidden_states.device().same_device(&self.device) {
            Ok(hidden_states.clone())
        } else {
            Ok(hidden_states.to_device(&self.device)?)
        }
    }

    pub fn supports_model_id(model_id: &str) -> bool {
        model_id == SUPPORTED_MODEL_ID
    }

    pub fn load_qwen35_0_8b_f16(model_id: &str, device: &Device) -> Result<Self> {
        if !Self::supports_model_id(model_id) {
            return Err(RuntimeError::External {
                context: "qwen35_fast",
                message: format!(
                    "qwen35_fast currently supports only {SUPPORTED_MODEL_ID}, got {model_id}"
                ),
            });
        }
        let topology = Qwen35FastTopology::qwen35_0_8b();
        if matches!(
            std::env::var("DOTCACHE_QWEN35_DISABLE_PREPARED_LOAD").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        ) {
            let source = HfHubModelSource::new()?;
            let artifacts = source.snapshot(model_id)?;
            let config: MinimalQwen35Config =
                serde_json::from_slice(&std::fs::read(&artifacts.config_path)?)?;
            let config = config.normalized();
            Qwen35FastTopology::validate_config(&config)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F16, device)?
            };
            let model = ModelForCausalLM::new(&config, vb)?;
            Ok(Self {
                config,
                topology,
                weights: MinimalQwen35Weights {
                    model_id: artifacts.model_id,
                    revision: artifacts.revision,
                    tokenizer_path: artifacts.tokenizer_path,
                    prepared_package_root: PathBuf::new(),
                },
                model,
                device: device.clone(),
            })
        } else {
            let package = Arc::new(PreparedModelPackage::resolve_or_build_qwen35_minimal(
                model_id, device,
            )?);
            let package_config: MinimalQwen35Config =
                serde_json::from_slice(&std::fs::read(package.config_path())?)?;
            let package_config = package_config.normalized();
            Qwen35FastTopology::validate_config(&package_config)?;
            let model = ModelForCausalLM::from_prepared(
                &package_config,
                PreparedTensorSource::new(package.clone(), device.clone()),
            )?;
            Ok(Self {
                config: package_config,
                topology,
                weights: MinimalQwen35Weights {
                    model_id: package.manifest().model_id.clone(),
                    revision: package.manifest().revision.clone(),
                    tokenizer_path: package.tokenizer_path(),
                    prepared_package_root: package.root().to_path_buf(),
                },
                model,
                device: device.clone(),
            })
        }
    }

    pub fn hidden_states_from_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_ids = if input_ids.device().same_device(&self.device) {
            input_ids.clone()
        } else {
            input_ids.to_device(&self.device)?
        };
        Ok(self.model.hidden_states_from_input_ids(&input_ids)?)
    }

    pub fn input_ids_tensor(&self, input_ids: &[u32]) -> Result<Tensor> {
        Tensor::from_slice(input_ids, (1, input_ids.len()), &self.device).map_err(Into::into)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    pub fn prefill_prompt_profiled(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<(Tensor, MinimalQwen35RuntimeProfile)> {
        let input_ids = self.input_ids_tensor(prompt_ids)?;
        let hidden_states = self.model.hidden_states_from_input_ids(&input_ids)?;
        Ok(self.model.forward_hidden_states_profiled(&hidden_states, 0)?)
    }

    pub fn prefill_next_token_profiled(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<(u32, MinimalQwen35RuntimeProfile)> {
        let (logits, profile) = self.prefill_prompt_profiled(prompt_ids)?;
        Ok((Self::argmax_last_token(&logits)?, profile))
    }

    pub fn decode_token_profiled(
        &mut self,
        token_id: u32,
    ) -> Result<(Tensor, MinimalQwen35RuntimeProfile)> {
        let input_ids = self.input_ids_tensor(&[token_id])?;
        let hidden_states = self.model.hidden_states_from_input_ids(&input_ids)?;
        let seqlen_offset = self.model.sequence_length();
        Ok(self
            .model
            .decode_hidden_states_profiled(&hidden_states, seqlen_offset)?)
    }

    pub fn decode_next_token_profiled(
        &mut self,
        token_id: u32,
    ) -> Result<(u32, MinimalQwen35RuntimeProfile)> {
        let (logits, profile) = self.decode_token_profiled(token_id)?;
        Ok((Self::argmax_last_token(&logits)?, profile))
    }

    pub fn decode_from_hidden_state(
        &mut self,
        hidden_state_t: &Tensor,
        cache: &mut MinimalQwen35KvCache,
    ) -> Result<Tensor> {
        let hidden_state_t = self.hidden_states_on_runner_device(hidden_state_t)?;
        self.model.restore_cache_state(cache)?;
        let seqlen_offset = cache.sequence_length();
        let logits = self
            .model
            .forward_hidden_states(&hidden_state_t, seqlen_offset)?
            .to_dtype(DType::F32)?;
        *cache = self.model.cache_state();
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_minimal::MinimalQwen35TextConfig;

    fn supported_config() -> MinimalQwen35Config {
        MinimalQwen35Config {
            text_config: MinimalQwen35TextConfig {
                vocab_size: SUPPORTED_VOCAB_SIZE,
                hidden_size: SUPPORTED_HIDDEN_SIZE,
                intermediate_size: SUPPORTED_INTERMEDIATE_SIZE,
                num_hidden_layers: SUPPORTED_NUM_HIDDEN_LAYERS,
                num_attention_heads: SUPPORTED_NUM_ATTENTION_HEADS,
                num_key_value_heads: SUPPORTED_NUM_KEY_VALUE_HEADS,
                hidden_act: candle_nn::Activation::Silu,
                max_position_embeddings: SUPPORTED_MAX_POSITION_EMBEDDINGS,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                attention_bias: false,
                attention_dropout: 0.0,
                head_dim: SUPPORTED_HEAD_DIM,
                linear_conv_kernel_dim: SUPPORTED_LINEAR_CONV_KERNEL_DIM,
                linear_key_head_dim: SUPPORTED_LINEAR_KEY_HEAD_DIM,
                linear_value_head_dim: SUPPORTED_LINEAR_VALUE_HEAD_DIM,
                linear_num_key_heads: SUPPORTED_LINEAR_NUM_KEY_HEADS,
                linear_num_value_heads: SUPPORTED_LINEAR_NUM_VALUE_HEADS,
                layer_types: SUPPORTED_LAYER_TYPES.iter().map(|s| s.to_string()).collect(),
                rope_parameters: None,
            },
        }
    }

    #[test]
    fn qwen35_fast_accepts_supported_topology() {
        Qwen35FastTopology::validate_config(&supported_config()).unwrap();
    }

    #[test]
    fn qwen35_fast_rejects_wrong_head_dim() {
        let mut config = supported_config();
        config.text_config.head_dim = 128;
        let err = Qwen35FastTopology::validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported Qwen3.5-0.8B config field head_dim"));
    }

    #[test]
    fn qwen35_fast_rejects_wrong_layer_pattern() {
        let mut config = supported_config();
        config.text_config.layer_types[3] = "linear_attention".to_string();
        let err = Qwen35FastTopology::validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported Qwen3.5-0.8B layer_types[3]"));
    }
}
