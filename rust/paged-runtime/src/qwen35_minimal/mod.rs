mod hip;
mod model;
mod with_tracing;

pub use model::{
    CacheState as MinimalQwen35KvCache, Config as MinimalQwen35Config, ModelForCausalLM,
    TextConfig as MinimalQwen35TextConfig,
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{HfHubModelSource, Result};

#[derive(Debug, Clone)]
pub struct MinimalQwen35Weights {
    pub model_id: String,
    pub revision: String,
}

#[derive(Debug)]
pub struct MinimalQwen35Runner {
    pub config: MinimalQwen35Config,
    pub weights: MinimalQwen35Weights,
    pub model: ModelForCausalLM,
    pub device: Device,
}

impl MinimalQwen35Runner {
    fn hidden_states_on_runner_device(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if hidden_states.device().same_device(&self.device) {
            Ok(hidden_states.clone())
        } else {
            Ok(hidden_states.to_device(&self.device)?)
        }
    }

    pub fn load_from_hf_0_8b_f16(model_id: &str, device: &Device) -> Result<Self> {
        let source = HfHubModelSource::new()?;
        let artifacts = source.snapshot(model_id)?;
        let config: MinimalQwen35Config =
            serde_json::from_slice(&std::fs::read(&artifacts.config_path)?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F16, device)?
        };
        let model = ModelForCausalLM::new(&config, vb)?;
        Ok(Self {
            config,
            weights: MinimalQwen35Weights {
                model_id: artifacts.model_id,
                revision: artifacts.revision,
            },
            model,
            device: device.clone(),
        })
    }

    pub fn hidden_states_from_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_ids = if input_ids.device().same_device(&self.device) {
            input_ids.clone()
        } else {
            input_ids.to_device(&self.device)?
        };
        Ok(self.model.hidden_states_from_input_ids(&input_ids)?)
    }

    pub fn prefill_from_hidden_states(
        &mut self,
        hidden_states: &Tensor,
    ) -> Result<(Tensor, MinimalQwen35KvCache)> {
        let hidden_states = self.hidden_states_on_runner_device(hidden_states)?;
        let logits = self
            .model
            .forward_hidden_states(&hidden_states, 0)?
            .to_dtype(DType::F32)?;
        Ok((logits, self.model.cache_state()))
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
