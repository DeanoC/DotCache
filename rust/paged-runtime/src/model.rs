#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Qwen2,
    Qwen35,
}

impl ModelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
        }
    }
}

impl std::str::FromStr for ModelFamily {
    type Err = crate::RuntimeError;

    fn from_str(value: &str) -> crate::Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "llama" => Ok(Self::Llama),
            "qwen2" => Ok(Self::Qwen2),
            "qwen35" | "qwen3.5" | "qwen3_5" => Ok(Self::Qwen35),
            other => Err(crate::RuntimeError::UnsupportedModelFamily {
                family: other.to_string(),
            }),
        }
    }
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    DenseControl,
    PagedControl,
    DotCacheExperimental,
    TorchControl,
}

impl RuntimeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DenseControl => "dense_control",
            Self::PagedControl => "paged_control",
            Self::DotCacheExperimental => "dotcache_experimental",
            Self::TorchControl => "torch_control",
        }
    }
}

impl std::fmt::Display for RuntimeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for RuntimeMode {
    type Err = crate::RuntimeError;

    fn from_str(value: &str) -> crate::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "dense_control" | "control_dense" => Ok(Self::DenseControl),
            "paged" | "paged_control" | "control_paged" => Ok(Self::PagedControl),
            "dotcache" | "dotcache_experimental" | "experimental" => {
                Ok(Self::DotCacheExperimental)
            }
            "torch" | "torch_control" | "aten_control" | "native_torch" => {
                Ok(Self::TorchControl)
            }
            other => Err(crate::RuntimeError::External {
                context: "runtime_mode",
                message: format!(
                    "unsupported runtime mode `{other}`, expected dense_control, paged_control, dotcache_experimental, or torch_control"
                ),
            }),
        }
    }
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeStageMetrics {
    pub tokenization_millis: f64,
    pub qkv_projection_millis: f64,
    pub kv_append_write_millis: f64,
    pub page_restore_millis: f64,
    pub page_spill_millis: f64,
    pub hybrid_cache_restore_millis: f64,
    pub hybrid_cache_store_millis: f64,
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
}

impl RuntimeStageMetrics {
    pub fn total_millis(&self) -> f64 {
        self.tokenization_millis
            + self.qkv_projection_millis
            + self.kv_append_write_millis
            + self.page_restore_millis
            + self.page_spill_millis
            + self.hybrid_cache_restore_millis
            + self.hybrid_cache_store_millis
            + self.layout_prepare_millis
            + self.attention_score_millis
            + self.attention_softmax_millis
            + self.attention_mix_millis
            + self.output_projection_millis
            + self.scheduler_planning_millis
            + self.transfer_millis
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.tokenization_millis += other.tokenization_millis;
        self.qkv_projection_millis += other.qkv_projection_millis;
        self.kv_append_write_millis += other.kv_append_write_millis;
        self.page_restore_millis += other.page_restore_millis;
        self.page_spill_millis += other.page_spill_millis;
        self.hybrid_cache_restore_millis += other.hybrid_cache_restore_millis;
        self.hybrid_cache_store_millis += other.hybrid_cache_store_millis;
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
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArchitecture {
    pub model_id: String,
    pub family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub eos_token_ids: Vec<u32>,
}

pub trait CausalLm {
    fn architecture(&self) -> &ModelArchitecture;
    fn reset(&mut self) -> crate::Result<()>;
    fn encode(&self, text: &str, add_special_tokens: bool) -> crate::Result<Vec<u32>>;
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> crate::Result<String>;
    fn forward_next_logits(&mut self, input_ids: &[u32]) -> crate::Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::{ModelFamily, RuntimeMode, RuntimeStageMetrics};

    #[test]
    fn qwen35_family_aliases_parse() {
        assert_eq!(
            "qwen35".parse::<ModelFamily>().unwrap(),
            ModelFamily::Qwen35
        );
        assert_eq!(
            "qwen3.5".parse::<ModelFamily>().unwrap(),
            ModelFamily::Qwen35
        );
        assert_eq!(
            "qwen3_5".parse::<ModelFamily>().unwrap(),
            ModelFamily::Qwen35
        );
    }

    #[test]
    fn runtime_mode_aliases_parse() {
        assert_eq!(
            "dense".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::DenseControl
        );
        assert_eq!(
            "paged_control".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::PagedControl
        );
        assert_eq!(
            "dotcache".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::DotCacheExperimental
        );
        assert_eq!(
            "torch_control".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::TorchControl
        );
    }

    #[test]
    fn runtime_stage_metrics_total_sums_components() {
        let metrics = RuntimeStageMetrics {
            tokenization_millis: 1.0,
            qkv_projection_millis: 2.0,
            kv_append_write_millis: 3.0,
            page_restore_millis: 4.0,
            page_spill_millis: 5.0,
            hybrid_cache_restore_millis: 6.0,
            hybrid_cache_store_millis: 7.0,
            layout_prepare_millis: 8.0,
            attention_score_millis: 9.0,
            attention_softmax_millis: 10.0,
            attention_mix_millis: 11.0,
            output_projection_millis: 12.0,
            full_attention_mask_prepare_millis: 0.0,
            full_attention_input_layout_millis: 0.0,
            full_attention_kv_materialize_millis: 0.0,
            full_attention_output_collect_millis: 0.0,
            full_attention_output_reshape_millis: 0.0,
            full_attention_gate_millis: 0.0,
            full_attention_kernel_execute_millis: 0.0,
            scheduler_planning_millis: 13.0,
            transfer_millis: 14.0,
            linear_attention_millis: 15.0,
            full_attention_millis: 16.0,
            mlp_millis: 17.0,
        };

        assert_eq!(metrics.total_millis(), 105.0);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GreedyGeneration {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
}

pub fn greedy_generate<M: CausalLm>(
    model: &mut M,
    prompt: &str,
    max_new_tokens: usize,
) -> crate::Result<GreedyGeneration> {
    let prompt_token_ids = model.encode(prompt, true)?;
    if prompt_token_ids.is_empty() {
        return Err(crate::RuntimeError::EmptyInput {
            context: "prompt encoding",
        });
    }

    model.reset()?;
    let mut logits = model.forward_next_logits(&prompt_token_ids)?;
    let mut generated_token_ids = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let next_token = argmax(&logits).ok_or(crate::RuntimeError::EmptyDecode)? as u32;
        generated_token_ids.push(next_token);

        if model.architecture().eos_token_ids.contains(&next_token) {
            break;
        }

        logits = model.forward_next_logits(&[next_token])?;
    }

    let mut all_token_ids = prompt_token_ids.clone();
    all_token_ids.extend_from_slice(&generated_token_ids);
    let text = model.decode(&all_token_ids, true)?;

    Ok(GreedyGeneration {
        prompt_token_ids,
        generated_token_ids,
        text,
    })
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index)
}
