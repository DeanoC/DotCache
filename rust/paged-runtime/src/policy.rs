use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{ModelFamily, Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptBucketPolicy {
    pub family: String,
    pub min_prompt_tokens: usize,
    pub max_prompt_tokens: usize,
    pub resident_page_budget: Option<usize>,
    pub resident_byte_budget: Option<usize>,
    pub restore_cooldown_window: Option<u64>,
    pub source: String,
}

impl PromptBucketPolicy {
    pub fn family_enum(&self) -> Result<ModelFamily> {
        self.family.parse()
    }

    pub fn matches(&self, family: ModelFamily, prompt_token_count: usize) -> bool {
        self.family_enum().ok() == Some(family)
            && prompt_token_count >= self.min_prompt_tokens
            && prompt_token_count <= self.max_prompt_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptBucketPolicyTable {
    pub version: String,
    pub policies: Vec<PromptBucketPolicy>,
}

impl PromptBucketPolicyTable {
    pub fn recommended(
        &self,
        family: ModelFamily,
        prompt_token_count: usize,
    ) -> Option<PromptBucketPolicy> {
        self.policies
            .iter()
            .filter(|policy| policy.family_enum().ok() == Some(family))
            .find(|policy| policy.matches(family, prompt_token_count))
            .cloned()
            .or_else(|| {
                self.policies
                    .iter()
                    .filter(|policy| policy.family_enum().ok() == Some(family))
                    .max_by_key(|policy| policy.max_prompt_tokens)
                    .cloned()
            })
    }
}

const DEFAULT_PROMPT_POLICY_JSON: &str = include_str!("../policies/default_prompt_policies.json");

static DEFAULT_PROMPT_POLICY_TABLE: OnceLock<PromptBucketPolicyTable> = OnceLock::new();

pub fn default_prompt_policy_table() -> Result<&'static PromptBucketPolicyTable> {
    if let Some(table) = DEFAULT_PROMPT_POLICY_TABLE.get() {
        return Ok(table);
    }

    let parsed =
        serde_json::from_str(DEFAULT_PROMPT_POLICY_JSON).map_err(|err| RuntimeError::External {
            context: "prompt_policy_table",
            message: err.to_string(),
        })?;
    let _ = DEFAULT_PROMPT_POLICY_TABLE.set(parsed);
    Ok(DEFAULT_PROMPT_POLICY_TABLE
        .get()
        .expect("prompt policy table should be initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_table_selects_matching_bucket() {
        let table = default_prompt_policy_table().expect("default policy table should load");
        let llama_short = table
            .recommended(ModelFamily::Llama, 24)
            .expect("llama short bucket");
        assert_eq!(llama_short.min_prompt_tokens, 0);
        assert_eq!(llama_short.max_prompt_tokens, 32);
        assert_eq!(llama_short.resident_page_budget, Some(2));
        assert_eq!(llama_short.resident_byte_budget, None);

        let qwen_long = table
            .recommended(ModelFamily::Qwen2, 400)
            .expect("qwen long bucket");
        assert_eq!(qwen_long.min_prompt_tokens, 129);
        assert_eq!(qwen_long.max_prompt_tokens, 512);
        assert_eq!(qwen_long.resident_page_budget, None);
        assert_eq!(qwen_long.resident_byte_budget, None);
    }

    #[test]
    fn default_policy_table_falls_back_to_largest_bucket() {
        let table = default_prompt_policy_table().expect("default policy table should load");
        let llama = table
            .recommended(ModelFamily::Llama, 2_048)
            .expect("llama fallback bucket");
        assert_eq!(llama.min_prompt_tokens, 129);
        assert_eq!(llama.max_prompt_tokens, 512);
        assert_eq!(llama.restore_cooldown_window, Some(8));
    }

    #[test]
    fn default_policy_table_captures_qwen_short_bucket_budget() {
        let table = default_prompt_policy_table().expect("default policy table should load");
        let qwen_short = table
            .recommended(ModelFamily::Qwen2, 24)
            .expect("qwen short bucket");
        assert_eq!(qwen_short.min_prompt_tokens, 0);
        assert_eq!(qwen_short.max_prompt_tokens, 32);
        assert_eq!(qwen_short.resident_page_budget, Some(2));
        assert_eq!(qwen_short.resident_byte_budget, Some(528));
    }
}
