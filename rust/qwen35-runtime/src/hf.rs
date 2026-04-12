use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use hf_hub::api::sync::{Api, ApiBuilder};
use serde::Deserialize;

use crate::{Result, RuntimeError};

#[derive(Debug, Clone)]
pub struct HfModelArtifacts {
    pub model_id: String,
    pub revision: String,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HfModelWeightIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

impl HfModelWeightIndex {
    pub fn unique_weight_filenames(&self) -> Vec<String> {
        self.weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[derive(Debug)]
pub struct HfHubModelSource {
    api: Api,
}

impl HfHubModelSource {
    pub fn new() -> Result<Self> {
        let api = ApiBuilder::from_env().build()?;
        Ok(Self { api })
    }

    pub fn snapshot(&self, model_id: &str) -> Result<HfModelArtifacts> {
        let repo = self.api.model(model_id.to_string());
        let info = repo.info()?;
        let filenames = info
            .siblings
            .iter()
            .map(|entry| entry.rfilename.as_str())
            .collect::<BTreeSet<_>>();

        let config_path = repo.get("config.json")?;
        let tokenizer_path = if filenames.contains("tokenizer.json") {
            repo.get("tokenizer.json")?
        } else {
            return Err(RuntimeError::MissingAsset {
                model_id: model_id.to_string(),
                filename: "tokenizer.json".to_string(),
            });
        };

        let weight_paths = if filenames.contains("model.safetensors.index.json") {
            let index_path = repo.get("model.safetensors.index.json")?;
            let index: HfModelWeightIndex = serde_json::from_slice(&fs::read(&index_path)?)?;
            index
                .unique_weight_filenames()
                .into_iter()
                .map(|filename| repo.get(&filename))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else if filenames.contains("model.safetensors") {
            vec![repo.get("model.safetensors")?]
        } else {
            return Err(RuntimeError::MissingAsset {
                model_id: model_id.to_string(),
                filename: "model.safetensors or model.safetensors.index.json".to_string(),
            });
        };

        Ok(HfModelArtifacts {
            model_id: model_id.to_string(),
            revision: info.sha,
            config_path,
            tokenizer_path,
            weight_paths,
        })
    }
}

