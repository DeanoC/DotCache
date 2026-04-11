use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::{BackendDevice, CandleDeviceSelector, Result, RuntimeError};

const DEFAULT_LUCE_REPO: &str = "/tmp/luce-megakernel";
const LUCE_REPO_ENV: &str = "DOTCACHE_QWEN35_LUCE_REPO";

#[derive(Debug, Clone)]
pub struct MegakernelControlBenchArgs<'a> {
    pub model_id: &'a str,
    pub prompt: &'a str,
    pub out_prefix: &'a Path,
    pub prompt_token_target: Option<usize>,
    pub device: &'a CandleDeviceSelector,
    pub warmup_runs: usize,
    pub max_new_tokens: usize,
    pub luce_repo: Option<&'a Path>,
}

#[derive(Debug, Clone)]
pub struct MegakernelControlRecord {
    pub record: Value,
    pub status: String,
    pub warning_message: Option<String>,
}

fn repo_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| RuntimeError::External {
            context: "megakernel_control",
            message: "failed to resolve repository root".to_string(),
        })
}

fn python_executable(repo_root: &Path) -> Result<PathBuf> {
    for candidate in [
        repo_root.join(".venv311/bin/python"),
        repo_root.join(".venv/bin/python"),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Ok(PathBuf::from("python3"))
}

fn luce_repo(args: &MegakernelControlBenchArgs<'_>) -> PathBuf {
    if let Some(path) = args.luce_repo {
        return path.to_path_buf();
    }
    if let Ok(path) = std::env::var(LUCE_REPO_ENV) {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(DEFAULT_LUCE_REPO)
}

fn luce_device(selector: &CandleDeviceSelector) -> Result<String> {
    match selector.backend_device() {
        BackendDevice::Cuda { ordinal } => Ok(format!("cuda:{ordinal}")),
        _ => Err(RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "megakernel_control requires a CUDA device, got {}",
                selector
            ),
        }),
    }
}

fn raw_out_prefix(out_prefix: &Path) -> PathBuf {
    PathBuf::from(format!(
        "{}.megakernel_control_raw",
        out_prefix.display()
    ))
}

pub fn run_qwen35_text_bench(
    args: &MegakernelControlBenchArgs<'_>,
) -> Result<MegakernelControlRecord> {
    if args.model_id != "Qwen/Qwen3.5-0.8B" {
        return Err(RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "megakernel_control currently supports only Qwen/Qwen3.5-0.8B, got {}",
                args.model_id
            ),
        });
    }

    let repo_root = repo_root()?;
    let python = python_executable(&repo_root)?;
    let luce_repo = luce_repo(args);
    if !luce_repo.exists() {
        return Err(RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "luce repo not found at {} (override with --luce-repo or {})",
                luce_repo.display(),
                LUCE_REPO_ENV
            ),
        });
    }

    let device = luce_device(args.device)?;
    let raw_prefix = raw_out_prefix(args.out_prefix);
    let summary_path = PathBuf::from(format!("{}.summary.json", raw_prefix.display()));

    let mut command = Command::new(python);
    command
        .current_dir(&repo_root)
        .env("PYTHONPATH", ".")
        .arg("benchmarks/bench_qwen35_luce_external.py")
        .arg(args.model_id)
        .arg(args.prompt)
        .arg(&raw_prefix)
        .arg("--luce-repo")
        .arg(&luce_repo)
        .arg("--device")
        .arg(device)
        .arg("--warmup-runs")
        .arg(args.warmup_runs.to_string())
        .arg("--max-new-tokens")
        .arg(args.max_new_tokens.to_string());
    if let Some(target) = args.prompt_token_target {
        command.arg("--prompt-token-target").arg(target.to_string());
    }

    let output = command.output().map_err(RuntimeError::from)?;
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !summary_path.is_file() {
        return Err(RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "luce control failed with status {} and did not write {}: {}",
                output.status,
                summary_path.display(),
                stderr
            ),
        });
    }

    let record: Value = serde_json::from_slice(&std::fs::read(&summary_path)?).map_err(|err| {
        RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "failed to parse luce summary {}: {err}",
                summary_path.display()
            ),
        }
    })?;

    let (status, warning_message) = if output.status.success() {
        ("completed".to_string(), None)
    } else {
        let warning = if stderr.is_empty() {
            format!("luce control exited with status {}", output.status)
        } else {
            format!("luce control exited with status {}: {}", output.status, stderr)
        };
        ("completed_with_warning".to_string(), Some(warning))
    };

    Ok(MegakernelControlRecord {
        record,
        status,
        warning_message,
    })
}
