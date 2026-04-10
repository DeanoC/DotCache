use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::DType;
use serde_json::Value;

use crate::{BackendDevice, CandleDeviceSelector, Result, RuntimeError};

fn repo_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| RuntimeError::External {
            context: "torch_control",
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
    Err(RuntimeError::External {
        context: "torch_control",
        message: "could not find a Python executable in .venv311 or .venv".to_string(),
    })
}

fn torch_backend_and_device(selector: &CandleDeviceSelector) -> (&'static str, String) {
    match selector.backend_device() {
        BackendDevice::Cpu => ("cpu_ref", "cpu".to_string()),
        BackendDevice::Metal { .. } => ("torch_mps", "mps".to_string()),
        BackendDevice::Cuda { ordinal } => {
            let device = if ordinal == 0 {
                "cuda".to_string()
            } else {
                format!("cuda:{ordinal}")
            };
            ("torch_cuda", device)
        }
        BackendDevice::Hip { ordinal } => {
            let device = if ordinal == 0 {
                "cuda".to_string()
            } else {
                format!("cuda:{ordinal}")
            };
            ("torch_cuda", device)
        }
    }
}

fn torch_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
        _ => "float32",
    }
}

fn parse_json_record(output: std::process::Output, context: &'static str) -> Result<Value> {
    if !output.status.success() {
        return Err(RuntimeError::External {
            context,
            message: format!(
                "python harness failed with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .ok_or_else(|| RuntimeError::External {
            context,
            message: format!("python harness did not emit a JSON record:\n{stdout}"),
        })?;

    serde_json::from_str(line).map_err(|err| RuntimeError::External {
        context,
        message: format!("failed to parse python harness JSON: {err}"),
    })
}

pub fn run_qwen35_text_bench(
    model_id: &str,
    prompt: &str,
    prompt_token_target: Option<usize>,
    device: &CandleDeviceSelector,
    dtype: DType,
    warmup_runs: usize,
    max_new_tokens: usize,
    profile_stages: bool,
) -> Result<Value> {
    let repo_root = repo_root()?;
    let python = python_executable(&repo_root)?;
    let (backend, torch_device) = torch_backend_and_device(device);

    let mut command = Command::new(python);
    command
        .current_dir(&repo_root)
        .env("PYTHONPATH", ".")
        .arg("benchmarks/bench_qwen35_text.py")
        .arg("--model-id")
        .arg(model_id)
        .arg("--device")
        .arg(&torch_device)
        .arg("--backend")
        .arg(backend)
        .arg("--torch-dtype")
        .arg(torch_dtype(dtype))
        .arg("--warmup-runs")
        .arg(warmup_runs.to_string())
        .arg("--max-new-tokens")
        .arg(max_new_tokens.to_string())
        .arg("--prompt-text")
        .arg(prompt)
        .arg("--continue-on-error");
    if let Some(target) = prompt_token_target {
        command.arg("--prompt-token-target").arg(target.to_string());
    }
    if profile_stages {
        command.arg("--profile-stages");
    }

    parse_json_record(
        command.output().map_err(RuntimeError::from)?,
        "torch_control bench",
    )
}

#[derive(Debug, Clone)]
pub struct TorchControlWorkloadArgs<'a> {
    pub model_id: &'a str,
    pub shared_prompt: &'a str,
    pub shared_prompt_token_target: Option<usize>,
    pub device: &'a CandleDeviceSelector,
    pub dtype: DType,
    pub warmup_runs: usize,
    pub total_sessions: usize,
    pub wave_size: usize,
    pub decode_rounds_per_wave: usize,
    pub max_new_tokens: usize,
    pub suffix_prefix: &'a str,
    pub stress_mode: bool,
    pub stress_suffix_repeats: usize,
    pub profile_stages: bool,
}

pub fn run_qwen35_text_workload(args: &TorchControlWorkloadArgs<'_>) -> Result<Value> {
    let repo_root = repo_root()?;
    let python = python_executable(&repo_root)?;
    let (backend, torch_device) = torch_backend_and_device(args.device);

    let mut command = Command::new(python);
    command
        .current_dir(&repo_root)
        .env("PYTHONPATH", ".")
        .arg("benchmarks/bench_qwen35_text_workload.py")
        .arg("--model-id")
        .arg(args.model_id)
        .arg("--device")
        .arg(&torch_device)
        .arg("--backend")
        .arg(backend)
        .arg("--torch-dtype")
        .arg(torch_dtype(args.dtype))
        .arg("--warmup-runs")
        .arg(args.warmup_runs.to_string())
        .arg("--total-sessions")
        .arg(args.total_sessions.to_string())
        .arg("--wave-size")
        .arg(args.wave_size.to_string())
        .arg("--decode-rounds-per-wave")
        .arg(args.decode_rounds_per_wave.to_string())
        .arg("--max-new-tokens")
        .arg(args.max_new_tokens.to_string())
        .arg("--suffix-prefix")
        .arg(args.suffix_prefix)
        .arg("--shared-prompt-text")
        .arg(args.shared_prompt)
        .arg("--continue-on-error");
    if let Some(target) = args.shared_prompt_token_target {
        command
            .arg("--shared-prompt-token-target")
            .arg(target.to_string());
    } else {
        command.arg("--shared-prompt-length").arg("1");
    }
    if args.stress_mode {
        command.arg("--stress");
    }
    if args.profile_stages {
        command.arg("--profile-stages");
    }
    command
        .arg("--stress-suffix-repeats")
        .arg(args.stress_suffix_repeats.to_string());

    parse_json_record(
        command.output().map_err(RuntimeError::from)?,
        "torch_control workload",
    )
}
