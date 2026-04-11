use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{BackendDevice, CandleDeviceSelector, Result, RuntimeError};

const DEFAULT_LUCE_REPO: &str = "/tmp/luce-megakernel";
const LUCE_REPO_ENV: &str = "DOTCACHE_QWEN35_LUCE_REPO";
const IMPL_ENV: &str = "DOTCACHE_MEGAKERNEL_CONTROL_IMPL";

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
    pub record: serde_json::Value,
    pub status: String,
    pub warning_message: Option<String>,
    pub attention_path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MegakernelBackend {
    InTreeMinimal,
    LuceExternal,
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

fn requested_backend(args: &MegakernelControlBenchArgs<'_>) -> MegakernelBackend {
    match std::env::var(IMPL_ENV).ok().as_deref() {
        Some("luce") | Some("luce_external") | Some("external") => {
            return MegakernelBackend::LuceExternal
        }
        Some("minimal") | Some("in_tree") | Some("in_tree_minimal") => {
            return MegakernelBackend::InTreeMinimal
        }
        _ => {}
    }
    if args.luce_repo.is_some() {
        MegakernelBackend::LuceExternal
    } else {
        MegakernelBackend::InTreeMinimal
    }
}

fn raw_out_prefix(out_prefix: &Path) -> PathBuf {
    PathBuf::from(format!(
        "{}.megakernel_control_raw",
        out_prefix.display()
    ))
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

fn run_external_luce(args: &MegakernelControlBenchArgs<'_>) -> Result<MegakernelControlRecord> {
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

    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summary_path)?).map_err(|err| {
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
        attention_path: "luce_external_megakernel".to_string(),
    })
}

#[cfg(feature = "qwen35-minimal")]
fn run_in_tree_minimal(args: &MegakernelControlBenchArgs<'_>) -> Result<MegakernelControlRecord> {
    use candle_core::{Device, IndexOp, Tensor};
    use crate::Qwen35FastRunner;
    use serde_json::json;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    #[derive(Debug, Clone, Default)]
    struct ProfileSummary {
        qkv_projection_millis: f64,
        kv_append_write_millis: f64,
        layout_prepare_millis: f64,
        attention_score_millis: f64,
        attention_softmax_millis: f64,
        attention_mix_millis: f64,
        output_projection_millis: f64,
        full_attention_mask_prepare_millis: f64,
        full_attention_input_layout_millis: f64,
        full_attention_kv_materialize_millis: f64,
        full_attention_output_collect_millis: f64,
        full_attention_output_reshape_millis: f64,
        full_attention_gate_millis: f64,
        full_attention_kernel_execute_millis: f64,
        scheduler_planning_millis: f64,
        transfer_millis: f64,
        linear_attention_millis: f64,
        full_attention_millis: f64,
        mlp_millis: f64,
    }

    fn env_flag_truthy(key: &str) -> bool {
        matches!(
            std::env::var(key).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
        )
    }

    fn resolve_minimal_device(selector: &CandleDeviceSelector) -> Result<Device> {
        match selector.backend_device() {
            BackendDevice::Cpu => Ok(Device::Cpu),
            BackendDevice::Cuda { ordinal } => {
                #[cfg(feature = "qwen35-minimal-cuda")]
                {
                    Ok(Device::new_cuda(ordinal)?)
                }
                #[cfg(not(feature = "qwen35-minimal-cuda"))]
                {
                    Err(RuntimeError::BackendUnavailable {
                        backend: "qwen35-minimal-cuda",
                        device: format!("cuda:{ordinal}"),
                    })
                }
            }
            BackendDevice::Hip { ordinal } => {
                #[cfg(feature = "qwen35-minimal-hip")]
                {
                    Ok(Device::new_hip(ordinal)?)
                }
                #[cfg(not(feature = "qwen35-minimal-hip"))]
                {
                    Err(RuntimeError::BackendUnavailable {
                        backend: "qwen35-minimal-hip",
                        device: format!("hip:{ordinal}"),
                    })
                }
            }
            BackendDevice::Metal { ordinal } => Err(RuntimeError::BackendUnavailable {
                backend: "megakernel_control",
                device: format!("metal:{ordinal}"),
            }),
        }
    }

    fn build_prompt_ids(
        tokenizer: &Tokenizer,
        prompt: &str,
        target: Option<usize>,
    ) -> Result<Vec<u32>> {
        let mut token_ids = tokenizer
            .encode(prompt, true)
            .map_err(|err| RuntimeError::External {
                context: "megakernel_control",
                message: format!("tokenizer encode failed: {err}"),
            })?
            .get_ids()
            .to_vec();
        if token_ids.is_empty() {
            return Err(RuntimeError::External {
                context: "megakernel_control",
                message: "prompt encoding produced no tokens".to_string(),
            });
        }
        if let Some(target) = target {
            match token_ids.len().cmp(&target) {
                std::cmp::Ordering::Greater => token_ids.truncate(target),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Less => {
                    let filler = tokenizer
                        .encode(format!(" {prompt}"), false)
                        .map_err(|err| RuntimeError::External {
                            context: "megakernel_control",
                            message: format!("tokenizer filler encode failed: {err}"),
                        })?;
                    let filler_ids = filler.get_ids();
                    if filler_ids.is_empty() {
                        return Err(RuntimeError::External {
                            context: "megakernel_control",
                            message: "prompt filler encoding produced no tokens".to_string(),
                        });
                    }
                    while token_ids.len() < target {
                        token_ids.extend_from_slice(filler_ids);
                    }
                    token_ids.truncate(target);
                }
            }
        }
        Ok(token_ids)
    }

    fn argmax_last_token(logits: &Tensor) -> Result<u32> {
        let last_token = match logits.dims() {
            [1, _vocab] => logits.i(0)?,
            [1, seq, _vocab] => logits.i((0, seq - 1))?,
            dims => {
                return Err(RuntimeError::External {
                    context: "megakernel_control",
                    message: format!("unexpected logits shape {dims:?}"),
                })
            }
        };
        last_token.argmax(candle_core::D::Minus1)?.to_scalar::<u32>().map_err(Into::into)
    }

    fn run_once(
        runner: &mut Qwen35FastRunner,
        tokenizer: &Tokenizer,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<(String, usize, f64, f64, f64, ProfileSummary)> {
        runner.clear_kv_cache();
        let prefill_started = Instant::now();
        let (mut logits, mut profile) = runner.prefill_prompt_profiled(prompt_ids)?;
        let prefill_millis = prefill_started.elapsed().as_secs_f64() * 1_000.0;
        let mut generated_ids = prompt_ids.to_vec();

        let decode_started = Instant::now();
        let mut next_token = argmax_last_token(&logits)?;
        for _ in 0..max_new_tokens {
            generated_ids.push(next_token);
            let (next_logits, step_profile) = runner.decode_token_profiled(next_token)?;
            logits = next_logits;
            profile.add_assign(&step_profile);
            next_token = argmax_last_token(&logits)?;
        }
        let decode_millis = decode_started.elapsed().as_secs_f64() * 1_000.0;
        let total_millis = prefill_millis + decode_millis;
        let generated_text = tokenizer
            .decode(&generated_ids, true)
            .map_err(|err| RuntimeError::External {
                context: "megakernel_control",
                message: format!("tokenizer decode failed: {err}"),
            })?;
        Ok((
            generated_text,
            generated_ids.len().saturating_sub(prompt_ids.len()),
            prefill_millis,
            decode_millis,
            total_millis,
            ProfileSummary {
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
            },
        ))
    }

    if !Qwen35FastRunner::supports_model_id(args.model_id) {
        return Err(RuntimeError::External {
            context: "megakernel_control",
            message: format!(
                "in-tree megakernel_control currently supports only {}, got {}",
                crate::qwen35_fast::SUPPORTED_MODEL_ID,
                args.model_id
            ),
        });
    }

    let device = resolve_minimal_device(args.device)?;
    let load_started = Instant::now();
    let mut runner = Qwen35FastRunner::load_qwen35_0_8b_f16(args.model_id, &device)?;
    let load_millis = load_started.elapsed().as_secs_f64() * 1_000.0;
    let tokenizer = Tokenizer::from_file(&runner.weights.tokenizer_path).map_err(|err| {
        RuntimeError::External {
            context: "megakernel_control",
            message: format!("failed to load tokenizer: {err}"),
        }
    })?;
    let prompt_ids = build_prompt_ids(&tokenizer, args.prompt, args.prompt_token_target)?;

    let mut warmup_millis = 0.0f64;
    for _ in 0..args.warmup_runs {
        let warmup_started = Instant::now();
        let _ = run_once(&mut runner, &tokenizer, &prompt_ids, args.max_new_tokens)?;
        warmup_millis += warmup_started.elapsed().as_secs_f64() * 1_000.0;
        runner.clear_kv_cache();
    }

    let (generated_text, generated_token_count, prefill_millis, decode_millis, total_millis, profile) =
        run_once(&mut runner, &tokenizer, &prompt_ids, args.max_new_tokens)?;
    runner.clear_kv_cache();

    let prefill_tps = if prefill_millis > 0.0 {
        prompt_ids.len() as f64 / (prefill_millis / 1_000.0)
    } else {
        0.0
    };
    let decode_tps = if decode_millis > 0.0 {
        generated_token_count as f64 / (decode_millis / 1_000.0)
    } else {
        0.0
    };
    let total_tokens = prompt_ids.len() + generated_token_count;
    let total_tps = if total_millis > 0.0 {
        total_tokens as f64 / (total_millis / 1_000.0)
    } else {
        0.0
    };

    let mut record = serde_json::Map::new();
    record.insert("model_id".to_string(), json!(runner.weights.model_id));
    record.insert("device".to_string(), json!(args.device.to_string()));
    record.insert(
        "backend_impl".to_string(),
        json!(crate::qwen35_fast::BACKEND_IMPL),
    );
    record.insert("backend_status".to_string(), json!("completed"));
    record.insert(
        "full_prefill_megakernel_requested".to_string(),
        json!(env_flag_truthy("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL")),
    );
    record.insert(
        "hip_persistent_full_prefill_requested".to_string(),
        json!(env_flag_truthy("CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL")),
    );
    record.insert("prompt_token_count".to_string(), json!(prompt_ids.len()));
    record.insert(
        "prompt_token_target".to_string(),
        json!(args.prompt_token_target),
    );
    record.insert(
        "generated_token_count".to_string(),
        json!(generated_token_count),
    );
    record.insert("warmup_runs".to_string(), json!(args.warmup_runs));
    record.insert("warmup_millis".to_string(), json!(warmup_millis));
    record.insert("max_new_tokens".to_string(), json!(args.max_new_tokens));
    record.insert("load_millis".to_string(), json!(load_millis));
    record.insert("prefill_millis".to_string(), json!(prefill_millis));
    record.insert("decode_millis".to_string(), json!(decode_millis));
    record.insert("total_millis".to_string(), json!(total_millis));
    record.insert("prefill_tokens_per_second".to_string(), json!(prefill_tps));
    record.insert("decode_tokens_per_second".to_string(), json!(decode_tps));
    record.insert("total_tokens_per_second".to_string(), json!(total_tps));
    record.insert(
        "stage_qkv_projection_millis".to_string(),
        json!(profile.qkv_projection_millis),
    );
    record.insert(
        "stage_kv_append_write_millis".to_string(),
        json!(profile.kv_append_write_millis),
    );
    record.insert(
        "stage_layout_prepare_millis".to_string(),
        json!(profile.layout_prepare_millis),
    );
    record.insert(
        "stage_attention_score_millis".to_string(),
        json!(profile.attention_score_millis),
    );
    record.insert(
        "stage_attention_softmax_millis".to_string(),
        json!(profile.attention_softmax_millis),
    );
    record.insert(
        "stage_attention_mix_millis".to_string(),
        json!(profile.attention_mix_millis),
    );
    record.insert(
        "stage_output_projection_millis".to_string(),
        json!(profile.output_projection_millis),
    );
    record.insert(
        "stage_full_attention_mask_prepare_millis".to_string(),
        json!(profile.full_attention_mask_prepare_millis),
    );
    record.insert(
        "stage_full_attention_input_layout_millis".to_string(),
        json!(profile.full_attention_input_layout_millis),
    );
    record.insert(
        "stage_full_attention_kv_materialize_millis".to_string(),
        json!(profile.full_attention_kv_materialize_millis),
    );
    record.insert(
        "stage_full_attention_output_collect_millis".to_string(),
        json!(profile.full_attention_output_collect_millis),
    );
    record.insert(
        "stage_full_attention_output_reshape_millis".to_string(),
        json!(profile.full_attention_output_reshape_millis),
    );
    record.insert(
        "stage_full_attention_gate_millis".to_string(),
        json!(profile.full_attention_gate_millis),
    );
    record.insert(
        "stage_full_attention_kernel_execute_millis".to_string(),
        json!(profile.full_attention_kernel_execute_millis),
    );
    record.insert(
        "stage_scheduler_planning_millis".to_string(),
        json!(profile.scheduler_planning_millis),
    );
    record.insert(
        "stage_transfer_millis".to_string(),
        json!(profile.transfer_millis),
    );
    record.insert(
        "stage_linear_attention_millis".to_string(),
        json!(profile.linear_attention_millis),
    );
    record.insert(
        "stage_full_attention_millis".to_string(),
        json!(profile.full_attention_millis),
    );
    record.insert("stage_mlp_millis".to_string(), json!(profile.mlp_millis));
    record.insert("generated_text".to_string(), json!(generated_text));
    record.insert("invalid_token_id".to_string(), serde_json::Value::Null);
    record.insert("invalid_token_step".to_string(), serde_json::Value::Null);
    record.insert(
        "terminated_due_to_invalid_token".to_string(),
        json!(false),
    );

    Ok(MegakernelControlRecord {
        record: serde_json::Value::Object(record),
        status: "completed".to_string(),
        warning_message: None,
        attention_path: "in_tree_megakernel_control".to_string(),
    })
}

#[cfg(not(feature = "qwen35-minimal"))]
fn run_in_tree_minimal(_args: &MegakernelControlBenchArgs<'_>) -> Result<MegakernelControlRecord> {
    Err(RuntimeError::BackendUnavailable {
        backend: "qwen35-minimal",
        device: "enable qwen35-minimal-cuda or qwen35-minimal-hip to use in-tree megakernel_control"
            .to_string(),
    })
}

pub fn run_qwen35_text_bench(
    args: &MegakernelControlBenchArgs<'_>,
) -> Result<MegakernelControlRecord> {
    match requested_backend(args) {
        MegakernelBackend::InTreeMinimal => run_in_tree_minimal(args),
        MegakernelBackend::LuceExternal => run_external_luce(args),
    }
}
