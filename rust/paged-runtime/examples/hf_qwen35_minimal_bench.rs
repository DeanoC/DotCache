#[cfg(feature = "qwen35-minimal")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candle_core::{DType, Device, Tensor};
    use candle_core::IndexOp;
    use dotcache_paged_runtime::{HfHubModelSource, MinimalQwen35KvCache, MinimalQwen35Runner};
    use serde::Serialize;
    use std::path::PathBuf;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    #[derive(Clone, Debug)]
    enum DeviceSelector {
        Cpu,
        Cuda(usize),
        Hip(usize),
    }

    impl std::fmt::Display for DeviceSelector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Cpu => f.write_str("cpu"),
                Self::Cuda(ordinal) => write!(f, "cuda:{ordinal}"),
                Self::Hip(ordinal) => write!(f, "hip:{ordinal}"),
            }
        }
    }

    impl std::str::FromStr for DeviceSelector {
        type Err = dotcache_paged_runtime::RuntimeError;

        fn from_str(value: &str) -> dotcache_paged_runtime::Result<Self> {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized == "cpu" {
                return Ok(Self::Cpu);
            }
            if let Some(rest) = normalized.strip_prefix("cuda") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| dotcache_paged_runtime::RuntimeError::External {
                        context: "device",
                        message: format!("invalid cuda device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Cuda(ordinal));
            }
            if let Some(rest) = normalized.strip_prefix("hip") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| dotcache_paged_runtime::RuntimeError::External {
                        context: "device",
                        message: format!("invalid hip device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Hip(ordinal));
            }
            Err(dotcache_paged_runtime::RuntimeError::External {
                context: "device",
                message: format!(
                    "unsupported device `{value}`, expected cpu, cuda[:ordinal], or hip[:ordinal]"
                ),
            })
        }
    }

    impl DeviceSelector {
        fn resolve(&self) -> dotcache_paged_runtime::Result<Device> {
            match self {
                Self::Cpu => Ok(Device::Cpu),
                Self::Cuda(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-cuda")]
                    {
                        Ok(Device::new_cuda(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-cuda"))]
                    {
                        Err(dotcache_paged_runtime::RuntimeError::BackendUnavailable {
                            backend: "cuda",
                            device: format!("cuda:{ordinal}"),
                        })
                    }
                }
                Self::Hip(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-hip")]
                    {
                        Ok(Device::new_hip(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-hip"))]
                    {
                        Err(dotcache_paged_runtime::RuntimeError::BackendUnavailable {
                            backend: "hip",
                            device: format!("hip:{ordinal}"),
                        })
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    struct BenchArgs {
        model_id: String,
        prompt: String,
        out_prefix: String,
        prompt_token_target: Option<usize>,
        warmup_runs: usize,
        max_new_tokens: usize,
        device: DeviceSelector,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct TimedRun {
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
    }

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

    #[derive(Debug)]
    struct RunResult {
        generated_text: String,
        generated_token_count: usize,
        timings: TimedRun,
        profile: ProfileSummary,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        model_id: String,
        device: String,
        prompt: String,
        prompt_token_count: usize,
        prompt_token_target: Option<usize>,
        generated_token_count: usize,
        warmup_runs: usize,
        warmup_millis: f64,
        max_new_tokens: usize,
        load_millis: f64,
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
        prefill_tokens_per_second: f64,
        decode_tokens_per_second: f64,
        total_tokens_per_second: f64,
        stage_qkv_projection_millis: f64,
        stage_kv_append_write_millis: f64,
        stage_layout_prepare_millis: f64,
        stage_attention_score_millis: f64,
        stage_attention_softmax_millis: f64,
        stage_attention_mix_millis: f64,
        stage_output_projection_millis: f64,
        stage_full_attention_mask_prepare_millis: f64,
        stage_full_attention_input_layout_millis: f64,
        stage_full_attention_kv_materialize_millis: f64,
        stage_full_attention_output_collect_millis: f64,
        stage_full_attention_output_reshape_millis: f64,
        stage_full_attention_gate_millis: f64,
        stage_full_attention_kernel_execute_millis: f64,
        stage_scheduler_planning_millis: f64,
        stage_transfer_millis: f64,
        stage_linear_attention_millis: f64,
        stage_full_attention_millis: f64,
        stage_mlp_millis: f64,
        generated_text: String,
    }

    fn parse_args() -> Result<BenchArgs, String> {
        let mut args = std::env::args().skip(1);
        let model_id = args.next().ok_or_else(|| {
            "usage: hf_qwen35_minimal_bench <model_id> <prompt> <out_prefix> [--prompt-token-target N] [--warmup-runs N] [--max-new-tokens N] [--device cpu|cuda[:ordinal]|hip[:ordinal]]".to_string()
        })?;
        let prompt = args.next().ok_or_else(|| "missing prompt".to_string())?;
        let out_prefix = args.next().ok_or_else(|| "missing out_prefix".to_string())?;
        let mut parsed = BenchArgs {
            model_id,
            prompt,
            out_prefix,
            prompt_token_target: None,
            warmup_runs: 1,
            max_new_tokens: 16,
            device: DeviceSelector::Cpu,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--prompt-token-target" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --prompt-token-target".to_string())?;
                    parsed.prompt_token_target = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid prompt token target `{value}`: {err}"))?,
                    );
                }
                "--warmup-runs" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --warmup-runs".to_string())?;
                    parsed.warmup_runs = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid warmup run count `{value}`: {err}"))?;
                }
                "--max-new-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --max-new-tokens".to_string())?;
                    parsed.max_new_tokens = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid max new token count `{value}`: {err}"))?;
                }
                "--device" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --device".to_string())?;
                    parsed.device = value.parse().map_err(|err| format!("{err}"))?;
                }
                other => return Err(format!("unknown flag `{other}`")),
            }
        }
        Ok(parsed)
    }

    fn build_prompt_ids(
        tokenizer: &Tokenizer,
        prompt: &str,
        target: Option<usize>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut token_ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err("prompt encoding produced no tokens".into());
        }
        if let Some(target) = target {
            match token_ids.len().cmp(&target) {
                std::cmp::Ordering::Greater => token_ids.truncate(target),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Less => {
                    let filler = tokenizer.encode(format!(" {prompt}"), false)?;
                    let filler_ids = filler.get_ids();
                    if filler_ids.is_empty() {
                        return Err("prompt filler encoding produced no tokens".into());
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

    fn argmax_last_token(logits: &Tensor) -> dotcache_paged_runtime::Result<u32> {
        let last_token = match logits.dims() {
            [1, _vocab] => logits.i(0)?,
            [1, seq, _vocab] => logits.i((0, seq - 1))?,
            dims => {
                return Err(dotcache_paged_runtime::RuntimeError::External {
                    context: "minimal-bench",
                    message: format!("unexpected logits shape {dims:?}"),
                });
            }
        };
        let values = last_token
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut best: Option<(usize, f32)> = None;
        for (index, value) in values.iter().copied().enumerate() {
            if value.is_nan() {
                continue;
            }
            match best {
                Some((_, best_value)) if value <= best_value => {}
                _ => best = Some((index, value)),
            }
        }
        let (index, _) = best.ok_or_else(|| dotcache_paged_runtime::RuntimeError::External {
            context: "minimal-bench",
            message: "all logits were NaN".to_string(),
        })?;
        Ok(index as u32)
    }

    fn run_once(
        runner: &mut MinimalQwen35Runner,
        tokenizer: &Tokenizer,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> dotcache_paged_runtime::Result<RunResult> {
        let input_ids = Tensor::from_vec(
            prompt_ids.to_vec(),
            (1, prompt_ids.len()),
            &Device::Cpu,
        )?;
        let hidden_states = runner.hidden_states_from_input_ids(&input_ids)?;

        let prefill_started = Instant::now();
        let (mut logits, mut profile) = runner.model.forward_hidden_states_profiled(&hidden_states, 0)?;
        let prefill_millis = prefill_started.elapsed().as_secs_f64() * 1_000.0;
        let mut cache: MinimalQwen35KvCache = runner.model.cache_state();
        let mut generated_ids = prompt_ids.to_vec();

        let decode_started = Instant::now();
        let mut next_token = argmax_last_token(&logits)?;
        for _ in 0..max_new_tokens {
            generated_ids.push(next_token);
            let decode_input = Tensor::from_vec(vec![next_token], (1, 1), &Device::Cpu)?;
            let hidden_state_t = runner.hidden_states_from_input_ids(&decode_input)?;
            let seqlen_offset = cache.sequence_length();
            runner.model.restore_cache_state(&cache)?;
            let (next_logits, step_profile) =
                runner.model.forward_hidden_states_profiled(&hidden_state_t, seqlen_offset)?;
            logits = next_logits;
            profile.add_assign(&step_profile);
            cache = runner.model.cache_state();
            next_token = argmax_last_token(&logits)?;
        }
        let decode_millis = decode_started.elapsed().as_secs_f64() * 1_000.0;
        let total_millis = prefill_millis + decode_millis;
        let generated_text = tokenizer.decode(&generated_ids, true)?;
        Ok(RunResult {
            generated_text,
            generated_token_count: generated_ids.len().saturating_sub(prompt_ids.len()),
            timings: TimedRun {
                prefill_millis,
                decode_millis,
                total_millis,
            },
            profile: ProfileSummary {
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
        })
    }

    let args = parse_args().map_err(|err| format!("{err}"))?;
    let source = HfHubModelSource::new()?;
    let artifacts = source.snapshot(&args.model_id)?;
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;
    let prompt_ids = build_prompt_ids(&tokenizer, &args.prompt, args.prompt_token_target)?;
    let device = args.device.resolve()?;

    let load_started = Instant::now();
    let mut runner = MinimalQwen35Runner::load_from_hf_0_8b_f16(&args.model_id, &device)?;
    let load_millis = load_started.elapsed().as_secs_f64() * 1_000.0;

    let mut warmup_millis = 0.0f64;
    for _ in 0..args.warmup_runs {
        runner.model.clear_kv_cache();
        let warmup_started = Instant::now();
        let _ = run_once(&mut runner, &tokenizer, &prompt_ids, args.max_new_tokens)?;
        warmup_millis += warmup_started.elapsed().as_secs_f64() * 1_000.0;
        runner.model.clear_kv_cache();
    }

    runner.model.clear_kv_cache();
    let run = run_once(&mut runner, &tokenizer, &prompt_ids, args.max_new_tokens)?;
    runner.model.clear_kv_cache();

    let prefill_tps = if run.timings.prefill_millis > 0.0 {
        prompt_ids.len() as f64 / (run.timings.prefill_millis / 1_000.0)
    } else {
        0.0
    };
    let decode_tps = if run.timings.decode_millis > 0.0 {
        run.generated_token_count as f64 / (run.timings.decode_millis / 1_000.0)
    } else {
        0.0
    };
    let total_tokens = prompt_ids.len() + run.generated_token_count;
    let total_tps = if run.timings.total_millis > 0.0 {
        total_tokens as f64 / (run.timings.total_millis / 1_000.0)
    } else {
        0.0
    };

    let summary = Summary {
        model_id: runner.weights.model_id.clone(),
        device: args.device.to_string(),
        prompt: args.prompt.clone(),
        prompt_token_count: prompt_ids.len(),
        prompt_token_target: args.prompt_token_target,
        generated_token_count: run.generated_token_count,
        warmup_runs: args.warmup_runs,
        warmup_millis,
        max_new_tokens: args.max_new_tokens,
        load_millis,
        prefill_millis: run.timings.prefill_millis,
        decode_millis: run.timings.decode_millis,
        total_millis: run.timings.total_millis,
        prefill_tokens_per_second: prefill_tps,
        decode_tokens_per_second: decode_tps,
        total_tokens_per_second: total_tps,
        stage_qkv_projection_millis: run.profile.qkv_projection_millis,
        stage_kv_append_write_millis: run.profile.kv_append_write_millis,
        stage_layout_prepare_millis: run.profile.layout_prepare_millis,
        stage_attention_score_millis: run.profile.attention_score_millis,
        stage_attention_softmax_millis: run.profile.attention_softmax_millis,
        stage_attention_mix_millis: run.profile.attention_mix_millis,
        stage_output_projection_millis: run.profile.output_projection_millis,
        stage_full_attention_mask_prepare_millis: run.profile.full_attention_mask_prepare_millis,
        stage_full_attention_input_layout_millis: run.profile.full_attention_input_layout_millis,
        stage_full_attention_kv_materialize_millis: run.profile.full_attention_kv_materialize_millis,
        stage_full_attention_output_collect_millis: run.profile.full_attention_output_collect_millis,
        stage_full_attention_output_reshape_millis: run.profile.full_attention_output_reshape_millis,
        stage_full_attention_gate_millis: run.profile.full_attention_gate_millis,
        stage_full_attention_kernel_execute_millis: run.profile.full_attention_kernel_execute_millis,
        stage_scheduler_planning_millis: run.profile.scheduler_planning_millis,
        stage_transfer_millis: run.profile.transfer_millis,
        stage_linear_attention_millis: run.profile.linear_attention_millis,
        stage_full_attention_millis: run.profile.full_attention_millis,
        stage_mlp_millis: run.profile.mlp_millis,
        generated_text: run.generated_text,
    };

    let summary_path = PathBuf::from(format!("{}.summary.json", args.out_prefix));
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!(
        "minimal bench summary written to {}",
        summary_path.display()
    );
    eprintln!(
        "device={} prompt_tokens={} generated_tokens={} prefill_ms={:.2} decode_ms={:.2} total_ms={:.2} prefill_tps={:.2} decode_tps={:.2}",
        args.device,
        prompt_ids.len(),
        run.generated_token_count,
        run.timings.prefill_millis,
        run.timings.decode_millis,
        run.timings.total_millis,
        prefill_tps,
        decode_tps,
    );
    println!("{}", summary.generated_text);
    Ok(())
}

#[cfg(not(feature = "qwen35-minimal"))]
fn main() {
    eprintln!("enable the `qwen35-minimal` feature to run this example");
}
