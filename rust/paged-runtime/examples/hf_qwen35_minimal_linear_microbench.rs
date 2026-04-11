#[cfg(feature = "qwen35-minimal")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candle_core::{Device, Tensor};
    use dotcache_paged_runtime::{
        MinimalQwen35LinearAttentionLayerSpec, MinimalQwen35Runner, RuntimeError,
    };
    use serde::Serialize;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    #[derive(Clone, Debug)]
    enum DeviceSelector {
        Cpu,
        Hip(usize),
    }

    impl std::fmt::Display for DeviceSelector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Cpu => f.write_str("cpu"),
                Self::Hip(ordinal) => write!(f, "hip:{ordinal}"),
            }
        }
    }

    impl std::str::FromStr for DeviceSelector {
        type Err = RuntimeError;

        fn from_str(value: &str) -> dotcache_paged_runtime::Result<Self> {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized == "cpu" {
                return Ok(Self::Cpu);
            }
            if let Some(rest) = normalized.strip_prefix("hip") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| RuntimeError::External {
                        context: "device",
                        message: format!("invalid hip device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Hip(ordinal));
            }
            Err(RuntimeError::External {
                context: "device",
                message: format!("unsupported device `{value}`, expected cpu or hip[:ordinal]"),
            })
        }
    }

    impl DeviceSelector {
        fn resolve(&self) -> dotcache_paged_runtime::Result<Device> {
            match self {
                Self::Cpu => Ok(Device::Cpu),
                Self::Hip(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-hip")]
                    {
                        Ok(Device::new_hip(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-hip"))]
                    {
                        Err(RuntimeError::BackendUnavailable {
                            backend: "hip",
                            device: format!("hip:{ordinal}"),
                        })
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        model_id: String,
        prompt: String,
        layer_id: Option<usize>,
        prompt_token_target: Option<usize>,
        repeats: usize,
        warmup_repeats: usize,
        device: DeviceSelector,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        model_id: String,
        device: String,
        layer_id: usize,
        layer_conv_dim: usize,
        layer_num_v_heads: usize,
        layer_num_k_heads: usize,
        layer_head_k_dim: usize,
        layer_head_v_dim: usize,
        layer_key_dim: usize,
        layer_value_dim: usize,
        layer_state_len: usize,
        layer_kernel_size: usize,
        prompt_token_count: usize,
        repeats: usize,
        warmup_repeats: usize,
        load_millis: f64,
        capture_millis: f64,
        mean_total_millis: f64,
        best_total_millis: f64,
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
        stage_linear_conv_millis: f64,
        stage_linear_chunk_prepare_millis: f64,
        stage_linear_chunk_solve_millis: f64,
        stage_linear_chunk_scan_millis: f64,
        stage_linear_chunk_index_millis: f64,
        stage_linear_chunk_local_attn_millis: f64,
        stage_linear_chunk_recurrent_read_millis: f64,
        stage_linear_chunk_state_update_millis: f64,
        stage_linear_recurrent_loop_millis: f64,
        stage_linear_full_kernel_pack_millis: f64,
        stage_linear_full_kernel_execute_millis: f64,
        stage_linear_full_kernel_unpack_millis: f64,
        fused_prefill_unique_input_bytes: u64,
        fused_prefill_unique_output_bytes: u64,
        fused_prefill_algorithmic_bytes: u64,
        fused_prefill_algorithmic_flops: u64,
        fused_prefill_algorithmic_arithmetic_intensity: f64,
        fused_prefill_achieved_gbytes_per_sec: f64,
        fused_prefill_achieved_gflops_per_sec: f64,
    }

    fn fused_prefill_work_estimate(
        spec: MinimalQwen35LinearAttentionLayerSpec,
        batch_size: usize,
        seq_len: usize,
        dtype_bytes: u64,
    ) -> (u64, u64, u64, u64) {
        let batch = batch_size as u64;
        let seq = seq_len as u64;
        let conv_dim = spec.conv_dim as u64;
        let num_heads = spec.num_v_heads as u64;
        let state_len = spec.state_len as u64;
        let kernel_size = spec.kernel_size as u64;
        let out_width = conv_dim + num_heads;

        let unique_input_bytes =
            batch * ((conv_dim * seq) + (conv_dim * state_len) + (conv_dim * kernel_size) + (num_heads * seq) + (2 * num_heads)) * dtype_bytes;
        let unique_output_bytes =
            batch * ((seq * out_width) + (conv_dim * state_len)) * dtype_bytes;

        let conv_boundary_steps = seq.min(state_len);
        let conv_steady_steps = seq.saturating_sub(conv_boundary_steps);
        let conv_boundary_reads = batch * conv_dim * conv_boundary_steps * kernel_size;
        let conv_steady_reads = batch * conv_dim * conv_steady_steps * kernel_size;
        let conv_weight_reads = batch * conv_dim * seq * kernel_size;
        let g_reads = batch * num_heads * seq * 3;
        let conv_writes = batch * conv_dim * seq;
        let g_writes = batch * num_heads * seq;
        let tail_writes = batch * conv_dim * state_len;
        let algorithmic_bytes = (conv_boundary_reads
            + conv_steady_reads
            + conv_weight_reads
            + g_reads
            + conv_writes
            + g_writes
            + tail_writes)
            * dtype_bytes;

        let conv_flops = batch * conv_dim * seq * ((kernel_size * 2) - 1 + 4);
        let g_flops = batch * num_heads * seq * 8;
        let algorithmic_flops = conv_flops + g_flops;

        (
            unique_input_bytes,
            unique_output_bytes,
            algorithmic_bytes,
            algorithmic_flops,
        )
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

    fn parse_args() -> Result<Args, Box<dyn std::error::Error + Send + Sync>> {
        let mut args = std::env::args().skip(1);
        let model_id = args.next().ok_or(
            "usage: hf_qwen35_minimal_linear_microbench <model_id> <prompt> [--layer-id N] [--prompt-token-target N] [--repeats N] [--warmup-repeats N] [--device cpu|hip[:ordinal]]",
        )?;
        let prompt = args.next().ok_or("missing prompt")?;
        let mut parsed = Args {
            model_id,
            prompt,
            layer_id: None,
            prompt_token_target: None,
            repeats: 5,
            warmup_repeats: 1,
            device: DeviceSelector::Cpu,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--layer-id" => {
                    parsed.layer_id = Some(
                        args.next()
                            .ok_or("missing value for --layer-id")?
                            .parse::<usize>()?,
                    );
                }
                "--prompt-token-target" => {
                    parsed.prompt_token_target = Some(
                        args.next()
                            .ok_or("missing value for --prompt-token-target")?
                            .parse::<usize>()?,
                    );
                }
                "--repeats" => {
                    parsed.repeats = args
                        .next()
                        .ok_or("missing value for --repeats")?
                        .parse::<usize>()?;
                }
                "--warmup-repeats" => {
                    parsed.warmup_repeats = args
                        .next()
                        .ok_or("missing value for --warmup-repeats")?
                        .parse::<usize>()?;
                }
                "--device" => {
                    parsed.device = args
                        .next()
                        .ok_or("missing value for --device")?
                        .parse::<DeviceSelector>()?;
                }
                other => return Err(format!("unexpected argument {other:?}").into()),
            }
        }
        Ok(parsed)
    }

    let args = parse_args()?;
    let source = dotcache_paged_runtime::HfHubModelSource::new()?;
    let artifacts = source.snapshot(&args.model_id)?;
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;
    let prompt_ids = build_prompt_ids(&tokenizer, args.prompt.as_str(), args.prompt_token_target)?;

    let device = args.device.resolve()?;
    let load_started = Instant::now();
    let mut runner = MinimalQwen35Runner::load_from_hf_0_8b_f16(&args.model_id, &device)?;
    let load_elapsed = load_started.elapsed().as_secs_f64() * 1_000.0;

    let linear_layer_ids = runner.model.linear_attention_layer_ids();
    let layer_id = args.layer_id.unwrap_or_else(|| linear_layer_ids[0]);
    if !linear_layer_ids.contains(&layer_id) {
        return Err(format!(
            "layer {layer_id} is not a minimal qwen3.5 linear-attention layer, available: {:?}",
            linear_layer_ids
        )
        .into());
    }
    let layer_spec = runner.model.linear_attention_layer_spec(layer_id)?;

    let input_ids = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device)?;
    let capture_started = Instant::now();
    if args.warmup_repeats > 0 {
        let _ = runner
            .model
            .bench_linear_attention_layer(&input_ids, layer_id, 0, args.warmup_repeats)?;
    }
    let result = runner
        .model
        .bench_linear_attention_layer(&input_ids, layer_id, 0, args.repeats)?;
    let capture_elapsed = capture_started.elapsed().as_secs_f64() * 1_000.0;

    let dtype_bytes = 2u64;
    let (
        fused_prefill_unique_input_bytes,
        fused_prefill_unique_output_bytes,
        fused_prefill_algorithmic_bytes,
        fused_prefill_algorithmic_flops,
    ) = fused_prefill_work_estimate(layer_spec, 1, prompt_ids.len(), dtype_bytes);
    let conv_seconds = result.mean_profile.kv_append_write_millis / 1_000.0;
    let fused_prefill_algorithmic_arithmetic_intensity = if fused_prefill_algorithmic_bytes > 0 {
        fused_prefill_algorithmic_flops as f64 / fused_prefill_algorithmic_bytes as f64
    } else {
        0.0
    };
    let fused_prefill_achieved_gbytes_per_sec = if conv_seconds > 0.0 {
        fused_prefill_algorithmic_bytes as f64 / conv_seconds / 1.0e9
    } else {
        0.0
    };
    let fused_prefill_achieved_gflops_per_sec = if conv_seconds > 0.0 {
        fused_prefill_algorithmic_flops as f64 / conv_seconds / 1.0e9
    } else {
        0.0
    };

    let summary = Summary {
        model_id: args.model_id,
        device: args.device.to_string(),
        layer_id,
        layer_conv_dim: layer_spec.conv_dim,
        layer_num_v_heads: layer_spec.num_v_heads,
        layer_num_k_heads: layer_spec.num_k_heads,
        layer_head_k_dim: layer_spec.head_k_dim,
        layer_head_v_dim: layer_spec.head_v_dim,
        layer_key_dim: layer_spec.key_dim,
        layer_value_dim: layer_spec.value_dim,
        layer_state_len: layer_spec.state_len,
        layer_kernel_size: layer_spec.kernel_size,
        prompt_token_count: prompt_ids.len(),
        repeats: args.repeats,
        warmup_repeats: args.warmup_repeats,
        load_millis: load_elapsed,
        capture_millis: capture_elapsed,
        mean_total_millis: result.mean_total_millis,
        best_total_millis: result.best_total_millis,
        stage_qkv_projection_millis: result.mean_profile.qkv_projection_millis,
        stage_kv_append_write_millis: result.mean_profile.kv_append_write_millis,
        stage_layout_prepare_millis: result.mean_profile.layout_prepare_millis,
        stage_attention_score_millis: result.mean_profile.attention_score_millis,
        stage_attention_softmax_millis: result.mean_profile.attention_softmax_millis,
        stage_attention_mix_millis: result.mean_profile.attention_mix_millis,
        stage_output_projection_millis: result.mean_profile.output_projection_millis,
        stage_full_attention_mask_prepare_millis: result
            .mean_profile
            .full_attention_mask_prepare_millis,
        stage_full_attention_input_layout_millis: result
            .mean_profile
            .full_attention_input_layout_millis,
        stage_full_attention_kv_materialize_millis: result
            .mean_profile
            .full_attention_kv_materialize_millis,
        stage_full_attention_output_collect_millis: result
            .mean_profile
            .full_attention_output_collect_millis,
        stage_full_attention_output_reshape_millis: result
            .mean_profile
            .full_attention_output_reshape_millis,
        stage_full_attention_gate_millis: result.mean_profile.full_attention_gate_millis,
        stage_full_attention_kernel_execute_millis: result
            .mean_profile
            .full_attention_kernel_execute_millis,
        stage_scheduler_planning_millis: result.mean_profile.scheduler_planning_millis,
        stage_transfer_millis: result.mean_profile.transfer_millis,
        stage_linear_attention_millis: result.mean_profile.linear_attention_millis,
        stage_full_attention_millis: result.mean_profile.full_attention_millis,
        stage_mlp_millis: result.mean_profile.mlp_millis,
        stage_linear_conv_millis: result.mean_profile.linear_conv_millis,
        stage_linear_chunk_prepare_millis: result.mean_profile.linear_chunk_prepare_millis,
        stage_linear_chunk_solve_millis: result.mean_profile.linear_chunk_solve_millis,
        stage_linear_chunk_scan_millis: result.mean_profile.linear_chunk_scan_millis,
        stage_linear_chunk_index_millis: result.mean_profile.linear_chunk_index_millis,
        stage_linear_chunk_local_attn_millis: result.mean_profile.linear_chunk_local_attn_millis,
        stage_linear_chunk_recurrent_read_millis: result
            .mean_profile
            .linear_chunk_recurrent_read_millis,
        stage_linear_chunk_state_update_millis: result
            .mean_profile
            .linear_chunk_state_update_millis,
        stage_linear_recurrent_loop_millis: result.mean_profile.linear_recurrent_loop_millis,
        stage_linear_full_kernel_pack_millis: result.mean_profile.linear_full_kernel_pack_millis,
        stage_linear_full_kernel_execute_millis: result
            .mean_profile
            .linear_full_kernel_execute_millis,
        stage_linear_full_kernel_unpack_millis: result
            .mean_profile
            .linear_full_kernel_unpack_millis,
        fused_prefill_unique_input_bytes,
        fused_prefill_unique_output_bytes,
        fused_prefill_algorithmic_bytes,
        fused_prefill_algorithmic_flops,
        fused_prefill_algorithmic_arithmetic_intensity,
        fused_prefill_achieved_gbytes_per_sec,
        fused_prefill_achieved_gflops_per_sec,
    };

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(not(feature = "qwen35-minimal"))]
fn main() {
    eprintln!("enable the `qwen35-minimal` feature to run this example");
}
