#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candle_core::{DType, Tensor};
    use candle_nn::VarBuilder;
    use dotcache_paged_runtime::{CandleDeviceSelector, HfHubModelSource};
    use serde::Serialize;
    use std::fs;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    #[derive(Debug)]
    struct Args {
        model_id: String,
        prompt: String,
        layer_id: Option<usize>,
        prompt_token_target: Option<usize>,
        repeats: usize,
        warmup_repeats: usize,
        device: CandleDeviceSelector,
        dtype: DType,
        summary_json: Option<String>,
        list_layers: bool,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        model_id: String,
        device: String,
        dtype: String,
        prompt_token_count: usize,
        prompt_token_target: Option<usize>,
        layer_id: usize,
        repeats: usize,
        warmup_repeats: usize,
        mean_total_millis: f64,
        best_total_millis: f64,
        iteration_total_millis: Vec<f64>,
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
        best_stage_linear_attention_millis: f64,
        load_millis: f64,
        capture_millis: f64,
    }

    fn parse_dtype(raw: &str) -> Result<DType, DynError> {
        match raw {
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "f32" => Ok(DType::F32),
            _ => Err(format!("unsupported --dtype {raw:?}, expected f16|bf16|f32").into()),
        }
    }

    fn build_prompt_token_ids(
        tokenizer: &Tokenizer,
        prompt: &str,
        prompt_token_target: Option<usize>,
    ) -> Result<Vec<u32>, DynError> {
        let encoding = tokenizer.encode(prompt, true)?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err("prompt encoding produced no tokens".into());
        }
        if let Some(target) = prompt_token_target {
            if target == 0 {
                return Err("--prompt-token-target must be at least 1".into());
            }
            if token_ids.len() > target {
                token_ids.truncate(target);
            } else if token_ids.len() < target {
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
        Ok(token_ids)
    }

    fn parse_args() -> Result<Args, DynError> {
        let mut args = std::env::args().skip(1);
        let model_id = args.next().ok_or(
            "usage: hf_qwen35_linear_microbench <model_id> <prompt> [--layer-id N] [--prompt-token-target N] [--repeats N] [--warmup-repeats N] [--device cpu|metal[:ordinal]|cuda[:ordinal]] [--dtype f16|bf16|f32] [--summary-json PATH] [--list-layers]",
        )?;
        let prompt = args.next().ok_or("missing prompt")?;
        let mut parsed = Args {
            model_id,
            prompt,
            layer_id: None,
            prompt_token_target: None,
            repeats: 10,
            warmup_repeats: 1,
            device: CandleDeviceSelector::Metal { ordinal: 0 },
            dtype: DType::F16,
            summary_json: None,
            list_layers: false,
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
                        .parse::<CandleDeviceSelector>()?;
                }
                "--dtype" => {
                    parsed.dtype = parse_dtype(&args.next().ok_or("missing value for --dtype")?)?;
                }
                "--summary-json" => {
                    parsed.summary_json =
                        Some(args.next().ok_or("missing value for --summary-json")?);
                }
                "--list-layers" => parsed.list_layers = true,
                other => return Err(format!("unexpected argument {other:?}").into()),
            }
        }
        Ok(parsed)
    }

    let args = parse_args()?;
    let load_start = Instant::now();
    let source = HfHubModelSource::new()?;
    let artifacts = source.snapshot(&args.model_id)?;
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;
    let config: candle_transformers::models::qwen3_5::Config =
        serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
    let config = config.normalized();
    let device = args.device.resolve()?;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, args.dtype, &device)?
    };
    let mut model =
        candle_transformers::models::qwen3_5::ModelForCausalLM::new(&config, var_builder)?;
    let load_millis = load_start.elapsed().as_secs_f64() * 1_000.0;

    let linear_layer_ids = model.linear_attention_layer_ids();
    if args.list_layers {
        println!("{}", serde_json::to_string_pretty(&linear_layer_ids)?);
        return Ok(());
    }
    let layer_id = args.layer_id.unwrap_or_else(|| linear_layer_ids[0]);
    if !linear_layer_ids.contains(&layer_id) {
        return Err(format!(
            "layer {layer_id} is not a Qwen3.5 linear-attention layer, available linear layers: {:?}",
            linear_layer_ids
        )
        .into());
    }

    let prompt_token_ids =
        build_prompt_token_ids(&tokenizer, &args.prompt, args.prompt_token_target)?;
    let input_ids = Tensor::from_vec(
        prompt_token_ids.clone(),
        (1, prompt_token_ids.len()),
        &device,
    )?;

    let capture_start = Instant::now();
    if args.warmup_repeats > 0 {
        let _ = model.bench_linear_attention_layer(&input_ids, layer_id, 0, args.warmup_repeats)?;
    }
    let result = model.bench_linear_attention_layer(&input_ids, layer_id, 0, args.repeats)?;
    let capture_millis = capture_start.elapsed().as_secs_f64() * 1_000.0;

    let summary = Summary {
        model_id: args.model_id,
        device: args.device.to_string(),
        dtype: format!("{:?}", args.dtype).to_lowercase(),
        prompt_token_count: prompt_token_ids.len(),
        prompt_token_target: args.prompt_token_target,
        layer_id,
        repeats: args.repeats,
        warmup_repeats: args.warmup_repeats,
        mean_total_millis: result.mean_total_millis,
        best_total_millis: result.best_total_millis,
        iteration_total_millis: result.iteration_total_millis,
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
        best_stage_linear_attention_millis: result.best_profile.linear_attention_millis,
        load_millis,
        capture_millis,
    };

    if let Some(path) = args.summary_json.as_ref() {
        fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
        eprintln!("microbench summary written to {path}");
    }

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
type DynError = Box<dyn std::error::Error + Send + Sync>;
