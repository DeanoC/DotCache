#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use std::time::Instant;

    use candle_core::DType;
    use dotcache_paged_runtime::{
        AttentionPathMode, BackendDevice, CandleCausalLm, CandleDeviceSelector, CausalLm,
        ModelFamily, PageModePolicy, PageModeSpec, RuntimeMode, RuntimeStageMetrics,
    };
    use serde::Serialize;
    use serde_json::Value;

    #[derive(Debug)]
    struct BenchArgs {
        family: ModelFamily,
        model_id: String,
        prompt: String,
        out_prefix: String,
        luce_repo: Option<PathBuf>,
        prompt_token_target: Option<usize>,
        device: CandleDeviceSelector,
        dtype: DType,
        runtime_mode: RuntimeMode,
        attention_path: Option<AttentionPathMode>,
        warmup_runs: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        serving_preset: Option<String>,
        default_key_page_mode: Option<PageModeSpec>,
        default_value_page_mode: Option<PageModeSpec>,
        key_layer_page_modes: Option<String>,
        value_layer_page_modes: Option<String>,
        sync_stage_profile: bool,
    }

    #[derive(Debug, Serialize)]
    struct BenchmarkSummary {
        model_id: String,
        family: String,
        device: String,
        dtype: String,
        runtime_mode: String,
        attention_path: String,
        prompt: String,
        prompt_token_count: usize,
        prompt_token_target: Option<usize>,
        generated_token_count: usize,
        warmup_runs: usize,
        warmup_millis: f64,
        max_new_tokens: usize,
        tokens_per_page: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        serving_preset: Option<String>,
        default_key_page_mode: String,
        default_value_page_mode: String,
        key_layer_page_mode_overrides: Vec<String>,
        value_layer_page_mode_overrides: Vec<String>,
        request_count: usize,
        prefill_request_count: usize,
        decode_request_count: usize,
        batch_decode_request_count: usize,
        spill_count: usize,
        restore_count: usize,
        spilled_bytes: usize,
        restored_bytes: usize,
        cooldown_hit_count: usize,
        physical_page_count: usize,
        virtual_page_count: usize,
        resident_physical_page_count: usize,
        spilled_physical_page_count: usize,
        resident_physical_byte_count: usize,
        spilled_physical_byte_count: usize,
        pinned_physical_page_count: usize,
        stage_tokenization_millis: f64,
        stage_qkv_projection_millis: f64,
        stage_kv_append_write_millis: f64,
        stage_page_restore_millis: f64,
        stage_page_spill_millis: f64,
        stage_hybrid_cache_restore_millis: f64,
        stage_hybrid_cache_store_millis: f64,
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
        stage_total_millis: f64,
        stage_profile_sync_enabled: bool,
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
        prefill_tokens_per_second: f64,
        decode_tokens_per_second: f64,
        total_tokens_per_second: f64,
        backend_status: Option<String>,
        warning_message: Option<String>,
        invalid_token_id: Option<i64>,
        invalid_token_step: Option<usize>,
        terminated_due_to_invalid_token: bool,
        text: String,
        trace_jsonl_path: String,
    }

    fn parse_override_arg(raw: &str) -> Result<Vec<(usize, PageModeSpec)>, String> {
        PageModePolicy::parse_overrides(raw).map_err(|err| err.to_string())
    }

    fn build_page_mode_policy(
        default_key: Option<PageModeSpec>,
        default_value: Option<PageModeSpec>,
        key_overrides: Option<&str>,
        value_overrides: Option<&str>,
    ) -> Result<PageModePolicy, String> {
        let mut policy = PageModePolicy::exact();
        if let Some(mode) = default_key {
            policy
                .set_default_key(mode)
                .map_err(|err| err.to_string())?;
        }
        if let Some(mode) = default_value {
            policy
                .set_default_value(mode)
                .map_err(|err| err.to_string())?;
        }
        if let Some(raw) = key_overrides {
            for (layer, mode) in parse_override_arg(raw)? {
                policy
                    .set_override(dotcache_paged_runtime::PageSideKind::Key, layer, mode)
                    .map_err(|err| err.to_string())?;
            }
        }
        if let Some(raw) = value_overrides {
            for (layer, mode) in parse_override_arg(raw)? {
                policy
                    .set_override(dotcache_paged_runtime::PageSideKind::Value, layer, mode)
                    .map_err(|err| err.to_string())?;
            }
        }
        Ok(policy)
    }

    fn describe_page_mode_policy(
        policy: Option<&PageModePolicy>,
    ) -> (String, String, Vec<String>, Vec<String>) {
        let Some(policy) = policy else {
            return (
                "exact".to_string(),
                "exact".to_string(),
                Vec::new(),
                Vec::new(),
            );
        };
        let key_overrides = policy
            .key_overrides()
            .iter()
            .map(|(layer, mode)| format!("{layer}={}", mode.describe()))
            .collect::<Vec<_>>();
        let value_overrides = policy
            .value_overrides()
            .iter()
            .map(|(layer, mode)| format!("{layer}={}", mode.describe()))
            .collect::<Vec<_>>();
        (
            policy.default_key().describe(),
            policy.default_value().describe(),
            key_overrides,
            value_overrides,
        )
    }

    fn validate_serving_preset_name(preset: &str) -> Result<(), String> {
        match preset {
            "m3-int8" => Ok(()),
            other => Err(format!(
                "invalid --serving-preset: {other} (expected m3-int8)"
            )),
        }
    }

    fn finalize_serving_preset(parsed: &mut BenchArgs) -> Result<(), String> {
        let Some(preset) = parsed.serving_preset.as_deref() else {
            return Ok(());
        };
        match preset {
            "m3-int8" => {
                if matches!(
                    parsed.runtime_mode,
                    RuntimeMode::DenseControl
                        | RuntimeMode::TorchControl
                        | RuntimeMode::MegakernelControl
                ) {
                    return Err("--serving-preset m3-int8 requires paged_control or dotcache_experimental".to_string());
                }
                if !matches!(parsed.attention_path, None | Some(AttentionPathMode::Paged)) {
                    return Err("--serving-preset m3-int8 is incompatible with --attention-path fused".to_string());
                }
                if parsed.default_key_page_mode.is_some()
                    || parsed.default_value_page_mode.is_some()
                    || parsed.key_layer_page_modes.is_some()
                    || parsed.value_layer_page_modes.is_some()
                {
                    return Err("--serving-preset m3-int8 cannot be combined with explicit page mode overrides".to_string());
                }
                parsed.attention_path = Some(AttentionPathMode::Paged);
                parsed.default_key_page_mode = Some(
                    "M3/affine/4/int8"
                        .parse::<PageModeSpec>()
                        .map_err(|err| err.to_string())?,
                );
                parsed.default_value_page_mode = Some(
                    "M3/affine/4/int8"
                        .parse::<PageModeSpec>()
                        .map_err(|err| err.to_string())?,
                );
                Ok(())
            }
            other => Err(format!(
                "invalid --serving-preset: {other} (expected m3-int8)"
            )),
        }
    }

    fn parse_args() -> Result<BenchArgs, String> {
        let mut args = std::env::args().skip(1);
        let family = args.next().ok_or_else(|| {
            "usage: hf_bench <family> <model_id> <prompt> <out_prefix> [--prompt-token-target N] [--device cpu|metal[:ordinal]|cuda[:ordinal]|hip[:ordinal]] [--dtype f16|bf16|f32] [--runtime-mode dense_control|paged_control|dotcache_experimental|torch_control|megakernel_control] [--attention-path paged|fused] [--warmup-runs N] [--max-new-tokens N] [--tokens-per-page N] [--resident-page-budget N] [--resident-byte-budget N] [--restore-cooldown N] [--serving-preset m3-int8] [--default-key-page-mode SPEC] [--default-value-page-mode SPEC] [--key-layer-page-modes LAYER=SPEC,...] [--value-layer-page-modes LAYER=SPEC,...] [--luce-repo PATH] [--sync-stage-profile]".to_string()
        })?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let prompt = args.next().ok_or_else(|| "missing prompt".to_string())?;
        let out_prefix = args
            .next()
            .ok_or_else(|| "missing out_prefix".to_string())?;

        let mut parsed = BenchArgs {
            family: family.parse().map_err(|err| format!("{err}"))?,
            model_id,
            prompt,
            out_prefix,
            luce_repo: None,
            prompt_token_target: None,
            device: CandleDeviceSelector::Cpu,
            dtype: DType::F32,
            runtime_mode: RuntimeMode::PagedControl,
            attention_path: None,
            warmup_runs: 1,
            max_new_tokens: 16,
            tokens_per_page: CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
            resident_page_budget: None,
            resident_byte_budget: None,
            restore_cooldown_window: None,
            serving_preset: None,
            default_key_page_mode: None,
            default_value_page_mode: None,
            key_layer_page_modes: None,
            value_layer_page_modes: None,
            sync_stage_profile: false,
        };

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--sync-stage-profile" => {
                    parsed.sync_stage_profile = true;
                }
                "--warmup-runs" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.warmup_runs = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --warmup-runs: {err}"))?;
                }
                "--max-new-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.max_new_tokens = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --max-new-tokens: {err}"))?;
                }
                "--tokens-per-page" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.tokens_per_page = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --tokens-per-page: {err}"))?;
                }
                "--prompt-token-target" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.prompt_token_target = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --prompt-token-target: {err}"))?,
                    );
                }
                "--device" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.device = value.parse::<CandleDeviceSelector>()?;
                }
                "--dtype" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.dtype = match value.as_str() {
                        "f16" | "float16" => DType::F16,
                        "bf16" => DType::BF16,
                        "f32" | "float32" => DType::F32,
                        other => {
                            return Err(format!(
                                "invalid --dtype: {other} (expected f16, bf16, or f32)"
                            ))
                        }
                    };
                }
                "--runtime-mode" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.runtime_mode = value
                        .parse::<RuntimeMode>()
                        .map_err(|err| format!("{err}"))?;
                }
                "--attention-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.attention_path = Some(value.parse::<AttentionPathMode>()?);
                }
                "--resident-page-budget" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.resident_page_budget = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --resident-page-budget: {err}"))?,
                    );
                }
                "--resident-byte-budget" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.resident_byte_budget = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --resident-byte-budget: {err}"))?,
                    );
                }
                "--restore-cooldown" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.restore_cooldown_window = Some(
                        value
                            .parse::<u64>()
                            .map_err(|err| format!("invalid --restore-cooldown: {err}"))?,
                    );
                }
                "--serving-preset" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    validate_serving_preset_name(&value)?;
                    parsed.serving_preset = Some(value);
                }
                "--default-key-page-mode" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.default_key_page_mode = Some(
                        value
                            .parse::<PageModeSpec>()
                            .map_err(|err| format!("{err}"))?,
                    );
                }
                "--default-value-page-mode" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.default_value_page_mode = Some(
                        value
                            .parse::<PageModeSpec>()
                            .map_err(|err| format!("{err}"))?,
                    );
                }
                "--key-layer-page-modes" => {
                    parsed.key_layer_page_modes = Some(
                        args.next()
                            .ok_or_else(|| format!("missing value for {flag}"))?,
                    );
                }
                "--value-layer-page-modes" => {
                    parsed.value_layer_page_modes = Some(
                        args.next()
                            .ok_or_else(|| format!("missing value for {flag}"))?,
                    );
                }
                "--luce-repo" => {
                    parsed.luce_repo = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| format!("missing value for {flag}"))?,
                    ));
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }

        if !std::env::args().any(|arg| arg == "--dtype") {
            parsed.dtype = match parsed.device.backend_device() {
                BackendDevice::Metal { .. }
                | BackendDevice::Cuda { .. }
                | BackendDevice::Hip { .. } => DType::F16,
                BackendDevice::Cpu => DType::F32,
            };
        }

        finalize_serving_preset(&mut parsed)?;

        Ok(parsed)
    }

    fn argmax(values: &[f32]) -> Option<usize> {
        values
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index)
    }

    fn millis(duration: std::time::Duration) -> f64 {
        duration.as_secs_f64() * 1_000.0
    }

    fn tokens_per_second(token_count: usize, duration: std::time::Duration) -> f64 {
        let seconds = duration.as_secs_f64();
        if seconds == 0.0 {
            token_count as f64
        } else {
            token_count as f64 / seconds
        }
    }

    fn aggregate_stage_metrics(model: &CandleCausalLm) -> RuntimeStageMetrics {
        let mut totals = RuntimeStageMetrics::default();
        for request in model.request_metrics() {
            totals.add_assign(request.stage_metrics());
        }
        totals
    }

    fn stage_profile_sync_enabled(explicit_flag: bool) -> bool {
        if explicit_flag {
            unsafe {
                std::env::set_var("CANDLE_QWEN35_PROFILE_SYNC", "1");
            }
            return true;
        }
        matches!(
            std::env::var("CANDLE_QWEN35_PROFILE_SYNC").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    fn json_string(record: &Value, key: &str) -> Result<String, Box<dyn std::error::Error>> {
        record
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("missing or invalid string field `{key}`").into())
    }

    fn json_usize(record: &Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
        record
            .get(key)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| format!("missing or invalid integer field `{key}`").into())
    }

    fn json_f64(record: &Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
        record
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing or invalid float field `{key}`").into())
    }

    fn json_f64_default(record: &Value, key: &str) -> f64 {
        record.get(key).and_then(Value::as_f64).unwrap_or(0.0)
    }

    fn json_string_opt(record: &Value, key: &str) -> Option<String> {
        record.get(key).and_then(Value::as_str).map(str::to_string)
    }

    fn json_i64_opt(record: &Value, key: &str) -> Option<i64> {
        record.get(key).and_then(Value::as_i64)
    }

    fn json_usize_opt(record: &Value, key: &str) -> Option<usize> {
        record.get(key).and_then(Value::as_u64).map(|value| value as usize)
    }

    fn json_bool_default(record: &Value, key: &str) -> bool {
        record.get(key).and_then(Value::as_bool).unwrap_or(false)
    }

    fn json_u32_vec(record: &Value, key: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let values = record
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing or invalid array field `{key}`"))?;
        values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|token| token as u32)
                    .ok_or_else(|| format!("invalid token value in `{key}`").into())
            })
            .collect()
    }

    fn python_stage_sum(record: &Value, stage_name: &str) -> f64 {
        json_f64_default(record, &format!("dense_prefill_stage_{stage_name}_ms"))
            + json_f64_default(record, &format!("dense_decode_stage_{stage_name}_ms"))
    }

    fn tokens_per_second_millis(token_count: usize, millis: f64) -> f64 {
        if millis == 0.0 {
            token_count as f64
        } else {
            token_count as f64 / (millis / 1_000.0)
        }
    }

    fn build_prompt_token_ids(
        model: &CandleCausalLm,
        prompt: &str,
        prompt_token_target: Option<usize>,
    ) -> Result<(Vec<u32>, std::time::Duration), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let mut token_ids = model.encode(prompt, true)?;
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
                let filler_ids = model.encode(&format!(" {}", prompt), false)?;
                if filler_ids.is_empty() {
                    return Err("prompt filler encoding produced no tokens".into());
                }
                while token_ids.len() < target {
                    token_ids.extend_from_slice(&filler_ids);
                }
                token_ids.truncate(target);
            }
        }
        Ok((token_ids, start.elapsed()))
    }

    #[derive(Debug)]
    struct BenchRunResult {
        generated_token_ids: Vec<u32>,
        prefill_elapsed: std::time::Duration,
        decode_elapsed: std::time::Duration,
        total_elapsed: std::time::Duration,
    }

    fn run_benchmark_pass(
        model: &mut CandleCausalLm,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<BenchRunResult, Box<dyn std::error::Error>> {
        let total_start = Instant::now();
        let prefill_start = Instant::now();
        let mut logits = model.forward_next_logits(prompt_token_ids)?;
        let prefill_elapsed = prefill_start.elapsed();

        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut decode_elapsed = std::time::Duration::ZERO;
        for _ in 0..max_new_tokens {
            let next_token = argmax(&logits).ok_or("empty decode logits")? as u32;
            generated_token_ids.push(next_token);
            if model.architecture().eos_token_ids.contains(&next_token) {
                break;
            }

            let decode_start = Instant::now();
            logits = model.forward_next_logits(&[next_token])?;
            decode_elapsed += decode_start.elapsed();
        }

        Ok(BenchRunResult {
            generated_token_ids,
            prefill_elapsed,
            decode_elapsed,
            total_elapsed: total_start.elapsed(),
        })
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let sync_stage_profile = stage_profile_sync_enabled(args.sync_stage_profile);

    if matches!(
        args.runtime_mode,
        RuntimeMode::TorchControl | RuntimeMode::MegakernelControl
    ) {
        if args.family != ModelFamily::Qwen35 {
            return Err(format!(
                "{} currently supports qwen35 only",
                args.runtime_mode
            )
            .into());
        }
        if args.runtime_mode == RuntimeMode::MegakernelControl {
            if args.attention_path.is_some() {
                return Err("--attention-path is not supported with megakernel_control".into());
            }
            if args.serving_preset.is_some()
                || args.default_key_page_mode.is_some()
                || args.default_value_page_mode.is_some()
                || args.key_layer_page_modes.is_some()
                || args.value_layer_page_modes.is_some()
            {
                return Err(
                    "page-mode controls are not supported with megakernel_control".into(),
                );
            }
        }

        let (record, backend_status, warning_message, attention_path, stage_metrics, text, generated_token_count, decode_request_count, prefill_millis, decode_millis, total_millis) =
            match args.runtime_mode {
                RuntimeMode::TorchControl => {
                    let record = dotcache_paged_runtime::torch_control::run_qwen35_text_bench(
                        &args.model_id,
                        &args.prompt,
                        args.prompt_token_target,
                        &args.device,
                        args.dtype,
                        args.warmup_runs,
                        args.max_new_tokens,
                        true,
                    )?;
                    if record.get("status").and_then(Value::as_str) == Some("error") {
                        return Err(format!(
                            "python torch_control bench failed: {}",
                            json_string(&record, "error_message")
                                .unwrap_or_else(|_| "unknown error".to_string())
                        )
                        .into());
                    }
                    let decode_steps = json_usize(&record, "decode_steps")?;
                    let generated_token_ids = json_u32_vec(&record, "dense_generated_ids")?;
                    let prefill_millis = json_f64(&record, "prefill_ms")?;
                    let decode_millis =
                        json_f64_default(&record, "dense_decode_ms_per_step") * decode_steps as f64;
                    let total_millis = prefill_millis + decode_millis;
                    let stage_metrics = RuntimeStageMetrics {
                        qkv_projection_millis: python_stage_sum(&record, "qkv_projection"),
                        kv_append_write_millis: python_stage_sum(&record, "kv_append_write"),
                        output_projection_millis: python_stage_sum(&record, "output_projection"),
                        linear_attention_millis: python_stage_sum(&record, "linear_attention"),
                        full_attention_millis: python_stage_sum(&record, "full_attention"),
                        mlp_millis: python_stage_sum(&record, "mlp"),
                        ..RuntimeStageMetrics::default()
                    };
                    let text = json_string_opt(&record, "dense_text").unwrap_or_default();
                    (
                        record,
                        None,
                        None,
                        "native_torch".to_string(),
                        stage_metrics,
                        text,
                        generated_token_ids.len(),
                        decode_steps,
                        prefill_millis,
                        decode_millis,
                        total_millis,
                    )
                }
                RuntimeMode::MegakernelControl => {
                    let control =
                        dotcache_paged_runtime::megakernel_control::run_qwen35_text_bench(
                            &dotcache_paged_runtime::megakernel_control::MegakernelControlBenchArgs {
                                model_id: &args.model_id,
                                prompt: &args.prompt,
                                out_prefix: std::path::Path::new(&args.out_prefix),
                                prompt_token_target: args.prompt_token_target,
                                device: &args.device,
                                warmup_runs: args.warmup_runs,
                                max_new_tokens: args.max_new_tokens,
                                luce_repo: args.luce_repo.as_deref(),
                            },
                        )?;
                    let record = control.record;
                    let stage_metrics = RuntimeStageMetrics {
                        qkv_projection_millis: json_f64_default(&record, "stage_qkv_projection_millis"),
                        kv_append_write_millis: json_f64_default(&record, "stage_kv_append_write_millis"),
                        layout_prepare_millis: json_f64_default(&record, "stage_layout_prepare_millis"),
                        attention_score_millis: json_f64_default(&record, "stage_attention_score_millis"),
                        attention_softmax_millis: json_f64_default(&record, "stage_attention_softmax_millis"),
                        attention_mix_millis: json_f64_default(&record, "stage_attention_mix_millis"),
                        output_projection_millis: json_f64_default(&record, "stage_output_projection_millis"),
                        full_attention_mask_prepare_millis: json_f64_default(&record, "stage_full_attention_mask_prepare_millis"),
                        full_attention_input_layout_millis: json_f64_default(&record, "stage_full_attention_input_layout_millis"),
                        full_attention_kv_materialize_millis: json_f64_default(&record, "stage_full_attention_kv_materialize_millis"),
                        full_attention_output_collect_millis: json_f64_default(&record, "stage_full_attention_output_collect_millis"),
                        full_attention_output_reshape_millis: json_f64_default(&record, "stage_full_attention_output_reshape_millis"),
                        full_attention_gate_millis: json_f64_default(&record, "stage_full_attention_gate_millis"),
                        full_attention_kernel_execute_millis: json_f64_default(&record, "stage_full_attention_kernel_execute_millis"),
                        scheduler_planning_millis: json_f64_default(&record, "stage_scheduler_planning_millis"),
                        transfer_millis: json_f64_default(&record, "stage_transfer_millis"),
                        linear_attention_millis: json_f64_default(&record, "stage_linear_attention_millis"),
                        full_attention_millis: json_f64_default(&record, "stage_full_attention_millis"),
                        mlp_millis: json_f64_default(&record, "stage_mlp_millis"),
                        ..RuntimeStageMetrics::default()
                    };
                    (
                        record.clone(),
                        Some(control.status),
                        control.warning_message,
                        control.attention_path,
                        stage_metrics,
                        json_string_opt(&record, "generated_text").unwrap_or_default(),
                        json_usize(&record, "generated_token_count")?,
                        json_usize(&record, "generated_token_count")?,
                        json_f64(&record, "prefill_millis")?,
                        json_f64(&record, "decode_millis")?,
                        json_f64(&record, "total_millis")?,
                    )
                }
                _ => unreachable!(),
            };

        let prompt_token_count = if args.runtime_mode == RuntimeMode::TorchControl {
            json_usize(&record, "prompt_length")?
        } else {
            json_usize(&record, "prompt_token_count")?
        };
        let trace_path = PathBuf::from(format!("{}.trace.jsonl", args.out_prefix));
        let summary_path = PathBuf::from(format!("{}.summary.json", args.out_prefix));
        std::fs::write(&trace_path, "")?;

        let summary = BenchmarkSummary {
            model_id: json_string_opt(&record, "model_id").unwrap_or_else(|| args.model_id.clone()),
            family: args.family.as_str().to_string(),
            device: args.device.to_string(),
            dtype: format!("{:?}", args.dtype).to_lowercase(),
            runtime_mode: args.runtime_mode.to_string(),
            attention_path,
            prompt: args.prompt.clone(),
            prompt_token_count,
            prompt_token_target: args.prompt_token_target,
            generated_token_count,
            warmup_runs: json_usize(&record, "warmup_runs")?,
            warmup_millis: json_f64(&record, "warmup_millis")?,
            max_new_tokens: args.max_new_tokens,
            tokens_per_page: args.tokens_per_page,
            resident_page_budget: None,
            resident_byte_budget: None,
            restore_cooldown_window: None,
            serving_preset: None,
            default_key_page_mode: "exact".to_string(),
            default_value_page_mode: "exact".to_string(),
            key_layer_page_mode_overrides: Vec::new(),
            value_layer_page_mode_overrides: Vec::new(),
            request_count: 1 + decode_request_count,
            prefill_request_count: 1,
            decode_request_count,
            batch_decode_request_count: 0,
            spill_count: 0,
            restore_count: 0,
            spilled_bytes: 0,
            restored_bytes: 0,
            cooldown_hit_count: 0,
            physical_page_count: 0,
            virtual_page_count: 0,
            resident_physical_page_count: 0,
            spilled_physical_page_count: 0,
            resident_physical_byte_count: 0,
            spilled_physical_byte_count: 0,
            pinned_physical_page_count: 0,
            stage_tokenization_millis: 0.0,
            stage_qkv_projection_millis: stage_metrics.qkv_projection_millis,
            stage_kv_append_write_millis: stage_metrics.kv_append_write_millis,
            stage_page_restore_millis: 0.0,
            stage_page_spill_millis: 0.0,
            stage_hybrid_cache_restore_millis: 0.0,
            stage_hybrid_cache_store_millis: 0.0,
            stage_layout_prepare_millis: 0.0,
            stage_attention_score_millis: 0.0,
            stage_attention_softmax_millis: 0.0,
            stage_attention_mix_millis: 0.0,
            stage_output_projection_millis: stage_metrics.output_projection_millis,
            stage_full_attention_mask_prepare_millis: 0.0,
            stage_full_attention_input_layout_millis: 0.0,
            stage_full_attention_kv_materialize_millis: 0.0,
            stage_full_attention_output_collect_millis: 0.0,
            stage_full_attention_output_reshape_millis: 0.0,
            stage_full_attention_gate_millis: 0.0,
            stage_full_attention_kernel_execute_millis: 0.0,
            stage_scheduler_planning_millis: 0.0,
            stage_transfer_millis: 0.0,
            stage_linear_attention_millis: stage_metrics.linear_attention_millis,
            stage_full_attention_millis: stage_metrics.full_attention_millis,
            stage_mlp_millis: stage_metrics.mlp_millis,
            stage_total_millis: stage_metrics.total_millis(),
            stage_profile_sync_enabled: sync_stage_profile,
            prefill_millis,
            decode_millis,
            total_millis,
            prefill_tokens_per_second: tokens_per_second_millis(prompt_token_count, prefill_millis),
            decode_tokens_per_second: tokens_per_second_millis(generated_token_count, decode_millis),
            total_tokens_per_second: tokens_per_second_millis(
                prompt_token_count + generated_token_count,
                total_millis,
            ),
            backend_status,
            warning_message,
            invalid_token_id: json_i64_opt(&record, "invalid_token_id"),
            invalid_token_step: json_usize_opt(&record, "invalid_token_step"),
            terminated_due_to_invalid_token: json_bool_default(
                &record,
                "terminated_due_to_invalid_token",
            ),
            text: text.clone(),
            trace_jsonl_path: trace_path.display().to_string(),
        };
        std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
        println!("{}", text);
        eprintln!("benchmark summary written to {}", summary_path.display());
        eprintln!("request trace written to {}", trace_path.display());
        eprintln!(
            "device={} runtime_mode={} attention_path={}",
            args.device,
            args.runtime_mode,
            summary.attention_path
        );
        eprintln!(
            "warmup runs={} warmup_ms={:.3}",
            summary.warmup_runs, summary.warmup_millis,
        );
        eprintln!(
            "timing prefill_ms={:.3} decode_ms={:.3} total_ms={:.3}",
            summary.prefill_millis, summary.decode_millis, summary.total_millis,
        );
        eprintln!(
            "throughput prefill_tps={:.3} decode_tps={:.3} total_tps={:.3}",
            summary.prefill_tokens_per_second,
            summary.decode_tokens_per_second,
            summary.total_tokens_per_second,
        );
        if let Some(status) = summary.backend_status.as_deref() {
            eprintln!("backend_status={status}");
        }
        if let Some(warning) = summary.warning_message.as_deref() {
            eprintln!("warning={warning}");
        }
        return Ok(());
    }

    let mut model = CandleCausalLm::from_hf_with_runtime_mode(
        &args.model_id,
        args.family,
        args.device.clone(),
        args.dtype,
        args.tokens_per_page,
        args.runtime_mode,
    )?;
    let page_mode_policy = build_page_mode_policy(
        args.default_key_page_mode.clone(),
        args.default_value_page_mode.clone(),
        args.key_layer_page_modes.as_deref(),
        args.value_layer_page_modes.as_deref(),
    )?;
    model.set_page_mode_policy(page_mode_policy);
    if let Some(attention_path) = args.attention_path {
        model.set_attention_path(attention_path);
    }
    if let Some(budget) = args.resident_page_budget {
        model.set_resident_physical_page_budget(Some(budget))?;
    }
    if let Some(budget) = args.resident_byte_budget {
        model.set_resident_physical_byte_budget(Some(budget))?;
    }
    if let Some(window) = args.restore_cooldown_window {
        model.set_restore_cooldown_window(window);
    }

    let (prompt_token_ids, tokenization_elapsed) =
        build_prompt_token_ids(&model, &args.prompt, args.prompt_token_target)?;
    if matches!(
        args.runtime_mode,
        RuntimeMode::PagedControl | RuntimeMode::DotCacheExperimental
    ) && args.resident_page_budget.is_none()
        && args.resident_byte_budget.is_none()
        && args.restore_cooldown_window.is_none()
    {
        if let Some(policy) =
            model.recommended_prompt_policy_for_token_count(prompt_token_ids.len())?
        {
            model.apply_prompt_policy(&policy)?;
        }
    }

    let warmup_start = Instant::now();
    for _ in 0..args.warmup_runs {
        model.reset()?;
        model.reset_cache_metrics();
        model.clear_request_metrics();
        let _ = run_benchmark_pass(&mut model, &prompt_token_ids, args.max_new_tokens)?;
    }
    let warmup_elapsed = warmup_start.elapsed();

    model.reset()?;
    model.reset_cache_metrics();
    model.clear_request_metrics();

    let run = run_benchmark_pass(&mut model, &prompt_token_ids, args.max_new_tokens)?;

    let mut all_token_ids = prompt_token_ids.clone();
    all_token_ids.extend_from_slice(&run.generated_token_ids);
    let text = model.decode(&all_token_ids, true)?;

    let session_id = model
        .active_session_id()
        .ok_or("active session should be available")?;
    let attention_path = model.attention_path();
    let session_metrics = model.session_metrics(session_id)?;
    let mut stage_metrics = aggregate_stage_metrics(&model);
    stage_metrics.tokenization_millis += millis(tokenization_elapsed);
    let cache = model.paged_cache();
    let (
        default_key_page_mode,
        default_value_page_mode,
        key_layer_page_mode_overrides,
        value_layer_page_mode_overrides,
    ) = describe_page_mode_policy(model.page_mode_policy());

    let trace_path = PathBuf::from(format!("{}.trace.jsonl", args.out_prefix));
    let summary_path = PathBuf::from(format!("{}.summary.json", args.out_prefix));
    model.write_request_metrics_jsonl(&trace_path)?;

    let summary = BenchmarkSummary {
        model_id: model.architecture().model_id.clone(),
        family: model.architecture().family.as_str().to_string(),
        device: args.device.to_string(),
        dtype: format!("{:?}", args.dtype).to_lowercase(),
        runtime_mode: model.runtime_mode().to_string(),
        attention_path: attention_path.to_string(),
        prompt: args.prompt.clone(),
        prompt_token_count: prompt_token_ids.len(),
        prompt_token_target: args.prompt_token_target,
        generated_token_count: run.generated_token_ids.len(),
        warmup_runs: args.warmup_runs,
        warmup_millis: millis(warmup_elapsed),
        max_new_tokens: args.max_new_tokens,
        tokens_per_page: args.tokens_per_page,
        resident_page_budget: model.resident_physical_page_budget(),
        resident_byte_budget: model.resident_physical_byte_budget(),
        restore_cooldown_window: args
            .restore_cooldown_window
            .or_else(|| model.restore_cooldown_window()),
        serving_preset: args.serving_preset.clone(),
        default_key_page_mode,
        default_value_page_mode,
        key_layer_page_mode_overrides,
        value_layer_page_mode_overrides,
        request_count: session_metrics.request_count,
        prefill_request_count: session_metrics.prefill_request_count,
        decode_request_count: session_metrics.decode_request_count,
        batch_decode_request_count: session_metrics.batch_decode_request_count,
        spill_count: session_metrics.spill_count,
        restore_count: session_metrics.restore_count,
        spilled_bytes: session_metrics.spilled_bytes,
        restored_bytes: session_metrics.restored_bytes,
        cooldown_hit_count: session_metrics.cooldown_hit_count,
        physical_page_count: cache.map(|cache| cache.physical_page_count()).unwrap_or(0),
        virtual_page_count: cache.map(|cache| cache.virtual_page_count()).unwrap_or(0),
        resident_physical_page_count: cache
            .map(|cache| cache.resident_physical_page_count())
            .unwrap_or(0),
        spilled_physical_page_count: cache
            .map(|cache| cache.spilled_physical_page_count())
            .unwrap_or(0),
        resident_physical_byte_count: cache
            .map(|cache| cache.resident_physical_byte_count())
            .unwrap_or(0),
        spilled_physical_byte_count: cache
            .map(|cache| cache.spilled_physical_byte_count())
            .unwrap_or(0),
        pinned_physical_page_count: cache
            .map(|cache| cache.pinned_physical_page_count())
            .unwrap_or(0),
        stage_tokenization_millis: stage_metrics.tokenization_millis,
        stage_qkv_projection_millis: stage_metrics.qkv_projection_millis,
        stage_kv_append_write_millis: stage_metrics.kv_append_write_millis,
        stage_page_restore_millis: stage_metrics.page_restore_millis,
        stage_page_spill_millis: stage_metrics.page_spill_millis,
        stage_hybrid_cache_restore_millis: stage_metrics.hybrid_cache_restore_millis,
        stage_hybrid_cache_store_millis: stage_metrics.hybrid_cache_store_millis,
        stage_layout_prepare_millis: stage_metrics.layout_prepare_millis,
        stage_attention_score_millis: stage_metrics.attention_score_millis,
        stage_attention_softmax_millis: stage_metrics.attention_softmax_millis,
        stage_attention_mix_millis: stage_metrics.attention_mix_millis,
        stage_output_projection_millis: stage_metrics.output_projection_millis,
        stage_full_attention_mask_prepare_millis: stage_metrics.full_attention_mask_prepare_millis,
        stage_full_attention_input_layout_millis: stage_metrics.full_attention_input_layout_millis,
        stage_full_attention_kv_materialize_millis: stage_metrics
            .full_attention_kv_materialize_millis,
        stage_full_attention_output_collect_millis: stage_metrics
            .full_attention_output_collect_millis,
        stage_full_attention_output_reshape_millis: stage_metrics
            .full_attention_output_reshape_millis,
        stage_full_attention_gate_millis: stage_metrics.full_attention_gate_millis,
        stage_full_attention_kernel_execute_millis: stage_metrics
            .full_attention_kernel_execute_millis,
        stage_scheduler_planning_millis: stage_metrics.scheduler_planning_millis,
        stage_transfer_millis: stage_metrics.transfer_millis,
        stage_linear_attention_millis: stage_metrics.linear_attention_millis,
        stage_full_attention_millis: stage_metrics.full_attention_millis,
        stage_mlp_millis: stage_metrics.mlp_millis,
        stage_total_millis: stage_metrics.total_millis(),
        stage_profile_sync_enabled: sync_stage_profile,
        prefill_millis: millis(run.prefill_elapsed),
        decode_millis: millis(run.decode_elapsed),
        total_millis: millis(run.total_elapsed),
        prefill_tokens_per_second: tokens_per_second(prompt_token_ids.len(), run.prefill_elapsed),
        decode_tokens_per_second: tokens_per_second(
            run.generated_token_ids.len(),
            run.decode_elapsed,
        ),
        total_tokens_per_second: tokens_per_second(
            prompt_token_ids.len() + run.generated_token_ids.len(),
            run.total_elapsed,
        ),
        backend_status: None,
        warning_message: None,
        invalid_token_id: None,
        invalid_token_step: None,
        terminated_due_to_invalid_token: false,
        text: text.clone(),
        trace_jsonl_path: trace_path.display().to_string(),
    };
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("{}", text);
    eprintln!("benchmark summary written to {}", summary_path.display());
    eprintln!("request trace written to {}", trace_path.display());
    eprintln!(
        "device={} runtime_mode={} attention_path={}",
        args.device,
        model.runtime_mode(),
        attention_path
    );
    eprintln!(
        "warmup runs={} warmup_ms={:.3}",
        summary.warmup_runs, summary.warmup_millis,
    );
    eprintln!(
        "timing prefill_ms={:.3} decode_ms={:.3} total_ms={:.3}",
        summary.prefill_millis, summary.decode_millis, summary.total_millis,
    );
    eprintln!(
        "throughput prefill_tps={:.3} decode_tps={:.3} total_tps={:.3}",
        summary.prefill_tokens_per_second,
        summary.decode_tokens_per_second,
        summary.total_tokens_per_second,
    );

    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
