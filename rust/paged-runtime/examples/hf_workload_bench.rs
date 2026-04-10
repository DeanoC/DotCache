#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use candle_core::DType;
    use dotcache_paged_runtime::{
        AttentionPathMode, BackendDevice, CandleCausalLm, CandleDeviceSelector, CausalLm,
        ModelFamily, RuntimeMode, RuntimeStageMetrics, SessionId, SessionRequestKind,
    };
    use serde::Serialize;
    use serde_json::Value;

    #[derive(Debug)]
    struct WorkloadArgs {
        family: ModelFamily,
        model_id: String,
        shared_prompt: String,
        out_prefix: String,
        shared_prompt_token_target: Option<usize>,
        device: CandleDeviceSelector,
        dtype: DType,
        runtime_mode: RuntimeMode,
        attention_path: Option<AttentionPathMode>,
        warmup_runs: usize,
        total_sessions: usize,
        wave_size: usize,
        decode_rounds_per_wave: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        suffix_prefix: String,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        sync_stage_profile: bool,
    }

    #[derive(Debug)]
    struct SessionWorkload {
        logical_index: usize,
        session_id: SessionId,
        arrival_wave: usize,
        attached_from_prefix: bool,
        suffix_text: String,
        suffix_token_ids: Vec<u32>,
        target_decode_tokens: usize,
        generated_token_ids: Vec<u32>,
        completed_by_eos: bool,
    }

    #[derive(Debug, Serialize)]
    struct SessionSummary {
        logical_index: usize,
        session_id: SessionId,
        arrival_wave: usize,
        attached_from_prefix: bool,
        suffix_text: String,
        suffix_token_count: usize,
        target_decode_tokens: usize,
        generated_token_count: usize,
        completed_by_eos: bool,
        text: String,
    }

    #[derive(Debug, Serialize)]
    struct WorkloadSummary {
        model_id: String,
        family: String,
        device: String,
        dtype: String,
        runtime_mode: String,
        attention_path: String,
        shared_prompt: String,
        shared_prompt_token_count: usize,
        shared_prompt_token_target: Option<usize>,
        warmup_runs: usize,
        warmup_millis: f64,
        total_sessions: usize,
        attached_session_count: usize,
        wave_size: usize,
        decode_rounds_per_wave: usize,
        peak_active_sessions: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        suffix_prefix: String,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        request_count: usize,
        prefill_request_count: usize,
        decode_request_count: usize,
        batch_decode_request_count: usize,
        total_input_token_count: usize,
        total_request_suffix_token_count: usize,
        total_generated_token_count: usize,
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
        cold_prefix_prefill_millis: f64,
        prefix_capture_millis: f64,
        seed_suffix_prefill_millis: f64,
        attached_suffix_prefill_millis: f64,
        attach_millis: f64,
        decode_millis: f64,
        close_millis: f64,
        release_prefix_millis: f64,
        total_millis: f64,
        cold_prefix_tokens_per_second: f64,
        request_prefill_tokens_per_second: f64,
        decode_tokens_per_second: f64,
        total_tokens_per_second: f64,
        sessions: Vec<SessionSummary>,
        trace_jsonl_path: String,
    }

    fn parse_args() -> Result<WorkloadArgs, String> {
        let mut args = std::env::args().skip(1);
        let family = args.next().ok_or_else(|| {
            "usage: hf_workload_bench <family> <model_id> <shared_prompt> <out_prefix> [--shared-prompt-token-target N] [--device cpu|metal[:ordinal]|cuda[:ordinal]] [--dtype f16|bf16|f32] [--runtime-mode dense_control|paged_control|dotcache_experimental|torch_control] [--attention-path paged|fused] [--warmup-runs N] [--total-sessions N] [--wave-size N] [--decode-rounds-per-wave N] [--max-new-tokens N] [--tokens-per-page N] [--suffix-prefix TEXT] [--stress] [--stress-suffix-repeats N] [--resident-page-budget N] [--resident-byte-budget N] [--restore-cooldown N] [--sync-stage-profile]".to_string()
        })?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let shared_prompt = args
            .next()
            .ok_or_else(|| "missing shared_prompt".to_string())?;
        let out_prefix = args
            .next()
            .ok_or_else(|| "missing out_prefix".to_string())?;

        let mut parsed = WorkloadArgs {
            family: family.parse().map_err(|err| format!("{err}"))?,
            model_id,
            shared_prompt,
            out_prefix,
            shared_prompt_token_target: None,
            device: CandleDeviceSelector::Cpu,
            dtype: DType::F32,
            runtime_mode: RuntimeMode::PagedControl,
            attention_path: None,
            warmup_runs: 1,
            total_sessions: 4,
            wave_size: 2,
            decode_rounds_per_wave: 1,
            max_new_tokens: 4,
            tokens_per_page: CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
            suffix_prefix: "session".to_string(),
            stress_mode: false,
            stress_suffix_repeats: 6,
            resident_page_budget: None,
            resident_byte_budget: None,
            restore_cooldown_window: None,
            sync_stage_profile: false,
        };

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--sync-stage-profile" => {
                    parsed.sync_stage_profile = true;
                }
                "--stress" => {
                    parsed.stress_mode = true;
                }
                "--warmup-runs" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.warmup_runs = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --warmup-runs: {err}"))?;
                }
                "--total-sessions" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.total_sessions = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --total-sessions: {err}"))?;
                }
                "--wave-size" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.wave_size = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --wave-size: {err}"))?;
                }
                "--decode-rounds-per-wave" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.decode_rounds_per_wave = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --decode-rounds-per-wave: {err}"))?;
                }
                "--max-new-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.max_new_tokens = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --max-new-tokens: {err}"))?;
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
                "--shared-prompt-token-target" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.shared_prompt_token_target =
                        Some(value.parse::<usize>().map_err(|err| {
                            format!("invalid --shared-prompt-token-target: {err}")
                        })?);
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
                "--tokens-per-page" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.tokens_per_page = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --tokens-per-page: {err}"))?;
                }
                "--suffix-prefix" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.suffix_prefix = value;
                }
                "--stress-suffix-repeats" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.stress_suffix_repeats = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --stress-suffix-repeats: {err}"))?;
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
                other => return Err(format!("unknown flag {other}")),
            }
        }

        if parsed.total_sessions == 0 {
            return Err("--total-sessions must be at least 1".to_string());
        }
        if parsed.wave_size == 0 {
            return Err("--wave-size must be at least 1".to_string());
        }
        if parsed.decode_rounds_per_wave == 0 {
            return Err("--decode-rounds-per-wave must be at least 1".to_string());
        }
        if parsed.max_new_tokens == 0 {
            return Err("--max-new-tokens must be at least 1".to_string());
        }
        if parsed.suffix_prefix.is_empty() {
            return Err("--suffix-prefix must not be empty".to_string());
        }
        if parsed.stress_suffix_repeats == 0 {
            return Err("--stress-suffix-repeats must be at least 1".to_string());
        }
        if !std::env::args().any(|arg| arg == "--dtype") {
            parsed.dtype = match parsed.device.backend_device() {
                BackendDevice::Metal { .. } | BackendDevice::Cuda { .. } => DType::F16,
                BackendDevice::Cpu => DType::F32,
            };
        }
        Ok(parsed)
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

    fn argmax(values: &[f32]) -> Option<usize> {
        values
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index)
    }

    fn millis(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1_000.0
    }

    fn tokens_per_second(token_count: usize, duration: Duration) -> f64 {
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

    fn tokens_per_second_millis(token_count: usize, millis: f64) -> f64 {
        if millis == 0.0 {
            token_count as f64
        } else {
            token_count as f64 / (millis / 1_000.0)
        }
    }

    fn python_stage_sum(record: &Value, stage_name: &str) -> f64 {
        record
            .get(&format!("dense_prefill_stage_{stage_name}_ms"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            + record
                .get(&format!("dense_decode_stage_{stage_name}_ms"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
    }

    fn build_prompt_token_ids(
        model: &CandleCausalLm,
        prompt: &str,
        prompt_token_target: Option<usize>,
    ) -> Result<(Vec<u32>, Duration), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let mut token_ids = model.encode(prompt, true)?;
        if token_ids.is_empty() {
            return Err("shared prompt encoding produced no tokens".into());
        }
        if let Some(target) = prompt_token_target {
            if target == 0 {
                return Err("--shared-prompt-token-target must be at least 1".into());
            }
            if token_ids.len() > target {
                token_ids.truncate(target);
            } else if token_ids.len() < target {
                let filler_ids = model.encode(&format!(" {}", prompt), false)?;
                if filler_ids.is_empty() {
                    return Err("shared prompt filler encoding produced no tokens".into());
                }
                while token_ids.len() < target {
                    token_ids.extend_from_slice(&filler_ids);
                }
                token_ids.truncate(target);
            }
        }
        Ok((token_ids, start.elapsed()))
    }

    fn suffix_text(
        prefix: &str,
        logical_index: usize,
        stress_mode: bool,
        stress_suffix_repeats: usize,
    ) -> String {
        if !stress_mode {
            return format!(" {}-{}", prefix, logical_index);
        }

        let mut text = String::new();
        for repeat in 0..stress_suffix_repeats {
            text.push_str(&format!(
                " {}-{}-segment-{} detail-{} load-{}",
                prefix, logical_index, repeat, logical_index, repeat
            ));
        }
        text
    }

    fn target_decode_tokens(logical_index: usize, max_new_tokens: usize) -> usize {
        if max_new_tokens == 1 {
            return 1;
        }
        let spread = std::cmp::min(max_new_tokens - 1, 2);
        max_new_tokens - (logical_index % (spread + 1))
    }

    fn prefill_session_chunked(
        model: &mut CandleCausalLm,
        session_id: SessionId,
        input_ids: &[u32],
        chunk_size: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut logits = Vec::new();
        for chunk in input_ids.chunks(chunk_size.max(1)) {
            logits = model.prefill_session(session_id, chunk)?;
        }
        Ok(logits)
    }

    fn prefill_session_suffix(
        model: &mut CandleCausalLm,
        session_id: SessionId,
        suffix_token_ids: &[u32],
        stress_mode: bool,
        tokens_per_page: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if stress_mode {
            prefill_session_chunked(
                model,
                session_id,
                suffix_token_ids,
                tokens_per_page.saturating_sub(1),
            )
        } else {
            Ok(model.prefill_session(session_id, suffix_token_ids)?)
        }
    }

    fn warmup_workload_pass(
        model: &mut CandleCausalLm,
        shared_prompt_token_ids: &[u32],
        suffix_prefix: &str,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        tokens_per_page: usize,
        warmup_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        model.reset()?;
        model.reset_cache_metrics();
        model.clear_request_metrics();

        let suffix_a = suffix_text(
            suffix_prefix,
            warmup_index * 2,
            stress_mode,
            stress_suffix_repeats,
        );
        let suffix_b = suffix_text(
            suffix_prefix,
            warmup_index * 2 + 1,
            stress_mode,
            stress_suffix_repeats,
        );
        let suffix_a_ids = model.encode(&suffix_a, false)?;
        let suffix_b_ids = model.encode(&suffix_b, false)?;
        if suffix_a_ids.is_empty() || suffix_b_ids.is_empty() {
            return Err("warmup suffix encoding produced no tokens".into());
        }

        let session_a;
        let session_b;
        let paged_prefix = if model.runtime_mode() == RuntimeMode::DenseControl {
            let active_session_id = model
                .active_session_id()
                .ok_or("active session should be available during warmup")?;
            session_a = active_session_id;
            let _ = model.prefill_session(session_a, shared_prompt_token_ids)?;
            session_b = model.fork_session(session_a)?;
            let _ = prefill_session_suffix(
                model,
                session_a,
                &suffix_a_ids,
                stress_mode,
                tokens_per_page,
            )?;
            let _ = prefill_session_suffix(
                model,
                session_b,
                &suffix_b_ids,
                stress_mode,
                tokens_per_page,
            )?;
            None
        } else {
            let _ = model.forward_next_logits(shared_prompt_token_ids)?;
            let seed_session_id = model
                .active_session_id()
                .ok_or("active session should be available during warmup")?;
            let prefix = model.capture_prefix(seed_session_id)?;
            model.close_session(seed_session_id)?;

            session_a = model.attach_prefix(&prefix)?;
            session_b = model.attach_prefix(&prefix)?;
            if stress_mode {
                let _ = prefill_session_chunked(
                    model,
                    session_a,
                    &suffix_a_ids,
                    tokens_per_page.saturating_sub(1),
                )?;
                let _ = prefill_session_chunked(
                    model,
                    session_b,
                    &suffix_b_ids,
                    tokens_per_page.saturating_sub(1),
                )?;
            } else {
                let _ = model.prefill_sessions_batch(&[
                    (session_a, suffix_a_ids.as_slice()),
                    (session_b, suffix_b_ids.as_slice()),
                ])?;
            }
            Some(prefix)
        };

        let decode_inputs = [
            (
                session_a,
                *suffix_a_ids.last().unwrap_or(&shared_prompt_token_ids[0]),
            ),
            (
                session_b,
                *suffix_b_ids.last().unwrap_or(&shared_prompt_token_ids[0]),
            ),
        ];
        let _ = model.forward_next_logits_batch(&decode_inputs)?;

        model.close_session(session_a)?;
        model.close_session(session_b)?;
        if let Some(prefix) = paged_prefix.as_ref() {
            model.release_prefix(prefix)?;
        }
        Ok(())
    }

    fn run_decode_round(
        model: &mut CandleCausalLm,
        active_session_ids: &mut Vec<SessionId>,
        workloads: &mut HashMap<SessionId, SessionWorkload>,
        logits_by_session: &mut HashMap<SessionId, Vec<f32>>,
        decode_elapsed: &mut Duration,
        close_elapsed: &mut Duration,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if active_session_ids.is_empty() {
            return Ok(0);
        }

        let mut continuing_session_ids = Vec::with_capacity(active_session_ids.len());
        let mut decode_requests = Vec::with_capacity(active_session_ids.len());
        let mut completed_session_ids = Vec::new();

        for &session_id in active_session_ids.iter() {
            let logits = logits_by_session
                .get(&session_id)
                .ok_or("missing logits for active session")?;
            let next_token = argmax(logits).ok_or("empty decode logits")? as u32;
            let workload = workloads
                .get_mut(&session_id)
                .ok_or("missing workload for active session")?;
            workload.generated_token_ids.push(next_token);
            let hit_eos = model.architecture().eos_token_ids.contains(&next_token);
            let hit_limit = workload.generated_token_ids.len() >= workload.target_decode_tokens;
            if hit_eos || hit_limit {
                workload.completed_by_eos = hit_eos;
                completed_session_ids.push(session_id);
            } else {
                decode_requests.push((session_id, next_token));
                continuing_session_ids.push(session_id);
            }
        }

        if !decode_requests.is_empty() {
            let decode_start = Instant::now();
            let batch_logits = model.forward_next_logits_batch(&decode_requests)?;
            *decode_elapsed += decode_start.elapsed();
            *logits_by_session = batch_logits.into_iter().collect();
        } else {
            logits_by_session.clear();
        }

        if !completed_session_ids.is_empty() {
            let close_start = Instant::now();
            for session_id in &completed_session_ids {
                logits_by_session.remove(session_id);
                model.close_session(*session_id)?;
            }
            *close_elapsed += close_start.elapsed();
        }

        *active_session_ids = continuing_session_ids;
        Ok(decode_requests.len())
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let sync_stage_profile = stage_profile_sync_enabled(args.sync_stage_profile);

    if args.runtime_mode == RuntimeMode::TorchControl {
        if args.family != ModelFamily::Qwen35 {
            return Err("torch_control currently supports qwen35 only".into());
        }
        let record = dotcache_paged_runtime::torch_control::run_qwen35_text_workload(
            &dotcache_paged_runtime::torch_control::TorchControlWorkloadArgs {
                model_id: &args.model_id,
                shared_prompt: &args.shared_prompt,
                shared_prompt_token_target: args.shared_prompt_token_target,
                device: &args.device,
                dtype: args.dtype,
                warmup_runs: args.warmup_runs,
                total_sessions: args.total_sessions,
                wave_size: args.wave_size,
                decode_rounds_per_wave: args.decode_rounds_per_wave,
                max_new_tokens: args.max_new_tokens,
                suffix_prefix: &args.suffix_prefix,
                stress_mode: args.stress_mode,
                stress_suffix_repeats: args.stress_suffix_repeats,
                profile_stages: true,
            },
        )?;
        if record.get("status").and_then(Value::as_str) == Some("error") {
            return Err(format!(
                "python torch_control workload failed: {}",
                json_string(&record, "error_message")
                    .unwrap_or_else(|_| "unknown error".to_string())
            )
            .into());
        }

        let trace_path = PathBuf::from(format!("{}.trace.jsonl", args.out_prefix));
        let summary_path = PathBuf::from(format!("{}.summary.json", args.out_prefix));
        std::fs::write(&trace_path, "")?;

        let session_values = record
            .get("sessions")
            .and_then(Value::as_array)
            .ok_or("missing or invalid sessions array")?;
        let sessions = session_values
            .iter()
            .map(|session| {
                let logical_index = session
                    .get("logical_index")
                    .and_then(Value::as_u64)
                    .ok_or("missing logical_index")? as usize;
                Ok(SessionSummary {
                    logical_index,
                    session_id: logical_index,
                    arrival_wave: session
                        .get("arrival_wave")
                        .and_then(Value::as_u64)
                        .ok_or("missing arrival_wave")? as usize,
                    attached_from_prefix: logical_index != 0,
                    suffix_text: suffix_text(
                        &args.suffix_prefix,
                        logical_index,
                        args.stress_mode,
                        args.stress_suffix_repeats,
                    ),
                    suffix_token_count: session
                        .get("suffix_token_count")
                        .and_then(Value::as_u64)
                        .ok_or("missing suffix_token_count")?
                        as usize,
                    target_decode_tokens: session
                        .get("target_decode_tokens")
                        .and_then(Value::as_u64)
                        .ok_or("missing target_decode_tokens")?
                        as usize,
                    generated_token_count: session
                        .get("generated_token_count")
                        .and_then(Value::as_u64)
                        .ok_or("missing generated_token_count")?
                        as usize,
                    completed_by_eos: session
                        .get("completed_by_eos")
                        .and_then(Value::as_bool)
                        .ok_or("missing completed_by_eos")?,
                    text: String::new(),
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

        let total_request_suffix_token_count = sessions
            .iter()
            .map(|session| session.suffix_token_count)
            .sum();
        let total_generated_token_count = json_usize(&record, "total_generated_token_count")?;
        let shared_prompt_token_count = json_usize(&record, "shared_prompt_token_count")?;
        let cold_prefix_prefill_millis = json_f64(&record, "cold_prefix_prefill_ms")?;
        let seed_suffix_prefill_millis = json_f64(&record, "seed_suffix_prefill_ms")?;
        let attached_suffix_prefill_millis = json_f64(&record, "attached_suffix_prefill_ms")?;
        let decode_millis = json_f64(&record, "decode_ms")?;
        let total_millis = json_f64(&record, "total_ms")?;
        let stage_qkv_projection_millis = python_stage_sum(&record, "qkv_projection");
        let stage_kv_append_write_millis = python_stage_sum(&record, "kv_append_write");
        let stage_output_projection_millis = python_stage_sum(&record, "output_projection");
        let stage_linear_attention_millis = python_stage_sum(&record, "linear_attention");
        let stage_full_attention_millis = python_stage_sum(&record, "full_attention");
        let stage_mlp_millis = python_stage_sum(&record, "mlp");
        let stage_total_millis = stage_qkv_projection_millis
            + stage_kv_append_write_millis
            + stage_output_projection_millis
            + stage_linear_attention_millis
            + stage_full_attention_millis
            + stage_mlp_millis;

        let summary = WorkloadSummary {
            model_id: json_string(&record, "model_id")?,
            family: args.family.as_str().to_string(),
            device: args.device.to_string(),
            dtype: format!("{:?}", args.dtype).to_lowercase(),
            runtime_mode: RuntimeMode::TorchControl.to_string(),
            attention_path: "native_torch".to_string(),
            shared_prompt: args.shared_prompt.clone(),
            shared_prompt_token_count,
            shared_prompt_token_target: args.shared_prompt_token_target,
            warmup_runs: json_usize(&record, "warmup_runs")?,
            warmup_millis: json_f64(&record, "warmup_ms")?,
            total_sessions: json_usize(&record, "total_sessions")?,
            attached_session_count: sessions
                .iter()
                .filter(|session| session.attached_from_prefix)
                .count(),
            wave_size: json_usize(&record, "wave_size")?,
            decode_rounds_per_wave: json_usize(&record, "decode_rounds_per_wave")?,
            peak_active_sessions: json_usize(&record, "peak_active_sessions")?,
            max_new_tokens: args.max_new_tokens,
            tokens_per_page: args.tokens_per_page,
            suffix_prefix: args.suffix_prefix.clone(),
            stress_mode: args.stress_mode,
            stress_suffix_repeats: args.stress_suffix_repeats,
            resident_page_budget: None,
            resident_byte_budget: None,
            restore_cooldown_window: None,
            request_count: 0,
            prefill_request_count: 0,
            decode_request_count: 0,
            batch_decode_request_count: 0,
            total_input_token_count: json_usize(&record, "total_input_token_count")?,
            total_request_suffix_token_count,
            total_generated_token_count,
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
            stage_qkv_projection_millis,
            stage_kv_append_write_millis,
            stage_layout_prepare_millis: 0.0,
            stage_attention_score_millis: 0.0,
            stage_attention_softmax_millis: 0.0,
            stage_attention_mix_millis: 0.0,
            stage_output_projection_millis,
            stage_full_attention_mask_prepare_millis: 0.0,
            stage_full_attention_input_layout_millis: 0.0,
            stage_full_attention_kv_materialize_millis: 0.0,
            stage_full_attention_output_collect_millis: 0.0,
            stage_full_attention_output_reshape_millis: 0.0,
            stage_full_attention_gate_millis: 0.0,
            stage_full_attention_kernel_execute_millis: 0.0,
            stage_scheduler_planning_millis: 0.0,
            stage_transfer_millis: 0.0,
            stage_linear_attention_millis,
            stage_full_attention_millis,
            stage_mlp_millis,
            stage_total_millis,
            stage_profile_sync_enabled: sync_stage_profile,
            cold_prefix_prefill_millis,
            prefix_capture_millis: 0.0,
            seed_suffix_prefill_millis,
            attached_suffix_prefill_millis,
            attach_millis: 0.0,
            decode_millis,
            close_millis: 0.0,
            release_prefix_millis: 0.0,
            total_millis,
            cold_prefix_tokens_per_second: tokens_per_second_millis(
                shared_prompt_token_count,
                cold_prefix_prefill_millis,
            ),
            request_prefill_tokens_per_second: tokens_per_second_millis(
                total_request_suffix_token_count,
                seed_suffix_prefill_millis + attached_suffix_prefill_millis,
            ),
            decode_tokens_per_second: tokens_per_second_millis(
                total_generated_token_count,
                decode_millis,
            ),
            total_tokens_per_second: json_f64(&record, "total_tokens_per_second")?,
            sessions,
            trace_jsonl_path: trace_path.display().to_string(),
        };
        std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
        println!(
            "workload: device={} runtime_mode={} attention_path={} warmup_runs={} warmup_ms={:.3} sessions={} peak_active={} total_ms={:.3} total_tps={:.3} spills={} restores={} cooldown_hits={}",
            args.device,
            RuntimeMode::TorchControl,
            "native_torch",
            summary.warmup_runs,
            summary.warmup_millis,
            summary.total_sessions,
            summary.peak_active_sessions,
            summary.total_millis,
            summary.total_tokens_per_second,
            summary.spill_count,
            summary.restore_count,
            summary.cooldown_hit_count,
        );
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

    let (shared_prompt_token_ids, mut tokenization_elapsed) =
        build_prompt_token_ids(&model, &args.shared_prompt, args.shared_prompt_token_target)?;

    let warmup_start = Instant::now();
    for warmup_index in 0..args.warmup_runs {
        warmup_workload_pass(
            &mut model,
            &shared_prompt_token_ids,
            &args.suffix_prefix,
            args.stress_mode,
            args.stress_suffix_repeats,
            args.tokens_per_page,
            warmup_index,
        )?;
    }
    let warmup_elapsed = warmup_start.elapsed();

    model.reset()?;
    model.reset_cache_metrics();
    model.clear_request_metrics();

    let total_start = Instant::now();

    let cold_prefix_prefill_start = Instant::now();
    let _ = model.forward_next_logits(&shared_prompt_token_ids)?;
    let cold_prefix_prefill_elapsed = cold_prefix_prefill_start.elapsed();

    let seed_session_id = model
        .active_session_id()
        .ok_or("active session should be available")?;

    let prefix_capture_start = Instant::now();
    let prefix = if model.runtime_mode() == RuntimeMode::DenseControl {
        None
    } else {
        Some(model.capture_prefix(seed_session_id)?)
    };
    let prefix_capture_elapsed = prefix_capture_start.elapsed();

    let mut workloads = HashMap::new();
    let mut logical_session_ids = Vec::with_capacity(args.total_sessions);
    let mut active_session_ids = Vec::new();
    let mut peak_active_sessions = 0usize;
    let mut close_elapsed = Duration::ZERO;

    let seed_suffix_text = suffix_text(
        &args.suffix_prefix,
        0,
        args.stress_mode,
        args.stress_suffix_repeats,
    );
    let encode_start = Instant::now();
    let seed_suffix_token_ids = model.encode(&seed_suffix_text, false)?;
    tokenization_elapsed += encode_start.elapsed();
    if seed_suffix_token_ids.is_empty() {
        return Err("seed suffix encoding produced no tokens".into());
    }
    let seed_suffix_prefill_start = Instant::now();
    let seed_session_id = if model.runtime_mode() == RuntimeMode::DenseControl {
        seed_session_id
    } else {
        let seed_close_start = Instant::now();
        model.close_session(seed_session_id)?;
        close_elapsed += seed_close_start.elapsed();
        model.attach_prefix(prefix.as_ref().ok_or("paged prefix should exist")?)?
    };
    let seed_logits = if model.runtime_mode() == RuntimeMode::DenseControl {
        prefill_session_suffix(
            &mut model,
            seed_session_id,
            &seed_suffix_token_ids,
            args.stress_mode,
            args.tokens_per_page,
        )?
    } else if args.stress_mode {
        prefill_session_chunked(
            &mut model,
            seed_session_id,
            &seed_suffix_token_ids,
            args.tokens_per_page.saturating_sub(1),
        )?
    } else {
        model.prefill_session(seed_session_id, &seed_suffix_token_ids)?
    };
    let seed_suffix_prefill_elapsed = seed_suffix_prefill_start.elapsed();

    workloads.insert(
        seed_session_id,
        SessionWorkload {
            logical_index: 0,
            session_id: seed_session_id,
            arrival_wave: 0,
            attached_from_prefix: model.runtime_mode() != RuntimeMode::DenseControl,
            suffix_text: seed_suffix_text,
            suffix_token_ids: seed_suffix_token_ids,
            target_decode_tokens: target_decode_tokens(0, args.max_new_tokens),
            generated_token_ids: Vec::with_capacity(args.max_new_tokens),
            completed_by_eos: false,
        },
    );
    logical_session_ids.push(seed_session_id);
    active_session_ids.push(seed_session_id);
    peak_active_sessions = peak_active_sessions.max(active_session_ids.len());

    let mut logits_by_session = HashMap::new();
    logits_by_session.insert(seed_session_id, seed_logits);

    let mut attach_elapsed = Duration::ZERO;
    let mut attached_suffix_prefill_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;
    let mut decode_input_token_count = 0usize;

    if args.total_sessions > 1 {
        decode_input_token_count += run_decode_round(
            &mut model,
            &mut active_session_ids,
            &mut workloads,
            &mut logits_by_session,
            &mut decode_elapsed,
            &mut close_elapsed,
        )?;
    }

    let mut pending_indices = (1..args.total_sessions).collect::<VecDeque<_>>();
    let mut wave_index = 1usize;
    while !pending_indices.is_empty() {
        let mut arrivals = Vec::new();
        let attach_start = Instant::now();
        for _ in 0..args.wave_size {
            let Some(logical_index) = pending_indices.pop_front() else {
                break;
            };
            let session_id = if model.runtime_mode() == RuntimeMode::DenseControl {
                model.fork_session(seed_session_id)?
            } else {
                model.attach_prefix(prefix.as_ref().ok_or("paged prefix should exist")?)?
            };
            logical_session_ids.push(session_id);
            arrivals.push((logical_index, session_id));
        }
        attach_elapsed += attach_start.elapsed();

        if !arrivals.is_empty() {
            let mut suffix_requests = Vec::with_capacity(arrivals.len());
            for &(logical_index, session_id) in &arrivals {
                let text = suffix_text(
                    &args.suffix_prefix,
                    logical_index,
                    args.stress_mode,
                    args.stress_suffix_repeats,
                );
                let encode_start = Instant::now();
                let token_ids = model.encode(&text, false)?;
                tokenization_elapsed += encode_start.elapsed();
                if token_ids.is_empty() {
                    return Err(format!(
                        "suffix encoding produced no tokens for logical session {logical_index}"
                    )
                    .into());
                }
                workloads.insert(
                    session_id,
                    SessionWorkload {
                        logical_index,
                        session_id,
                        arrival_wave: wave_index,
                        attached_from_prefix: true,
                        suffix_text: text,
                        suffix_token_ids: token_ids.clone(),
                        target_decode_tokens: target_decode_tokens(
                            logical_index,
                            args.max_new_tokens,
                        ),
                        generated_token_ids: Vec::with_capacity(args.max_new_tokens),
                        completed_by_eos: false,
                    },
                );
                suffix_requests.push((session_id, token_ids));
            }

            let suffix_prefill_start = Instant::now();
            if model.runtime_mode() == RuntimeMode::DenseControl {
                for (session_id, token_ids) in suffix_requests {
                    let logits = prefill_session_suffix(
                        &mut model,
                        session_id,
                        &token_ids,
                        args.stress_mode,
                        args.tokens_per_page,
                    )?;
                    logits_by_session.insert(session_id, logits);
                    active_session_ids.push(session_id);
                }
            } else if args.stress_mode {
                for (session_id, token_ids) in suffix_requests {
                    let logits = prefill_session_chunked(
                        &mut model,
                        session_id,
                        &token_ids,
                        args.tokens_per_page.saturating_sub(1),
                    )?;
                    logits_by_session.insert(session_id, logits);
                    active_session_ids.push(session_id);
                }
            } else {
                let batched_requests = suffix_requests
                    .iter()
                    .map(|(session_id, token_ids)| (*session_id, token_ids.as_slice()))
                    .collect::<Vec<_>>();
                let batch_logits = model.prefill_sessions_batch(&batched_requests)?;
                for (session_id, logits) in batch_logits {
                    logits_by_session.insert(session_id, logits);
                    active_session_ids.push(session_id);
                }
            }
            attached_suffix_prefill_elapsed += suffix_prefill_start.elapsed();
            peak_active_sessions = peak_active_sessions.max(active_session_ids.len());
        }

        for _ in 0..args.decode_rounds_per_wave {
            decode_input_token_count += run_decode_round(
                &mut model,
                &mut active_session_ids,
                &mut workloads,
                &mut logits_by_session,
                &mut decode_elapsed,
                &mut close_elapsed,
            )?;
            if active_session_ids.is_empty() {
                break;
            }
        }

        wave_index += 1;
    }

    while !active_session_ids.is_empty() {
        decode_input_token_count += run_decode_round(
            &mut model,
            &mut active_session_ids,
            &mut workloads,
            &mut logits_by_session,
            &mut decode_elapsed,
            &mut close_elapsed,
        )?;
    }

    let release_prefix_start = Instant::now();
    if let Some(prefix) = prefix.as_ref() {
        model.release_prefix(prefix)?;
    }
    let release_prefix_elapsed = release_prefix_start.elapsed();
    let total_elapsed = total_start.elapsed();

    let mut ordered_workloads = logical_session_ids
        .into_iter()
        .map(|session_id| workloads.remove(&session_id).ok_or("missing workload"))
        .collect::<Result<Vec<_>, _>>()?;
    ordered_workloads.sort_by_key(|workload| workload.logical_index);

    let sessions = ordered_workloads
        .iter()
        .map(|workload| {
            let mut all_token_ids = shared_prompt_token_ids.clone();
            all_token_ids.extend_from_slice(&workload.suffix_token_ids);
            all_token_ids.extend_from_slice(&workload.generated_token_ids);
            let text = model.decode(&all_token_ids, true)?;
            Ok(SessionSummary {
                logical_index: workload.logical_index,
                session_id: workload.session_id,
                arrival_wave: workload.arrival_wave,
                attached_from_prefix: workload.attached_from_prefix,
                suffix_text: workload.suffix_text.clone(),
                suffix_token_count: workload.suffix_token_ids.len(),
                target_decode_tokens: workload.target_decode_tokens,
                generated_token_count: workload.generated_token_ids.len(),
                completed_by_eos: workload.completed_by_eos,
                text,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    let request_metrics = model.request_metrics();
    let request_count = request_metrics.len();
    let prefill_request_count = request_metrics
        .iter()
        .filter(|request| request.kind() == SessionRequestKind::Prefill)
        .count();
    let decode_request_count = request_metrics
        .iter()
        .filter(|request| request.kind() == SessionRequestKind::Decode)
        .count();
    let batch_decode_request_count = request_metrics
        .iter()
        .filter(|request| request.kind() == SessionRequestKind::BatchDecode)
        .count();
    let total_input_token_count = request_metrics
        .iter()
        .map(|request| request.input_token_count())
        .sum::<usize>();

    let total_request_suffix_token_count = sessions
        .iter()
        .map(|session| session.suffix_token_count)
        .sum();
    let total_generated_token_count = sessions
        .iter()
        .map(|session| session.generated_token_count)
        .sum();

    let attention_path = model.attention_path();
    let mut stage_metrics = aggregate_stage_metrics(&model);
    stage_metrics.tokenization_millis += millis(tokenization_elapsed);
    let cache_metrics = model.cache_metrics().cloned().unwrap_or_default();
    let cache = model.paged_cache();

    let trace_path = PathBuf::from(format!("{}.trace.jsonl", args.out_prefix));
    let summary_path = PathBuf::from(format!("{}.summary.json", args.out_prefix));
    model.write_request_metrics_jsonl(&trace_path)?;

    let summary = WorkloadSummary {
        model_id: model.architecture().model_id.clone(),
        family: model.architecture().family.as_str().to_string(),
        device: args.device.to_string(),
        dtype: format!("{:?}", args.dtype).to_lowercase(),
        runtime_mode: model.runtime_mode().to_string(),
        attention_path: attention_path.to_string(),
        shared_prompt: args.shared_prompt.clone(),
        shared_prompt_token_count: shared_prompt_token_ids.len(),
        shared_prompt_token_target: args.shared_prompt_token_target,
        warmup_runs: args.warmup_runs,
        warmup_millis: millis(warmup_elapsed),
        total_sessions: args.total_sessions,
        attached_session_count: sessions
            .iter()
            .filter(|session| session.attached_from_prefix)
            .count(),
        wave_size: args.wave_size,
        decode_rounds_per_wave: args.decode_rounds_per_wave,
        peak_active_sessions,
        max_new_tokens: args.max_new_tokens,
        tokens_per_page: args.tokens_per_page,
        suffix_prefix: args.suffix_prefix.clone(),
        stress_mode: args.stress_mode,
        stress_suffix_repeats: args.stress_suffix_repeats,
        resident_page_budget: args.resident_page_budget,
        resident_byte_budget: args.resident_byte_budget,
        restore_cooldown_window: args
            .restore_cooldown_window
            .or_else(|| model.restore_cooldown_window()),
        request_count,
        prefill_request_count,
        decode_request_count,
        batch_decode_request_count,
        total_input_token_count,
        total_request_suffix_token_count,
        total_generated_token_count,
        spill_count: cache_metrics.spill_count,
        restore_count: cache_metrics.restore_count,
        spilled_bytes: cache_metrics.spilled_bytes,
        restored_bytes: cache_metrics.restored_bytes,
        cooldown_hit_count: cache_metrics.cooldown_hit_count,
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
        cold_prefix_prefill_millis: millis(cold_prefix_prefill_elapsed),
        prefix_capture_millis: millis(prefix_capture_elapsed),
        seed_suffix_prefill_millis: millis(seed_suffix_prefill_elapsed),
        attached_suffix_prefill_millis: millis(attached_suffix_prefill_elapsed),
        attach_millis: millis(attach_elapsed),
        decode_millis: millis(decode_elapsed),
        close_millis: millis(close_elapsed),
        release_prefix_millis: millis(release_prefix_elapsed),
        total_millis: millis(total_elapsed),
        cold_prefix_tokens_per_second: tokens_per_second(
            shared_prompt_token_ids.len(),
            cold_prefix_prefill_elapsed,
        ),
        request_prefill_tokens_per_second: tokens_per_second(
            total_request_suffix_token_count,
            seed_suffix_prefill_elapsed + attached_suffix_prefill_elapsed,
        ),
        decode_tokens_per_second: tokens_per_second(decode_input_token_count, decode_elapsed),
        total_tokens_per_second: tokens_per_second(total_input_token_count, total_elapsed),
        sessions,
        trace_jsonl_path: trace_path.display().to_string(),
    };

    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    println!(
        "workload: device={} runtime_mode={} attention_path={} warmup_runs={} warmup_ms={:.3} sessions={} peak_active={} total_ms={:.3} total_tps={:.3} spills={} restores={} cooldown_hits={}",
        args.device,
        model.runtime_mode(),
        attention_path,
        summary.warmup_runs,
        summary.warmup_millis,
        summary.total_sessions,
        summary.peak_active_sessions,
        summary.total_millis,
        summary.total_tokens_per_second,
        summary.spill_count,
        summary.restore_count,
        summary.cooldown_hit_count,
    );
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
