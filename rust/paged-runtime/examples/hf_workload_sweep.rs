#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::{HashMap, VecDeque};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use candle_core::DType;
    use dotcache_paged_runtime::{
        AttentionPathMode, CandleCausalLm, CandleDeviceSelector, CausalLm, ModelFamily, SessionId,
        SessionRequestKind,
    };
    use serde::Serialize;

    #[derive(Debug, Clone)]
    struct WorkloadConfig {
        family: ModelFamily,
        model_id: String,
        shared_prompt: String,
        device: CandleDeviceSelector,
        attention_path: AttentionPathMode,
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
        attention_path: String,
        shared_prompt: String,
        shared_prompt_token_count: usize,
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
        peak_resident_physical_page_count: usize,
        peak_resident_physical_byte_count: usize,
        physical_page_count: usize,
        virtual_page_count: usize,
        resident_physical_page_count: usize,
        spilled_physical_page_count: usize,
        resident_physical_byte_count: usize,
        spilled_physical_byte_count: usize,
        pinned_physical_page_count: usize,
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

    #[derive(Debug, Serialize)]
    struct SweepVariantSummary {
        name: String,
        summary_path: String,
        trace_jsonl_path: String,
        device: String,
        attention_path: String,
        total_sessions: usize,
        wave_size: usize,
        decode_rounds_per_wave: usize,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        peak_active_sessions: usize,
        total_generated_token_count: usize,
        total_millis: f64,
        total_tokens_per_second: f64,
        spill_count: usize,
        restore_count: usize,
        cooldown_hit_count: usize,
    }

    #[derive(Debug, Serialize)]
    struct SweepIndex {
        model_id: String,
        family: String,
        shared_prompt: String,
        device: String,
        attention_path: String,
        warmup_runs: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        suffix_prefix: String,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        variant_count: usize,
        variants: Vec<SweepVariantSummary>,
    }

    #[derive(Debug)]
    struct SweepArgs {
        family: ModelFamily,
        model_id: String,
        shared_prompt: String,
        out_dir: PathBuf,
        device: CandleDeviceSelector,
        attention_path: Option<AttentionPathMode>,
        warmup_runs: usize,
        total_sessions_list: Vec<usize>,
        wave_sizes: Vec<usize>,
        decode_rounds_per_wave_list: Vec<usize>,
        max_new_tokens: usize,
        tokens_per_page: usize,
        suffix_prefix: String,
        stress_mode: bool,
        stress_suffix_repeats: usize,
        resident_page_budgets: Vec<Option<usize>>,
        resident_byte_budgets: Vec<Option<usize>>,
        resident_byte_budgets_explicit: bool,
        restore_cooldowns: Vec<Option<u64>>,
    }

    fn parse_usize_list(value: &str, flag: &str) -> Result<Vec<usize>, String> {
        let values = value
            .split(',')
            .map(|entry| {
                let entry = entry.trim();
                entry
                    .parse::<usize>()
                    .map_err(|err| format!("invalid {flag} entry `{entry}`: {err}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().any(|&value| value == 0) {
            return Err(format!("{flag} values must all be at least 1"));
        }
        Ok(values)
    }

    fn parse_optional_usize_list(value: &str, flag: &str) -> Result<Vec<Option<usize>>, String> {
        value
            .split(',')
            .map(|entry| {
                let entry = entry.trim();
                if entry.eq_ignore_ascii_case("none") {
                    Ok(None)
                } else {
                    entry
                        .parse::<usize>()
                        .map(Some)
                        .map_err(|err| format!("invalid {flag} entry `{entry}`: {err}"))
                }
            })
            .collect()
    }

    fn parse_optional_u64_list(value: &str, flag: &str) -> Result<Vec<Option<u64>>, String> {
        value
            .split(',')
            .map(|entry| {
                let entry = entry.trim();
                if entry.eq_ignore_ascii_case("none") {
                    Ok(None)
                } else {
                    entry
                        .parse::<u64>()
                        .map(Some)
                        .map_err(|err| format!("invalid {flag} entry `{entry}`: {err}"))
                }
            })
            .collect()
    }

    fn parse_args() -> Result<SweepArgs, String> {
        let mut args = std::env::args().skip(1);
        let family = args.next().ok_or_else(|| {
            "usage: hf_workload_sweep <family> <model_id> <shared_prompt> <out_dir> [--device cpu|metal[:ordinal]|cuda[:ordinal]] [--attention-path paged|fused] [--warmup-runs N] [--total-sessions-list LIST] [--wave-sizes LIST] [--decode-rounds-per-wave-list LIST] [--max-new-tokens N] [--tokens-per-page N] [--suffix-prefix TEXT] [--stress] [--stress-suffix-repeats N] [--resident-page-budgets LIST] [--resident-byte-budgets LIST] [--restore-cooldowns LIST]".to_string()
        })?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let shared_prompt = args
            .next()
            .ok_or_else(|| "missing shared_prompt".to_string())?;
        let out_dir = args
            .next()
            .ok_or_else(|| "missing out_dir".to_string())
            .map(PathBuf::from)?;

        let mut parsed = SweepArgs {
            family: family.parse().map_err(|err| format!("{err}"))?,
            model_id,
            shared_prompt,
            out_dir,
            device: CandleDeviceSelector::Cpu,
            attention_path: None,
            warmup_runs: 1,
            total_sessions_list: vec![4, 6],
            wave_sizes: vec![2, 3],
            decode_rounds_per_wave_list: vec![1, 2],
            max_new_tokens: 4,
            tokens_per_page: CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
            suffix_prefix: "session".to_string(),
            stress_mode: false,
            stress_suffix_repeats: 6,
            resident_page_budgets: vec![None, Some(2), Some(1)],
            resident_byte_budgets: vec![None],
            resident_byte_budgets_explicit: false,
            restore_cooldowns: vec![Some(8), Some(32)],
        };

        while let Some(flag) = args.next() {
            match flag.as_str() {
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
                "--total-sessions-list" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.total_sessions_list = parse_usize_list(&value, "--total-sessions-list")?;
                }
                "--wave-sizes" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.wave_sizes = parse_usize_list(&value, "--wave-sizes")?;
                }
                "--decode-rounds-per-wave-list" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.decode_rounds_per_wave_list =
                        parse_usize_list(&value, "--decode-rounds-per-wave-list")?;
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
                "--resident-page-budgets" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.resident_page_budgets =
                        parse_optional_usize_list(&value, "--resident-page-budgets")?;
                }
                "--resident-byte-budgets" => {
                    parsed.resident_byte_budgets_explicit = true;
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.resident_byte_budgets =
                        parse_optional_usize_list(&value, "--resident-byte-budgets")?;
                }
                "--restore-cooldowns" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    parsed.restore_cooldowns =
                        parse_optional_u64_list(&value, "--restore-cooldowns")?;
                }
                other => return Err(format!("unknown flag {other}")),
            }
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
        Ok(parsed)
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

    fn warmup_workload_pass(
        model: &mut CandleCausalLm,
        shared_prompt_token_ids: &[u32],
        config: &WorkloadConfig,
        warmup_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        model.reset()?;
        model.reset_cache_metrics();
        model.clear_request_metrics();

        let _ = model.forward_next_logits(shared_prompt_token_ids)?;
        let seed_session_id = model
            .active_session_id()
            .ok_or("active session should be available during warmup")?;
        let prefix = model.capture_prefix(seed_session_id)?;
        model.close_session(seed_session_id)?;

        let session_a = model.attach_prefix(&prefix)?;
        let session_b = model.attach_prefix(&prefix)?;
        let suffix_a = suffix_text(
            &config.suffix_prefix,
            warmup_index * 2,
            config.stress_mode,
            config.stress_suffix_repeats,
        );
        let suffix_b = suffix_text(
            &config.suffix_prefix,
            warmup_index * 2 + 1,
            config.stress_mode,
            config.stress_suffix_repeats,
        );
        let suffix_a_ids = model.encode(&suffix_a, false)?;
        let suffix_b_ids = model.encode(&suffix_b, false)?;
        if suffix_a_ids.is_empty() || suffix_b_ids.is_empty() {
            return Err("warmup suffix encoding produced no tokens".into());
        }

        if config.stress_mode {
            let _ = prefill_session_chunked(
                model,
                session_a,
                &suffix_a_ids,
                config.tokens_per_page.saturating_sub(1),
            )?;
            let _ = prefill_session_chunked(
                model,
                session_b,
                &suffix_b_ids,
                config.tokens_per_page.saturating_sub(1),
            )?;
        } else {
            let _ = model.prefill_sessions_batch(&[
                (session_a, suffix_a_ids.as_slice()),
                (session_b, suffix_b_ids.as_slice()),
            ])?;
        }

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
        model.release_prefix(&prefix)?;
        Ok(())
    }

    fn push_unique_budget(budgets: &mut Vec<Option<usize>>, budget: Option<usize>) {
        if !budgets.contains(&budget) {
            budgets.push(budget);
        }
    }

    fn auto_byte_budgets_from_peak(peak_bytes: usize, stress_mode: bool) -> Vec<Option<usize>> {
        let mut budgets = vec![None];
        if peak_bytes == 0 {
            return budgets;
        }

        if stress_mode {
            push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes * 2 / 3)));
            push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes / 3)));
        } else {
            push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes * 3 / 4)));
            push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes / 2)));
        }
        budgets
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

    fn variant_slug(config: &WorkloadConfig) -> String {
        format!(
            "device-{}_attn-{}_stress-{}_repeats-{}_sessions-{}_wave-{}_rounds-{}_pages-{}_bytes-{}_cooldown-{}",
            config.device.slug(),
            config.attention_path,
            if config.stress_mode { "on" } else { "off" },
            config.stress_suffix_repeats,
            config.total_sessions,
            config.wave_size,
            config.decode_rounds_per_wave,
            config
                .resident_page_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            config
                .resident_byte_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            config
                .restore_cooldown_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
        )
    }

    fn run_workload(
        config: &WorkloadConfig,
        out_prefix: &Path,
    ) -> Result<WorkloadSummary, Box<dyn std::error::Error>> {
        let mut model = CandleCausalLm::from_hf_with_paging(
            &config.model_id,
            config.family,
            config.device.clone(),
            DType::F32,
            config.tokens_per_page,
        )?;
        model.set_attention_path(config.attention_path);
        if let Some(budget) = config.resident_page_budget {
            model.set_resident_physical_page_budget(Some(budget))?;
        }
        if let Some(budget) = config.resident_byte_budget {
            model.set_resident_physical_byte_budget(Some(budget))?;
        }
        if let Some(window) = config.restore_cooldown_window {
            model.set_restore_cooldown_window(window);
        }

        let shared_prompt_token_ids = model.encode(&config.shared_prompt, true)?;
        if shared_prompt_token_ids.is_empty() {
            return Err("shared prompt encoding produced no tokens".into());
        }

        let warmup_start = Instant::now();
        for warmup_index in 0..config.warmup_runs {
            warmup_workload_pass(&mut model, &shared_prompt_token_ids, config, warmup_index)?;
        }
        let warmup_elapsed = warmup_start.elapsed();

        model.reset()?;
        model.reset_cache_metrics();
        model.clear_request_metrics();

        let total_start = Instant::now();
        let mut peak_resident_physical_page_count = model.resident_physical_page_count();
        let mut peak_resident_physical_byte_count = model.resident_physical_byte_count();

        let cold_prefix_prefill_start = Instant::now();
        let _ = model.forward_next_logits(&shared_prompt_token_ids)?;
        let cold_prefix_prefill_elapsed = cold_prefix_prefill_start.elapsed();
        peak_resident_physical_page_count =
            peak_resident_physical_page_count.max(model.resident_physical_page_count());
        peak_resident_physical_byte_count =
            peak_resident_physical_byte_count.max(model.resident_physical_byte_count());

        let seed_session_id = model
            .active_session_id()
            .ok_or("active session should be available")?;

        let prefix_capture_start = Instant::now();
        let prefix = model.capture_prefix(seed_session_id)?;
        let prefix_capture_elapsed = prefix_capture_start.elapsed();

        let mut workloads = HashMap::new();
        let mut logical_session_ids = Vec::with_capacity(config.total_sessions);
        let mut active_session_ids = Vec::new();
        let mut peak_active_sessions = 0usize;
        let mut close_elapsed = Duration::ZERO;

        let seed_close_start = Instant::now();
        model.close_session(seed_session_id)?;
        close_elapsed += seed_close_start.elapsed();

        let seed_suffix_text = suffix_text(
            &config.suffix_prefix,
            0,
            config.stress_mode,
            config.stress_suffix_repeats,
        );
        let seed_suffix_token_ids = model.encode(&seed_suffix_text, false)?;
        if seed_suffix_token_ids.is_empty() {
            return Err("seed suffix encoding produced no tokens".into());
        }
        let seed_session_id = model.attach_prefix(&prefix)?;
        let seed_suffix_prefill_start = Instant::now();
        let seed_logits = if config.stress_mode {
            prefill_session_chunked(
                &mut model,
                seed_session_id,
                &seed_suffix_token_ids,
                config.tokens_per_page.saturating_sub(1),
            )?
        } else {
            model.prefill_session(seed_session_id, &seed_suffix_token_ids)?
        };
        let seed_suffix_prefill_elapsed = seed_suffix_prefill_start.elapsed();
        peak_resident_physical_page_count =
            peak_resident_physical_page_count.max(model.resident_physical_page_count());
        peak_resident_physical_byte_count =
            peak_resident_physical_byte_count.max(model.resident_physical_byte_count());

        workloads.insert(
            seed_session_id,
            SessionWorkload {
                logical_index: 0,
                session_id: seed_session_id,
                arrival_wave: 0,
                attached_from_prefix: true,
                suffix_text: seed_suffix_text,
                suffix_token_ids: seed_suffix_token_ids,
                target_decode_tokens: target_decode_tokens(0, config.max_new_tokens),
                generated_token_ids: Vec::with_capacity(config.max_new_tokens),
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

        if config.total_sessions > 1 {
            decode_input_token_count += run_decode_round(
                &mut model,
                &mut active_session_ids,
                &mut workloads,
                &mut logits_by_session,
                &mut decode_elapsed,
                &mut close_elapsed,
            )?;
            peak_resident_physical_page_count =
                peak_resident_physical_page_count.max(model.resident_physical_page_count());
            peak_resident_physical_byte_count =
                peak_resident_physical_byte_count.max(model.resident_physical_byte_count());
        }

        let mut pending_indices = (1..config.total_sessions).collect::<VecDeque<_>>();
        let mut wave_index = 1usize;
        while !pending_indices.is_empty() {
            let mut arrivals = Vec::new();
            let attach_start = Instant::now();
            for _ in 0..config.wave_size {
                let Some(logical_index) = pending_indices.pop_front() else {
                    break;
                };
                let session_id = model.attach_prefix(&prefix)?;
                logical_session_ids.push(session_id);
                arrivals.push((logical_index, session_id));
            }
            attach_elapsed += attach_start.elapsed();

            if !arrivals.is_empty() {
                let mut suffix_requests = Vec::with_capacity(arrivals.len());
                for &(logical_index, session_id) in &arrivals {
                    let text = suffix_text(
                        &config.suffix_prefix,
                        logical_index,
                        config.stress_mode,
                        config.stress_suffix_repeats,
                    );
                    let token_ids = model.encode(&text, false)?;
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
                                config.max_new_tokens,
                            ),
                            generated_token_ids: Vec::with_capacity(config.max_new_tokens),
                            completed_by_eos: false,
                        },
                    );
                    suffix_requests.push((session_id, token_ids));
                }

                let suffix_prefill_start = Instant::now();
                if config.stress_mode {
                    for (session_id, token_ids) in suffix_requests {
                        let logits = prefill_session_chunked(
                            &mut model,
                            session_id,
                            &token_ids,
                            config.tokens_per_page.saturating_sub(1),
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
                peak_resident_physical_page_count =
                    peak_resident_physical_page_count.max(model.resident_physical_page_count());
                peak_resident_physical_byte_count =
                    peak_resident_physical_byte_count.max(model.resident_physical_byte_count());
            }

            for _ in 0..config.decode_rounds_per_wave {
                decode_input_token_count += run_decode_round(
                    &mut model,
                    &mut active_session_ids,
                    &mut workloads,
                    &mut logits_by_session,
                    &mut decode_elapsed,
                    &mut close_elapsed,
                )?;
                peak_resident_physical_page_count =
                    peak_resident_physical_page_count.max(model.resident_physical_page_count());
                peak_resident_physical_byte_count =
                    peak_resident_physical_byte_count.max(model.resident_physical_byte_count());
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
            peak_resident_physical_page_count =
                peak_resident_physical_page_count.max(model.resident_physical_page_count());
            peak_resident_physical_byte_count =
                peak_resident_physical_byte_count.max(model.resident_physical_byte_count());
        }

        let release_prefix_start = Instant::now();
        model.release_prefix(&prefix)?;
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
        let cache_metrics = model.cache_metrics().cloned().unwrap_or_default();
        let cache = model
            .paged_cache()
            .ok_or("paged cache should be available for workload summary")?;

        let trace_path = PathBuf::from(format!("{}.trace.jsonl", out_prefix.display()));
        let summary_path = PathBuf::from(format!("{}.summary.json", out_prefix.display()));
        model.write_request_metrics_jsonl(&trace_path)?;

        let summary = WorkloadSummary {
            model_id: model.architecture().model_id.clone(),
            family: model.architecture().family.as_str().to_string(),
            device: config.device.to_string(),
            attention_path: attention_path.to_string(),
            shared_prompt: config.shared_prompt.clone(),
            shared_prompt_token_count: shared_prompt_token_ids.len(),
            warmup_runs: config.warmup_runs,
            warmup_millis: millis(warmup_elapsed),
            total_sessions: config.total_sessions,
            attached_session_count: config.total_sessions,
            wave_size: config.wave_size,
            decode_rounds_per_wave: config.decode_rounds_per_wave,
            peak_active_sessions,
            max_new_tokens: config.max_new_tokens,
            tokens_per_page: config.tokens_per_page,
            suffix_prefix: config.suffix_prefix.clone(),
            stress_mode: config.stress_mode,
            stress_suffix_repeats: config.stress_suffix_repeats,
            resident_page_budget: config.resident_page_budget,
            resident_byte_budget: config.resident_byte_budget,
            restore_cooldown_window: config
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
            peak_resident_physical_page_count,
            peak_resident_physical_byte_count,
            physical_page_count: cache.physical_page_count(),
            virtual_page_count: cache.virtual_page_count(),
            resident_physical_page_count: cache.resident_physical_page_count(),
            spilled_physical_page_count: cache.spilled_physical_page_count(),
            resident_physical_byte_count: cache.resident_physical_byte_count(),
            spilled_physical_byte_count: cache.spilled_physical_byte_count(),
            pinned_physical_page_count: cache.pinned_physical_page_count(),
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
        Ok(summary)
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    std::fs::create_dir_all(&args.out_dir)?;
    let attention_path = args
        .attention_path
        .unwrap_or_else(|| AttentionPathMode::default_for_selector(&args.device));

    let resident_byte_budgets = if args.resident_byte_budgets_explicit {
        args.resident_byte_budgets.clone()
    } else {
        let calibration_prefix = args.out_dir.join(".workload-byte-calibration");
        let calibration_summary = run_workload(
            &WorkloadConfig {
                family: args.family,
                model_id: args.model_id.clone(),
                shared_prompt: args.shared_prompt.clone(),
                device: args.device.clone(),
                attention_path,
                warmup_runs: args.warmup_runs,
                total_sessions: args.total_sessions_list[0],
                wave_size: args.wave_sizes[0],
                decode_rounds_per_wave: args.decode_rounds_per_wave_list[0],
                max_new_tokens: args.max_new_tokens,
                tokens_per_page: args.tokens_per_page,
                suffix_prefix: args.suffix_prefix.clone(),
                stress_mode: args.stress_mode,
                stress_suffix_repeats: args.stress_suffix_repeats,
                resident_page_budget: None,
                resident_byte_budget: None,
                restore_cooldown_window: args.restore_cooldowns[0],
            },
            &calibration_prefix,
        )?;
        let budgets = auto_byte_budgets_from_peak(
            calibration_summary.peak_resident_physical_byte_count,
            args.stress_mode,
        );
        let _ = std::fs::remove_file(format!("{}.summary.json", calibration_prefix.display()));
        let _ = std::fs::remove_file(format!("{}.trace.jsonl", calibration_prefix.display()));
        eprintln!(
            "auto-derived resident byte budgets from peak {} bytes: {:?}",
            calibration_summary.peak_resident_physical_byte_count, budgets
        );
        budgets
    };

    let mut variants = Vec::new();
    for &total_sessions in &args.total_sessions_list {
        for &wave_size in &args.wave_sizes {
            for &decode_rounds_per_wave in &args.decode_rounds_per_wave_list {
                for &resident_page_budget in &args.resident_page_budgets {
                    for &resident_byte_budget in &resident_byte_budgets {
                        for &restore_cooldown_window in &args.restore_cooldowns {
                            let config = WorkloadConfig {
                                family: args.family,
                                model_id: args.model_id.clone(),
                                shared_prompt: args.shared_prompt.clone(),
                                device: args.device.clone(),
                                attention_path,
                                warmup_runs: args.warmup_runs,
                                total_sessions,
                                wave_size,
                                decode_rounds_per_wave,
                                max_new_tokens: args.max_new_tokens,
                                tokens_per_page: args.tokens_per_page,
                                suffix_prefix: args.suffix_prefix.clone(),
                                stress_mode: args.stress_mode,
                                stress_suffix_repeats: args.stress_suffix_repeats,
                                resident_page_budget,
                                resident_byte_budget,
                                restore_cooldown_window,
                            };
                            let name = variant_slug(&config);
                            let out_prefix = args.out_dir.join(&name);
                            let summary = run_workload(&config, &out_prefix)?;
                            println!(
                                "{name}: sessions={} wave={} rounds={} total_ms={:.3} total_tps={:.3} spills={} restores={} cooldown_hits={}",
                                summary.total_sessions,
                                summary.wave_size,
                                summary.decode_rounds_per_wave,
                                summary.total_millis,
                                summary.total_tokens_per_second,
                                summary.spill_count,
                                summary.restore_count,
                                summary.cooldown_hit_count,
                            );
                            variants.push(SweepVariantSummary {
                                name,
                                summary_path: format!("{}.summary.json", out_prefix.display()),
                                trace_jsonl_path: summary.trace_jsonl_path.clone(),
                                device: summary.device.clone(),
                                attention_path: summary.attention_path.clone(),
                                total_sessions: summary.total_sessions,
                                wave_size: summary.wave_size,
                                decode_rounds_per_wave: summary.decode_rounds_per_wave,
                                stress_mode: summary.stress_mode,
                                stress_suffix_repeats: summary.stress_suffix_repeats,
                                resident_page_budget,
                                resident_byte_budget,
                                restore_cooldown_window,
                                peak_active_sessions: summary.peak_active_sessions,
                                total_generated_token_count: summary.total_generated_token_count,
                                total_millis: summary.total_millis,
                                total_tokens_per_second: summary.total_tokens_per_second,
                                spill_count: summary.spill_count,
                                restore_count: summary.restore_count,
                                cooldown_hit_count: summary.cooldown_hit_count,
                            });
                        }
                    }
                }
            }
        }
    }

    variants.sort_by(|lhs, rhs| {
        rhs.total_tokens_per_second
            .total_cmp(&lhs.total_tokens_per_second)
            .then_with(|| lhs.total_millis.total_cmp(&rhs.total_millis))
    });
    let index = SweepIndex {
        model_id: args.model_id,
        family: args.family.as_str().to_string(),
        shared_prompt: args.shared_prompt,
        device: args.device.to_string(),
        attention_path: attention_path.to_string(),
        warmup_runs: args.warmup_runs,
        max_new_tokens: args.max_new_tokens,
        tokens_per_page: args.tokens_per_page,
        suffix_prefix: args.suffix_prefix,
        stress_mode: args.stress_mode,
        stress_suffix_repeats: args.stress_suffix_repeats,
        variant_count: variants.len(),
        variants,
    };
    let index_path = args.out_dir.join("index.json");
    std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;
    eprintln!("wrote workload sweep index to {}", index_path.display());
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
