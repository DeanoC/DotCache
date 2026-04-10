#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use candle_core::DType;
    use dotcache_paged_runtime::{
        AttentionPathMode, CandleCausalLm, CandleDeviceSelector, CausalLm, ModelFamily,
        RuntimeMode, SessionRequestKind,
    };
    use serde::Serialize;

    #[derive(Debug, Clone)]
    struct BenchmarkConfig {
        family: ModelFamily,
        model_id: String,
        prompt: String,
        device: CandleDeviceSelector,
        runtime_mode: RuntimeMode,
        attention_path: AttentionPathMode,
        warmup_runs: usize,
        max_new_tokens: usize,
        batch_size: usize,
        tokens_per_page: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Serialize)]
    struct BenchmarkSummary {
        model_id: String,
        family: String,
        device: String,
        runtime_mode: String,
        attention_path: String,
        prompt: String,
        batch_size: usize,
        warmup_runs: usize,
        warmup_millis: f64,
        session_ids: Vec<usize>,
        session_count: usize,
        prompt_token_count_per_session: usize,
        total_prefill_token_count: usize,
        generated_token_count_per_session: Vec<usize>,
        total_generated_token_count: usize,
        total_input_token_count: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        request_count: usize,
        prefill_request_count: usize,
        decode_request_count: usize,
        batch_decode_request_count: usize,
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
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
        prefill_tokens_per_second: f64,
        decode_tokens_per_second: f64,
        total_tokens_per_second: f64,
        texts: Vec<String>,
        trace_jsonl_path: String,
    }

    #[derive(Debug, Serialize)]
    struct SweepVariantSummary {
        name: String,
        summary_path: String,
        trace_jsonl_path: String,
        device: String,
        runtime_mode: String,
        attention_path: String,
        batch_size: usize,
        total_millis: f64,
        total_tokens_per_second: f64,
        total_generated_token_count: usize,
        spill_count: usize,
        restore_count: usize,
        cooldown_hit_count: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Serialize)]
    struct SweepIndex {
        model_id: String,
        family: String,
        prompt: String,
        device: String,
        runtime_mode: String,
        attention_path: String,
        warmup_runs: usize,
        max_new_tokens: usize,
        tokens_per_page: usize,
        variant_count: usize,
        variants: Vec<SweepVariantSummary>,
    }

    #[derive(Debug)]
    struct SweepArgs {
        family: ModelFamily,
        model_id: String,
        prompt: String,
        out_dir: PathBuf,
        device: CandleDeviceSelector,
        runtime_mode: RuntimeMode,
        attention_path: Option<AttentionPathMode>,
        warmup_runs: usize,
        max_new_tokens: usize,
        batch_size: usize,
        tokens_per_page: usize,
        resident_page_budgets: Vec<Option<usize>>,
        resident_byte_budgets: Vec<Option<usize>>,
        resident_byte_budgets_explicit: bool,
        restore_cooldowns: Vec<Option<u64>>,
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
            "usage: hf_bench_sweep <family> <model_id> <prompt> <out_dir> [--device cpu|metal[:ordinal]|cuda[:ordinal]|hip[:ordinal]] [--runtime-mode dense_control|paged_control|dotcache_experimental] [--attention-path paged|fused] [--warmup-runs N] [--max-new-tokens N] [--batch-size N] [--tokens-per-page N] [--resident-page-budgets LIST] [--resident-byte-budgets LIST] [--restore-cooldowns LIST]".to_string()
        })?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let prompt = args.next().ok_or_else(|| "missing prompt".to_string())?;
        let out_dir = args
            .next()
            .ok_or_else(|| "missing out_dir".to_string())
            .map(PathBuf::from)?;

        let mut parsed = SweepArgs {
            family: family.parse().map_err(|err| format!("{err}"))?,
            model_id,
            prompt,
            out_dir,
            device: CandleDeviceSelector::Cpu,
            runtime_mode: RuntimeMode::PagedControl,
            attention_path: None,
            warmup_runs: 1,
            max_new_tokens: 16,
            batch_size: 1,
            tokens_per_page: CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
            resident_page_budgets: vec![None, Some(2)],
            resident_byte_budgets: vec![None],
            resident_byte_budgets_explicit: false,
            restore_cooldowns: vec![Some(8)],
        };

        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--warmup-runs" => {
                    parsed.warmup_runs = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --warmup-runs: {err}"))?;
                }
                "--max-new-tokens" => {
                    parsed.max_new_tokens = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --max-new-tokens: {err}"))?;
                }
                "--batch-size" => {
                    parsed.batch_size = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --batch-size: {err}"))?;
                    if parsed.batch_size == 0 {
                        return Err("--batch-size must be at least 1".to_string());
                    }
                }
                "--device" => {
                    parsed.device = value.parse::<CandleDeviceSelector>()?;
                }
                "--runtime-mode" => {
                    parsed.runtime_mode = value
                        .parse::<RuntimeMode>()
                        .map_err(|err| format!("{err}"))?;
                }
                "--attention-path" => {
                    parsed.attention_path = Some(value.parse::<AttentionPathMode>()?);
                }
                "--tokens-per-page" => {
                    parsed.tokens_per_page = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --tokens-per-page: {err}"))?;
                }
                "--resident-page-budgets" => {
                    parsed.resident_page_budgets =
                        parse_optional_usize_list(&value, "--resident-page-budgets")?;
                }
                "--resident-byte-budgets" => {
                    parsed.resident_byte_budgets_explicit = true;
                    parsed.resident_byte_budgets =
                        parse_optional_usize_list(&value, "--resident-byte-budgets")?;
                }
                "--restore-cooldowns" => {
                    parsed.restore_cooldowns =
                        parse_optional_u64_list(&value, "--restore-cooldowns")?;
                }
                other => return Err(format!("unknown flag {other}")),
            }
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

    fn variant_slug(
        device: &CandleDeviceSelector,
        runtime_mode: RuntimeMode,
        attention_path: AttentionPathMode,
        batch_size: usize,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    ) -> String {
        format!(
            "device-{}_mode-{}_attn-{}_batch-{}_pages-{}_bytes-{}_cooldown-{}",
            device.slug(),
            runtime_mode,
            attention_path,
            batch_size,
            resident_page_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            resident_byte_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            restore_cooldown_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
        )
    }

    fn push_unique_budget(budgets: &mut Vec<Option<usize>>, budget: Option<usize>) {
        if !budgets.contains(&budget) {
            budgets.push(budget);
        }
    }

    fn auto_byte_budgets_from_peak(peak_bytes: usize) -> Vec<Option<usize>> {
        let mut budgets = vec![None];
        if peak_bytes == 0 {
            return budgets;
        }

        push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes * 3 / 4)));
        push_unique_budget(&mut budgets, Some(std::cmp::max(1, peak_bytes / 2)));
        budgets
    }

    #[derive(Debug)]
    struct BenchmarkRunResult {
        session_ids: Vec<usize>,
        generated_token_count_per_session: Vec<usize>,
        total_generated_token_count: usize,
        total_input_token_count: usize,
        texts: Vec<String>,
        prefill_elapsed: Duration,
        decode_elapsed: Duration,
        total_elapsed: Duration,
        peak_resident_physical_page_count: usize,
        peak_resident_physical_byte_count: usize,
    }

    fn run_benchmark_pass(
        model: &mut CandleCausalLm,
        prompt_token_ids: &[u32],
        config: &BenchmarkConfig,
    ) -> Result<BenchmarkRunResult, Box<dyn std::error::Error>> {
        model.reset()?;
        model.reset_cache_metrics();
        model.clear_request_metrics();

        let mut session_ids = Vec::with_capacity(config.batch_size);
        let active_session_id = model
            .active_session_id()
            .ok_or("active session should be available after reset")?;
        session_ids.push(active_session_id);
        for _ in 1..config.batch_size {
            session_ids.push(model.create_session()?);
        }

        let total_start = Instant::now();
        let mut peak_resident_physical_page_count = model.resident_physical_page_count();
        let mut peak_resident_physical_byte_count = model.resident_physical_byte_count();
        let prefill_start = Instant::now();
        let prefill_requests = session_ids
            .iter()
            .map(|&session_id| (session_id, prompt_token_ids))
            .collect::<Vec<_>>();
        let prefill_logits = model.prefill_sessions_batch(&prefill_requests)?;
        let prefill_elapsed = prefill_start.elapsed();
        peak_resident_physical_page_count =
            peak_resident_physical_page_count.max(model.resident_physical_page_count());
        peak_resident_physical_byte_count =
            peak_resident_physical_byte_count.max(model.resident_physical_byte_count());

        let session_index_by_id = session_ids
            .iter()
            .enumerate()
            .map(|(index, &session_id)| (session_id, index))
            .collect::<HashMap<_, _>>();
        let mut logits_by_session = prefill_logits.into_iter().collect::<HashMap<_, _>>();
        let mut generated_token_ids = session_ids
            .iter()
            .map(|_| Vec::with_capacity(config.max_new_tokens))
            .collect::<Vec<_>>();
        let mut live_session_ids = session_ids.clone();
        let mut decode_elapsed = Duration::ZERO;
        for _ in 0..config.max_new_tokens {
            if live_session_ids.is_empty() {
                break;
            }

            let mut decode_requests = Vec::with_capacity(live_session_ids.len());
            let mut continuing_session_ids = Vec::with_capacity(live_session_ids.len());
            for &session_id in &live_session_ids {
                let logits = logits_by_session
                    .get(&session_id)
                    .ok_or("missing logits for live session")?;
                let next_token = argmax(logits).ok_or("empty decode logits")? as u32;
                let session_index = *session_index_by_id
                    .get(&session_id)
                    .ok_or("missing session index")?;
                generated_token_ids[session_index].push(next_token);
                if !model.architecture().eos_token_ids.contains(&next_token) {
                    decode_requests.push((session_id, next_token));
                    continuing_session_ids.push(session_id);
                }
            }

            if decode_requests.is_empty() {
                break;
            }

            let decode_start = Instant::now();
            let batch_logits = model.forward_next_logits_batch(&decode_requests)?;
            decode_elapsed += decode_start.elapsed();
            logits_by_session = batch_logits.into_iter().collect::<HashMap<_, _>>();
            live_session_ids = continuing_session_ids;
            peak_resident_physical_page_count =
                peak_resident_physical_page_count.max(model.resident_physical_page_count());
            peak_resident_physical_byte_count =
                peak_resident_physical_byte_count.max(model.resident_physical_byte_count());
        }

        let texts = generated_token_ids
            .iter()
            .map(|session_generated_token_ids| {
                let mut all_token_ids = prompt_token_ids.to_vec();
                all_token_ids.extend_from_slice(session_generated_token_ids);
                model.decode(&all_token_ids, true)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let request_metrics = model.request_metrics();
        let total_input_token_count = request_metrics
            .iter()
            .map(|request| request.input_token_count())
            .sum::<usize>();
        let generated_token_count_per_session = generated_token_ids
            .iter()
            .map(std::vec::Vec::len)
            .collect::<Vec<_>>();
        let total_generated_token_count = generated_token_count_per_session.iter().sum();

        Ok(BenchmarkRunResult {
            session_ids,
            generated_token_count_per_session,
            total_generated_token_count,
            total_input_token_count,
            texts,
            prefill_elapsed,
            decode_elapsed,
            total_elapsed: total_start.elapsed(),
            peak_resident_physical_page_count,
            peak_resident_physical_byte_count,
        })
    }

    fn run_benchmark(
        config: &BenchmarkConfig,
        out_prefix: &Path,
    ) -> Result<BenchmarkSummary, Box<dyn std::error::Error>> {
        let mut model = CandleCausalLm::from_hf_with_runtime_mode(
            &config.model_id,
            config.family,
            config.device.clone(),
            DType::F32,
            config.tokens_per_page,
            config.runtime_mode,
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

        let prompt_token_ids = model.encode(&config.prompt, true)?;
        if prompt_token_ids.is_empty() {
            return Err("prompt encoding produced no tokens".into());
        }
        let warmup_start = Instant::now();
        for _ in 0..config.warmup_runs {
            let _ = run_benchmark_pass(&mut model, &prompt_token_ids, config)?;
        }
        let warmup_elapsed = warmup_start.elapsed();

        let run = run_benchmark_pass(&mut model, &prompt_token_ids, config)?;

        let request_metrics = model.request_metrics();
        let attention_path = model.attention_path();
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
        let cache_metrics = model.cache_metrics().cloned().unwrap_or_default();
        let cache = model.paged_cache();
        let trace_path = PathBuf::from(format!("{}.trace.jsonl", out_prefix.display()));
        let summary_path = PathBuf::from(format!("{}.summary.json", out_prefix.display()));
        model.write_request_metrics_jsonl(&trace_path)?;

        let total_prefill_token_count = prompt_token_ids.len() * run.session_ids.len();
        let summary = BenchmarkSummary {
            model_id: model.architecture().model_id.clone(),
            family: model.architecture().family.as_str().to_string(),
            device: config.device.to_string(),
            runtime_mode: model.runtime_mode().to_string(),
            attention_path: attention_path.to_string(),
            prompt: config.prompt.clone(),
            batch_size: config.batch_size,
            warmup_runs: config.warmup_runs,
            warmup_millis: millis(warmup_elapsed),
            session_ids: run.session_ids,
            session_count: model.session_count().unwrap_or(config.batch_size),
            prompt_token_count_per_session: prompt_token_ids.len(),
            total_prefill_token_count,
            generated_token_count_per_session: run.generated_token_count_per_session,
            total_generated_token_count: run.total_generated_token_count,
            total_input_token_count: run.total_input_token_count,
            max_new_tokens: config.max_new_tokens,
            tokens_per_page: config.tokens_per_page,
            resident_page_budget: config.resident_page_budget,
            resident_byte_budget: config.resident_byte_budget,
            restore_cooldown_window: config
                .restore_cooldown_window
                .or_else(|| model.restore_cooldown_window()),
            request_count,
            prefill_request_count,
            decode_request_count,
            batch_decode_request_count,
            spill_count: cache_metrics.spill_count,
            restore_count: cache_metrics.restore_count,
            spilled_bytes: cache_metrics.spilled_bytes,
            restored_bytes: cache_metrics.restored_bytes,
            cooldown_hit_count: cache_metrics.cooldown_hit_count,
            peak_resident_physical_page_count: run.peak_resident_physical_page_count,
            peak_resident_physical_byte_count: run.peak_resident_physical_byte_count,
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
            prefill_millis: millis(run.prefill_elapsed),
            decode_millis: millis(run.decode_elapsed),
            total_millis: millis(run.total_elapsed),
            prefill_tokens_per_second: tokens_per_second(
                total_prefill_token_count,
                run.prefill_elapsed,
            ),
            decode_tokens_per_second: tokens_per_second(
                request_metrics
                    .iter()
                    .filter(|request| request.kind() == SessionRequestKind::BatchDecode)
                    .map(|request| request.input_token_count())
                    .sum::<usize>(),
                run.decode_elapsed,
            ),
            total_tokens_per_second: tokens_per_second(
                run.total_input_token_count,
                run.total_elapsed,
            ),
            texts: run.texts,
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
        let calibration_prefix = args.out_dir.join(".bench-byte-calibration");
        let calibration_summary = run_benchmark(
            &BenchmarkConfig {
                family: args.family,
                model_id: args.model_id.clone(),
                prompt: args.prompt.clone(),
                device: args.device.clone(),
                runtime_mode: args.runtime_mode,
                attention_path,
                warmup_runs: args.warmup_runs,
                max_new_tokens: args.max_new_tokens,
                batch_size: args.batch_size,
                tokens_per_page: args.tokens_per_page,
                resident_page_budget: None,
                resident_byte_budget: None,
                restore_cooldown_window: args.restore_cooldowns[0],
            },
            &calibration_prefix,
        )?;
        let budgets =
            auto_byte_budgets_from_peak(calibration_summary.peak_resident_physical_byte_count);
        let _ = std::fs::remove_file(format!("{}.summary.json", calibration_prefix.display()));
        let _ = std::fs::remove_file(format!("{}.trace.jsonl", calibration_prefix.display()));
        eprintln!(
            "auto-derived resident byte budgets from peak {} bytes: {:?}",
            calibration_summary.peak_resident_physical_byte_count, budgets
        );
        budgets
    };

    let mut variants = Vec::new();
    for &resident_page_budget in &args.resident_page_budgets {
        for &resident_byte_budget in &resident_byte_budgets {
            for &restore_cooldown_window in &args.restore_cooldowns {
                let name = variant_slug(
                    &args.device,
                    args.runtime_mode,
                    attention_path,
                    args.batch_size,
                    resident_page_budget,
                    resident_byte_budget,
                    restore_cooldown_window,
                );
                let out_prefix = args.out_dir.join(&name);
                let summary = run_benchmark(
                    &BenchmarkConfig {
                        family: args.family,
                        model_id: args.model_id.clone(),
                        prompt: args.prompt.clone(),
                        device: args.device.clone(),
                        runtime_mode: args.runtime_mode,
                        attention_path,
                        warmup_runs: args.warmup_runs,
                        max_new_tokens: args.max_new_tokens,
                        batch_size: args.batch_size,
                        tokens_per_page: args.tokens_per_page,
                        resident_page_budget,
                        resident_byte_budget,
                        restore_cooldown_window,
                    },
                    &out_prefix,
                )?;
                println!(
                    "{name}: batch_size={} total_ms={:.3} total_tps={:.3} generated_tokens={} spills={} restores={} cooldown_hits={}",
                    summary.batch_size,
                    summary.total_millis,
                    summary.total_tokens_per_second,
                    summary.total_generated_token_count,
                    summary.spill_count,
                    summary.restore_count,
                    summary.cooldown_hit_count,
                );
                variants.push(SweepVariantSummary {
                    name,
                    summary_path: format!("{}.summary.json", out_prefix.display()),
                    trace_jsonl_path: summary.trace_jsonl_path.clone(),
                    device: summary.device.clone(),
                    runtime_mode: summary.runtime_mode.clone(),
                    attention_path: summary.attention_path.clone(),
                    batch_size: summary.batch_size,
                    total_millis: summary.total_millis,
                    total_tokens_per_second: summary.total_tokens_per_second,
                    total_generated_token_count: summary.total_generated_token_count,
                    spill_count: summary.spill_count,
                    restore_count: summary.restore_count,
                    cooldown_hit_count: summary.cooldown_hit_count,
                    resident_page_budget,
                    resident_byte_budget,
                    restore_cooldown_window,
                });
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
        prompt: args.prompt,
        device: args.device.to_string(),
        runtime_mode: args.runtime_mode.to_string(),
        attention_path: attention_path.to_string(),
        warmup_runs: args.warmup_runs,
        max_new_tokens: args.max_new_tokens,
        tokens_per_page: args.tokens_per_page,
        variant_count: variants.len(),
        variants,
    };
    let index_path = args.out_dir.join("index.json");
    std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;
    eprintln!("wrote sweep index to {}", index_path.display());
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
