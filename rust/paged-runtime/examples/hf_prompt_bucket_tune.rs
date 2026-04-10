#[cfg(feature = "hf")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    use dotcache_paged_runtime::HfHubModelSource;
    use serde::{Deserialize, Serialize};
    use tokenizers::Tokenizer;

    type AppError = Box<dyn std::error::Error + Send + Sync>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Bench,
        Workload,
        All,
    }

    impl Mode {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "bench" => Ok(Self::Bench),
                "workload" => Ok(Self::Workload),
                "all" => Ok(Self::All),
                other => Err(format!(
                    "invalid mode `{other}`, expected `bench`, `workload`, or `all`"
                )),
            }
        }

        fn as_str(self) -> &'static str {
            match self {
                Self::Bench => "bench",
                Self::Workload => "workload",
                Self::All => "all",
            }
        }

        fn execution_modes(self) -> Vec<Self> {
            match self {
                Self::Bench => vec![Self::Bench],
                Self::Workload => vec![Self::Workload],
                Self::All => vec![Self::Bench, Self::Workload],
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        mode: Mode,
        family: String,
        model_id: String,
        base_prompt: String,
        out_dir: PathBuf,
        token_buckets: Vec<usize>,
        top_k: usize,
        common_tune_args: Vec<String>,
        bench_tune_args: Vec<String>,
        workload_tune_args: Vec<String>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
    struct PolicyKey {
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Deserialize)]
    struct TuneSummary {
        coarse_report_path: String,
        refined_report_path: String,
        policy_report_path: String,
        coarse_best_variant: String,
        refined_best_variant: String,
        recommended_policy: Option<PolicyKey>,
        recommended_balanced_score: Option<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct PolicyAggregate {
        policy: PolicyKey,
        balanced_score: f64,
    }

    #[derive(Debug, Deserialize)]
    struct PolicyReport {
        best_complete_policy: Option<PolicyAggregate>,
        best_partial_policy: Option<PolicyAggregate>,
    }

    #[derive(Debug, Serialize)]
    struct RegimeBucketSummary {
        mode: String,
        tune_summary_path: String,
        policy_report_path: String,
        coarse_best_variant: String,
        refined_best_variant: String,
        recommended_policy: Option<PolicyKey>,
        recommended_balanced_score: Option<f64>,
    }

    #[derive(Debug, Serialize)]
    struct BucketSummary {
        token_bucket_label: String,
        token_bucket_lower_bound: usize,
        token_bucket_upper_bound: usize,
        prompt_character_count: usize,
        prompt_whitespace_token_count: usize,
        actual_prompt_token_count: usize,
        prompt_repeat_count: usize,
        prompt: String,
        bench: Option<RegimeBucketSummary>,
        workload: Option<RegimeBucketSummary>,
        combined_policy_report_path: Option<String>,
        recommended_policy: Option<PolicyKey>,
        recommended_balanced_score: Option<f64>,
    }

    #[derive(Debug, Serialize)]
    struct RecommendedPolicyAggregate {
        policy: PolicyKey,
        bucket_count: usize,
        average_balanced_score: Option<f64>,
    }

    #[derive(Debug, Serialize)]
    struct BucketReport {
        mode: String,
        family: String,
        model_id: String,
        base_prompt: String,
        token_buckets: Vec<usize>,
        policy_shifted_across_buckets: bool,
        stable_recommended_policy: Option<PolicyKey>,
        best_cross_bucket_policy: Option<RecommendedPolicyAggregate>,
        buckets: Vec<BucketSummary>,
    }

    fn parse_csv_usize(value: &str) -> Result<Vec<usize>, String> {
        let parsed = value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| {
                item.parse::<usize>()
                    .map_err(|err| format!("invalid value `{item}`: {err}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if parsed.is_empty() {
            return Err("list must not be empty".to_string());
        }
        if parsed.iter().any(|value| *value == 0) {
            return Err("list values must be at least 1".to_string());
        }
        Ok(parsed)
    }

    fn split_arg_string(value: &str) -> Vec<String> {
        value
            .split_whitespace()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    }

    fn parse_args() -> Result<Args, String> {
        let mut args = std::env::args().skip(1);
        let mode = Mode::parse(
            &args.next().ok_or_else(|| {
                "usage: hf_prompt_bucket_tune <bench|workload|all> <family> <model_id> <base_prompt> <out_dir> [--token-buckets CSV] [--top-k N] [--bench-args \"...\"] [--workload-args \"...\"] [common tune args]".to_string()
            })?,
        )?;
        let family = args.next().ok_or_else(|| "missing family".to_string())?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let base_prompt = args
            .next()
            .ok_or_else(|| "missing base_prompt".to_string())?;
        let out_dir = args
            .next()
            .ok_or_else(|| "missing out_dir".to_string())
            .map(PathBuf::from)?;

        let mut token_buckets = vec![32, 128, 512];
        let mut top_k = 2usize;
        let mut common_tune_args = Vec::new();
        let mut bench_tune_args = Vec::new();
        let mut workload_tune_args = Vec::new();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--token-buckets" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --token-buckets".to_string())?;
                    token_buckets = parse_csv_usize(&value)?;
                }
                "--top-k" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --top-k".to_string())?;
                    top_k = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --top-k: {err}"))?;
                    if top_k == 0 {
                        return Err("--top-k must be at least 1".to_string());
                    }
                }
                "--bench-args" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --bench-args".to_string())?;
                    bench_tune_args.extend(split_arg_string(&value));
                }
                "--workload-args" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --workload-args".to_string())?;
                    workload_tune_args.extend(split_arg_string(&value));
                }
                _ => common_tune_args.push(arg),
            }
        }

        token_buckets.sort_unstable();
        token_buckets.dedup();

        Ok(Args {
            mode,
            family,
            model_id,
            base_prompt,
            out_dir,
            token_buckets,
            top_k,
            common_tune_args,
            bench_tune_args,
            workload_tune_args,
        })
    }

    fn manifest_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
    }

    fn cargo_feature_string(args: &[String]) -> &'static str {
        let mut device = "cpu";
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if arg == "--device" {
                if let Some(value) = iter.next() {
                    device = value.split(':').next().unwrap_or("cpu");
                }
                break;
            }
        }

        match device {
            "metal" => "candle,candle-metal",
            "cuda" => "candle,candle-cuda",
            "hip" => "candle,candle-hip",
            _ => "candle",
        }
    }

    fn run_example(example: &str, args: &[String]) -> Result<(), AppError> {
        let status = Command::new("cargo")
            .arg("run")
            .arg("--manifest-path")
            .arg(manifest_path())
            .arg("--features")
            .arg(cargo_feature_string(args))
            .arg("--example")
            .arg(example)
            .arg("--")
            .args(args)
            .status()?;
        if !status.success() {
            return Err(format!("example `{example}` exited with status {status}").into());
        }
        Ok(())
    }

    fn prompt_whitespace_token_count(prompt: &str) -> usize {
        prompt.split_whitespace().count()
    }

    fn tokenizer_token_count(tokenizer: &Tokenizer, prompt: &str) -> Result<usize, AppError> {
        Ok(tokenizer.encode(prompt, true)?.get_ids().len())
    }

    fn repeated_prompt(base_prompt: &str, repeat_count: usize) -> String {
        std::iter::repeat_n(base_prompt, repeat_count)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn prompt_for_token_bucket(
        tokenizer: &Tokenizer,
        base_prompt: &str,
        lower_bound: usize,
        upper_bound: usize,
    ) -> Result<(usize, String, usize), AppError> {
        let mut count_cache = BTreeMap::new();
        let mut token_count_for_repeat = |repeat_count: usize| -> Result<usize, AppError> {
            if let Some(count) = count_cache.get(&repeat_count) {
                return Ok(*count);
            }
            let prompt = repeated_prompt(base_prompt, repeat_count);
            let count = tokenizer_token_count(tokenizer, &prompt)?;
            count_cache.insert(repeat_count, count);
            Ok(count)
        };

        let first_count = token_count_for_repeat(1)?;
        if first_count > upper_bound {
            return Err(format!(
                "base prompt already tokenizes to {first_count} tokens, which does not fit the `{lower_bound}-{upper_bound}` bucket"
            )
            .into());
        }

        let mut low_repeat = 1usize;
        let mut low_count = first_count;
        while low_count < lower_bound {
            low_repeat = low_repeat.saturating_mul(2);
            low_count = token_count_for_repeat(low_repeat)?;
            if low_count > upper_bound {
                break;
            }
        }

        let mut search_low = 1usize;
        let mut search_high = low_repeat;
        if low_count <= upper_bound {
            search_high = low_repeat;
            while token_count_for_repeat(search_high)? <= upper_bound {
                search_low = search_high;
                search_high = search_high.saturating_mul(2);
                if search_high == search_low {
                    break;
                }
            }
        }

        let mut best_repeat = None;
        let mut lo = search_low;
        let mut hi = search_high;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let count = token_count_for_repeat(mid)?;
            if count <= upper_bound {
                if count >= lower_bound {
                    best_repeat = Some((mid, count));
                }
                lo = mid.saturating_add(1);
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }

        let (repeat_count, actual_token_count) = best_repeat.ok_or_else(|| {
            format!(
                "could not build a prompt in the `{lower_bound}-{upper_bound}` token bucket from base prompt `{base_prompt}`"
            )
        })?;
        let prompt = repeated_prompt(base_prompt, repeat_count);
        Ok((repeat_count, prompt, actual_token_count))
    }

    fn read_tune_summary(path: &PathBuf) -> Result<TuneSummary, AppError> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn read_policy_report(path: &PathBuf) -> Result<PolicyReport, AppError> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn mode_tune_args(mode: Mode, args: &Args) -> Vec<String> {
        let mut combined = args.common_tune_args.clone();
        match mode {
            Mode::Bench => combined.extend(args.bench_tune_args.clone()),
            Mode::Workload => combined.extend(args.workload_tune_args.clone()),
            Mode::All => {}
        }
        combined
    }

    fn format_policy(policy: &PolicyKey) -> String {
        format!(
            "pages-{}_bytes-{}_cooldown-{}",
            policy
                .resident_page_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            policy
                .resident_byte_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            policy
                .restore_cooldown_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
        )
    }

    fn effective_bucket_policy(bucket: &BucketSummary) -> (Option<PolicyKey>, Option<f64>) {
        (
            bucket.recommended_policy.clone(),
            bucket.recommended_balanced_score,
        )
    }

    fn markdown_summary(report: &BucketReport) -> String {
        let mut output = String::new();
        output.push_str("# Prompt Bucket Tune\n\n");
        output.push_str(&format!(
            "- mode: `{}`\n- family: `{}`\n- model: `{}`\n- token buckets: `{}`\n- policy shifted across buckets: `{}`\n",
            report.mode,
            report.family,
            report.model_id,
            report
                .token_buckets
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(","),
            report.policy_shifted_across_buckets,
        ));
        if let Some(policy) = &report.stable_recommended_policy {
            output.push_str(&format!(
                "- stable recommended policy: `{}`\n",
                format_policy(policy),
            ));
        }
        if let Some(aggregate) = &report.best_cross_bucket_policy {
            output.push_str(&format!(
                "- best cross-bucket policy: `{}` across `{}` bucket(s)\n",
                format_policy(&aggregate.policy),
                aggregate.bucket_count,
            ));
            if let Some(score) = aggregate.average_balanced_score {
                output.push_str(&format!("- best cross-bucket score: `{score:.3}`\n"));
            }
        }
        output.push_str(
            "\n| Bucket | Tokens | Bench Policy | Workload Policy | Recommended Policy | Score |\n",
        );
        output.push_str("| --- | ---: | --- | --- | --- | ---: |\n");
        for bucket in &report.buckets {
            let bench_policy = bucket
                .bench
                .as_ref()
                .and_then(|summary| summary.recommended_policy.as_ref())
                .map(format_policy)
                .unwrap_or_else(|| "-".to_string());
            let workload_policy = bucket
                .workload
                .as_ref()
                .and_then(|summary| summary.recommended_policy.as_ref())
                .map(format_policy)
                .unwrap_or_else(|| "-".to_string());
            let recommended_policy = bucket
                .recommended_policy
                .as_ref()
                .map(format_policy)
                .unwrap_or_else(|| "-".to_string());
            let score = bucket
                .recommended_balanced_score
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            output.push_str(&format!(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                bucket.token_bucket_label,
                bucket.actual_prompt_token_count,
                bench_policy,
                workload_policy,
                recommended_policy,
                score,
            ));
        }
        output
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    fs::create_dir_all(&args.out_dir)?;

    let artifacts = HfHubModelSource::new()?.snapshot(&args.model_id)?;
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;

    let mut buckets = Vec::new();
    let mut lower_bound = 0usize;
    for upper_bound in &args.token_buckets {
        let next_lower = if lower_bound == 0 { 0 } else { lower_bound + 1 };
        let token_bucket_label = format!("{next_lower}-{upper_bound}");
        let (repeat_count, prompt, actual_prompt_token_count) =
            prompt_for_token_bucket(&tokenizer, &args.base_prompt, next_lower, *upper_bound)?;
        let bucket_dir = args
            .out_dir
            .join(format!("bucket-{}", token_bucket_label.replace('-', "_")));
        fs::create_dir_all(&bucket_dir)?;

        let mut bench = None;
        let mut workload = None;
        let mut cross_regime_report_inputs = Vec::new();

        for execution_mode in args.mode.execution_modes() {
            let regime_dir = bucket_dir.join(execution_mode.as_str());
            fs::create_dir_all(&regime_dir)?;

            let mut tune_args = vec![
                execution_mode.as_str().to_string(),
                args.family.clone(),
                args.model_id.clone(),
                prompt.clone(),
                regime_dir.display().to_string(),
                "--top-k".to_string(),
                args.top_k.to_string(),
            ];
            tune_args.extend(mode_tune_args(execution_mode, &args));
            run_example("hf_policy_tune", &tune_args)?;

            let tune_summary_path = regime_dir.join("tune.json");
            let tune_summary = read_tune_summary(&tune_summary_path)?;
            cross_regime_report_inputs.push(tune_summary.coarse_report_path.clone());
            cross_regime_report_inputs.push(tune_summary.refined_report_path.clone());
            let regime_summary = RegimeBucketSummary {
                mode: execution_mode.as_str().to_string(),
                tune_summary_path: tune_summary_path.display().to_string(),
                policy_report_path: tune_summary.policy_report_path.clone(),
                coarse_best_variant: tune_summary.coarse_best_variant,
                refined_best_variant: tune_summary.refined_best_variant,
                recommended_policy: tune_summary.recommended_policy,
                recommended_balanced_score: tune_summary.recommended_balanced_score,
            };

            match execution_mode {
                Mode::Bench => bench = Some(regime_summary),
                Mode::Workload => workload = Some(regime_summary),
                Mode::All => {}
            }
        }

        let (recommended_policy, recommended_balanced_score, combined_policy_report_path) =
            if cross_regime_report_inputs.len() > 2 {
                let cross_regime_prefix = bucket_dir.join("policy-cross-regime");
                let mut policy_args = cross_regime_report_inputs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                policy_args.push("--out-prefix".to_string());
                policy_args.push(cross_regime_prefix.display().to_string());
                run_example("hf_policy_report", &policy_args)?;
                let policy_report_path =
                    PathBuf::from(format!("{}.json", cross_regime_prefix.display()));
                let policy_report = read_policy_report(&policy_report_path)?;
                let recommended = policy_report
                    .best_complete_policy
                    .or(policy_report.best_partial_policy);
                (
                    recommended
                        .as_ref()
                        .map(|aggregate| aggregate.policy.clone()),
                    recommended
                        .as_ref()
                        .map(|aggregate| aggregate.balanced_score),
                    Some(policy_report_path.display().to_string()),
                )
            } else {
                let single = bench
                    .as_ref()
                    .or(workload.as_ref())
                    .map(|summary| {
                        (
                            summary.recommended_policy.clone(),
                            summary.recommended_balanced_score,
                            Some(summary.policy_report_path.clone()),
                        )
                    })
                    .unwrap_or((None, None, None));
                single
            };

        buckets.push(BucketSummary {
            token_bucket_label,
            token_bucket_lower_bound: next_lower,
            token_bucket_upper_bound: *upper_bound,
            prompt_character_count: prompt.len(),
            prompt_whitespace_token_count: prompt_whitespace_token_count(&prompt),
            actual_prompt_token_count,
            prompt_repeat_count: repeat_count,
            prompt,
            bench,
            workload,
            combined_policy_report_path,
            recommended_policy,
            recommended_balanced_score,
        });
        lower_bound = *upper_bound;
    }

    let mut aggregates: BTreeMap<PolicyKey, (usize, f64, usize)> = BTreeMap::new();
    for bucket in &buckets {
        let (policy, score) = effective_bucket_policy(bucket);
        if let Some(policy) = policy {
            let entry = aggregates.entry(policy).or_insert((0, 0.0, 0));
            entry.0 += 1;
            if let Some(score) = score {
                entry.1 += score;
                entry.2 += 1;
            }
        }
    }

    let unique_policies = buckets
        .iter()
        .filter_map(|bucket| bucket.recommended_policy.clone())
        .collect::<BTreeSet<_>>();
    let stable_recommended_policy = if unique_policies.len() == 1 {
        unique_policies.into_iter().next()
    } else {
        None
    };
    let policy_shifted_across_buckets = stable_recommended_policy.is_none();

    let best_cross_bucket_policy = aggregates
        .into_iter()
        .map(|(policy, (bucket_count, score_sum, score_count))| {
            let average_balanced_score = if score_count == 0 {
                None
            } else {
                Some(score_sum / score_count as f64)
            };
            RecommendedPolicyAggregate {
                policy,
                bucket_count,
                average_balanced_score,
            }
        })
        .max_by(|left, right| {
            left.bucket_count.cmp(&right.bucket_count).then_with(|| {
                left.average_balanced_score
                    .partial_cmp(&right.average_balanced_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

    let report = BucketReport {
        mode: args.mode.as_str().to_string(),
        family: args.family,
        model_id: args.model_id,
        base_prompt: args.base_prompt,
        token_buckets: args.token_buckets,
        policy_shifted_across_buckets,
        stable_recommended_policy,
        best_cross_bucket_policy,
        buckets,
    };

    let report_prefix = args.out_dir.join("bucket-report");
    let report_json_path = PathBuf::from(format!("{}.json", report_prefix.display()));
    let report_md_path = PathBuf::from(format!("{}.md", report_prefix.display()));
    let markdown = markdown_summary(&report);
    fs::write(&report_json_path, serde_json::to_string_pretty(&report)?)?;
    fs::write(&report_md_path, markdown.clone())?;
    print!("{markdown}");
    eprintln!(
        "wrote prompt bucket report to {} and {}",
        report_json_path.display(),
        report_md_path.display()
    );

    Ok(())
}

#[cfg(not(feature = "hf"))]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    Err("hf_prompt_bucket_tune requires the `hf` feature".into())
}
