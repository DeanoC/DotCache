#[cfg(feature = "hf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy)]
    enum Mode {
        Bench,
        Workload,
    }

    impl Mode {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "bench" => Ok(Self::Bench),
                "workload" => Ok(Self::Workload),
                other => Err(format!(
                    "invalid mode `{other}`, expected `bench` or `workload`"
                )),
            }
        }

        fn sweep_example(self) -> &'static str {
            match self {
                Self::Bench => "hf_bench_sweep",
                Self::Workload => "hf_workload_sweep",
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        mode: Mode,
        family: String,
        model_id: String,
        prompt: String,
        out_dir: PathBuf,
        top_k: usize,
        sweep_args: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct ComparisonReport {
        best_throughput_variant: String,
        variants: Vec<VariantReport>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct VariantReport {
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct PolicyKey {
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Clone, Deserialize)]
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
    struct TuneSummary {
        mode: String,
        family: String,
        model_id: String,
        top_k: usize,
        coarse_report_path: String,
        refined_report_path: String,
        policy_report_path: String,
        coarse_best_variant: String,
        refined_best_variant: String,
        recommended_policy: Option<PolicyKey>,
        recommended_balanced_score: Option<f64>,
        refined_page_budgets: Vec<Option<usize>>,
        refined_byte_budgets: Vec<Option<usize>>,
        refined_restore_cooldowns: Vec<Option<u64>>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut args = std::env::args().skip(1);
        let mode = Mode::parse(
            &args.next().ok_or_else(|| {
                "usage: hf_policy_tune <bench|workload> <family> <model_id> <prompt> <out_dir> [--top-k N] [extra sweep args]".to_string()
            })?,
        )?;
        let family = args.next().ok_or_else(|| "missing family".to_string())?;
        let model_id = args.next().ok_or_else(|| "missing model_id".to_string())?;
        let prompt = args.next().ok_or_else(|| "missing prompt".to_string())?;
        let out_dir = args
            .next()
            .ok_or_else(|| "missing out_dir".to_string())
            .map(PathBuf::from)?;

        let mut top_k = 2usize;
        let mut sweep_args = Vec::new();
        while let Some(arg) = args.next() {
            if arg == "--top-k" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --top-k".to_string())?;
                top_k = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --top-k: {err}"))?;
                if top_k == 0 {
                    return Err("--top-k must be at least 1".to_string());
                }
            } else {
                sweep_args.push(arg);
            }
        }

        Ok(Args {
            mode,
            family,
            model_id,
            prompt,
            out_dir,
            top_k,
            sweep_args,
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

    fn run_example(example: &str, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
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

    fn optional_usize_csv(values: &[Option<usize>]) -> String {
        values
            .iter()
            .map(|value| {
                value
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string())
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn optional_u64_csv(values: &[Option<u64>]) -> String {
        values
            .iter()
            .map(|value| {
                value
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string())
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn coarse_sweep_args(args: &Args, out_dir: &PathBuf) -> Vec<String> {
        let mut sweep_args = vec![
            args.family.clone(),
            args.model_id.clone(),
            args.prompt.clone(),
            out_dir.display().to_string(),
        ];
        sweep_args.extend(args.sweep_args.clone());
        sweep_args
    }

    fn neighbor_page_budgets(budget: Option<usize>) -> Vec<Option<usize>> {
        let mut budgets = BTreeSet::new();
        match budget {
            Some(value) => {
                budgets.insert(Some(value));
                if value > 1 {
                    budgets.insert(Some(value - 1));
                }
                budgets.insert(Some(value + 1));
            }
            None => {
                budgets.insert(None);
                budgets.insert(Some(1));
                budgets.insert(Some(2));
            }
        }
        budgets.into_iter().collect()
    }

    fn neighbor_byte_budgets(budget: Option<usize>) -> Vec<Option<usize>> {
        let mut budgets = BTreeSet::new();
        match budget {
            Some(value) => {
                budgets.insert(Some(value));
                budgets.insert(Some(std::cmp::max(1, value / 2)));
                budgets.insert(Some(std::cmp::max(1, value * 3 / 4)));
                budgets.insert(Some(std::cmp::max(1, value * 5 / 4)));
            }
            None => {
                budgets.insert(None);
            }
        }
        budgets.into_iter().collect()
    }

    fn neighbor_cooldowns(value: Option<u64>) -> Vec<Option<u64>> {
        let mut cooldowns = BTreeSet::new();
        match value {
            Some(value) => {
                cooldowns.insert(Some(value));
                cooldowns.insert(Some(std::cmp::max(1, value / 2)));
                cooldowns.insert(Some(value.saturating_mul(2)));
            }
            None => {
                cooldowns.insert(None);
                cooldowns.insert(Some(8));
            }
        }
        cooldowns.into_iter().collect()
    }

    fn refine_budgets(
        report: &ComparisonReport,
        top_k: usize,
    ) -> (Vec<Option<usize>>, Vec<Option<usize>>, Vec<Option<u64>>) {
        let top_variants = report
            .variants
            .iter()
            .take(top_k)
            .cloned()
            .collect::<Vec<_>>();

        let mut page_budgets = BTreeSet::new();
        let mut byte_budgets = BTreeSet::new();
        let mut cooldowns = BTreeSet::new();
        for variant in &top_variants {
            for budget in neighbor_page_budgets(variant.resident_page_budget) {
                page_budgets.insert(budget);
            }
            for budget in neighbor_byte_budgets(variant.resident_byte_budget) {
                byte_budgets.insert(budget);
            }
            for cooldown in neighbor_cooldowns(variant.restore_cooldown_window) {
                cooldowns.insert(cooldown);
            }
        }

        (
            page_budgets.into_iter().collect(),
            byte_budgets.into_iter().collect(),
            cooldowns.into_iter().collect(),
        )
    }

    fn refined_sweep_args(
        args: &Args,
        out_dir: &PathBuf,
        page_budgets: &[Option<usize>],
        byte_budgets: &[Option<usize>],
        cooldowns: &[Option<u64>],
    ) -> Vec<String> {
        let mut filtered_args = Vec::new();
        let mut iter = args.sweep_args.iter();
        while let Some(arg) = iter.next() {
            if matches!(
                arg.as_str(),
                "--resident-page-budgets" | "--resident-byte-budgets" | "--restore-cooldowns"
            ) {
                let _ = iter.next();
                continue;
            }
            filtered_args.push(arg.clone());
        }

        let mut sweep_args = vec![
            args.family.clone(),
            args.model_id.clone(),
            args.prompt.clone(),
            out_dir.display().to_string(),
        ];
        sweep_args.extend(filtered_args);
        sweep_args.push("--resident-page-budgets".to_string());
        sweep_args.push(optional_usize_csv(page_budgets));
        sweep_args.push("--resident-byte-budgets".to_string());
        sweep_args.push(optional_usize_csv(byte_budgets));
        sweep_args.push("--restore-cooldowns".to_string());
        sweep_args.push(optional_u64_csv(cooldowns));
        sweep_args
    }

    fn read_comparison_report(
        path: &PathBuf,
    ) -> Result<ComparisonReport, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn read_policy_report(path: &PathBuf) -> Result<PolicyReport, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn markdown_summary(summary: &TuneSummary) -> String {
        let mut output = String::new();
        output.push_str("# Policy Tune\n\n");
        output.push_str(&format!(
            "- mode: `{}`\n- family: `{}`\n- model: `{}`\n- coarse best: `{}`\n- refined best: `{}`\n",
            summary.mode,
            summary.family,
            summary.model_id,
            summary.coarse_best_variant,
            summary.refined_best_variant,
        ));
        if let Some(policy) = &summary.recommended_policy {
            output.push_str(&format!(
                "- recommended policy: `pages-{}_bytes-{}_cooldown-{}`\n",
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
            ));
        }
        if let Some(score) = summary.recommended_balanced_score {
            output.push_str(&format!("- balanced score: `{score:.3}`\n"));
        }
        output.push_str(&format!(
            "- refined page budgets: `{}`\n- refined byte budgets: `{}`\n- refined cooldowns: `{}`\n",
            optional_usize_csv(&summary.refined_page_budgets),
            optional_usize_csv(&summary.refined_byte_budgets),
            optional_u64_csv(&summary.refined_restore_cooldowns),
        ));
        output
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    fs::create_dir_all(&args.out_dir)?;

    let coarse_dir = args.out_dir.join("coarse");
    fs::create_dir_all(&coarse_dir)?;
    run_example(
        args.mode.sweep_example(),
        &coarse_sweep_args(&args, &coarse_dir),
    )?;
    let coarse_report_prefix = coarse_dir.join("workload-report");
    run_example(
        "hf_workload_report",
        &[
            coarse_dir.display().to_string(),
            "--out-prefix".to_string(),
            coarse_report_prefix.display().to_string(),
        ],
    )?;
    let coarse_report_path = PathBuf::from(format!("{}.json", coarse_report_prefix.display()));
    let coarse_report = read_comparison_report(&coarse_report_path)?;

    let (refined_page_budgets, refined_byte_budgets, refined_restore_cooldowns) =
        refine_budgets(&coarse_report, args.top_k);

    let refined_dir = args.out_dir.join("refined");
    fs::create_dir_all(&refined_dir)?;
    run_example(
        args.mode.sweep_example(),
        &refined_sweep_args(
            &args,
            &refined_dir,
            &refined_page_budgets,
            &refined_byte_budgets,
            &refined_restore_cooldowns,
        ),
    )?;
    let refined_report_prefix = refined_dir.join("workload-report");
    run_example(
        "hf_workload_report",
        &[
            refined_dir.display().to_string(),
            "--out-prefix".to_string(),
            refined_report_prefix.display().to_string(),
        ],
    )?;
    let refined_report_path = PathBuf::from(format!("{}.json", refined_report_prefix.display()));
    let refined_report = read_comparison_report(&refined_report_path)?;

    let policy_prefix = args.out_dir.join("policy");
    run_example(
        "hf_policy_report",
        &[
            coarse_report_path.display().to_string(),
            refined_report_path.display().to_string(),
            "--out-prefix".to_string(),
            policy_prefix.display().to_string(),
        ],
    )?;
    let policy_report_path = PathBuf::from(format!("{}.json", policy_prefix.display()));
    let policy_report = read_policy_report(&policy_report_path)?;

    let recommended = policy_report
        .best_complete_policy
        .or(policy_report.best_partial_policy);
    let summary = TuneSummary {
        mode: match args.mode {
            Mode::Bench => "bench".to_string(),
            Mode::Workload => "workload".to_string(),
        },
        family: args.family,
        model_id: args.model_id,
        top_k: args.top_k,
        coarse_report_path: coarse_report_path.display().to_string(),
        refined_report_path: refined_report_path.display().to_string(),
        policy_report_path: policy_report_path.display().to_string(),
        coarse_best_variant: coarse_report.best_throughput_variant,
        refined_best_variant: refined_report.best_throughput_variant,
        recommended_policy: recommended
            .as_ref()
            .map(|aggregate| aggregate.policy.clone()),
        recommended_balanced_score: recommended
            .as_ref()
            .map(|aggregate| aggregate.balanced_score),
        refined_page_budgets,
        refined_byte_budgets,
        refined_restore_cooldowns,
    };

    let summary_prefix = args.out_dir.join("tune");
    let summary_json_path = PathBuf::from(format!("{}.json", summary_prefix.display()));
    let summary_md_path = PathBuf::from(format!("{}.md", summary_prefix.display()));
    let markdown = markdown_summary(&summary);
    fs::write(&summary_json_path, serde_json::to_string_pretty(&summary)?)?;
    fs::write(&summary_md_path, markdown.clone())?;
    print!("{markdown}");
    eprintln!(
        "wrote tune summary to {} and {}",
        summary_json_path.display(),
        summary_md_path.display()
    );

    Ok(())
}

#[cfg(not(feature = "hf"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("hf_policy_tune requires the `hf` feature".into())
}
