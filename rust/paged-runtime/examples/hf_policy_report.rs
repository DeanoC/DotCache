#[cfg(feature = "hf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};

    #[derive(Debug)]
    struct Args {
        inputs: Vec<PathBuf>,
        out_prefix: Option<PathBuf>,
    }

    #[derive(Debug, Deserialize)]
    struct ComparisonReport {
        model_id: String,
        family: String,
        variants: Vec<VariantReport>,
    }

    #[derive(Debug, Deserialize)]
    struct VariantReport {
        kind: String,
        name: String,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        total_tokens_per_second: f64,
        churn_events_per_generated_token: f64,
        spill_per_generated_token: f64,
        restore_per_generated_token: f64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
    struct PolicyKey {
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct PolicyObservation {
        source_report: String,
        variant_name: String,
        kind: String,
        throughput: f64,
        throughput_ratio: f64,
        churn_per_generated_token: f64,
        churn_ratio: f64,
        spill_per_generated_token: f64,
        restore_per_generated_token: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct PolicyAggregate {
        policy: PolicyKey,
        coverage: usize,
        report_count: usize,
        complete_coverage: bool,
        average_throughput_ratio: f64,
        average_churn_ratio: f64,
        balanced_score: f64,
        observations: Vec<PolicyObservation>,
    }

    #[derive(Debug, Serialize)]
    struct PolicyReport {
        model_id: String,
        family: String,
        report_count: usize,
        best_complete_policy: Option<PolicyAggregate>,
        best_partial_policy: Option<PolicyAggregate>,
        policies: Vec<PolicyAggregate>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut inputs = Vec::new();
        let mut out_prefix = None;
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-prefix" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --out-prefix".to_string())?;
                    out_prefix = Some(PathBuf::from(value));
                }
                _ => inputs.push(PathBuf::from(arg)),
            }
        }

        if inputs.is_empty() {
            return Err(
                "usage: hf_policy_report <report.json|report.md-prefix|dir>... [--out-prefix PATH]"
                    .to_string(),
            );
        }

        Ok(Args { inputs, out_prefix })
    }

    fn default_out_prefix(inputs: &[PathBuf]) -> Option<PathBuf> {
        if inputs.len() != 1 {
            return None;
        }
        let input = &inputs[0];
        if input.is_dir() {
            return Some(input.join("policy-report"));
        }
        if input.extension().and_then(|ext| ext.to_str()) == Some("json") {
            let stem = input.file_stem()?.to_str()?;
            return Some(input.parent()?.join(format!("{stem}.policy")));
        }
        Some(input.clone())
    }

    fn collect_report_paths(
        input: &Path,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if input.is_dir() {
            let mut entries = fs::read_dir(input)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with("report.json"))
                })
                .collect::<Vec<_>>();
            entries.sort();
            out.extend(entries);
            return Ok(());
        }

        if input.exists() {
            let name = input
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if name.ends_with("report.json") {
                out.push(input.to_path_buf());
                return Ok(());
            }
            return Err(format!("unsupported input file `{}`", input.display()).into());
        }

        let json_path = PathBuf::from(format!("{}.json", input.display()));
        if json_path.exists() {
            out.push(json_path);
            return Ok(());
        }

        Err(format!(
            "input `{}` did not resolve to a report json, prefix, or directory",
            input.display()
        )
        .into())
    }

    fn ratio(value: f64, best: f64, invert: bool) -> f64 {
        if best <= 0.0 {
            return if invert { 1.0 } else { 0.0 };
        }
        if invert {
            if value <= 0.0 {
                1.0
            } else {
                (best / value).min(1.0)
            }
        } else {
            (value / best).min(1.0)
        }
    }

    fn markdown_report(report: &PolicyReport) -> String {
        let mut output = String::new();
        output.push_str("# Policy Report\n\n");
        output.push_str(&format!(
            "- model: `{}` ({})\n- input reports: {}\n",
            report.model_id, report.family, report.report_count
        ));
        if let Some(best) = &report.best_complete_policy {
            output.push_str(&format!(
                "- best complete policy: `pages-{}_bytes-{}_cooldown-{}`\n",
                best.policy
                    .resident_page_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                best.policy
                    .resident_byte_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                best.policy
                    .restore_cooldown_window
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            ));
        } else {
            output.push_str("- best complete policy: none shared across all reports\n");
        }
        if let Some(best) = &report.best_partial_policy {
            output.push_str(&format!(
                "- best partial policy: `pages-{}_bytes-{}_cooldown-{}`\n\n",
                best.policy
                    .resident_page_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                best.policy
                    .resident_byte_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                best.policy
                    .restore_cooldown_window
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            ));
        } else {
            output.push('\n');
        }

        output.push_str(
            "| policy | coverage | balanced | avg throughput ratio | avg churn ratio |\n",
        );
        output.push_str("| --- | ---: | ---: | ---: | ---: |\n");
        for policy in &report.policies {
            output.push_str(&format!(
                "| `pages-{}_bytes-{}_cooldown-{}` | {}/{} | {:.3} | {:.3} | {:.3} |\n",
                policy
                    .policy
                    .resident_page_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                policy
                    .policy
                    .resident_byte_budget
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                policy
                    .policy
                    .restore_cooldown_window
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                policy.coverage,
                policy.report_count,
                policy.balanced_score,
                policy.average_throughput_ratio,
                policy.average_churn_ratio,
            ));
        }

        output
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let mut report_paths = Vec::new();
    for input in &args.inputs {
        collect_report_paths(input, &mut report_paths)?;
    }
    report_paths.sort();
    report_paths.dedup();

    if report_paths.is_empty() {
        return Err("no report json files found".into());
    }

    let mut parsed_reports = Vec::with_capacity(report_paths.len());
    for report_path in &report_paths {
        let report: ComparisonReport = serde_json::from_slice(&fs::read(report_path)?)?;
        parsed_reports.push((report_path.display().to_string(), report));
    }

    let model_id = parsed_reports[0].1.model_id.clone();
    let family = parsed_reports[0].1.family.clone();

    let mut aggregate_map: BTreeMap<PolicyKey, Vec<PolicyObservation>> = BTreeMap::new();
    let mut policy_sources: BTreeMap<PolicyKey, BTreeSet<String>> = BTreeMap::new();

    for (report_name, report) in &parsed_reports {
        let best_throughput = report
            .variants
            .iter()
            .map(|variant| variant.total_tokens_per_second)
            .fold(0.0, f64::max);
        let best_churn = report
            .variants
            .iter()
            .map(|variant| variant.churn_events_per_generated_token)
            .fold(f64::INFINITY, f64::min);
        let best_churn = if best_churn.is_infinite() {
            0.0
        } else {
            best_churn
        };

        for variant in &report.variants {
            let policy = PolicyKey {
                resident_page_budget: variant.resident_page_budget,
                resident_byte_budget: variant.resident_byte_budget,
                restore_cooldown_window: variant.restore_cooldown_window,
            };
            let observation = PolicyObservation {
                source_report: report_name.clone(),
                variant_name: variant.name.clone(),
                kind: variant.kind.clone(),
                throughput: variant.total_tokens_per_second,
                throughput_ratio: ratio(variant.total_tokens_per_second, best_throughput, false),
                churn_per_generated_token: variant.churn_events_per_generated_token,
                churn_ratio: ratio(variant.churn_events_per_generated_token, best_churn, true),
                spill_per_generated_token: variant.spill_per_generated_token,
                restore_per_generated_token: variant.restore_per_generated_token,
            };
            aggregate_map
                .entry(policy.clone())
                .or_default()
                .push(observation);
            policy_sources
                .entry(policy)
                .or_default()
                .insert(report_name.clone());
        }
    }

    let report_count = parsed_reports.len();
    let mut policies = aggregate_map
        .into_iter()
        .map(|(policy, observations)| {
            let coverage = policy_sources
                .get(&policy)
                .map(BTreeSet::len)
                .unwrap_or_default();
            let average_throughput_ratio = observations
                .iter()
                .map(|observation| observation.throughput_ratio)
                .sum::<f64>()
                / observations.len() as f64;
            let average_churn_ratio = observations
                .iter()
                .map(|observation| observation.churn_ratio)
                .sum::<f64>()
                / observations.len() as f64;
            let coverage_ratio = coverage as f64 / report_count as f64;
            let balanced_score =
                coverage_ratio * ((average_throughput_ratio + average_churn_ratio) / 2.0);

            PolicyAggregate {
                policy,
                coverage,
                report_count,
                complete_coverage: coverage == report_count,
                average_throughput_ratio,
                average_churn_ratio,
                balanced_score,
                observations,
            }
        })
        .collect::<Vec<_>>();

    policies.sort_by(|lhs, rhs| rhs.balanced_score.total_cmp(&lhs.balanced_score));

    let best_complete_policy = policies
        .iter()
        .filter(|policy| policy.complete_coverage)
        .max_by(|lhs, rhs| lhs.balanced_score.total_cmp(&rhs.balanced_score))
        .cloned();
    let best_partial_policy = policies
        .iter()
        .max_by(|lhs, rhs| lhs.balanced_score.total_cmp(&rhs.balanced_score))
        .cloned();

    let report = PolicyReport {
        model_id,
        family,
        report_count,
        best_complete_policy,
        best_partial_policy,
        policies,
    };

    let markdown = markdown_report(&report);
    print!("{markdown}");

    let out_prefix = args.out_prefix.or_else(|| default_out_prefix(&args.inputs));
    if let Some(out_prefix) = out_prefix {
        let json_path = PathBuf::from(format!("{}.json", out_prefix.display()));
        let md_path = PathBuf::from(format!("{}.md", out_prefix.display()));
        fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
        fs::write(&md_path, markdown)?;
        eprintln!(
            "wrote policy report to {} and {}",
            json_path.display(),
            md_path.display()
        );
    }

    Ok(())
}

#[cfg(not(feature = "hf"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("hf_policy_report requires the `hf` feature".into())
}
