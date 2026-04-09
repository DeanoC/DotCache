#[cfg(feature = "hf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Debug)]
    struct Args {
        inputs: Vec<PathBuf>,
        out_prefix: Option<PathBuf>,
    }

    #[derive(Debug, Deserialize)]
    struct SweepIndex {
        variants: Vec<SweepVariantSummary>,
    }

    #[derive(Debug, Deserialize)]
    struct SweepVariantSummary {
        summary_path: String,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum ReportKind {
        Workload,
        Bench,
    }

    #[derive(Debug, Serialize)]
    struct ComparisonReport {
        summary_count: usize,
        kinds: Vec<ReportKind>,
        model_id: String,
        family: String,
        best_throughput_variant: String,
        best_latency_variant: String,
        lowest_spill_variant: String,
        lowest_restore_variant: String,
        lowest_churn_variant: String,
        variants: Vec<VariantReport>,
    }

    #[derive(Debug, Serialize)]
    struct VariantReport {
        kind: ReportKind,
        name: String,
        summary_path: String,
        trace_jsonl_path: String,
        total_sessions: Option<usize>,
        batch_size: Option<usize>,
        wave_size: Option<usize>,
        decode_rounds_per_wave: Option<usize>,
        stress_mode: Option<bool>,
        stress_suffix_repeats: Option<usize>,
        resident_page_budget: Option<usize>,
        resident_byte_budget: Option<usize>,
        restore_cooldown_window: Option<u64>,
        total_input_token_count: usize,
        total_generated_token_count: usize,
        total_millis: f64,
        total_tokens_per_second: f64,
        request_prefill_tokens_per_second: f64,
        decode_tokens_per_second: f64,
        spill_count: usize,
        restore_count: usize,
        cooldown_hit_count: usize,
        spilled_bytes: usize,
        restored_bytes: usize,
        spill_per_generated_token: f64,
        restore_per_generated_token: f64,
        churn_events_per_generated_token: f64,
        spilled_bytes_per_generated_token: f64,
        restored_bytes_per_generated_token: f64,
        prefill_millis: f64,
        decode_millis: f64,
        prefill_share: f64,
        decode_share: f64,
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
                "usage: hf_workload_report <summary|index|dir|prefix>... [--out-prefix PATH]"
                    .to_string(),
            );
        }

        Ok(Args { inputs, out_prefix })
    }

    fn normalized_rate(count: usize, total_generated_token_count: usize) -> f64 {
        if total_generated_token_count == 0 {
            count as f64
        } else {
            count as f64 / total_generated_token_count as f64
        }
    }

    fn summary_name(summary_path: &Path) -> String {
        let filename = summary_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("summary.json");
        filename
            .strip_suffix(".summary.json")
            .or_else(|| filename.strip_suffix(".json"))
            .unwrap_or(filename)
            .to_string()
    }

    fn default_out_prefix(inputs: &[PathBuf]) -> Option<PathBuf> {
        if inputs.len() != 1 {
            return None;
        }
        let input = &inputs[0];
        if input.is_dir() {
            return Some(input.join("workload-report"));
        }
        if input.extension().and_then(|ext| ext.to_str()) == Some("json") {
            let stem = input.file_stem()?.to_str()?;
            if stem == "index" {
                return Some(input.parent()?.join("workload-report"));
            }
            let stripped = stem.strip_suffix(".summary").unwrap_or(stem);
            return Some(input.parent()?.join(format!("{stripped}.report")));
        }

        Some(input.clone())
    }

    fn collect_summary_paths(
        input: &Path,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if input.is_dir() {
            let mut entries = fs::read_dir(input)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".summary.json"))
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
            if name.ends_with(".summary.json") {
                out.push(input.to_path_buf());
                return Ok(());
            }
            if name == "index.json" {
                let index: SweepIndex = serde_json::from_slice(&fs::read(input)?)?;
                let base_dir = input.parent().unwrap_or_else(|| Path::new("."));
                out.extend(index.variants.into_iter().map(|variant| {
                    let path = PathBuf::from(variant.summary_path);
                    if path.is_absolute() {
                        path
                    } else {
                        base_dir.join(path)
                    }
                }));
                return Ok(());
            }
            return Err(format!("unsupported input file `{}`", input.display()).into());
        }

        let summary_path = PathBuf::from(format!("{}.summary.json", input.display()));
        if summary_path.exists() {
            out.push(summary_path);
            return Ok(());
        }

        Err(format!(
            "input `{}` did not resolve to a summary, index, or directory",
            input.display()
        )
        .into())
    }

    fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
        value
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing string field `{key}`").into())
    }

    fn usize_field(value: &Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let raw = value
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("missing integer field `{key}`"))?;
        usize::try_from(raw).map_err(|_| format!("field `{key}` overflowed usize").into())
    }

    fn optional_usize_field(
        value: &Value,
        key: &str,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        match value.get(key) {
            Some(Value::Null) | None => Ok(None),
            Some(other) => {
                let raw = other
                    .as_u64()
                    .ok_or_else(|| format!("field `{key}` was not an integer"))?;
                Ok(Some(
                    usize::try_from(raw).map_err(|_| format!("field `{key}` overflowed usize"))?,
                ))
            }
        }
    }

    fn optional_u64_field(
        value: &Value,
        key: &str,
    ) -> Result<Option<u64>, Box<dyn std::error::Error>> {
        match value.get(key) {
            Some(Value::Null) | None => Ok(None),
            Some(other) => other
                .as_u64()
                .map(Some)
                .ok_or_else(|| format!("field `{key}` was not an integer").into()),
        }
    }

    fn optional_bool_field(
        value: &Value,
        key: &str,
    ) -> Result<Option<bool>, Box<dyn std::error::Error>> {
        match value.get(key) {
            Some(Value::Null) | None => Ok(None),
            Some(other) => other
                .as_bool()
                .map(Some)
                .ok_or_else(|| format!("field `{key}` was not a bool").into()),
        }
    }

    fn f64_field(value: &Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
        value
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing float field `{key}`").into())
    }

    fn report_kind(value: &Value) -> Result<ReportKind, Box<dyn std::error::Error>> {
        if value.get("shared_prompt").is_some() || value.get("total_sessions").is_some() {
            return Ok(ReportKind::Workload);
        }
        if value.get("prompt").is_some() {
            return Ok(ReportKind::Bench);
        }
        Err("could not infer summary kind".into())
    }

    fn variant_from_summary(
        summary_path: &Path,
        value: &Value,
    ) -> Result<VariantReport, Box<dyn std::error::Error>> {
        let kind = report_kind(value)?;
        let total_generated_token_count = usize_field(value, "total_generated_token_count")?;
        let spill_count = usize_field(value, "spill_count")?;
        let restore_count = usize_field(value, "restore_count")?;
        let spilled_bytes = usize_field(value, "spilled_bytes")?;
        let restored_bytes = usize_field(value, "restored_bytes")?;
        let total_millis = f64_field(value, "total_millis")?;
        let decode_millis = f64_field(value, "decode_millis")?;
        let prefill_millis = match kind {
            ReportKind::Workload => {
                f64_field(value, "cold_prefix_prefill_millis")?
                    + f64_field(value, "seed_suffix_prefill_millis")?
                    + f64_field(value, "attached_suffix_prefill_millis")?
            }
            ReportKind::Bench => f64_field(value, "prefill_millis")?,
        };
        let prefill_share = if total_millis == 0.0 {
            0.0
        } else {
            prefill_millis / total_millis
        };
        let decode_share = if total_millis == 0.0 {
            0.0
        } else {
            decode_millis / total_millis
        };

        let total_sessions = match kind {
            ReportKind::Workload => Some(usize_field(value, "total_sessions")?),
            ReportKind::Bench => Some(usize_field(value, "session_count")?),
        };

        Ok(VariantReport {
            kind,
            name: summary_name(summary_path),
            summary_path: summary_path.display().to_string(),
            trace_jsonl_path: str_field(value, "trace_jsonl_path")?.to_string(),
            total_sessions,
            batch_size: optional_usize_field(value, "batch_size")?,
            wave_size: optional_usize_field(value, "wave_size")?,
            decode_rounds_per_wave: optional_usize_field(value, "decode_rounds_per_wave")?,
            stress_mode: optional_bool_field(value, "stress_mode")?,
            stress_suffix_repeats: optional_usize_field(value, "stress_suffix_repeats")?,
            resident_page_budget: optional_usize_field(value, "resident_page_budget")?,
            resident_byte_budget: optional_usize_field(value, "resident_byte_budget")?,
            restore_cooldown_window: optional_u64_field(value, "restore_cooldown_window")?,
            total_input_token_count: usize_field(value, "total_input_token_count")?,
            total_generated_token_count,
            total_millis,
            total_tokens_per_second: f64_field(value, "total_tokens_per_second")?,
            request_prefill_tokens_per_second: f64_field(value, "prefill_tokens_per_second")
                .or_else(|_| f64_field(value, "request_prefill_tokens_per_second"))?,
            decode_tokens_per_second: f64_field(value, "decode_tokens_per_second")?,
            spill_count,
            restore_count,
            cooldown_hit_count: usize_field(value, "cooldown_hit_count")?,
            spilled_bytes,
            restored_bytes,
            spill_per_generated_token: normalized_rate(spill_count, total_generated_token_count),
            restore_per_generated_token: normalized_rate(
                restore_count,
                total_generated_token_count,
            ),
            churn_events_per_generated_token: normalized_rate(
                spill_count + restore_count,
                total_generated_token_count,
            ),
            spilled_bytes_per_generated_token: normalized_rate(
                spilled_bytes,
                total_generated_token_count,
            ),
            restored_bytes_per_generated_token: normalized_rate(
                restored_bytes,
                total_generated_token_count,
            ),
            prefill_millis,
            decode_millis,
            prefill_share,
            decode_share,
        })
    }

    fn markdown_report(report: &ComparisonReport) -> String {
        let mut output = String::new();
        output.push_str("# Benchmark Report\n\n");
        output.push_str(&format!(
            "- model: `{}` ({})\n- summaries: {}\n- best throughput: `{}`\n- best latency: `{}`\n- lowest spill pressure: `{}`\n- lowest restore pressure: `{}`\n- lowest overall churn: `{}`\n\n",
            report.model_id,
            report.family,
            report.summary_count,
            report.best_throughput_variant,
            report.best_latency_variant,
            report.lowest_spill_variant,
            report.lowest_restore_variant,
            report.lowest_churn_variant,
        ));
        output.push_str("| variant | kind | tps | ms | spill/token | restore/token | churn/token | sessions | batch | page budget | byte budget | cooldown |\n");
        output.push_str(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        );
        for variant in &report.variants {
            let kind = match variant.kind {
                ReportKind::Workload => "workload",
                ReportKind::Bench => "bench",
            };
            let total_sessions = variant
                .total_sessions
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string());
            let batch_size = variant
                .batch_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string());
            let page_budget = variant
                .resident_page_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string());
            let byte_budget = variant
                .resident_byte_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string());
            let cooldown = variant
                .restore_cooldown_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string());
            output.push_str(&format!(
                "| `{}` | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {} | {} | {} | {} |\n",
                variant.name,
                kind,
                variant.total_tokens_per_second,
                variant.total_millis,
                variant.spill_per_generated_token,
                variant.restore_per_generated_token,
                variant.churn_events_per_generated_token,
                total_sessions,
                batch_size,
                page_budget,
                byte_budget,
                cooldown,
            ));
        }
        output
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let mut summary_paths = Vec::new();
    for input in &args.inputs {
        collect_summary_paths(input, &mut summary_paths)?;
    }
    summary_paths.sort();
    summary_paths.dedup();

    if summary_paths.is_empty() {
        return Err("no summaries found".into());
    }

    let first_value: Value = serde_json::from_slice(&fs::read(&summary_paths[0])?)?;
    let model_id = str_field(&first_value, "model_id")?.to_string();
    let family = str_field(&first_value, "family")?.to_string();

    let mut variants = Vec::with_capacity(summary_paths.len());
    let mut kinds = Vec::new();
    for summary_path in &summary_paths {
        let value: Value = serde_json::from_slice(&fs::read(summary_path)?)?;
        let variant = variant_from_summary(summary_path, &value)?;
        if !kinds
            .iter()
            .any(|kind| std::mem::discriminant(kind) == std::mem::discriminant(&variant.kind))
        {
            kinds.push(variant.kind.clone());
        }
        variants.push(variant);
    }

    variants.sort_by(|lhs, rhs| {
        rhs.total_tokens_per_second
            .total_cmp(&lhs.total_tokens_per_second)
    });

    let best_throughput_variant = variants
        .iter()
        .max_by(|lhs, rhs| {
            lhs.total_tokens_per_second
                .total_cmp(&rhs.total_tokens_per_second)
        })
        .map(|variant| variant.name.clone())
        .unwrap();
    let best_latency_variant = variants
        .iter()
        .min_by(|lhs, rhs| lhs.total_millis.total_cmp(&rhs.total_millis))
        .map(|variant| variant.name.clone())
        .unwrap();
    let lowest_spill_variant = variants
        .iter()
        .min_by(|lhs, rhs| {
            lhs.spill_per_generated_token
                .total_cmp(&rhs.spill_per_generated_token)
        })
        .map(|variant| variant.name.clone())
        .unwrap();
    let lowest_restore_variant = variants
        .iter()
        .min_by(|lhs, rhs| {
            lhs.restore_per_generated_token
                .total_cmp(&rhs.restore_per_generated_token)
        })
        .map(|variant| variant.name.clone())
        .unwrap();
    let lowest_churn_variant = variants
        .iter()
        .min_by(|lhs, rhs| {
            lhs.churn_events_per_generated_token
                .total_cmp(&rhs.churn_events_per_generated_token)
        })
        .map(|variant| variant.name.clone())
        .unwrap();

    let report = ComparisonReport {
        summary_count: variants.len(),
        kinds,
        model_id,
        family,
        best_throughput_variant,
        best_latency_variant,
        lowest_spill_variant,
        lowest_restore_variant,
        lowest_churn_variant,
        variants,
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
            "wrote benchmark report to {} and {}",
            json_path.display(),
            md_path.display()
        );
    }
    Ok(())
}

#[cfg(not(feature = "hf"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("hf_workload_report requires the `hf` feature".into())
}
