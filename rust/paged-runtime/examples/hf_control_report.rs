#[cfg(feature = "hf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Debug)]
    struct Args {
        rust_inputs: Vec<PathBuf>,
        python_jsonl: PathBuf,
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

    #[derive(Debug, Clone, Copy, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum RustSummaryKind {
        Bench,
        Workload,
    }

    #[derive(Debug, Clone)]
    struct RustSummaryRecord {
        summary_path: PathBuf,
        kind: RustSummaryKind,
        model_id: String,
        family: String,
        device: String,
        runtime_mode: String,
        attention_path: String,
        prompt_token_count: usize,
        total_sessions: usize,
        wave_size: usize,
        decode_rounds_per_wave: usize,
        max_new_tokens: usize,
        warmup_millis: f64,
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
        total_tokens_per_second: f64,
        stage_total_millis: f64,
        stage_qkv_projection_millis: f64,
        stage_kv_append_write_millis: f64,
        stage_layout_prepare_millis: f64,
        stage_attention_score_millis: f64,
        stage_attention_softmax_millis: f64,
        stage_attention_mix_millis: f64,
        stage_output_projection_millis: f64,
        stage_scheduler_planning_millis: f64,
        stage_transfer_millis: f64,
        stage_linear_attention_millis: f64,
        stage_full_attention_millis: f64,
        stage_mlp_millis: f64,
    }

    #[derive(Debug, Clone)]
    struct PythonDenseRecord {
        kind: RustSummaryKind,
        prompt_length: usize,
        total_sessions: usize,
        wave_size: usize,
        decode_rounds_per_wave: usize,
        max_new_tokens: usize,
        warmup_millis: f64,
        prefill_millis: f64,
        decode_millis: f64,
        total_millis: f64,
        total_tokens_per_second: f64,
        stage_qkv_projection_millis: f64,
        stage_kv_append_write_millis: f64,
        stage_output_projection_millis: f64,
        stage_linear_conv_millis: f64,
        stage_linear_attention_millis: f64,
        stage_full_attention_millis: f64,
        stage_mlp_millis: f64,
    }

    #[derive(Debug, Serialize)]
    struct ControlEquivalenceReport {
        python_jsonl_path: String,
        rust_summary_count: usize,
        python_case_count: usize,
        matched_case_count: usize,
        within_twenty_percent_count: usize,
        variants: Vec<ControlVariantReport>,
    }

    #[derive(Debug, Serialize)]
    struct ControlVariantReport {
        summary_path: String,
        kind: RustSummaryKind,
        model_id: String,
        family: String,
        device: String,
        runtime_mode: String,
        attention_path: String,
        prompt_token_count: usize,
        matched_python_prompt_length: Option<usize>,
        within_twenty_percent: bool,
        rust_warmup_millis: f64,
        python_warmup_millis: Option<f64>,
        rust_prefill_millis: f64,
        python_prefill_millis: Option<f64>,
        prefill_delta_millis: Option<f64>,
        prefill_delta_ratio: Option<f64>,
        rust_decode_millis: f64,
        python_decode_millis: Option<f64>,
        decode_delta_millis: Option<f64>,
        decode_delta_ratio: Option<f64>,
        rust_total_millis: f64,
        python_total_millis: Option<f64>,
        total_delta_millis: Option<f64>,
        total_delta_ratio: Option<f64>,
        rust_total_tokens_per_second: f64,
        python_total_tokens_per_second: Option<f64>,
        total_tokens_per_second_delta: Option<f64>,
        total_tokens_per_second_delta_ratio: Option<f64>,
        stage_total_millis: f64,
        stage_qkv_projection_millis: f64,
        stage_kv_append_write_millis: f64,
        stage_layout_prepare_millis: f64,
        stage_attention_score_millis: f64,
        stage_attention_softmax_millis: f64,
        stage_attention_mix_millis: f64,
        stage_output_projection_millis: f64,
        stage_scheduler_planning_millis: f64,
        stage_transfer_millis: f64,
        stage_linear_attention_millis: f64,
        stage_full_attention_millis: f64,
        stage_mlp_millis: f64,
        python_stage_qkv_projection_millis: Option<f64>,
        python_stage_kv_append_write_millis: Option<f64>,
        python_stage_output_projection_millis: Option<f64>,
        python_stage_linear_conv_millis: Option<f64>,
        python_stage_linear_attention_millis: Option<f64>,
        python_stage_full_attention_millis: Option<f64>,
        python_stage_mlp_millis: Option<f64>,
        stage_qkv_projection_delta_ratio: Option<f64>,
        stage_kv_append_write_delta_ratio: Option<f64>,
        stage_output_projection_delta_ratio: Option<f64>,
        stage_linear_attention_delta_ratio: Option<f64>,
        stage_full_attention_delta_ratio: Option<f64>,
        stage_mlp_delta_ratio: Option<f64>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut rust_inputs = Vec::new();
        let mut python_jsonl = None;
        let mut out_prefix = None;
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--python-jsonl" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --python-jsonl".to_string())?;
                    python_jsonl = Some(PathBuf::from(value));
                }
                "--out-prefix" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --out-prefix".to_string())?;
                    out_prefix = Some(PathBuf::from(value));
                }
                _ => rust_inputs.push(PathBuf::from(arg)),
            }
        }

        if rust_inputs.is_empty() {
            return Err(
                "usage: hf_control_report <rust-summary|index|dir|prefix>... --python-jsonl <path> [--out-prefix PATH]"
                    .to_string(),
            );
        }

        let python_jsonl =
            python_jsonl.ok_or_else(|| "missing required --python-jsonl <path>".to_string())?;

        Ok(Args {
            rust_inputs,
            python_jsonl,
            out_prefix,
        })
    }

    fn default_out_prefix(inputs: &[PathBuf]) -> Option<PathBuf> {
        if inputs.len() != 1 {
            return None;
        }
        let input = &inputs[0];
        if input.is_dir() {
            return Some(input.join("control-equivalence-report"));
        }
        if input.extension().and_then(|ext| ext.to_str()) == Some("json") {
            let stem = input.file_stem()?.to_str()?;
            if stem == "index" {
                return Some(input.parent()?.join("control-equivalence-report"));
            }
            let stripped = stem.strip_suffix(".summary").unwrap_or(stem);
            return Some(input.parent()?.join(format!("{stripped}.control-report")));
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
            return Err(format!("unsupported rust input file `{}`", input.display()).into());
        }

        let summary_path = PathBuf::from(format!("{}.summary.json", input.display()));
        if summary_path.exists() {
            out.push(summary_path);
            return Ok(());
        }

        Err(format!(
            "rust input `{}` did not resolve to a summary, index, or directory",
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

    fn optional_str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
        value.get(key).and_then(Value::as_str)
    }

    fn usize_field(value: &Value, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let raw = value
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("missing integer field `{key}`"))?;
        usize::try_from(raw).map_err(|_| format!("field `{key}` overflowed usize").into())
    }

    fn f64_field(value: &Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
        value
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing numeric field `{key}`").into())
    }

    fn optional_f64_field(value: &Value, key: &str) -> f64 {
        value.get(key).and_then(Value::as_f64).unwrap_or(0.0)
    }

    fn python_stage_sum(value: &Value, stage_name: &str) -> f64 {
        optional_f64_field(value, &format!("dense_prefill_stage_{stage_name}_ms"))
            + optional_f64_field(value, &format!("dense_decode_stage_{stage_name}_ms"))
    }

    fn parse_rust_summary(path: &Path) -> Result<RustSummaryRecord, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_slice(&fs::read(path)?)?;
        let kind = if value.get("prompt_token_count").is_some() {
            RustSummaryKind::Bench
        } else {
            RustSummaryKind::Workload
        };
        let prompt_token_count = if value.get("prompt_token_count").is_some() {
            usize_field(&value, "prompt_token_count")?
        } else {
            usize_field(&value, "shared_prompt_token_count")?
        };
        let prefill_millis = if value.get("prefill_millis").is_some() {
            f64_field(&value, "prefill_millis")?
        } else {
            f64_field(&value, "cold_prefix_prefill_millis")?
        };

        Ok(RustSummaryRecord {
            summary_path: path.to_path_buf(),
            kind,
            model_id: str_field(&value, "model_id")?.to_string(),
            family: str_field(&value, "family")?.to_string(),
            device: str_field(&value, "device")?.to_string(),
            runtime_mode: optional_str_field(&value, "runtime_mode")
                .unwrap_or("paged_control")
                .to_string(),
            attention_path: optional_str_field(&value, "attention_path")
                .unwrap_or("paged")
                .to_string(),
            prompt_token_count,
            total_sessions: if matches!(kind, RustSummaryKind::Workload) {
                usize_field(&value, "total_sessions")?
            } else {
                1
            },
            wave_size: if matches!(kind, RustSummaryKind::Workload) {
                usize_field(&value, "wave_size")?
            } else {
                1
            },
            decode_rounds_per_wave: if matches!(kind, RustSummaryKind::Workload) {
                usize_field(&value, "decode_rounds_per_wave")?
            } else {
                0
            },
            max_new_tokens: usize_field(&value, "max_new_tokens")?,
            warmup_millis: optional_f64_field(&value, "warmup_millis"),
            prefill_millis,
            decode_millis: f64_field(&value, "decode_millis")?,
            total_millis: f64_field(&value, "total_millis")?,
            total_tokens_per_second: f64_field(&value, "total_tokens_per_second")?,
            stage_total_millis: optional_f64_field(&value, "stage_total_millis"),
            stage_qkv_projection_millis: optional_f64_field(&value, "stage_qkv_projection_millis"),
            stage_kv_append_write_millis: optional_f64_field(
                &value,
                "stage_kv_append_write_millis",
            ),
            stage_layout_prepare_millis: optional_f64_field(&value, "stage_layout_prepare_millis"),
            stage_attention_score_millis: optional_f64_field(
                &value,
                "stage_attention_score_millis",
            ),
            stage_attention_softmax_millis: optional_f64_field(
                &value,
                "stage_attention_softmax_millis",
            ),
            stage_attention_mix_millis: optional_f64_field(&value, "stage_attention_mix_millis"),
            stage_output_projection_millis: optional_f64_field(
                &value,
                "stage_output_projection_millis",
            ),
            stage_scheduler_planning_millis: optional_f64_field(
                &value,
                "stage_scheduler_planning_millis",
            ),
            stage_transfer_millis: optional_f64_field(&value, "stage_transfer_millis"),
            stage_linear_attention_millis: optional_f64_field(
                &value,
                "stage_linear_attention_millis",
            ),
            stage_full_attention_millis: optional_f64_field(&value, "stage_full_attention_millis"),
            stage_mlp_millis: optional_f64_field(&value, "stage_mlp_millis"),
        })
    }

    fn parse_python_jsonl(
        path: &Path,
    ) -> Result<
        BTreeMap<(String, usize, usize, usize, usize, usize), PythonDenseRecord>,
        Box<dyn std::error::Error>,
    > {
        let mut by_case = BTreeMap::new();
        for (line_index, line) in fs::read_to_string(path)?.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).map_err(|err| {
                format!(
                    "failed to parse python jsonl line {} in {}: {err}",
                    line_index + 1,
                    path.display()
                )
            })?;
            let Some(benchmark) = value.get("benchmark").and_then(Value::as_str) else {
                continue;
            };
            if value
                .get("status")
                .and_then(Value::as_str)
                .is_some_and(|status| status == "error")
            {
                continue;
            }
            let (kind, prompt_length) = match benchmark {
                "qwen35_text" => (
                    RustSummaryKind::Bench,
                    usize::try_from(
                        value
                            .get("prompt_length")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
                    .map_err(|_| "python prompt_length overflowed usize")?,
                ),
                "qwen35_text_workload" => (
                    RustSummaryKind::Workload,
                    usize::try_from(
                        value
                            .get("shared_prompt_token_count")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
                    .map_err(|_| "python shared_prompt_token_count overflowed usize")?,
                ),
                _ => continue,
            };
            if prompt_length == 0 {
                continue;
            }
            let record = match kind {
                RustSummaryKind::Bench => {
                    let generated_token_count = value
                        .get("dense_generated_ids")
                        .and_then(Value::as_array)
                        .map(|ids| ids.len())
                        .unwrap_or_else(|| {
                            let decode_steps = value
                                .get("decode_steps")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            usize::try_from(decode_steps).unwrap_or(0) + 1
                        });
                    let prefill_millis = f64_field(&value, "prefill_ms")?;
                    let decode_steps = value
                        .get("decode_steps")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as f64;
                    let dense_decode_ms_per_step =
                        optional_f64_field(&value, "dense_decode_ms_per_step");
                    let decode_millis = dense_decode_ms_per_step * decode_steps;
                    let total_millis = prefill_millis + decode_millis;
                    let total_tokens_per_second = if total_millis == 0.0 {
                        (prompt_length + generated_token_count) as f64
                    } else {
                        (prompt_length + generated_token_count) as f64 / (total_millis / 1_000.0)
                    };
                    PythonDenseRecord {
                        kind,
                        prompt_length,
                        total_sessions: 1,
                        wave_size: 1,
                        decode_rounds_per_wave: 0,
                        max_new_tokens: usize::try_from(
                            value
                                .get("decode_steps")
                                .and_then(Value::as_u64)
                                .unwrap_or(0),
                        )
                        .unwrap_or(0)
                            + 1,
                        warmup_millis: optional_f64_field(&value, "warmup_ms"),
                        prefill_millis,
                        decode_millis,
                        total_millis,
                        total_tokens_per_second,
                        stage_qkv_projection_millis: python_stage_sum(&value, "qkv_projection"),
                        stage_kv_append_write_millis: python_stage_sum(&value, "kv_append_write"),
                        stage_output_projection_millis: python_stage_sum(
                            &value,
                            "output_projection",
                        ),
                        stage_linear_conv_millis: python_stage_sum(&value, "linear_conv"),
                        stage_linear_attention_millis: python_stage_sum(
                            &value,
                            "linear_attention",
                        ),
                        stage_full_attention_millis: python_stage_sum(&value, "full_attention"),
                        stage_mlp_millis: python_stage_sum(&value, "mlp"),
                    }
                }
                RustSummaryKind::Workload => PythonDenseRecord {
                    kind,
                    prompt_length,
                    total_sessions: usize::try_from(
                        value
                            .get("total_sessions")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
                    .map_err(|_| "python total_sessions overflowed usize")?,
                    wave_size: usize::try_from(
                        value.get("wave_size").and_then(Value::as_u64).unwrap_or(0),
                    )
                    .map_err(|_| "python wave_size overflowed usize")?,
                    decode_rounds_per_wave: usize::try_from(
                        value
                            .get("decode_rounds_per_wave")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
                    .map_err(|_| "python decode_rounds_per_wave overflowed usize")?,
                    max_new_tokens: usize::try_from(
                        value
                            .get("max_new_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
                    .map_err(|_| "python max_new_tokens overflowed usize")?,
                    warmup_millis: optional_f64_field(&value, "warmup_ms"),
                    prefill_millis: optional_f64_field(&value, "cold_prefix_prefill_ms")
                        + optional_f64_field(&value, "seed_suffix_prefill_ms")
                        + optional_f64_field(&value, "attached_suffix_prefill_ms"),
                    decode_millis: optional_f64_field(&value, "decode_ms"),
                    total_millis: f64_field(&value, "total_ms")?,
                    total_tokens_per_second: f64_field(&value, "total_tokens_per_second")?,
                    stage_qkv_projection_millis: python_stage_sum(&value, "qkv_projection"),
                    stage_kv_append_write_millis: python_stage_sum(&value, "kv_append_write"),
                    stage_output_projection_millis: python_stage_sum(
                        &value,
                        "output_projection",
                    ),
                    stage_linear_conv_millis: python_stage_sum(&value, "linear_conv"),
                    stage_linear_attention_millis: python_stage_sum(
                        &value,
                        "linear_attention",
                    ),
                    stage_full_attention_millis: python_stage_sum(&value, "full_attention"),
                    stage_mlp_millis: python_stage_sum(&value, "mlp"),
                },
            };
            by_case.insert(
                (
                    match record.kind {
                        RustSummaryKind::Bench => "bench".to_string(),
                        RustSummaryKind::Workload => "workload".to_string(),
                    },
                    record.prompt_length,
                    record.total_sessions,
                    record.wave_size,
                    record.decode_rounds_per_wave,
                    record.max_new_tokens,
                ),
                record,
            );
        }
        Ok(by_case)
    }

    fn ratio_delta(rust_value: f64, python_value: f64) -> Option<f64> {
        if python_value == 0.0 {
            None
        } else {
            Some((rust_value - python_value) / python_value)
        }
    }

    fn inverse_ratio_delta(rust_value: f64, python_value: f64) -> Option<f64> {
        if python_value == 0.0 {
            None
        } else {
            Some((python_value - rust_value) / python_value)
        }
    }

    fn render_markdown(report: &ControlEquivalenceReport) -> String {
        let mut lines = Vec::new();
        lines.push("# Control Equivalence Report".to_string());
        lines.push(String::new());
        lines.push(format!("- Rust summaries: {}", report.rust_summary_count));
        lines.push(format!(
            "- Python dense cases: {}",
            report.python_case_count
        ));
        lines.push(format!("- Matched cases: {}", report.matched_case_count));
        lines.push(format!(
            "- Within 20%: {}",
            report.within_twenty_percent_count
        ));
        lines.push(String::new());
        lines.push(
            "| Summary | Mode | Prompt Tokens | Matched | Total Δ | Prefill Δ | Decode Δ | TPS Δ |"
                .to_string(),
        );
        lines.push("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |".to_string());
        for variant in &report.variants {
            let matched = if variant.matched_python_prompt_length.is_some() {
                "yes"
            } else {
                "no"
            };
            let total_delta = variant
                .total_delta_ratio
                .map(|value| format!("{:+.1}%", value * 100.0))
                .unwrap_or_else(|| "n/a".to_string());
            let prefill_delta = variant
                .prefill_delta_ratio
                .map(|value| format!("{:+.1}%", value * 100.0))
                .unwrap_or_else(|| "n/a".to_string());
            let decode_delta = variant
                .decode_delta_ratio
                .map(|value| format!("{:+.1}%", value * 100.0))
                .unwrap_or_else(|| "n/a".to_string());
            let tps_delta = variant
                .total_tokens_per_second_delta_ratio
                .map(|value| format!("{:+.1}%", value * 100.0))
                .unwrap_or_else(|| "n/a".to_string());
            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                Path::new(&variant.summary_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(&variant.summary_path),
                variant.runtime_mode,
                variant.prompt_token_count,
                matched,
                total_delta,
                prefill_delta,
                decode_delta,
                tps_delta,
            ));
        }
        lines.push(String::new());
        lines.push("## Stage Deltas".to_string());
        lines.push(String::new());
        lines.push(
            "| Summary | Linear Attn Δ | Full Attn Δ | QKV Δ | KV Append Δ | Out Proj Δ | MLP Δ |"
                .to_string(),
        );
        lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |".to_string());
        for variant in &report.variants {
            let stage = |value: Option<f64>| {
                value
                    .map(|value| format!("{:+.1}%", value * 100.0))
                    .unwrap_or_else(|| "n/a".to_string())
            };
            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} |",
                Path::new(&variant.summary_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(&variant.summary_path),
                stage(variant.stage_linear_attention_delta_ratio),
                stage(variant.stage_full_attention_delta_ratio),
                stage(variant.stage_qkv_projection_delta_ratio),
                stage(variant.stage_kv_append_write_delta_ratio),
                stage(variant.stage_output_projection_delta_ratio),
                stage(variant.stage_mlp_delta_ratio),
            ));
        }
        lines.push(String::new());
        lines.join("\n")
    }

    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;

    let mut rust_summary_paths = Vec::new();
    for input in &args.rust_inputs {
        collect_summary_paths(input, &mut rust_summary_paths)?;
    }
    rust_summary_paths.sort();
    rust_summary_paths.dedup();

    let rust_summaries = rust_summary_paths
        .iter()
        .map(|path| parse_rust_summary(path))
        .collect::<Result<Vec<_>, _>>()?;
    let python_cases = parse_python_jsonl(&args.python_jsonl)?;

    let mut matched_case_count = 0usize;
    let mut within_twenty_percent_count = 0usize;
    let mut variants = Vec::with_capacity(rust_summaries.len());

    for rust in rust_summaries {
        let python = python_cases.get(&(
            match rust.kind {
                RustSummaryKind::Bench => "bench".to_string(),
                RustSummaryKind::Workload => "workload".to_string(),
            },
            rust.prompt_token_count,
            rust.total_sessions,
            rust.wave_size,
            rust.decode_rounds_per_wave,
            rust.max_new_tokens,
        ));
        if python.is_some() {
            matched_case_count += 1;
        }
        let prefill_delta_ratio =
            python.and_then(|python| ratio_delta(rust.prefill_millis, python.prefill_millis));
        let decode_delta_ratio =
            python.and_then(|python| ratio_delta(rust.decode_millis, python.decode_millis));
        let total_delta_ratio =
            python.and_then(|python| ratio_delta(rust.total_millis, python.total_millis));
        let total_tokens_per_second_delta_ratio = python.and_then(|python| {
            inverse_ratio_delta(rust.total_tokens_per_second, python.total_tokens_per_second)
        });
        let stage_qkv_projection_delta_ratio = python.and_then(|python| {
            ratio_delta(
                rust.stage_qkv_projection_millis,
                python.stage_qkv_projection_millis,
            )
        });
        let stage_kv_append_write_delta_ratio = python.and_then(|python| {
            ratio_delta(
                rust.stage_kv_append_write_millis,
                python.stage_kv_append_write_millis,
            )
        });
        let stage_output_projection_delta_ratio = python.and_then(|python| {
            ratio_delta(
                rust.stage_output_projection_millis,
                python.stage_output_projection_millis,
            )
        });
        let stage_linear_attention_delta_ratio = python.and_then(|python| {
            ratio_delta(
                rust.stage_linear_attention_millis,
                python.stage_linear_attention_millis,
            )
        });
        let stage_full_attention_delta_ratio = python.and_then(|python| {
            ratio_delta(
                rust.stage_full_attention_millis,
                python.stage_full_attention_millis,
            )
        });
        let stage_mlp_delta_ratio =
            python.and_then(|python| ratio_delta(rust.stage_mlp_millis, python.stage_mlp_millis));
        let within_twenty_percent = total_delta_ratio
            .map(|ratio| ratio.abs() <= 0.20)
            .unwrap_or(false);
        if within_twenty_percent {
            within_twenty_percent_count += 1;
        }

        variants.push(ControlVariantReport {
            summary_path: rust.summary_path.display().to_string(),
            kind: rust.kind.clone(),
            model_id: rust.model_id.clone(),
            family: rust.family.clone(),
            device: rust.device.clone(),
            runtime_mode: rust.runtime_mode.clone(),
            attention_path: rust.attention_path.clone(),
            prompt_token_count: rust.prompt_token_count,
            matched_python_prompt_length: python.map(|python| python.prompt_length),
            within_twenty_percent,
            rust_warmup_millis: rust.warmup_millis,
            python_warmup_millis: python.map(|python| python.warmup_millis),
            rust_prefill_millis: rust.prefill_millis,
            python_prefill_millis: python.map(|python| python.prefill_millis),
            prefill_delta_millis: python.map(|python| rust.prefill_millis - python.prefill_millis),
            prefill_delta_ratio,
            rust_decode_millis: rust.decode_millis,
            python_decode_millis: python.map(|python| python.decode_millis),
            decode_delta_millis: python.map(|python| rust.decode_millis - python.decode_millis),
            decode_delta_ratio,
            rust_total_millis: rust.total_millis,
            python_total_millis: python.map(|python| python.total_millis),
            total_delta_millis: python.map(|python| rust.total_millis - python.total_millis),
            total_delta_ratio,
            rust_total_tokens_per_second: rust.total_tokens_per_second,
            python_total_tokens_per_second: python.map(|python| python.total_tokens_per_second),
            total_tokens_per_second_delta: python
                .map(|python| rust.total_tokens_per_second - python.total_tokens_per_second),
            total_tokens_per_second_delta_ratio,
            stage_total_millis: rust.stage_total_millis,
            stage_qkv_projection_millis: rust.stage_qkv_projection_millis,
            stage_kv_append_write_millis: rust.stage_kv_append_write_millis,
            stage_layout_prepare_millis: rust.stage_layout_prepare_millis,
            stage_attention_score_millis: rust.stage_attention_score_millis,
            stage_attention_softmax_millis: rust.stage_attention_softmax_millis,
            stage_attention_mix_millis: rust.stage_attention_mix_millis,
            stage_output_projection_millis: rust.stage_output_projection_millis,
            stage_scheduler_planning_millis: rust.stage_scheduler_planning_millis,
            stage_transfer_millis: rust.stage_transfer_millis,
            stage_linear_attention_millis: rust.stage_linear_attention_millis,
            stage_full_attention_millis: rust.stage_full_attention_millis,
            stage_mlp_millis: rust.stage_mlp_millis,
            python_stage_qkv_projection_millis: python
                .map(|python| python.stage_qkv_projection_millis),
            python_stage_kv_append_write_millis: python
                .map(|python| python.stage_kv_append_write_millis),
            python_stage_output_projection_millis: python
                .map(|python| python.stage_output_projection_millis),
            python_stage_linear_conv_millis: python.map(|python| python.stage_linear_conv_millis),
            python_stage_linear_attention_millis: python
                .map(|python| python.stage_linear_attention_millis),
            python_stage_full_attention_millis: python
                .map(|python| python.stage_full_attention_millis),
            python_stage_mlp_millis: python.map(|python| python.stage_mlp_millis),
            stage_qkv_projection_delta_ratio,
            stage_kv_append_write_delta_ratio,
            stage_output_projection_delta_ratio,
            stage_linear_attention_delta_ratio,
            stage_full_attention_delta_ratio,
            stage_mlp_delta_ratio,
        });
    }

    variants.sort_by(|lhs, rhs| {
        lhs.prompt_token_count
            .cmp(&rhs.prompt_token_count)
            .then_with(|| lhs.runtime_mode.cmp(&rhs.runtime_mode))
            .then_with(|| lhs.summary_path.cmp(&rhs.summary_path))
    });

    let report = ControlEquivalenceReport {
        python_jsonl_path: args.python_jsonl.display().to_string(),
        rust_summary_count: variants.len(),
        python_case_count: python_cases.len(),
        matched_case_count,
        within_twenty_percent_count,
        variants,
    };

    let out_prefix = args
        .out_prefix
        .or_else(|| default_out_prefix(&args.rust_inputs))
        .unwrap_or_else(|| PathBuf::from("control-equivalence-report"));
    let json_path = out_prefix.with_extension("json");
    let md_path = out_prefix.with_extension("md");
    fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
    fs::write(&md_path, render_markdown(&report))?;

    println!(
        "control report: rust_summaries={} python_cases={} matched={} within_20pct={} json={} md={}",
        report.rust_summary_count,
        report.python_case_count,
        report.matched_case_count,
        report.within_twenty_percent_count,
        json_path.display(),
        md_path.display(),
    );
    Ok(())
}

#[cfg(not(feature = "hf"))]
fn main() {
    eprintln!("enable the `hf` feature to run this example");
}
