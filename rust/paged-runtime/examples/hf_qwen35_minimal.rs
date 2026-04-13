#[cfg(feature = "qwen35-minimal")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::time::Instant;

    use candle_core::{DType, Device, IndexOp, Tensor};
    use dotcache_paged_runtime::{
        MinimalQwen35LoadMode, MinimalQwen35Runner, Result, RuntimeError,
    };
    use tokenizers::Tokenizer;

    #[derive(Clone, Debug)]
    enum DeviceSelector {
        Cpu,
        Cuda(usize),
        Hip(usize),
    }

    impl std::fmt::Display for DeviceSelector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Cpu => f.write_str("cpu"),
                Self::Cuda(ordinal) => write!(f, "cuda:{ordinal}"),
                Self::Hip(ordinal) => write!(f, "hip:{ordinal}"),
            }
        }
    }

    impl std::str::FromStr for DeviceSelector {
        type Err = RuntimeError;

        fn from_str(value: &str) -> Result<Self> {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized == "cpu" {
                return Ok(Self::Cpu);
            }
            if let Some(rest) = normalized.strip_prefix("cuda") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| RuntimeError::External {
                        context: "device",
                        message: format!("invalid cuda device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Cuda(ordinal));
            }
            if let Some(rest) = normalized.strip_prefix("hip") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| RuntimeError::External {
                        context: "device",
                        message: format!("invalid hip device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Hip(ordinal));
            }
            Err(RuntimeError::External {
                context: "device",
                message: format!(
                    "unsupported device `{value}`, expected cpu, cuda[:ordinal], or hip[:ordinal]"
                ),
            })
        }
    }

    impl DeviceSelector {
        fn resolve(&self) -> Result<Device> {
            match self {
                Self::Cpu => Ok(Device::Cpu),
                Self::Cuda(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-cuda")]
                    {
                        Ok(Device::new_cuda(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-cuda"))]
                    {
                        Err(RuntimeError::BackendUnavailable {
                            backend: "cuda",
                            device: format!("cuda:{ordinal}"),
                        })
                    }
                }
                Self::Hip(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-hip")]
                    {
                        Ok(Device::new_hip(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-hip"))]
                    {
                        Err(RuntimeError::BackendUnavailable {
                            backend: "hip",
                            device: format!("hip:{ordinal}"),
                        })
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum LoadMode {
        Native,
        Direct,
        HipDirect,
    }

    impl LoadMode {
        fn runner_mode(self) -> Option<MinimalQwen35LoadMode> {
            match self {
                Self::Native => Some(MinimalQwen35LoadMode::NativeStore),
                Self::Direct => None,
                Self::HipDirect => Some(MinimalQwen35LoadMode::HipDirect),
            }
        }
    }

    impl std::str::FromStr for LoadMode {
        type Err = RuntimeError;

        fn from_str(value: &str) -> Result<Self> {
            match value.trim().to_ascii_lowercase().as_str() {
                "native" => Ok(Self::Native),
                "direct" => Ok(Self::Direct),
                "hip-direct" | "direct-hip" => Ok(Self::HipDirect),
                other => Err(RuntimeError::External {
                    context: "load-mode",
                    message: format!(
                        "unsupported load mode `{other}`, expected native, direct, or hip-direct"
                    ),
                }),
            }
        }
    }

    fn argmax_last_token(logits: &Tensor) -> Result<u32> {
        let last_token = match logits.dims() {
            [1, _vocab] => logits.i(0)?,
            [1, seq, _vocab] => logits.i((0, seq - 1))?,
            dims => {
                return Err(RuntimeError::External {
                    context: "qwen35-minimal-example",
                    message: format!("unexpected logits shape {dims:?}"),
                });
            }
        };
        let values = last_token
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut best: Option<(usize, f32)> = None;
        let mut nan_count = 0usize;
        for (index, value) in values.iter().copied().enumerate() {
            if value.is_nan() {
                nan_count += 1;
                continue;
            }
            match best {
                Some((_, best_value)) if value <= best_value => {}
                _ => best = Some((index, value)),
            }
        }
        let (index, _) = best.ok_or_else(|| RuntimeError::External {
            context: "last-token logits",
            message: format!(
                "all logits were NaN for shape {:?} ({} values)",
                logits.dims(),
                nan_count
            ),
        })?;
        if nan_count > 0 {
            eprintln!(
                "warning: skipped {nan_count} NaN logits when computing argmax for shape {:?}",
                logits.dims()
            );
        }
        Ok(index as u32)
    }

    fn max_logit_delta(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
        let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if lhs.len() != rhs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "logit delta",
                expected: lhs.len(),
                got: rhs.len(),
            });
        }
        let mut max_delta = 0.0f32;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            max_delta = max_delta.max((lhs - rhs).abs());
        }
        Ok(max_delta)
    }

    fn logit_nan_count(logits: &Tensor) -> Result<usize> {
        let values = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(values.iter().filter(|value| value.is_nan()).count())
    }

    fn report_linear_nan_trace(runner: &mut MinimalQwen35Runner, input_ids: &Tensor) -> Result<()> {
        for layer_id in runner.linear_attention_layer_ids() {
            let trace = runner.trace_linear_attention_layer(input_ids, layer_id, 0)?;
            let output_nans = logit_nan_count(trace.layer_output.tensor())?;
            let state_nans = logit_nan_count(trace.recurrent_state.tensor())?;
            if output_nans > 0 || state_nans > 0 {
                eprintln!(
                    "warning: linear layer {layer_id} emitted NaNs output={} recurrent_state={}",
                    output_nans, state_nans
                );
                return Ok(());
            }
        }
        eprintln!("warning: no linear layer trace emitted NaNs despite NaN prefill logits");
        Ok(())
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn print_hip_counters(label: &str) {
        let counters = candle_core::hip::transfer_counters();
        eprintln!(
            "hip_transfers[{label}] h2d={} d2h={} d2d={}",
            counters.host_to_device_bytes,
            counters.device_to_host_bytes,
            counters.device_to_device_bytes,
        );
    }

    let mut args = std::env::args().skip(1);
    let model_id = args.next().ok_or(
        "usage: hf_qwen35_minimal <model_id> <prompt> [max_new_tokens] [--device cpu|cuda[:ordinal]|hip[:ordinal]] [--load-mode native|direct|hip-direct] [--device-only]",
    )?;
    let prompt = args.next().ok_or("missing prompt")?;
    let mut positional = Vec::new();
    let mut device_selector = DeviceSelector::Cpu;
    let mut load_mode = LoadMode::Native;
    let mut device_only = false;
    while let Some(arg) = args.next() {
        if arg == "--device" {
            let value = args.next().ok_or("missing value for --device")?;
            device_selector = value.parse()?;
        } else if arg == "--load-mode" {
            let value = args.next().ok_or("missing value for --load-mode")?;
            load_mode = value.parse()?;
        } else if arg == "--device-only" {
            device_only = true;
        } else {
            positional.push(arg);
        }
    }
    let max_new_tokens = positional
        .first()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);

    let cpu_device = Device::Cpu;
    let target_device = device_selector.resolve()?;
    let (mut cpu_runner, cpu_load_elapsed) = if device_only {
        (None, std::time::Duration::ZERO)
    } else {
        let cpu_load_started = Instant::now();
        let cpu_runner = match load_mode.runner_mode() {
            Some(mode) => MinimalQwen35Runner::load_with_mode(&model_id, &cpu_device, mode)?,
            None => MinimalQwen35Runner::load_from_hf_direct_f16(&model_id, &cpu_device)?,
        };
        (Some(cpu_runner), cpu_load_started.elapsed())
    };

    let device_load_started = Instant::now();
    let mut device_runner = match load_mode.runner_mode() {
        Some(mode) => MinimalQwen35Runner::load_with_mode(&model_id, &target_device, mode)?,
        None => MinimalQwen35Runner::load_from_hf_direct_f16(&model_id, &target_device)?,
    };
    let device_load_elapsed = device_load_started.elapsed();
    let tokenizer = Tokenizer::from_file(&device_runner.weights.tokenizer_path)?;
    let prompt_ids = tokenizer.encode(prompt.as_str(), true)?.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(RuntimeError::EmptyInput { context: "prompt" }.into());
    }

    let input_ids = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &cpu_device)?;
    let hidden_states = if let Some(cpu_runner) = cpu_runner.as_ref() {
        cpu_runner.hidden_states_from_input_ids(&input_ids)?
    } else {
        device_runner.hidden_states_from_input_ids_direct(&input_ids)?
    };

    let (mut cpu_logits, mut cpu_cache, cpu_prefill_elapsed) =
        if let Some(cpu_runner) = cpu_runner.as_mut() {
            let cpu_prefill_started = Instant::now();
            let (cpu_logits, cpu_cache) = cpu_runner.prefill_from_hidden_states(&hidden_states)?;
            (
                Some(cpu_logits),
                Some(cpu_cache),
                cpu_prefill_started.elapsed(),
            )
        } else {
            (None, None, std::time::Duration::ZERO)
        };

    #[cfg(feature = "qwen35-minimal-hip")]
    if target_device.is_hip() {
        candle_core::hip::reset_transfer_counters();
    }
    let device_prefill_started = Instant::now();
    let (mut device_logits, mut device_cache) =
        device_runner.prefill_from_hidden_states(&hidden_states)?;
    let device_prefill_elapsed = device_prefill_started.elapsed();
    #[cfg(feature = "qwen35-minimal-hip")]
    if target_device.is_hip()
        && matches!(
            std::env::var("DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    {
        print_hip_counters("prefill");
    }
    let cpu_prefill_nans = match cpu_logits.as_ref() {
        Some(cpu_logits) => logit_nan_count(cpu_logits.tensor())?,
        None => 0,
    };
    let device_prefill_nans = logit_nan_count(device_logits.tensor())?;
    if cpu_prefill_nans > 0 || device_prefill_nans > 0 {
        eprintln!(
            "warning: prefill logits contain NaNs cpu={} device={}",
            cpu_prefill_nans, device_prefill_nans
        );
        if let Some(cpu_runner) = cpu_runner.as_mut() {
            report_linear_nan_trace(cpu_runner, &input_ids)?;
        }
    }

    let prefill_delta = match cpu_logits.as_ref() {
        Some(cpu_logits) => max_logit_delta(cpu_logits.tensor(), device_logits.tensor())?,
        None => f32::NAN,
    };
    let mut generated_ids = prompt_ids.clone();
    let mut max_decode_delta = if device_only { f32::NAN } else { 0.0f32 };
    let mut cpu_decode_elapsed = std::time::Duration::ZERO;
    let mut device_decode_elapsed = std::time::Duration::ZERO;
    let mut next_token = match cpu_logits.as_ref() {
        Some(cpu_logits) => argmax_last_token(cpu_logits.tensor())?,
        None => argmax_last_token(device_logits.tensor())?,
    };
    for _ in 0..max_new_tokens {
        generated_ids.push(next_token);

        let decode_input = Tensor::from_vec(vec![next_token], (1, 1), &cpu_device)?;
        let device_hidden_state = device_runner.hidden_states_from_input_ids_direct(&decode_input)?;
        if let (Some(cpu_runner), Some(cpu_cache)) = (cpu_runner.as_mut(), cpu_cache.as_mut()) {
            let cpu_hidden_state = cpu_runner.hidden_states_from_input_ids(&decode_input)?;
            let cpu_decode_started = Instant::now();
            cpu_logits = Some(cpu_runner.decode_from_hidden_state(&cpu_hidden_state, cpu_cache)?);
            cpu_decode_elapsed += cpu_decode_started.elapsed();
        }

        #[cfg(feature = "qwen35-minimal-hip")]
        if target_device.is_hip() {
            candle_core::hip::reset_transfer_counters();
        }
        let device_decode_started = Instant::now();
        device_logits =
            device_runner.decode_from_hidden_state(&device_hidden_state, &mut device_cache)?;
        device_decode_elapsed += device_decode_started.elapsed();
        #[cfg(feature = "qwen35-minimal-hip")]
        if target_device.is_hip()
            && matches!(
                std::env::var("DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        {
            print_hip_counters("decode-step");
        }
        let cpu_decode_nans = match cpu_logits.as_ref() {
            Some(cpu_logits) => logit_nan_count(cpu_logits.tensor())?,
            None => 0,
        };
        let device_decode_nans = logit_nan_count(device_logits.tensor())?;
        if cpu_decode_nans > 0 || device_decode_nans > 0 {
            eprintln!(
                "warning: decode logits contain NaNs cpu={} device={}",
                cpu_decode_nans, device_decode_nans
            );
        }

        if let Some(cpu_logits) = cpu_logits.as_ref() {
            max_decode_delta = max_decode_delta.max(max_logit_delta(
                cpu_logits.tensor(),
                device_logits.tensor(),
            )?);
            next_token = argmax_last_token(cpu_logits.tensor())?;
        } else {
            next_token = argmax_last_token(device_logits.tensor())?;
        }
    }

    let generated_text = tokenizer.decode(&generated_ids, true)?;
    println!("{generated_text}");
    eprintln!(
        "device={device_selector} device_only={} prompt_tokens={} generated_tokens={} cpu_load_ms={:.2} device_load_ms={:.2} cpu_prefill_ms={:.2} device_prefill_ms={:.2} cpu_decode_ms={:.2} device_decode_ms={:.2} prefill_max_delta={:.6} decode_max_delta={:.6}",
        device_only,
        prompt_ids.len(),
        generated_ids.len().saturating_sub(prompt_ids.len()),
        cpu_load_elapsed.as_secs_f64() * 1000.0,
        device_load_elapsed.as_secs_f64() * 1000.0,
        cpu_prefill_elapsed.as_secs_f64() * 1000.0,
        device_prefill_elapsed.as_secs_f64() * 1000.0,
        cpu_decode_elapsed.as_secs_f64() * 1000.0,
        device_decode_elapsed.as_secs_f64() * 1000.0,
        prefill_delta,
        max_decode_delta,
    );
    Ok(())
}

#[cfg(not(feature = "qwen35-minimal"))]
fn main() {
    eprintln!("enable the `qwen35-minimal` feature to run this example");
}
