#[cfg(feature = "qwen35-minimal")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::time::Instant;

    use candle_core::{DType, Device, IndexOp, Tensor};
    use dotcache_paged_runtime::{MinimalQwen35LoadMode, MinimalQwen35Runner, Result, RuntimeError};
    use serde::Serialize;
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

        fn cpu_reference_runner_mode(self) -> Option<MinimalQwen35LoadMode> {
            match self {
                Self::HipDirect => None,
                _ => self.runner_mode(),
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

    impl std::fmt::Display for LoadMode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Native => f.write_str("native"),
                Self::Direct => f.write_str("direct"),
                Self::HipDirect => f.write_str("hip-direct"),
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum OracleMode {
        Cpu,
        NativeDevice,
        None,
        Pytorch,
    }

    impl OracleMode {
        fn default_for(load_mode: LoadMode, device_selector: &DeviceSelector) -> Self {
            match (load_mode, device_selector) {
                (LoadMode::HipDirect, DeviceSelector::Hip(_)) => Self::NativeDevice,
                _ => Self::Cpu,
            }
        }
    }

    impl std::str::FromStr for OracleMode {
        type Err = RuntimeError;

        fn from_str(value: &str) -> Result<Self> {
            match value.trim().to_ascii_lowercase().as_str() {
                "cpu" => Ok(Self::Cpu),
                "native-device" | "native" | "device" => Ok(Self::NativeDevice),
                "none" => Ok(Self::None),
                "pytorch" => Ok(Self::Pytorch),
                other => Err(RuntimeError::External {
                    context: "oracle",
                    message: format!(
                        "unsupported oracle `{other}`, expected cpu, native-device, none, or pytorch"
                    ),
                }),
            }
        }
    }

    impl std::fmt::Display for OracleMode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Cpu => f.write_str("cpu"),
                Self::NativeDevice => f.write_str("native-device"),
                Self::None => f.write_str("none"),
                Self::Pytorch => f.write_str("pytorch"),
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct RunRecord {
        model_id: String,
        prompt: String,
        device: String,
        load_mode: String,
        oracle: String,
        oracle_device: String,
        device_only: bool,
        prompt_token_count: usize,
        generated_token_count: usize,
        max_new_tokens: usize,
        cpu_load_ms: f64,
        device_load_ms: f64,
        cpu_prefill_ms: f64,
        device_prefill_ms: f64,
        cpu_decode_ms: f64,
        device_decode_ms: f64,
        oracle_load_ms: f64,
        oracle_prefill_ms: f64,
        oracle_decode_ms: f64,
        prefill_max_delta: f32,
        prefill_cache_max_delta: Option<f32>,
        decode_max_delta: f32,
        decode_input_hidden_max_delta: Option<f32>,
        decode_step_cache_max_delta: Option<f32>,
        generated_text: String,
        hip_trace_candle_fallback: bool,
        hip_print_transfers: bool,
        full_prefill_megakernel_requested: bool,
        hip_persistent_full_prefill_requested: bool,
    }

    fn env_flag_truthy(key: &str) -> bool {
        matches!(
            std::env::var(key).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
        )
    }

    fn device_label(device: &Device) -> &'static str {
        if device.is_cpu() {
            "cpu"
        } else if device.is_cuda() {
            "cuda"
        } else if device.is_hip() {
            "hip"
        } else if device.is_metal() {
            "metal"
        } else {
            "unknown"
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

    fn cache_max_delta(
        lhs: &dotcache_qwen35_runtime::MinimalQwen35KvCache,
        rhs: &dotcache_qwen35_runtime::MinimalQwen35KvCache,
    ) -> Result<Option<f32>> {
        let mut max_delta: Option<f32> = None;
        for line in lhs.layer_max_abs_deltas(rhs)? {
            for field in line.split_whitespace() {
                if let Some((key, value)) = field.split_once('=') {
                    if key.ends_with("_max_abs_delta") {
                        let parsed = value.parse::<f32>().map_err(|err| RuntimeError::External {
                            context: "cache delta parse",
                            message: format!("failed to parse `{field}`: {err}"),
                        })?;
                        max_delta = Some(match max_delta {
                            Some(current) => current.max(parsed),
                            None => parsed,
                        });
                    }
                }
            }
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

    fn trace_decode_input_delta_enabled() -> bool {
        matches!(
            std::env::var("DOTCACHE_QWEN35_TRACE_DECODE_INPUT_DELTA").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    fn trace_cross_runner_cache_delta_enabled() -> bool {
        matches!(
            std::env::var("DOTCACHE_QWEN35_TRACE_CROSS_RUNNER_CACHE_DELTA").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    fn trace_cross_runner_generic_decode_delta_enabled() -> bool {
        matches!(
            std::env::var("DOTCACHE_QWEN35_TRACE_CROSS_RUNNER_GENERIC_DECODE_DELTA").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    fn trace_cross_runner_model_only_decode_delta_enabled() -> bool {
        matches!(
            std::env::var("DOTCACHE_QWEN35_TRACE_CROSS_RUNNER_MODEL_ONLY_DECODE_DELTA").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    fn trace_cross_runner_cache_delta(
        label: &str,
        oracle: &dotcache_qwen35_runtime::MinimalQwen35KvCache,
        device: &dotcache_qwen35_runtime::MinimalQwen35KvCache,
    ) -> Result<()> {
        if !trace_cross_runner_cache_delta_enabled() {
            return Ok(());
        }
        for line in oracle.layer_max_abs_deltas(device)? {
            eprintln!("cross-runner-cache-delta[{label}] {line}");
        }
        Ok(())
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
        "usage: hf_qwen35_minimal <model_id> <prompt> [max_new_tokens] [--device cpu|cuda[:ordinal]|hip[:ordinal]] [--load-mode native|direct|hip-direct] [--oracle cpu|native-device|none|pytorch] [--device-only] [--record-json <path>]",
    )?;
    let prompt = args.next().ok_or("missing prompt")?;
    let mut positional = Vec::new();
    let mut device_selector = DeviceSelector::Cpu;
    let mut load_mode = LoadMode::Native;
    let mut oracle_mode: Option<OracleMode> = None;
    let mut device_only = false;
    let mut record_json_path: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg == "--device" {
            let value = args.next().ok_or("missing value for --device")?;
            device_selector = value.parse()?;
        } else if arg == "--load-mode" {
            let value = args.next().ok_or("missing value for --load-mode")?;
            load_mode = value.parse()?;
        } else if arg == "--oracle" {
            let value = args.next().ok_or("missing value for --oracle")?;
            oracle_mode = Some(value.parse()?);
        } else if arg == "--device-only" {
            device_only = true;
        } else if arg == "--record-json" {
            record_json_path = Some(args.next().ok_or("missing value for --record-json")?);
        } else {
            positional.push(arg);
        }
    }
    let max_new_tokens = positional
        .first()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let oracle_mode = oracle_mode.unwrap_or_else(|| OracleMode::default_for(load_mode, &device_selector));

    let cpu_device = Device::Cpu;
    let target_device = device_selector.resolve()?;
    let (oracle_device, oracle_runner_mode) = match oracle_mode {
        OracleMode::Cpu => (
            cpu_device.clone(),
            load_mode.cpu_reference_runner_mode(),
        ),
        OracleMode::NativeDevice => (
            target_device.clone(),
            Some(MinimalQwen35LoadMode::NativeStore),
        ),
        OracleMode::None => (cpu_device.clone(), None),
        OracleMode::Pytorch => {
            return Err(RuntimeError::External {
                context: "oracle",
                message: "pytorch oracle requested but Python torch/transformers integration is not implemented in this harness yet".to_string(),
            }
            .into())
        }
    };
    let (mut oracle_runner, oracle_load_elapsed) = if device_only || matches!(oracle_mode, OracleMode::None) {
        (None, std::time::Duration::ZERO)
    } else {
        let oracle_load_started = Instant::now();
        let oracle_runner = match oracle_runner_mode {
            Some(mode) => MinimalQwen35Runner::load_with_mode(&model_id, &oracle_device, mode)?,
            None => MinimalQwen35Runner::load_from_hf_direct_f16(&model_id, &oracle_device)?,
        };
        (Some(oracle_runner), oracle_load_started.elapsed())
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
    let oracle_input_ids = if oracle_device.location() == cpu_device.location() {
        input_ids.clone()
    } else {
        input_ids.to_device(&oracle_device)?
    };
    let (mut oracle_logits, mut oracle_cache, oracle_prefill_elapsed) =
        if let Some(oracle_runner) = oracle_runner.as_mut() {
            let oracle_hidden_states = oracle_runner.hidden_states_from_input_ids(&oracle_input_ids)?;
            let oracle_prefill_started = Instant::now();
            match oracle_runner.prefill_from_hidden_states(&oracle_hidden_states) {
                Ok((oracle_logits, oracle_cache)) => (
                    Some(oracle_logits),
                    Some(oracle_cache),
                    oracle_prefill_started.elapsed(),
                ),
                Err(err) => {
                    eprintln!("warning: disabling oracle path after prefill failure: {err}");
                    (None, None, std::time::Duration::ZERO)
                }
            }
        } else {
            (None, None, std::time::Duration::ZERO)
        };

    #[cfg(feature = "qwen35-minimal-hip")]
    if target_device.is_hip() {
        candle_core::hip::reset_transfer_counters();
    }
    let device_hidden_states = device_runner.hidden_states_from_input_ids_direct(&input_ids)?;
    let device_prefill_started = Instant::now();
    let (mut device_logits, mut device_cache) =
        device_runner.prefill_from_hidden_states(&device_hidden_states)?;
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
    let oracle_prefill_nans = match oracle_logits.as_ref() {
        Some(oracle_logits) => logit_nan_count(oracle_logits.tensor())?,
        None => 0,
    };
    let device_prefill_nans = logit_nan_count(device_logits.tensor())?;
    if oracle_prefill_nans > 0 || device_prefill_nans > 0 {
        eprintln!(
            "warning: prefill logits contain NaNs oracle={} device={}",
            oracle_prefill_nans, device_prefill_nans
        );
        if let Some(oracle_runner) = oracle_runner.as_mut() {
            report_linear_nan_trace(oracle_runner, &oracle_input_ids)?;
        }
    }

    let prefill_delta = match oracle_logits.as_ref() {
        Some(oracle_logits) => max_logit_delta(oracle_logits.tensor(), device_logits.tensor())?,
        None => f32::NAN,
    };
    let prefill_cache_max_delta = match oracle_cache.as_ref() {
        Some(oracle_cache) => cache_max_delta(oracle_cache, &device_cache)?,
        None => None,
    };
    if let (Some(oracle_cache), true) = (oracle_cache.as_ref(), oracle_device.location() == target_device.location()) {
        trace_cross_runner_cache_delta("prefill", oracle_cache, &device_cache)?;
    }
    let mut generated_ids = prompt_ids.clone();
    let mut max_decode_delta = if device_only { f32::NAN } else { 0.0f32 };
    let mut max_decode_input_hidden_delta: Option<f32> = None;
    let mut max_decode_step_cache_delta: Option<f32> = None;
    let mut oracle_decode_elapsed = std::time::Duration::ZERO;
    let mut device_decode_elapsed = std::time::Duration::ZERO;
    let mut oracle_reference_enabled = oracle_logits.is_some() && oracle_cache.is_some();
    let mut next_token = match oracle_logits.as_ref() {
        Some(oracle_logits) => argmax_last_token(oracle_logits.tensor())?,
        None => argmax_last_token(device_logits.tensor())?,
    };
    for _ in 0..max_new_tokens {
        generated_ids.push(next_token);

        let decode_input = Tensor::from_vec(vec![next_token], (1, 1), &cpu_device)?;
        let device_hidden_state = device_runner.hidden_states_from_input_ids_direct(&decode_input)?;
        let device_pre_decode_cache = if trace_cross_runner_generic_decode_delta_enabled() {
            Some(device_cache.clone())
        } else {
            None
        };
        if oracle_reference_enabled {
            if let (Some(oracle_runner), Some(oracle_cache_ref)) = (oracle_runner.as_mut(), oracle_cache.as_mut()) {
            let oracle_decode_input = if oracle_device.location() == cpu_device.location() {
                decode_input.clone()
            } else {
                decode_input.to_device(&oracle_device)?
            };
            let oracle_hidden_state = oracle_runner.hidden_states_from_input_ids(&oracle_decode_input)?;
            let input_delta = max_logit_delta(
                oracle_hidden_state.tensor(),
                device_hidden_state.tensor(),
            )?;
            max_decode_input_hidden_delta = Some(match max_decode_input_hidden_delta {
                Some(current) => current.max(input_delta),
                None => input_delta,
            });
            if trace_decode_input_delta_enabled() && oracle_device.location() == target_device.location() {
                eprintln!(
                    "decode-input-hidden-delta token={} max_abs_delta={:.6}",
                    next_token, input_delta
                );
            }
            let oracle_decode_started = Instant::now();
            match oracle_runner.decode_from_hidden_state(&oracle_hidden_state, oracle_cache_ref) {
                Ok(logits) => {
                    oracle_logits = Some(logits);
                    oracle_decode_elapsed += oracle_decode_started.elapsed();
                }
                Err(err) => {
                    eprintln!(
                        "warning: disabling oracle path after decode failure: {err}"
                    );
                    oracle_logits = None;
                    oracle_cache = None;
                    oracle_reference_enabled = false;
                }
            }
        }
        }

        #[cfg(feature = "qwen35-minimal-hip")]
        if target_device.is_hip() {
            candle_core::hip::reset_transfer_counters();
        }
        let device_decode_started = Instant::now();
        device_logits =
            device_runner.decode_from_hidden_state(&device_hidden_state, &mut device_cache)?;
        device_decode_elapsed += device_decode_started.elapsed();
        if trace_cross_runner_generic_decode_delta_enabled()
            && oracle_device.location() == target_device.location()
        {
            if let (Some(oracle_logits_ref), Some(oracle_cache_ref)) =
                (oracle_logits.as_ref(), oracle_cache.as_ref())
            {
                let mut generic_cache = device_pre_decode_cache
                    .clone()
                    .unwrap_or_else(|| device_cache.clone());
                let generic_logits = device_runner
                    .decode_from_hidden_state_generic_only(&device_hidden_state, &mut generic_cache)?;
                let generic_logit_delta =
                    max_logit_delta(oracle_logits_ref.tensor(), generic_logits.tensor())?;
                eprintln!(
                    "cross-runner-generic-decode-delta logits_max_abs_delta={:.6}",
                    generic_logit_delta
                );
                trace_cross_runner_cache_delta(
                    "generic-decode-step",
                    oracle_cache_ref,
                    &generic_cache,
                )?;
            }
        }
        if trace_cross_runner_model_only_decode_delta_enabled()
            && oracle_device.location() == target_device.location()
        {
            if let (Some(oracle_logits_ref), Some(oracle_cache_ref)) =
                (oracle_logits.as_ref(), oracle_cache.as_ref())
            {
                let mut model_only_cache = device_pre_decode_cache
                    .clone()
                    .unwrap_or_else(|| device_cache.clone());
                let model_only_logits = device_runner
                    .decode_from_hidden_state_model_only(&device_hidden_state, &mut model_only_cache)?;
                let model_only_logit_delta =
                    max_logit_delta(oracle_logits_ref.tensor(), model_only_logits.tensor())?;
                eprintln!(
                    "cross-runner-model-only-decode-delta logits_max_abs_delta={:.6}",
                    model_only_logit_delta
                );
                trace_cross_runner_cache_delta(
                    "model-only-decode-step",
                    oracle_cache_ref,
                    &model_only_cache,
                )?;
            }
        }
        if let Some(oracle_cache_ref) = oracle_cache.as_ref() {
            if let Some(cache_delta) = cache_max_delta(oracle_cache_ref, &device_cache)? {
                max_decode_step_cache_delta = Some(match max_decode_step_cache_delta {
                    Some(current) => current.max(cache_delta),
                    None => cache_delta,
                });
            }
            if oracle_device.location() == target_device.location() {
                trace_cross_runner_cache_delta("decode-step", oracle_cache_ref, &device_cache)?;
            }
        }
        #[cfg(feature = "qwen35-minimal-hip")]
        if target_device.is_hip()
            && matches!(
                std::env::var("DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        {
            print_hip_counters("decode-step");
        }
        let oracle_decode_nans = match oracle_logits.as_ref() {
            Some(oracle_logits) => logit_nan_count(oracle_logits.tensor())?,
            None => 0,
        };
        let device_decode_nans = logit_nan_count(device_logits.tensor())?;
        if oracle_decode_nans > 0 || device_decode_nans > 0 {
            eprintln!(
                "warning: decode logits contain NaNs oracle={} device={}",
                oracle_decode_nans, device_decode_nans
            );
        }

        if let Some(oracle_logits) = oracle_logits.as_ref() {
            max_decode_delta = max_decode_delta.max(max_logit_delta(
                oracle_logits.tensor(),
                device_logits.tensor(),
            )?);
            next_token = argmax_last_token(oracle_logits.tensor())?;
        } else {
            next_token = argmax_last_token(device_logits.tensor())?;
        }
    }

    let generated_text = tokenizer.decode(&generated_ids, true)?;
    let record = RunRecord {
        model_id: model_id.clone(),
        prompt: prompt.clone(),
        device: device_selector.to_string(),
        load_mode: load_mode.to_string(),
        oracle: oracle_mode.to_string(),
        oracle_device: match oracle_mode {
            OracleMode::None => "none".to_string(),
            _ => device_label(&oracle_device).to_string(),
        },
        device_only,
        prompt_token_count: prompt_ids.len(),
        generated_token_count: generated_ids.len().saturating_sub(prompt_ids.len()),
        max_new_tokens,
        cpu_load_ms: if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_load_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
        device_load_ms: device_load_elapsed.as_secs_f64() * 1000.0,
        cpu_prefill_ms: if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_prefill_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
        device_prefill_ms: device_prefill_elapsed.as_secs_f64() * 1000.0,
        cpu_decode_ms: if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_decode_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
        device_decode_ms: device_decode_elapsed.as_secs_f64() * 1000.0,
        oracle_load_ms: oracle_load_elapsed.as_secs_f64() * 1000.0,
        oracle_prefill_ms: oracle_prefill_elapsed.as_secs_f64() * 1000.0,
        oracle_decode_ms: oracle_decode_elapsed.as_secs_f64() * 1000.0,
        prefill_max_delta: prefill_delta,
        prefill_cache_max_delta,
        decode_max_delta: max_decode_delta,
        decode_input_hidden_max_delta: max_decode_input_hidden_delta,
        decode_step_cache_max_delta: max_decode_step_cache_delta,
        generated_text: generated_text.clone(),
        hip_trace_candle_fallback: env_flag_truthy("DOTCACHE_HIP_TRACE_CANDLE_FALLBACK"),
        hip_print_transfers: env_flag_truthy("DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS"),
        full_prefill_megakernel_requested: env_flag_truthy("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL"),
        hip_persistent_full_prefill_requested: env_flag_truthy(
            "CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL",
        ),
    };
    if let Some(record_json_path) = record_json_path.as_ref() {
        std::fs::write(record_json_path, serde_json::to_string_pretty(&record)?)?;
        eprintln!("run record written to {record_json_path}");
    }
    println!("{generated_text}");
    eprintln!(
        "device={device_selector} device_only={} prompt_tokens={} generated_tokens={} cpu_load_ms={:.2} device_load_ms={:.2} cpu_prefill_ms={:.2} device_prefill_ms={:.2} cpu_decode_ms={:.2} device_decode_ms={:.2} prefill_max_delta={:.6} decode_max_delta={:.6}",
        device_only,
        prompt_ids.len(),
        generated_ids.len().saturating_sub(prompt_ids.len()),
        if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_load_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
        device_load_elapsed.as_secs_f64() * 1000.0,
        if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_prefill_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
        device_prefill_elapsed.as_secs_f64() * 1000.0,
        if matches!(oracle_mode, OracleMode::Cpu) {
            oracle_decode_elapsed.as_secs_f64() * 1000.0
        } else {
            f64::NAN
        },
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
