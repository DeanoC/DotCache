#![cfg(feature = "candle")]

use std::ffi::OsString;
use std::fs;
use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_5::{
    Config as NativeQwen35Config, ModelForCausalLM as NativeQwen35Model,
};
use dotcache_paged_runtime::{
    greedy_generate, CandleCausalLm, CandleDeviceSelector, CausalLm, HfHubModelSource, ModelFamily,
    RuntimeMode,
};
use tokenizers::Tokenizer;

const QWEN35_MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
const LONG_PROMPT_TOKENS: usize = 2_048;
const VERY_LONG_PROMPT_TOKENS: usize = 8_192;
const PARITY_DECODE_STEPS: usize = 2;
const MAX_KERNEL_LOGIT_DELTA: f32 = 0.05;
const MAX_TRACE_DELTA: f32 = 0.05;
const QWEN35_EXPERIMENT_ENV_KEYS: [&str; 15] = [
    "CANDLE_QWEN35_DELTA_STATE_KERNEL",
    "CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL",
    "CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL",
    "CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL",
    "CANDLE_QWEN35_DELTA_CHUNK_STEP_2D_KERNEL",
    "CANDLE_QWEN35_DELTA_CHUNK_SPLIT_KERNEL",
    "CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL",
    "CANDLE_QWEN35_DELTA_FULL_KERNEL",
    "CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL",
    "CANDLE_QWEN35_DELTA_SCAN_MODE",
    "CANDLE_QWEN35_FULL_BLOCKWISE_ATTN",
    "CANDLE_QWEN35_FULL_SDPA_CHUNKED",
    "CANDLE_QWEN35_FULL_EAGER_TORCHLIKE",
    "CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL",
    "CANDLE_QWEN35_LINEAR_PACKED_PREFILL",
];

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index)
        .unwrap()
}

fn assert_logits_close_with_tolerance(step: &str, lhs: &[f32], rhs: &[f32], tolerance: f32) {
    assert_eq!(lhs.len(), rhs.len(), "{step} logits length mismatch");

    let mut max_delta = 0.0f32;
    let mut max_index = 0usize;
    for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        if delta > max_delta {
            max_delta = delta;
            max_index = index;
        }
    }

    assert!(
        max_delta <= tolerance,
        "{step} logits diverged: max_delta={max_delta} at index {max_index}, lhs={}, rhs={}",
        lhs[max_index],
        rhs[max_index],
    );
    assert_eq!(
        argmax(lhs),
        argmax(rhs),
        "{step} argmax token diverged despite close logits",
    );
}

fn experiment_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvResetGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvResetGuard {
    fn clear_qwen35_experiment_vars() -> Self {
        let saved = QWEN35_EXPERIMENT_ENV_KEYS
            .iter()
            .map(|key| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for key in QWEN35_EXPERIMENT_ENV_KEYS {
            unsafe {
                std::env::remove_var(key);
            }
        }
        Self { saved }
    }

    fn set(&self, key: &'static str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }
}

impl Drop for EnvResetGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn build_prompt_token_ids(
    model: &CandleCausalLm,
    prompt: &str,
    prompt_token_target: usize,
) -> dotcache_paged_runtime::Result<Vec<u32>> {
    let mut token_ids = model.encode(prompt, true)?;
    assert!(
        !token_ids.is_empty(),
        "prompt encoding produced no tokens for `{prompt}`"
    );
    if token_ids.len() > prompt_token_target {
        token_ids.truncate(prompt_token_target);
        return Ok(token_ids);
    }

    let filler_ids = model.encode(&format!(" {}", prompt), false)?;
    assert!(
        !filler_ids.is_empty(),
        "prompt filler encoding produced no tokens for `{prompt}`"
    );
    while token_ids.len() < prompt_token_target {
        token_ids.extend_from_slice(&filler_ids);
    }
    token_ids.truncate(prompt_token_target);
    Ok(token_ids)
}

fn build_prompt_token_ids_with_tokenizer(
    tokenizer: &Tokenizer,
    prompt: &str,
    prompt_token_target: usize,
) -> dotcache_paged_runtime::Result<Vec<u32>> {
    let mut token_ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    assert!(
        !token_ids.is_empty(),
        "prompt encoding produced no tokens for `{prompt}`"
    );
    if token_ids.len() > prompt_token_target {
        token_ids.truncate(prompt_token_target);
        return Ok(token_ids);
    }

    let filler_ids = tokenizer
        .encode(format!(" {}", prompt), false)?
        .get_ids()
        .to_vec();
    assert!(
        !filler_ids.is_empty(),
        "prompt filler encoding produced no tokens for `{prompt}`"
    );
    while token_ids.len() < prompt_token_target {
        token_ids.extend_from_slice(&filler_ids);
    }
    token_ids.truncate(prompt_token_target);
    Ok(token_ids)
}

fn assert_tensor_close_with_tolerance(
    step: &str,
    lhs: &Tensor,
    rhs: &Tensor,
    tolerance: f32,
) -> dotcache_paged_runtime::Result<()> {
    assert_eq!(lhs.shape(), rhs.shape(), "{step} tensor shape mismatch");
    let lhs = lhs.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let mut max_delta = 0.0f32;
    let mut max_index = 0usize;
    for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        if delta > max_delta {
            max_delta = delta;
            max_index = index;
        }
    }

    assert!(
        max_delta <= tolerance,
        "{step} tensor diverged: max_delta={max_delta} at flat index {max_index}, lhs={}, rhs={}",
        lhs[max_index],
        rhs[max_index],
    );
    Ok(())
}

fn logits_tensor_to_vec(logits: &Tensor) -> dotcache_paged_runtime::Result<Vec<f32>> {
    Ok(logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?)
}

fn load_native_qwen35_model(
    selector: CandleDeviceSelector,
) -> dotcache_paged_runtime::Result<(NativeQwen35Model, Tokenizer, candle_core::Device)> {
    let source = HfHubModelSource::new()?;
    let artifacts = source.snapshot(QWEN35_MODEL_ID)?;
    let device = selector.resolve()?;
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path)?;
    let config: NativeQwen35Config = serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F32, &device)?
    };
    let model = NativeQwen35Model::new(&config, var_builder)?;
    Ok((model, tokenizer, device))
}

fn compare_dense_control_logits_with_kernel_env(
    selector: CandleDeviceSelector,
    prompt: &str,
    prompt_token_target: usize,
    enabled_envs: &[(&'static str, &'static str)],
) -> dotcache_paged_runtime::Result<()> {
    let _lock = experiment_env_lock().lock().unwrap();
    let env_guard = EnvResetGuard::clear_qwen35_experiment_vars();

    let mut baseline = CandleCausalLm::from_hf_with_runtime_mode(
        QWEN35_MODEL_ID,
        ModelFamily::Qwen35,
        selector.clone(),
        DType::F32,
        CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
        RuntimeMode::DenseControl,
    )?;
    let prompt_ids = build_prompt_token_ids(&baseline, prompt, prompt_token_target)?;
    baseline.reset()?;
    let mut baseline_logits = baseline.forward_next_logits(&prompt_ids)?;

    for (key, value) in enabled_envs {
        env_guard.set(key, value);
    }
    let mut experimental = CandleCausalLm::from_hf_with_runtime_mode(
        QWEN35_MODEL_ID,
        ModelFamily::Qwen35,
        selector,
        DType::F32,
        CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
        RuntimeMode::DenseControl,
    )?;
    experimental.reset()?;
    let mut experimental_logits = experimental.forward_next_logits(&prompt_ids)?;

    assert_logits_close_with_tolerance(
        "prompt baseline vs experimental",
        &baseline_logits,
        &experimental_logits,
        MAX_KERNEL_LOGIT_DELTA,
    );

    let mut next_token = argmax(&baseline_logits) as u32;
    for step in 0..PARITY_DECODE_STEPS {
        baseline_logits = baseline.forward_next_logits(&[next_token])?;
        experimental_logits = experimental.forward_next_logits(&[next_token])?;
        assert_logits_close_with_tolerance(
            &format!("decode step {} baseline vs experimental", step + 1),
            &baseline_logits,
            &experimental_logits,
            MAX_KERNEL_LOGIT_DELTA,
        );
        next_token = argmax(&baseline_logits) as u32;
    }

    Ok(())
}

fn compare_native_linear_trace_with_kernel_env(
    selector: CandleDeviceSelector,
    prompt: &str,
    prompt_token_target: usize,
    target_layer: usize,
    enabled_envs: &[(&'static str, &'static str)],
) -> dotcache_paged_runtime::Result<()> {
    let _lock = experiment_env_lock().lock().unwrap();
    let env_guard = EnvResetGuard::clear_qwen35_experiment_vars();

    let (mut baseline, tokenizer, baseline_device) = load_native_qwen35_model(selector.clone())?;
    let prompt_ids =
        build_prompt_token_ids_with_tokenizer(&tokenizer, prompt, prompt_token_target)?;
    let baseline_input_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &baseline_device)?;
    let baseline_trace =
        baseline.trace_linear_attention_layer(&baseline_input_ids, target_layer, 0)?;

    for (key, value) in enabled_envs {
        env_guard.set(key, value);
    }
    let (mut experimental, _tokenizer, experimental_device) = load_native_qwen35_model(selector)?;
    let experimental_input_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &experimental_device)?;
    let experimental_trace =
        experimental.trace_linear_attention_layer(&experimental_input_ids, target_layer, 0)?;

    assert_eq!(baseline_trace.layer_id, experimental_trace.layer_id);
    assert_eq!(
        baseline_trace.sequence_length,
        experimental_trace.sequence_length
    );
    assert_tensor_close_with_tolerance(
        "linear layer output baseline vs experimental",
        &baseline_trace.layer_output,
        &experimental_trace.layer_output,
        MAX_TRACE_DELTA,
    )?;
    assert_tensor_close_with_tolerance(
        "recurrent state baseline vs experimental",
        &baseline_trace.recurrent_state,
        &experimental_trace.recurrent_state,
        MAX_TRACE_DELTA,
    )?;
    Ok(())
}

fn compare_native_linear_traces_with_kernel_env(
    selector: CandleDeviceSelector,
    prompt: &str,
    prompt_token_target: usize,
    target_layers: &[usize],
    enabled_envs: &[(&'static str, &'static str)],
) -> dotcache_paged_runtime::Result<()> {
    let _lock = experiment_env_lock().lock().unwrap();
    let env_guard = EnvResetGuard::clear_qwen35_experiment_vars();

    let (mut baseline, tokenizer, baseline_device) = load_native_qwen35_model(selector.clone())?;
    let prompt_ids =
        build_prompt_token_ids_with_tokenizer(&tokenizer, prompt, prompt_token_target)?;
    let baseline_input_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &baseline_device)?;

    for (key, value) in enabled_envs {
        env_guard.set(key, value);
    }
    let (mut experimental, _tokenizer, experimental_device) = load_native_qwen35_model(selector)?;
    let experimental_input_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &experimental_device)?;

    for &target_layer in target_layers {
        let baseline_trace =
            baseline.trace_linear_attention_layer(&baseline_input_ids, target_layer, 0)?;
        let experimental_trace =
            experimental.trace_linear_attention_layer(&experimental_input_ids, target_layer, 0)?;

        assert_eq!(baseline_trace.layer_id, experimental_trace.layer_id);
        assert_eq!(
            baseline_trace.sequence_length,
            experimental_trace.sequence_length
        );
        assert_tensor_close_with_tolerance(
            &format!("layer {target_layer} output baseline vs experimental"),
            &baseline_trace.layer_output,
            &experimental_trace.layer_output,
            MAX_TRACE_DELTA,
        )?;
        assert_tensor_close_with_tolerance(
            &format!("layer {target_layer} recurrent state baseline vs experimental"),
            &baseline_trace.recurrent_state,
            &experimental_trace.recurrent_state,
            MAX_TRACE_DELTA,
        )?;
    }
    Ok(())
}

fn compare_native_linear_decode_traces_with_kernel_env(
    selector: CandleDeviceSelector,
    prompt: &str,
    prompt_token_target: usize,
    target_layers: &[usize],
    decode_steps: usize,
    enabled_envs: &[(&'static str, &'static str)],
) -> dotcache_paged_runtime::Result<()> {
    let _lock = experiment_env_lock().lock().unwrap();
    let env_guard = EnvResetGuard::clear_qwen35_experiment_vars();

    let (mut baseline, tokenizer, baseline_device) = load_native_qwen35_model(selector.clone())?;
    let prompt_ids =
        build_prompt_token_ids_with_tokenizer(&tokenizer, prompt, prompt_token_target)?;
    let baseline_prompt_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &baseline_device)?;
    let baseline_prefill_logits = baseline.forward(&baseline_prompt_ids, 0)?;

    for (key, value) in enabled_envs {
        env_guard.set(key, value);
    }
    let (mut experimental, _tokenizer, experimental_device) = load_native_qwen35_model(selector)?;
    let experimental_prompt_ids =
        Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &experimental_device)?;
    let experimental_prefill_logits = experimental.forward(&experimental_prompt_ids, 0)?;

    let mut baseline_logits = logits_tensor_to_vec(&baseline_prefill_logits)?;
    let mut experimental_logits = logits_tensor_to_vec(&experimental_prefill_logits)?;
    assert_logits_close_with_tolerance(
        "decode prefill baseline vs experimental",
        &baseline_logits,
        &experimental_logits,
        MAX_KERNEL_LOGIT_DELTA,
    );

    let mut next_token = argmax(&baseline_logits) as u32;
    let mut seqlen_offset = prompt_ids.len();
    for step in 0..decode_steps {
        let baseline_step_ids = Tensor::from_slice(&[next_token], (1, 1), &baseline_device)?;
        let (baseline_step_logits, baseline_traces, _) = baseline
            .forward_profiled_with_linear_traces(
                &baseline_step_ids,
                seqlen_offset,
                target_layers,
            )?;

        let experimental_step_ids =
            Tensor::from_slice(&[next_token], (1, 1), &experimental_device)?;
        let (experimental_step_logits, experimental_traces, _) = experimental
            .forward_profiled_with_linear_traces(
                &experimental_step_ids,
                seqlen_offset,
                target_layers,
            )?;

        baseline_logits = logits_tensor_to_vec(&baseline_step_logits)?;
        experimental_logits = logits_tensor_to_vec(&experimental_step_logits)?;
        assert_logits_close_with_tolerance(
            &format!("decode step {} logits baseline vs experimental", step + 1),
            &baseline_logits,
            &experimental_logits,
            MAX_KERNEL_LOGIT_DELTA,
        );
        assert_eq!(
            baseline_traces.len(),
            experimental_traces.len(),
            "decode step {} trace count mismatch",
            step + 1
        );
        for (baseline_trace, experimental_trace) in
            baseline_traces.iter().zip(experimental_traces.iter())
        {
            assert_eq!(baseline_trace.layer_id, experimental_trace.layer_id);
            assert_tensor_close_with_tolerance(
                &format!(
                    "decode step {} layer {} output baseline vs experimental",
                    step + 1,
                    baseline_trace.layer_id
                ),
                &baseline_trace.layer_output,
                &experimental_trace.layer_output,
                MAX_TRACE_DELTA,
            )?;
            assert_tensor_close_with_tolerance(
                &format!(
                    "decode step {} layer {} recurrent state baseline vs experimental",
                    step + 1,
                    baseline_trace.layer_id
                ),
                &baseline_trace.recurrent_state,
                &experimental_trace.recurrent_state,
                MAX_TRACE_DELTA,
            )?;
        }

        next_token = argmax(&baseline_logits) as u32;
        seqlen_offset += 1;
    }
    Ok(())
}

#[test]
#[ignore = "downloads Qwen3.5-0.8B and runs a live dense-control smoke test"]
fn qwen35_dense_control_smoke_runs_on_cpu() -> dotcache_paged_runtime::Result<()> {
    let mut model = CandleCausalLm::from_hf_with_runtime_mode(
        QWEN35_MODEL_ID,
        ModelFamily::Qwen35,
        CandleDeviceSelector::Cpu,
        DType::F32,
        CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
        RuntimeMode::DenseControl,
    )?;
    let generation = greedy_generate(&mut model, "hello", 1)?;
    assert!(generation.text.starts_with("hello"));
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and runs a live dense-control CUDA smoke test"]
fn qwen35_dense_control_smoke_runs_on_cuda() -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    let mut model = CandleCausalLm::from_hf_with_runtime_mode(
        QWEN35_MODEL_ID,
        ModelFamily::Qwen35,
        selector,
        DType::F32,
        CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
        RuntimeMode::DenseControl,
    )?;
    let generation = greedy_generate(&mut model, "hello", 1)?;
    assert!(generation.text.starts_with("hello"));
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full-attention prefill megakernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_full_attention_megakernel_matches_baseline_on_cuda_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL", "1")],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the packed linear prefill kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_linear_packed_prefill_matches_baseline_on_cuda_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_LINEAR_PACKED_PREFILL", "1")],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the recurrent DeltaNet prefill kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_recurrent_prefill_kernel_matches_baseline_on_cuda_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the chunk-step kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_chunk_step_kernel_matches_baseline_on_cuda_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the windowed chunk-step kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_chunk_windowed_kernel_matches_baseline_on_cuda_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the chunk-scan kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_chunk_scan_kernel_matches_baseline_on_cuda_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_CHUNK_SCAN_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the state-scan kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_state_scan_kernel_matches_baseline_on_cuda_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the chunk-fused kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_chunk_fused_kernel_matches_baseline_on_cuda_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full DeltaNet kernel against the dense-control CUDA baseline"]
fn qwen35_dense_control_full_kernel_matches_baseline_on_cuda_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Cuda { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("cuda device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the DeltaNet state-update kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_state_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_STATE_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the DeltaNet state-scan kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_state_scan_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the DeltaNet chunk-fused kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_chunk_fused_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the fully GPU-owned chunk-step DeltaNet kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_chunk_step_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the fully GPU-owned chunk-step DeltaNet kernel against the dense-control Metal baseline at 8192 tokens"]
fn qwen35_dense_control_chunk_step_kernel_matches_baseline_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the legacy 1d chunk-step kernel against the default long-context Metal baseline"]
fn qwen35_dense_control_chunk_step_1d_fallback_matches_default_2d_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_2D_KERNEL", "0"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the legacy non-windowed chunk-step path against the default windowed Metal baseline"]
fn qwen35_dense_control_chunk_step_non_windowed_matches_default_windowed_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL", "1"),
            ("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL", "0"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the linear packed-prefill opt-out path against the default dense-control Metal baseline"]
fn qwen35_dense_control_linear_packed_prefill_opt_out_matches_default_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_LINEAR_PACKED_PREFILL", "0")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full-attention prefill megakernel against the dense-control Metal baseline"]
fn qwen35_dense_control_full_attention_megakernel_matches_baseline_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full-attention prefill megakernel opt-out path against the default dense-control Metal baseline"]
fn qwen35_dense_control_full_attention_megakernel_opt_out_matches_default_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL", "0")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full DeltaNet GPU kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_full_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the fully GPU-owned recurrent DeltaNet prefill kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_recurrent_prefill_kernel_matches_baseline_on_metal_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &[
            ("CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL", "1"),
            ("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE", "1"),
        ],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates the full DeltaNet GPU kernel against the dense-control Metal baseline at 8192 tokens"]
fn qwen35_dense_control_full_kernel_matches_baseline_on_metal_very_long_prompt(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_dense_control_logits_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        VERY_LONG_PROMPT_TOKENS,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates linear-attention outputs/state for the full DeltaNet GPU kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_full_kernel_matches_baseline_linear_trace_on_metal(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    compare_native_linear_trace_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        0,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates representative linear-attention layers for the full DeltaNet GPU kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_full_kernel_matches_baseline_representative_linear_traces_on_metal(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    let (baseline, _tokenizer, _device) = load_native_qwen35_model(selector.clone())?;
    let linear_layer_ids = baseline.linear_attention_layer_ids();
    assert!(
        !linear_layer_ids.is_empty(),
        "qwen3.5 model should expose linear-attention layers"
    );
    let representative_layers = if linear_layer_ids.len() == 1 {
        vec![linear_layer_ids[0]]
    } else {
        let mut layers = vec![
            linear_layer_ids[0],
            linear_layer_ids[linear_layer_ids.len() / 2],
            *linear_layer_ids.last().unwrap(),
        ];
        layers.dedup();
        layers
    };

    compare_native_linear_traces_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &representative_layers,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and validates representative linear-attention decode traces for the full DeltaNet GPU kernel against the dense-control Metal baseline"]
fn qwen35_dense_control_full_kernel_matches_baseline_representative_linear_decode_traces_on_metal(
) -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    let (baseline, _tokenizer, _device) = load_native_qwen35_model(selector.clone())?;
    let linear_layer_ids = baseline.linear_attention_layer_ids();
    assert!(
        !linear_layer_ids.is_empty(),
        "qwen3.5 model should expose linear-attention layers"
    );
    let representative_layers = if linear_layer_ids.len() == 1 {
        vec![linear_layer_ids[0]]
    } else {
        let mut layers = vec![
            linear_layer_ids[0],
            linear_layer_ids[linear_layer_ids.len() / 2],
            *linear_layer_ids.last().unwrap(),
        ];
        layers.dedup();
        layers
    };

    compare_native_linear_decode_traces_with_kernel_env(
        selector,
        "hello from the qwen35 correctness harness",
        LONG_PROMPT_TOKENS,
        &representative_layers,
        PARITY_DECODE_STEPS,
        &[("CANDLE_QWEN35_DELTA_FULL_KERNEL", "1")],
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "downloads Qwen3.5-0.8B and runs a live dense-control Metal smoke test"]
fn qwen35_dense_control_smoke_runs_on_metal() -> dotcache_paged_runtime::Result<()> {
    let selector = CandleDeviceSelector::Metal { ordinal: 0 };
    if selector.resolve().is_err() {
        return Ok(());
    }

    let mut model = CandleCausalLm::from_hf_with_runtime_mode(
        QWEN35_MODEL_ID,
        ModelFamily::Qwen35,
        selector,
        DType::F32,
        CandleCausalLm::DEFAULT_TOKENS_PER_PAGE,
        RuntimeMode::DenseControl,
    )?;
    let generation = greedy_generate(&mut model, "hello", 1)?;
    assert!(generation.text.starts_with("hello"));
    Ok(())
}
