#![cfg(feature = "candle")]

use std::fs;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache as NativeLlamaCache, Config as NativeLlamaConfig};
use dotcache_paged_runtime::{
    CandleCausalLm, CandleDeviceSelector, CausalLm, HfHubModelSource, ModelFamily,
    SessionRequestKind,
};
use serde::Deserialize;

const MODEL_ID: &str = "trl-internal-testing/tiny-random-LlamaForCausalLM";
const QWEN2_MODEL_ID: &str = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5";
const TOKENS_PER_PAGE: usize = 2;
const DECODE_STEPS: usize = 3;
const MAX_LOGIT_DELTA: f32 = 0.015;
const MAX_DEVICE_LOGIT_DELTA: f32 = 0.02;

#[derive(Debug, Clone, Deserialize)]
struct CompatLlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f64,
    #[serde(default = "default_llama_rope_theta")]
    rope_theta: f32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<candle_transformers::models::llama::LlamaEosToks>,
    rope_scaling: Option<candle_transformers::models::llama::Llama3RopeConfig>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

impl CompatLlamaConfig {
    fn into_runtime(self) -> NativeLlamaConfig {
        NativeLlamaConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            use_flash_attn: false,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self
                .max_position_embeddings
                .unwrap_or(candle_transformers::models::llama::DEFAULT_MAX_SEQ_LEN),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompatQwen2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    sliding_window: Option<usize>,
    max_window_layers: usize,
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    use_sliding_window: bool,
    hidden_act: candle_nn::Activation,
}

impl CompatQwen2Config {
    fn into_runtime(self) -> candle_transformers::models::qwen2::Config {
        candle_transformers::models::qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window.unwrap_or(self.max_position_embeddings),
            max_window_layers: self.max_window_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window,
            hidden_act: self.hidden_act,
        }
    }
}

fn default_llama_rope_theta() -> f32 {
    10_000.0
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index)
        .unwrap()
}

fn assert_logits_close(step: &str, paged: &[f32], native: &[f32]) {
    assert_eq!(paged.len(), native.len(), "{step} logits length mismatch");

    let mut max_delta = 0.0f32;
    let mut max_index = 0usize;
    for (index, (paged, native)) in paged.iter().zip(native.iter()).enumerate() {
        let delta = (paged - native).abs();
        if delta > max_delta {
            max_delta = delta;
            max_index = index;
        }
    }

    assert!(
        max_delta <= MAX_LOGIT_DELTA,
        "{step} logits diverged: max_delta={max_delta} at index {max_index}, paged={}, native={}",
        paged[max_index],
        native[max_index],
    );
    assert_eq!(
        argmax(paged),
        argmax(native),
        "{step} argmax token diverged despite close logits",
    );
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

fn prefill_session_chunked(
    model: &mut CandleCausalLm,
    session_id: usize,
    input_ids: &[u32],
    chunk_size: usize,
) -> dotcache_paged_runtime::Result<Vec<f32>> {
    let mut logits = Vec::new();
    for chunk in input_ids.chunks(chunk_size.max(1)) {
        logits = model.prefill_session(session_id, chunk)?;
    }
    Ok(logits)
}

fn stress_suffix_text(logical_index: usize, repeats: usize) -> String {
    let mut text = String::new();
    for repeat in 0..repeats {
        text.push_str(&format!(
            " workload-{logical_index}-segment-{repeat} detail-{logical_index} load-{repeat}"
        ));
    }
    text
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live parity validation"]
fn llama_paged_logits_match_native_candle_on_tiny_hf() -> dotcache_paged_runtime::Result<()> {
    let source = HfHubModelSource::new()?;
    let artifacts = source.snapshot(MODEL_ID)?;
    let prompt = "one two three four five six seven eight";

    let mut paged = CandleCausalLm::from_artifacts_with_paging(
        artifacts.clone(),
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;

    let prompt_ids = paged.encode(prompt, true)?;
    assert!(
        prompt_ids.len() > TOKENS_PER_PAGE,
        "prompt should cross a page boundary, got {} tokens",
        prompt_ids.len()
    );

    let device = Device::Cpu;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F32, &device)?
    };
    let config: CompatLlamaConfig = serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
    let runtime_config = config.into_runtime();
    let native_model =
        candle_transformers::models::llama::Llama::load(var_builder, &runtime_config)?;
    let mut native_cache = NativeLlamaCache::new(true, DType::F32, &runtime_config, &device)?;

    paged.reset()?;

    let native_prompt_logits = native_model
        .forward(
            &Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &device)?,
            0,
            &mut native_cache,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let paged_prompt_logits = paged.forward_next_logits(&prompt_ids)?;
    assert_logits_close("prompt", &paged_prompt_logits, &native_prompt_logits);

    let mut next_token = argmax(&native_prompt_logits) as u32;
    for step in 0..DECODE_STEPS {
        let native_logits = native_model
            .forward(
                &Tensor::from_slice(&[next_token], (1, 1), &device)?,
                prompt_ids.len() + step,
                &mut native_cache,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let paged_logits = paged.forward_next_logits(&[next_token])?;
        assert_logits_close(
            &format!("decode step {}", step + 1),
            &paged_logits,
            &native_logits,
        );
        next_token = argmax(&native_logits) as u32;
    }

    let cache = paged
        .paged_cache()
        .expect("llama parity test should expose a paged cache");
    assert!(cache.physical_page_count() > 0);
    assert!(cache.virtual_page_count() > 0);

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live Metal-vs-CPU validation"]
fn llama_paged_logits_match_between_cpu_and_metal_on_tiny_hf() -> dotcache_paged_runtime::Result<()>
{
    let prompt = "one two three four five six seven eight";
    if (CandleDeviceSelector::Metal { ordinal: 0 })
        .resolve()
        .is_err()
    {
        eprintln!("metal device is unavailable on this host, skipping");
        return Ok(());
    }

    let mut cpu = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let mut metal = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Metal { ordinal: 0 },
        DType::F32,
        TOKENS_PER_PAGE,
    )?;

    let prompt_ids = cpu.encode(prompt, true)?;
    cpu.reset()?;
    metal.reset()?;

    let mut cpu_logits = cpu.forward_next_logits(&prompt_ids)?;
    let mut metal_logits = metal.forward_next_logits(&prompt_ids)?;
    assert_logits_close_with_tolerance(
        "prompt cpu vs metal",
        &cpu_logits,
        &metal_logits,
        MAX_DEVICE_LOGIT_DELTA,
    );

    let mut next_token = argmax(&cpu_logits) as u32;
    for step in 0..DECODE_STEPS {
        cpu_logits = cpu.forward_next_logits(&[next_token])?;
        metal_logits = metal.forward_next_logits(&[next_token])?;
        assert_logits_close_with_tolerance(
            &format!("decode step {} cpu vs metal", step + 1),
            &cpu_logits,
            &metal_logits,
            MAX_DEVICE_LOGIT_DELTA,
        );
        next_token = argmax(&cpu_logits) as u32;
    }

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live multi-session validation"]
fn llama_model_api_can_fork_and_switch_sessions() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let root_session_id = model
        .active_session_id()
        .expect("llama should expose sessions");
    let prompt_logits = model.forward_next_logits(&prompt_ids)?;
    assert_eq!(
        model.session_position(root_session_id)?,
        prompt_ids.len() as u32
    );

    let fork_session_id = model.fork_active_session()?;
    assert_eq!(model.session_count(), Some(2));
    assert_eq!(
        model.resolve_session_physical_page_ids(root_session_id, 0, 0)?,
        model.resolve_session_physical_page_ids(fork_session_id, 0, 0)?,
    );

    let next_token = argmax(&prompt_logits) as u32;
    model.set_active_session(fork_session_id)?;
    let fork_logits = model.forward_next_logits(&[next_token])?;
    assert_eq!(
        model.session_position(fork_session_id)?,
        prompt_ids.len() as u32 + 1
    );

    model.set_active_session(root_session_id)?;
    let root_logits = model.forward_next_logits(&[next_token])?;
    assert_eq!(
        model.session_position(root_session_id)?,
        prompt_ids.len() as u32 + 1
    );
    assert_logits_close("forked decode", &fork_logits, &root_logits);

    let root_pages = model.resolve_session_physical_page_ids(root_session_id, 0, 0)?;
    let fork_pages = model.resolve_session_physical_page_ids(fork_session_id, 0, 0)?;
    assert_eq!(root_pages.len(), fork_pages.len());
    assert_eq!(
        root_pages[..root_pages.len() - 1],
        fork_pages[..fork_pages.len() - 1]
    );
    assert_ne!(root_pages.last(), fork_pages.last());

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live batched multi-session validation"]
fn llama_model_api_can_batch_decode_multiple_sessions() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let root_session_id = model
        .active_session_id()
        .expect("llama should expose sessions");
    let prompt_logits = model.forward_next_logits(&prompt_ids)?;
    let fork_session_id = model.fork_active_session()?;
    let next_token = argmax(&prompt_logits) as u32;

    model.set_active_session(fork_session_id)?;
    let sequential_fork_logits = model.forward_next_logits(&[next_token])?;
    model.set_active_session(root_session_id)?;
    let sequential_root_logits = model.forward_next_logits(&[next_token])?;

    let mut batched = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    batched.reset()?;
    let batched_root_session_id = batched.active_session_id().unwrap();
    let batched_prompt_logits = batched.forward_next_logits(&prompt_ids)?;
    assert_logits_close("batched prompt", &batched_prompt_logits, &prompt_logits);
    let batched_fork_session_id = batched.fork_active_session()?;
    let batched_results = batched.forward_next_logits_batch(&[
        (batched_root_session_id, next_token),
        (batched_fork_session_id, next_token),
    ])?;

    assert_eq!(batched_results.len(), 2);
    assert_eq!(batched_results[0].0, batched_root_session_id);
    assert_eq!(batched_results[1].0, batched_fork_session_id);
    assert_logits_close(
        "batched root",
        &batched_results[0].1,
        &sequential_root_logits,
    );
    assert_logits_close(
        "batched fork",
        &batched_results[1].1,
        &sequential_fork_logits,
    );
    assert_eq!(
        batched.session_position(batched_root_session_id)?,
        prompt_ids.len() as u32 + 1
    );
    assert_eq!(
        batched.session_position(batched_fork_session_id)?,
        prompt_ids.len() as u32 + 1
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live prefix attach validation"]
fn llama_model_api_can_attach_a_prefilled_prefix() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let source_session_id = model.active_session_id().unwrap();
    let prompt_logits = model.prefill_active_session(&prompt_ids)?;
    let prefix = model.capture_active_prefix()?;
    let attached_a = model.attach_prefix(&prefix)?;
    let attached_b = model.attach_prefix(&prefix)?;

    assert_eq!(model.session_position(attached_a)?, prompt_ids.len() as u32);
    assert_eq!(model.session_position(attached_b)?, prompt_ids.len() as u32);
    assert_eq!(
        model.resolve_session_physical_page_ids(source_session_id, 0, 0)?,
        model.resolve_session_physical_page_ids(attached_a, 0, 0)?,
    );
    assert_eq!(
        model.resolve_session_physical_page_ids(attached_a, 0, 0)?,
        model.resolve_session_physical_page_ids(attached_b, 0, 0)?,
    );

    let next_token = argmax(&prompt_logits) as u32;
    model.set_active_session(attached_a)?;
    let attached_a_logits = model.forward_next_logits(&[next_token])?;
    model.set_active_session(attached_b)?;
    let attached_b_logits = model.forward_next_logits(&[next_token])?;
    assert_logits_close(
        "attached prefix decode",
        &attached_a_logits,
        &attached_b_logits,
    );

    let attached_a_pages = model.resolve_session_physical_page_ids(attached_a, 0, 0)?;
    let attached_b_pages = model.resolve_session_physical_page_ids(attached_b, 0, 0)?;
    assert_eq!(
        attached_a_pages[..attached_a_pages.len() - 1],
        attached_b_pages[..attached_b_pages.len() - 1]
    );
    assert_ne!(attached_a_pages.last(), attached_b_pages.last());

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live resident-budget stress validation"]
fn llama_model_api_can_prefill_attached_prefix_suffix_under_resident_page_budget(
) -> dotcache_paged_runtime::Result<()> {
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    model.set_resident_physical_page_budget(Some(1))?;

    let shared_prompt = "hello";
    let shared_prompt_ids = model.encode(shared_prompt, true)?;
    assert!(!shared_prompt_ids.is_empty());

    model.reset()?;
    let _ = model.prefill_active_session(&shared_prompt_ids)?;
    let seed_session_id = model.active_session_id().unwrap();
    let prefix = model.capture_prefix(seed_session_id)?;
    model.close_session(seed_session_id)?;

    let suffix_ids = model.encode(&stress_suffix_text(0, 3), false)?;
    assert!(
        suffix_ids.len() > TOKENS_PER_PAGE,
        "suffix should span multiple pages, got {} tokens",
        suffix_ids.len()
    );

    let attached_session_id = model.attach_prefix(&prefix)?;
    let chunk_size = TOKENS_PER_PAGE.saturating_sub(1).max(1);
    let logits = prefill_session_chunked(&mut model, attached_session_id, &suffix_ids, chunk_size)?;

    assert!(!logits.is_empty());
    assert_eq!(
        model.session_position(attached_session_id)?,
        (shared_prompt_ids.len() + suffix_ids.len()) as u32
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live workload-style resident-budget stress validation"]
fn llama_model_api_runs_multi_session_stress_workload_under_resident_page_budget(
) -> dotcache_paged_runtime::Result<()> {
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    model.set_resident_physical_page_budget(Some(1))?;

    let shared_prompt_ids = model.encode("hello", true)?;
    let chunk_size = TOKENS_PER_PAGE.saturating_sub(1).max(1);

    model.reset()?;
    let seed_prefill_logits = model.prefill_active_session(&shared_prompt_ids)?;
    let seed_source_session_id = model.active_session_id().unwrap();
    let prefix = model.capture_prefix(seed_source_session_id)?;
    model.close_session(seed_source_session_id)?;

    let seed_session_id = model.attach_prefix(&prefix)?;
    let seed_suffix_ids = model.encode(&stress_suffix_text(0, 3), false)?;
    let mut seed_logits =
        prefill_session_chunked(&mut model, seed_session_id, &seed_suffix_ids, chunk_size)?;

    let next_seed_token = argmax(&seed_logits) as u32;
    let decode_results = model.forward_next_logits_batch(&[(seed_session_id, next_seed_token)])?;
    seed_logits = decode_results
        .into_iter()
        .find(|(session_id, _)| *session_id == seed_session_id)
        .map(|(_, logits)| logits)
        .expect("seed session logits should be returned");
    assert!(!seed_prefill_logits.is_empty());
    assert!(!seed_logits.is_empty());

    let session_one_id = model.attach_prefix(&prefix)?;
    let session_two_id = model.attach_prefix(&prefix)?;
    let session_one_suffix_ids = model.encode(&stress_suffix_text(1, 3), false)?;
    let session_two_suffix_ids = model.encode(&stress_suffix_text(2, 3), false)?;

    let session_one_logits = prefill_session_chunked(
        &mut model,
        session_one_id,
        &session_one_suffix_ids,
        chunk_size,
    )?;
    let session_two_logits = prefill_session_chunked(
        &mut model,
        session_two_id,
        &session_two_suffix_ids,
        chunk_size,
    )?;

    assert!(!session_one_logits.is_empty());
    assert!(!session_two_logits.is_empty());
    assert_eq!(
        model.session_position(session_one_id)?,
        (shared_prompt_ids.len() + session_one_suffix_ids.len()) as u32
    );
    assert_eq!(
        model.session_position(session_two_id)?,
        (shared_prompt_ids.len() + session_two_suffix_ids.len()) as u32
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live batched prefill validation"]
fn llama_model_api_can_batch_prefill_sessions() -> dotcache_paged_runtime::Result<()> {
    let prompt_a = "one two three four";
    let prompt_b = "five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_a_ids = model.encode(prompt_a, true)?;
    let prompt_b_ids = model.encode(prompt_b, true)?;

    model.reset()?;
    let session_a = model.active_session_id().unwrap();
    let session_b = model.create_session()?;

    let batch_logits = model.prefill_sessions_batch(&[
        (session_a, prompt_a_ids.as_slice()),
        (session_b, prompt_b_ids.as_slice()),
    ])?;

    let mut sequential = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    sequential.reset()?;
    let sequential_a = sequential.active_session_id().unwrap();
    let sequential_b = sequential.create_session()?;
    let sequential_a_logits = sequential.prefill_session(sequential_a, &prompt_a_ids)?;
    let sequential_b_logits = sequential.prefill_session(sequential_b, &prompt_b_ids)?;

    assert_eq!(batch_logits[0].0, session_a);
    assert_eq!(batch_logits[1].0, session_b);
    assert_logits_close(
        "batched prefill a",
        &batch_logits[0].1,
        &sequential_a_logits,
    );
    assert_logits_close(
        "batched prefill b",
        &batch_logits[1].1,
        &sequential_b_logits,
    );
    assert_eq!(
        model.session_position(session_a)?,
        prompt_a_ids.len() as u32
    );
    assert_eq!(
        model.session_position(session_b)?,
        prompt_b_ids.len() as u32
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live request/session metrics validation"]
fn llama_model_api_tracks_request_and_session_metrics() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;

    model.reset()?;
    model.reset_cache_metrics();
    model.clear_request_metrics();
    let session_id = model.active_session_id().unwrap();
    let _ = model.prefill_active_session(&prompt_ids)?;
    let decode_token = 0u32;
    let _ = model.forward_next_logits(&[decode_token])?;

    let request_metrics = model.request_metrics();
    assert_eq!(request_metrics.len(), 2);
    assert_eq!(request_metrics[0].kind(), SessionRequestKind::Prefill);
    assert_eq!(request_metrics[0].session_ids(), &[session_id]);
    assert_eq!(request_metrics[0].input_token_count(), prompt_ids.len());
    assert_eq!(request_metrics[1].kind(), SessionRequestKind::Decode);
    assert_eq!(request_metrics[1].session_ids(), &[session_id]);
    assert_eq!(request_metrics[1].input_token_count(), 1);

    let session_metrics = model.session_metrics(session_id)?;
    assert_eq!(session_metrics.request_count, 2);
    assert_eq!(session_metrics.prefill_request_count, 1);
    assert_eq!(session_metrics.decode_request_count, 1);
    assert_eq!(session_metrics.input_token_count, prompt_ids.len() + 1);
    assert!(model.cache_metrics().is_some());

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live JSONL trace export validation"]
fn llama_model_api_exports_request_metrics_as_jsonl() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;

    model.reset()?;
    model.clear_request_metrics();
    let session_id = model.active_session_id().unwrap();
    let _ = model.prefill_active_session(&prompt_ids)?;
    let _ = model.forward_next_logits(&[0u32])?;

    let jsonl = model.export_request_metrics_jsonl()?;
    let lines = jsonl.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);

    let prefill: serde_json::Value = serde_json::from_str(lines[0])?;
    let decode: serde_json::Value = serde_json::from_str(lines[1])?;

    assert_eq!(prefill["kind"], "Prefill");
    assert_eq!(prefill["session_ids"], serde_json::json!([session_id]));
    assert_eq!(prefill["input_token_count"], prompt_ids.len());
    assert_eq!(prefill["session_metrics"][0]["session_id"], session_id);

    assert_eq!(decode["kind"], "Decode");
    assert_eq!(decode["session_ids"], serde_json::json!([session_id]));
    assert_eq!(decode["input_token_count"], 1);
    assert_eq!(
        decode["session_metrics"][0]["metrics"]["request_count"],
        serde_json::json!(2)
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live reclaim validation"]
fn llama_model_api_reclaims_physical_pages_after_session_close_and_prefix_release(
) -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let source_session_id = model.active_session_id().unwrap();
    let _ = model.prefill_active_session(&prompt_ids)?;
    let prefix = model.capture_active_prefix()?;
    let attached_session_id = model.attach_prefix(&prefix)?;

    assert!(
        model
            .paged_cache()
            .expect("llama should expose paged cache")
            .physical_page_count()
            > 0
    );

    model.close_session(source_session_id)?;
    model.close_session(attached_session_id)?;
    assert!(
        model
            .paged_cache()
            .expect("llama should expose paged cache")
            .physical_page_count()
            > 0
    );

    model.release_prefix(&prefix)?;
    assert_eq!(
        model
            .paged_cache()
            .expect("llama should expose paged cache")
            .physical_page_count(),
        0
    );

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live spill and restore validation"]
fn llama_model_api_restores_spilled_prefix_pages_on_decode() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six seven eight";
    let mut model = CandleCausalLm::from_hf_with_paging(
        MODEL_ID,
        ModelFamily::Llama,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let _ = model.prefill_active_session(&prompt_ids)?;
    let prefix = model.capture_active_prefix()?;
    let attached_session_id = model.attach_prefix(&prefix)?;
    let spilled = model.spill_prefix(&prefix)?;
    assert!(spilled > 0);
    assert!(model.spilled_physical_page_count() > 0);

    model.set_active_session(attached_session_id)?;
    let next_token = 0u32;
    let _ = model.forward_next_logits(&[next_token])?;
    assert_eq!(model.spilled_physical_page_count(), 0);
    assert!(model.resident_physical_page_count() > 0);

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live qwen2 parity validation"]
fn qwen2_paged_logits_match_native_candle_on_tiny_hf() -> dotcache_paged_runtime::Result<()> {
    let source = HfHubModelSource::new()?;
    let artifacts = source.snapshot(QWEN2_MODEL_ID)?;
    let prompt = "one two three four five six";

    let mut paged = CandleCausalLm::from_artifacts_with_paging(
        artifacts.clone(),
        ModelFamily::Qwen2,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;

    let prompt_ids = paged.encode(prompt, true)?;
    assert!(
        prompt_ids.len() > TOKENS_PER_PAGE,
        "prompt should cross a page boundary, got {} tokens",
        prompt_ids.len()
    );

    let device = Device::Cpu;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&artifacts.weight_paths, DType::F32, &device)?
    };
    let config: CompatQwen2Config = serde_json::from_slice(&fs::read(&artifacts.config_path)?)?;
    let runtime_config = config.into_runtime();
    let mut native_model =
        candle_transformers::models::qwen2::ModelForCausalLM::new(&runtime_config, var_builder)?;

    paged.reset()?;

    let native_prompt_logits = native_model
        .forward(
            &Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &device)?,
            0,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let paged_prompt_logits = paged.forward_next_logits(&prompt_ids)?;
    assert_logits_close("qwen2 prompt", &paged_prompt_logits, &native_prompt_logits);

    let mut next_token = argmax(&native_prompt_logits) as u32;
    for step in 0..DECODE_STEPS {
        let native_logits = native_model
            .forward(
                &Tensor::from_slice(&[next_token], (1, 1), &device)?,
                prompt_ids.len() + step,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let paged_logits = paged.forward_next_logits(&[next_token])?;
        assert_logits_close(
            &format!("qwen2 decode step {}", step + 1),
            &paged_logits,
            &native_logits,
        );
        next_token = argmax(&native_logits) as u32;
    }

    let cache = paged
        .paged_cache()
        .expect("qwen2 parity test should expose a paged cache");
    assert!(cache.physical_page_count() > 0);
    assert!(cache.virtual_page_count() > 0);

    Ok(())
}

#[test]
#[ignore = "downloads a tiny Hugging Face checkpoint for live qwen2 batched multi-session validation"]
fn qwen2_model_api_can_batch_decode_multiple_sessions() -> dotcache_paged_runtime::Result<()> {
    let prompt = "one two three four five six";
    let mut model = CandleCausalLm::from_hf_with_paging(
        QWEN2_MODEL_ID,
        ModelFamily::Qwen2,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    let prompt_ids = model.encode(prompt, true)?;
    assert!(prompt_ids.len() > TOKENS_PER_PAGE);

    model.reset()?;
    let root_session_id = model
        .active_session_id()
        .expect("qwen2 should expose sessions");
    let prompt_logits = model.forward_next_logits(&prompt_ids)?;
    let fork_session_id = model.fork_active_session()?;
    let next_token = argmax(&prompt_logits) as u32;

    model.set_active_session(fork_session_id)?;
    let sequential_fork_logits = model.forward_next_logits(&[next_token])?;
    model.set_active_session(root_session_id)?;
    let sequential_root_logits = model.forward_next_logits(&[next_token])?;

    let mut batched = CandleCausalLm::from_hf_with_paging(
        QWEN2_MODEL_ID,
        ModelFamily::Qwen2,
        CandleDeviceSelector::Cpu,
        DType::F32,
        TOKENS_PER_PAGE,
    )?;
    batched.reset()?;
    let batched_root_session_id = batched.active_session_id().unwrap();
    let batched_prompt_logits = batched.forward_next_logits(&prompt_ids)?;
    assert_logits_close(
        "qwen2 batched prompt",
        &batched_prompt_logits,
        &prompt_logits,
    );
    let batched_fork_session_id = batched.fork_active_session()?;
    let batched_results = batched.forward_next_logits_batch(&[
        (batched_root_session_id, next_token),
        (batched_fork_session_id, next_token),
    ])?;

    assert_eq!(batched_results.len(), 2);
    assert_eq!(batched_results[0].0, batched_root_session_id);
    assert_eq!(batched_results[1].0, batched_fork_session_id);
    assert_logits_close(
        "qwen2 batched root",
        &batched_results[0].1,
        &sequential_root_logits,
    );
    assert_logits_close(
        "qwen2 batched fork",
        &batched_results[1].1,
        &sequential_fork_logits,
    );
    assert_eq!(
        batched.session_position(batched_root_session_id)?,
        prompt_ids.len() as u32 + 1
    );
    assert_eq!(
        batched.session_position(batched_fork_session_id)?,
        prompt_ids.len() as u32 + 1
    );

    Ok(())
}
