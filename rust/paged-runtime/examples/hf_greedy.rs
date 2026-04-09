#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::DType;
    use dotcache_paged_runtime::{
        greedy_generate, AttentionPathMode, CandleCausalLm, CandleDeviceSelector, ModelFamily,
        RuntimeMode,
    };

    let mut args = std::env::args().skip(1);
    let family = args.next().ok_or(
        "usage: hf_greedy <family> <model_id> <prompt> [max_new_tokens] [tokens_per_page] [trace_jsonl_path] [--device cpu|metal[:ordinal]|cuda[:ordinal]] [--runtime-mode dense_control|paged_control|dotcache_experimental] [--attention-path paged|fused]",
    )?;
    let model_id = args.next().ok_or("missing model_id")?;
    let prompt = args.next().ok_or("missing prompt")?;
    let mut positional = Vec::new();
    let mut device = CandleDeviceSelector::Cpu;
    let mut runtime_mode = RuntimeMode::PagedControl;
    let mut attention_path = None;
    while let Some(arg) = args.next() {
        if arg == "--device" {
            let value = args.next().ok_or("missing value for --device")?;
            device = value.parse::<CandleDeviceSelector>()?;
        } else if arg == "--runtime-mode" {
            let value = args.next().ok_or("missing value for --runtime-mode")?;
            runtime_mode = value.parse::<RuntimeMode>()?;
        } else if arg == "--attention-path" {
            let value = args.next().ok_or("missing value for --attention-path")?;
            attention_path = Some(value.parse::<AttentionPathMode>()?);
        } else {
            positional.push(arg);
        }
    }

    let max_new_tokens = positional
        .first()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let (tokens_per_page, trace_jsonl_path) = match (positional.get(1), positional.get(2)) {
        (None, None) => (CandleCausalLm::DEFAULT_TOKENS_PER_PAGE, None),
        (Some(value), None) => match value.parse::<usize>() {
            Ok(tokens_per_page) => (tokens_per_page, None),
            Err(_) => (CandleCausalLm::DEFAULT_TOKENS_PER_PAGE, Some(value.clone())),
        },
        (Some(tokens_per_page), Some(trace_jsonl_path)) => (
            tokens_per_page.parse::<usize>()?,
            Some(trace_jsonl_path.clone()),
        ),
        (None, Some(_)) => unreachable!("trace path cannot be provided without the preceding slot"),
    };

    let family: ModelFamily = family.parse()?;
    let mut model = CandleCausalLm::from_hf_with_runtime_mode(
        &model_id,
        family,
        device.clone(),
        DType::F32,
        tokens_per_page,
        runtime_mode,
    )?;
    if let Some(attention_path) = attention_path {
        model.set_attention_path(attention_path);
    }
    let generation = greedy_generate(&mut model, &prompt, max_new_tokens)?;
    let attention_path = model.attention_path();

    println!("{}", generation.text);
    eprintln!(
        "device={device} runtime_mode={} attention_path={attention_path}",
        model.runtime_mode()
    );
    if let Some(cache) = model.paged_cache() {
        eprintln!(
            "paged physical_pages={} virtual_pages={} tokens={} tokens_per_page={}",
            cache.physical_page_count(),
            cache.virtual_page_count(),
            cache.total_token_count(),
            cache.tokens_per_page(),
        );
    }
    if let Some(trace_jsonl_path) = trace_jsonl_path {
        model.write_request_metrics_jsonl(&trace_jsonl_path)?;
        eprintln!("wrote request trace to {}", trace_jsonl_path);
    }
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the `candle` feature to run this example");
}
