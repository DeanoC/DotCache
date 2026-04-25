"""Profile a single cert decode step at 64K using torch profiler.

Runs a few warmup steps, then captures one fully-traced step with
torch.profiler so we can see exactly which kernels eat time.

Output: a .json Chrome trace and a printed self_cuda_time summary.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch


def resolve_fp16_key_cache_blocks(spec: str, seq_len: int, decode_budget: int, block_size: int) -> int | None:
    text = str(spec).strip().lower()
    if text in ("", "paper"):
        return 512
    if text in ("full", "none"):
        return None
    if text in ("inf", "infinite"):
        return (int(seq_len) + int(decode_budget) + int(block_size) - 1) // int(block_size)
    cap = int(text)
    if cap < 0:
        raise ValueError(f"fp16 key cache blocks must be >= 0, got {cap}")
    return cap


def build_prefill(tokenizer, context_tokens: int) -> str:
    FILLER = (
        "The history of mathematics spans thousands of years and encompasses many "
        "different cultures and civilizations. "
    )
    question = "\nContinue:"
    ft = len(tokenizer.encode(FILLER, add_special_tokens=False))
    qt = len(tokenizer.encode(question, add_special_tokens=False))
    avail = context_tokens - qt - 50
    nb = max(avail // ft, 2)
    return FILLER * nb + question


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-length", type=int, default=65536)
    ap.add_argument("--warmup-steps", type=int, default=24)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--output", default="benchmarks/results/perf_tests_20260422/cert_step_trace.json")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--use-int4-values", action="store_true")
    ap.add_argument("--fp16-key-cache-blocks", default="3584",
                    help="integer, full, or infinite. 'infinite' is profiling-only.")
    ap.add_argument("--fp16-value-cache-blocks", default="1536",
                    help="integer, full, or infinite. 'infinite' is profiling-only.")
    ap.add_argument("--fast-attend", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )
    from dotcache.config import DotCacheConfig

    os.environ["DOTCACHE_FAST_ATTEND"] = "1" if args.fast_attend else "0"
    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", token=token,
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, cfg)
    device = next(model.parameters()).device

    prompt = build_prefill(tokenizer, args.context_length)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_length).to(device)
    seq_len = ids["input_ids"].shape[1]

    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    decode_budget = int(args.warmup_steps) + int(args.profile_steps) + 8
    key_cap = resolve_fp16_key_cache_blocks(
        args.fp16_key_cache_blocks, seq_len, decode_budget, block_size=16,
    )
    value_cap = resolve_fp16_key_cache_blocks(
        args.fp16_value_cache_blocks, seq_len, decode_budget, block_size=16,
    )
    if args.use_int4_values:
        tiered = create_tiered_cache_int4v_from_model(
            past_kv,
            layer_ids,
            group_size=16,
            max_new_tokens=decode_budget,
            fp16_key_cache_capacity=key_cap,
            fp16_value_cache_capacity=value_cap,
        )
        if str(args.fp16_key_cache_blocks).strip().lower() in ("inf", "infinite"):
            active_blocks = max(c.active_blocks for c in tiered.values())
            for c in tiered.values():
                c.ensure_fp16_keys_resident(range(active_blocks))
        if str(args.fp16_value_cache_blocks).strip().lower() in ("inf", "infinite"):
            active_blocks = max(c.active_blocks for c in tiered.values())
            for c in tiered.values():
                c.ensure_fp16_values_resident(range(active_blocks))
    else:
        tiered = create_tiered_cache_from_model(
            past_kv,
            layer_ids,
            max_new_tokens=decode_budget,
            fp16_key_cache_capacity=key_cap,
        )
    del past_kv, out
    gc.collect(); torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        v_tolerance=0.05,
        collect_stats=False, append_kv=True, top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=False, eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
        phase_timings=None,
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_tok

    # Warmup
    for _ in range(args.warmup_steps):
        with torch.inference_mode():
            out = model(input_ids=current_input, use_cache=False,
                        cache_position=cache_position,
                        position_ids=cache_position.unsqueeze(0))
        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    torch.cuda.synchronize()
    print(f"Profiling {args.profile_steps} cert decode step(s)...")

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1),
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.profile_steps + 1):  # +1 for the "warmup" step in schedule
            with torch.inference_mode():
                out = model(input_ids=current_input, use_cache=False,
                            cache_position=cache_position,
                            position_ids=cache_position.unsqueeze(0))
            tid = out.logits[:, -1, :].argmax(dim=-1)
            current_input = tid.view(1, 1)
            cache_position = cache_position + 1
            prof.step()

    # Write chrome trace + print summary.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(args.output)
    print(f"Wrote {args.output}")

    print("\n=== Top 30 by self_cuda_time_total (μs) ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    ))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
