"""Profile a single cert decode step at 64K full-mirror using torch profiler.

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
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    from dotcache.config import DotCacheConfig

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading Llama-3.1-8B (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", token=token)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B", quantization_config=quant, device_map="auto", token=token,
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
    tiered = create_tiered_cache_from_model(past_kv, layer_ids, fp16_key_cache_capacity=None)
    del past_kv, out
    gc.collect(); torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered, layer_epsilons={},
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
