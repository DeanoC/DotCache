"""64K certified decode throughput WITHOUT phase timers.

The breakdown bench's `_PhaseTimer` syncs the CUDA stream at every region
exit (~128 syncs/step), serialising kernel pipelining. This bench removes
all phase timing and reports the production tok/s the paper would measure.

Toggles the split-K hybrid kernel via DOTCACHE_FAST_ATTEND and runs both
configurations from a single model load.
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
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


def timed_decode(model, adapter, cache_position, first_token, device, steps: int, warmup: int):
    """Run `warmup` untimed steps, then `steps` timed steps. Returns per-step ms list."""
    current_input = first_token

    for _ in range(warmup):
        with torch.inference_mode():
            out = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    # Timed — one sync at the end; no per-step sync inside the loop.
    per_step_ms: list[float] = []
    for _ in range(steps):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        with torch.inference_mode():
            out = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        end_evt.record()
        # Record only — do not sync yet (events sit on the stream).
        per_step_ms.append((start_evt, end_evt))
        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    torch.cuda.synchronize()
    ms_list = [s.elapsed_time(e) for s, e in per_step_ms]
    return ms_list, current_input, cache_position


def run_config(model, adapter, device, tokenizer, ctx_len: int, steps: int, warmup: int, fast: bool):
    os.environ["DOTCACHE_FAST_ATTEND"] = "1" if fast else "0"

    from dotcache.integrations.llama import (
        CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    prompt = build_prefill(tokenizer, ctx_len)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(device)
    seq_len = ids["input_ids"].shape[1]

    adapter.clear()
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    _env_cap = os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS")
    _cap = None if _env_cap is None or _env_cap == "" else int(_env_cap)
    tiered = create_tiered_cache_from_model(past_kv, layer_ids, fp16_key_cache_capacity=_cap)
    del past_kv
    gc.collect(); torch.cuda.empty_cache()

    _per_kv_group = os.environ.get("DOTCACHE_PER_KV_GROUP_TOPK", "0") == "1"
    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        v_tolerance=0.5,
        collect_stats=False,  # << off
        append_kv=True,
        top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=False,  # << off, cosmetic on hot path
        eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
        per_kv_group_topk=_per_kv_group,
        phase_timings=None,  # << no sync-per-region
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    ms_list, _, _ = timed_decode(model, adapter, cache_position, first_token, device, steps, warmup)
    adapter.clear()
    gc.collect(); torch.cuda.empty_cache()
    return ms_list, seq_len


def run_dense(model, adapter, device, tokenizer, ctx_len: int, steps: int, warmup: int):
    prompt = build_prefill(tokenizer, ctx_len)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(device)
    seq_len = ids["input_ids"].shape[1]

    adapter.clear()
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token

    for _ in range(warmup):
        with torch.inference_mode():
            out = model(input_ids=current_input, past_key_values=past_kv,
                        use_cache=True,
                        cache_position=cache_position,
                        position_ids=cache_position.unsqueeze(0))
        past_kv = out.past_key_values
        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    per_step = []
    for _ in range(steps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.inference_mode():
            out = model(input_ids=current_input, past_key_values=past_kv,
                        use_cache=True,
                        cache_position=cache_position,
                        position_ids=cache_position.unsqueeze(0))
        e.record()
        per_step.append((s, e))
        past_kv = out.past_key_values
        tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1
    torch.cuda.synchronize()
    ms_list = [s.elapsed_time(e) for s, e in per_step]
    del past_kv
    gc.collect(); torch.cuda.empty_cache()
    return ms_list, seq_len


def summarise(label: str, ms: list[float]):
    ms_sorted = sorted(ms)
    n = len(ms_sorted)
    mean = statistics.mean(ms)
    p50 = ms_sorted[n // 2]
    p95 = ms_sorted[min(n - 1, int(0.95 * (n - 1)))]
    p99 = ms_sorted[min(n - 1, int(0.99 * (n - 1)))]
    print(f"  {label:<24}  mean {mean:>7.2f} ms  p50 {p50:>7.2f} ms  p95 {p95:>7.2f} ms  p99 {p99:>7.2f} ms  "
          f"tok/s {1000/mean:>6.2f}")
    return mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--context-length", type=int, default=65536)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=16)
    ap.add_argument("--modes", default="dense,slow,fast",
                    help="comma-separated subset of {dense, slow, fast}")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

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

    print(f"Context: {args.context_length}  decode steps: {args.decode_steps} (warmup {args.warmup_steps})\n")
    print(f"=== Per-step decode ms (no phase timer) ===")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}

    if "dense" in modes:
        ms, _ = run_dense(model, adapter, device, tokenizer,
                          args.context_length, args.decode_steps, args.warmup_steps)
        results["dense"] = summarise("dense (baseline)", ms)

    if "slow" in modes:
        ms, _ = run_config(model, adapter, device, tokenizer,
                           args.context_length, args.decode_steps, args.warmup_steps, fast=False)
        results["slow"] = summarise("cert FAST_ATTEND=0", ms)

    if "fast" in modes:
        ms, _ = run_config(model, adapter, device, tokenizer,
                           args.context_length, args.decode_steps, args.warmup_steps, fast=True)
        results["fast"] = summarise("cert FAST_ATTEND=1", ms)

    print(f"\n=== Ratios ===")
    if "dense" in results and "slow" in results:
        print(f"  cert-slow / dense: {results['slow']/results['dense']:.2f}x")
    if "dense" in results and "fast" in results:
        print(f"  cert-fast / dense: {results['fast']/results['dense']:.2f}x")
    if "slow" in results and "fast" in results:
        print(f"  cert-slow / cert-fast: {results['slow']/results['fast']:.2f}x")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
