"""32K certified attention benchmark.

Runs certified attention at 32K context inside the LLaMA forward pass,
comparing against dense (standard HF) attention.

Key result: 32K fits on RTX 5090 (32GB) with both dense and certified.
Certified achieves 78.7% skip rate at 32K vs 51% at 8K.
"""
from __future__ import annotations

import os
import sys
import gc
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.build_heterogeneous_context import build_context

LAYER_EPS = {}
for _lid in range(6):
    LAYER_EPS[_lid] = 1e-3
for _lid in range(6, 29):
    LAYER_EPS[_lid] = 1e-4
for _lid in range(29, 32):
    LAYER_EPS[_lid] = 1e-5


def run_32k_benchmark(model, adapter, tokenizer, prompt_text, ctx_len, gen_steps):
    """Run certified vs dense at a given context length."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {seq_len} tokens")

    # === Phase 1: Dense prefill + build tiered caches ===
    adapter.set_mode("dense")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_kv = outputs.past_key_values
    peak_prefill = torch.cuda.max_memory_allocated()
    print(f"  Prefill: {prefill_ms:.0f} ms, peak {peak_prefill/1e9:.2f} GB")

    # Build tiered caches
    adapter.load_certified_cache(past_kv, layer_epsilons=LAYER_EPS, default_epsilon=1e-4)
    tiered_vram = sum(c.vram_bytes() for c in adapter.certified_state.tiered_caches.values())
    tiered_cpu = sum(c.cpu_bytes() for c in adapter.certified_state.tiered_caches.values())
    print(f"  Tiered: {tiered_vram/1e6:.0f} MB VRAM, {tiered_cpu/1e6:.0f} MB CPU")

    # Free prefill KV
    del outputs, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 2: Certified decode ===
    adapter.set_mode("certified")
    adapter.certified_state.collect_stats = False
    cache_position = torch.tensor([seq_len], dtype=torch.long, device="cuda")
    current_input = first_token

    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            out = model(input_ids=current_input, use_cache=False,
                        cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
        current_input = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache_position = cache_position + 1

    # Stats sample
    adapter.certified_state.collect_stats = True
    adapter.certified_state.clear_step_stats()
    with torch.inference_mode():
        out = model(input_ids=current_input, use_cache=False,
                    cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
    current_input = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache_position = cache_position + 1
    stats = adapter.certified_state.aggregate_step_stats()
    skip_rate = stats["skip_rate"]

    # Timed decode
    adapter.certified_state.collect_stats = False
    cert_tokens = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(gen_steps):
        with torch.inference_mode():
            out = model(input_ids=current_input, use_cache=False,
                        cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
        tid = out.logits[:, -1, :].argmax(dim=-1).item()
        cert_tokens.append(tid)
        current_input = torch.tensor([[tid]], dtype=torch.long, device="cuda")
        cache_position = cache_position + 1
    torch.cuda.synchronize()
    cert_time = time.perf_counter() - t0
    cert_peak = torch.cuda.max_memory_allocated()

    # === Phase 3: Dense decode for comparison ===
    adapter.clear()
    adapter.set_mode("dense")
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        outputs_d = model(**inputs, use_cache=True)
    past_kv_d = outputs_d.past_key_values
    first_token_d = outputs_d.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del outputs_d
    gc.collect()
    torch.cuda.empty_cache()

    dense_tokens = []
    current_input = first_token_d
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(gen_steps):
        with torch.inference_mode():
            out = model(input_ids=current_input, past_key_values=past_kv_d, use_cache=True)
        tid = out.logits[:, -1, :].argmax(dim=-1).item()
        dense_tokens.append(tid)
        past_kv_d = out.past_key_values
        current_input = torch.tensor([[tid]], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    dense_time = time.perf_counter() - t0
    dense_peak = torch.cuda.max_memory_allocated()

    del past_kv_d
    gc.collect()
    torch.cuda.empty_cache()

    speedup = (dense_time - cert_time) / dense_time
    return {
        "seq_len": seq_len,
        "gen_steps": gen_steps,
        "prefill_ms": prefill_ms,
        "peak_prefill_gb": peak_prefill / 1e9,
        "dense_ms_per_step": dense_time / gen_steps * 1000,
        "dense_peak_gb": dense_peak / 1e9,
        "cert_ms_per_step": cert_time / gen_steps * 1000,
        "cert_peak_gb": cert_peak / 1e9,
        "speedup": speedup,
        "skip_rate": skip_rate,
        "tiered_vram_mb": tiered_vram / 1e6,
        "tiered_cpu_mb": tiered_cpu / 1e6,
        "vram_saved_gb": (dense_peak - cert_peak) / 1e9,
        "dense_text": tokenizer.decode(dense_tokens, skip_special_tokens=True)[:80],
        "cert_text": tokenizer.decode(cert_tokens, skip_special_tokens=True)[:80],
        "per_layer_skip": stats.get("per_layer_skip_rate", {}),
    }


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()
    print(f"Model VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    config = DotCacheConfig(head_dim=model.config.hidden_size // model.config.num_attention_heads)
    adapter = LlamaDotCacheModelAdapter(model, config)

    results = []
    gen_steps = 32

    for ctx_len, label in [(8192, "8K"), (16384, "16K"), (32768, "32K")]:
        print(f"\n{'='*60}")
        print(f"  {label} context, {gen_steps} decode steps")
        print(f"{'='*60}")

        prompt = build_context(ctx_len, question_idx=0)
        r = run_32k_benchmark(model, adapter, tokenizer, prompt, ctx_len, gen_steps)
        r["context"] = label
        results.append(r)

        print(f"  Dense:     {r['dense_ms_per_step']:.1f} ms/step, peak {r['dense_peak_gb']:.2f} GB")
        print(f"  Certified: {r['cert_ms_per_step']:.1f} ms/step, peak {r['cert_peak_gb']:.2f} GB")
        print(f"  Speedup:   {r['speedup']:+.1%}")
        print(f"  Skip rate: {r['skip_rate']:.1%}")
        print(f"  VRAM saved: {r['vram_saved_gb']:.2f} GB")

        adapter.clear()
        adapter.set_mode("dense")
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print(f"CONTEXT SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ctx':>5} {'Dense ms':>9} {'Cert ms':>8} {'Speedup':>8} {'Skip':>6} {'Dense GB':>9} {'Cert GB':>8} {'Saved':>6}")
    for r in results:
        print(f"  {r['context']:>3} {r['dense_ms_per_step']:>8.1f} {r['cert_ms_per_step']:>7.1f} "
              f"{r['speedup']:>7.1%} {r['skip_rate']:>5.1%} "
              f"{r['dense_peak_gb']:>8.2f} {r['cert_peak_gb']:>7.2f} {r['vram_saved_gb']:>5.2f}")

    out_path = Path("benchmarks/results/certified_context_scaling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
