"""Integrated certified attention benchmark.

Runs certified attention INSIDE the LLaMA forward pass via the adapter,
comparing against dense (standard HF) attention. This eliminates the
64× Python dispatch overhead from bench_tiered_vs_oracle.py.

Measures: token match, cosine similarity, wall-clock latency, VRAM usage.
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

# Per-layer epsilon profile for LLaMA 3.1-8B (calibrated for cos≥0.999)
LAYER_EPS = {}
for _lid in range(6):
    LAYER_EPS[_lid] = 1e-3
for _lid in range(6, 29):
    LAYER_EPS[_lid] = 1e-4
for _lid in range(29, 32):
    LAYER_EPS[_lid] = 1e-5


def run_dense_generation(model, tokenizer, prompt_text, ctx_len, gen_steps):
    """Run standard HF generation (dense attention) and return tokens + hidden states."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values

    current_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(gen_steps):
        with torch.inference_mode():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
        token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
        generated_tokens.append(token_id)
        past_kv = outputs.past_key_values
        current_token = torch.tensor([[token_id]], dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    dense_time = time.perf_counter() - t0

    return {
        "tokens": generated_tokens,
        "time_s": dense_time,
        "seq_len": seq_len,
        "past_kv": past_kv,
    }


def run_certified_generation(model, adapter, tokenizer, prompt_text, ctx_len, gen_steps):
    """Run generation with certified attention inside the forward pass."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Phase 1: Dense prefill to get KV cache + first token
    adapter.set_mode("dense")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prefill_kv = outputs.past_key_values

    # Phase 2: Build tiered caches from prefill KV
    adapter.load_certified_cache(
        prefill_kv,
        layer_epsilons=LAYER_EPS,
        default_epsilon=1e-4,
        block_size=16,
    )

    tiered_vram = sum(c.vram_bytes() for c in adapter.certified_state.tiered_caches.values())
    tiered_cpu = sum(c.cpu_bytes() for c in adapter.certified_state.tiered_caches.values())

    # Free prefill KV (now in tiered storage)
    del prefill_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: Certified decode
    adapter.set_mode("certified")
    adapter.reset_runtime_metrics()
    current_input = first_token
    cache_position = torch.tensor([seq_len], dtype=torch.long, device="cuda")

    # Warmup: 3 steps (JIT compile Triton kernels)
    for _ in range(min(3, gen_steps)):
        with torch.inference_mode():
            warmup_out = model(
                input_ids=current_input,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        current_input = warmup_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache_position = cache_position + 1

    # Collect stats for one step (before timed run)
    adapter.certified_state.collect_stats = True
    adapter.certified_state.clear_step_stats()
    with torch.inference_mode():
        stat_out = model(
            input_ids=current_input,
            use_cache=False,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
        )
    current_input = stat_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache_position = cache_position + 1
    sample_stats = adapter.certified_state.aggregate_step_stats()

    # Timed run: disable stats to avoid GPU sync overhead
    adapter.certified_state.collect_stats = False
    generated_tokens = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(gen_steps):
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
        generated_tokens.append(token_id)
        current_input = torch.tensor([[token_id]], dtype=torch.long, device="cuda")
        cache_position = cache_position + 1

    torch.cuda.synchronize()
    certified_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated()

    skip_rates = [sample_stats.get("skip_rate", 0.0)]

    return {
        "tokens": generated_tokens,
        "time_s": certified_time,
        "seq_len": seq_len,
        "tiered_vram_mb": tiered_vram / 1e6,
        "tiered_cpu_mb": tiered_cpu / 1e6,
        "peak_vram_gb": peak_vram / 1e9,
        "skip_rate_mean": float(np.mean(skip_rates)) if skip_rates else 0.0,
        "skip_rate_min": float(np.min(skip_rates)) if skip_rates else 0.0,
        "sample_stats": sample_stats,
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
    model_vram = torch.cuda.memory_allocated()
    print(f"Model VRAM: {model_vram / 1e9:.2f} GB")

    # Install adapter (head_dim=128 for LLaMA 3.1-8B)
    config = DotCacheConfig(head_dim=model.config.hidden_size // model.config.num_attention_heads)
    adapter = LlamaDotCacheModelAdapter(model, config)

    results = []
    gen_steps = 32
    ctx_len = 8192

    prompts = [
        ("hetero_8k", build_context(ctx_len, question_idx=0)),
        ("code_8k", Path("dotcache/integrations/llama.py").read_text()[:30000]),
    ]

    for pname, ptxt in prompts:
        print(f"\n{'='*70}")
        print(f"  {pname}: {gen_steps} decode steps at {ctx_len} context")
        print(f"{'='*70}")

        # Dense baseline
        print(f"\n  [Dense] Running...")
        torch.cuda.reset_peak_memory_stats()
        dense = run_dense_generation(model, tokenizer, ptxt, ctx_len, gen_steps)
        dense_peak = torch.cuda.max_memory_allocated()
        print(f"  [Dense] {gen_steps} steps in {dense['time_s']*1000:.1f} ms "
              f"({dense['time_s']/gen_steps*1000:.1f} ms/step)")
        print(f"  [Dense] Peak VRAM: {dense_peak/1e9:.2f} GB")
        print(f"  [Dense] Text: {tokenizer.decode(dense['tokens'][:20], skip_special_tokens=True)[:80]}")

        # Clean up dense KV
        del dense["past_kv"]
        gc.collect()
        torch.cuda.empty_cache()

        # Certified
        print(f"\n  [Certified] Running...")
        certified = run_certified_generation(model, adapter, tokenizer, ptxt, ctx_len, gen_steps)
        print(f"  [Certified] {gen_steps} steps in {certified['time_s']*1000:.1f} ms "
              f"({certified['time_s']/gen_steps*1000:.1f} ms/step)")
        print(f"  [Certified] Peak VRAM: {certified['peak_vram_gb']:.2f} GB")
        print(f"  [Certified] Skip rate: {certified['skip_rate_mean']:.1%}")
        print(f"  [Certified] Tiered: {certified['tiered_vram_mb']:.0f} MB VRAM, "
              f"{certified['tiered_cpu_mb']:.0f} MB CPU")
        print(f"  [Certified] Text: {tokenizer.decode(certified['tokens'][:20], skip_special_tokens=True)[:80]}")

        # Compare
        token_match = sum(1 for a, b in zip(dense["tokens"], certified["tokens"]) if a == b)
        dense_ms = dense["time_s"] * 1000
        cert_ms = certified["time_s"] * 1000
        speedup = (dense_ms - cert_ms) / dense_ms

        print(f"\n  --- Comparison ---")
        print(f"  Token match: {token_match}/{gen_steps}")
        print(f"  Dense: {dense_ms:.1f} ms total, {dense_ms/gen_steps:.1f} ms/step")
        print(f"  Certified: {cert_ms:.1f} ms total, {cert_ms/gen_steps:.1f} ms/step")
        print(f"  Speedup: {speedup:+.1%}")

        r = {
            "prompt": pname,
            "context": f"{ctx_len//1024}K",
            "gen_steps": gen_steps,
            "token_match": f"{token_match}/{gen_steps}",
            "dense_ms": dense_ms,
            "dense_ms_per_step": dense_ms / gen_steps,
            "certified_ms": cert_ms,
            "certified_ms_per_step": cert_ms / gen_steps,
            "speedup": speedup,
            "skip_rate_mean": certified["skip_rate_mean"],
            "tiered_vram_mb": certified["tiered_vram_mb"],
            "tiered_cpu_mb": certified["tiered_cpu_mb"],
            "peak_vram_gb": certified["peak_vram_gb"],
            "dense_text": tokenizer.decode(dense["tokens"], skip_special_tokens=True)[:80],
            "certified_text": tokenizer.decode(certified["tokens"], skip_special_tokens=True)[:80],
        }
        results.append(r)

        # Clean up
        adapter.clear()
        adapter.set_mode("dense")
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Prompt':<15} {'Match':>7} {'Dense ms/step':>14} {'Cert ms/step':>13} {'Speedup':>8} {'Skip':>6} {'VRAM MB':>8}")
    for r in results:
        print(f"  {r['prompt']:<13} {r['token_match']:>7} {r['dense_ms_per_step']:>13.1f} "
              f"{r['certified_ms_per_step']:>12.1f} {r['speedup']:>7.1%} "
              f"{r['skip_rate_mean']:>5.1%} {r['tiered_vram_mb']:>7.0f}")

    out_path = Path("benchmarks/results/certified_integrated.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
