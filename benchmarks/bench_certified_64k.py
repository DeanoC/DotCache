"""64K certified attention benchmark with INT8 model + chunked prefill.

INT8 model (9 GB) + chunked prefill (8K chunks) + tiered KV cache
enables 64K context on 32GB GPU.
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

CHUNK_SIZE = 8192


def chunked_prefill(model, input_ids, chunk_size=CHUNK_SIZE):
    """Prefill in chunks to avoid MLP activation OOM at long contexts."""
    seq_len = input_ids.shape[1]
    past_kv = None
    first_token = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]

        with torch.inference_mode():
            if past_kv is None:
                out = model(input_ids=chunk_ids, use_cache=True)
            else:
                pos_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
                out = model(input_ids=chunk_ids, past_key_values=past_kv,
                           use_cache=True, position_ids=pos_ids)
            past_kv = out.past_key_values

        if end == seq_len:
            first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        del out
        gc.collect()
        torch.cuda.empty_cache()

    return past_kv, first_token


def run_benchmark(model, adapter, tokenizer, ctx_len, gen_steps=32):
    """Run certified vs dense at a given context length."""
    prompt = build_context(ctx_len, question_idx=0)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {seq_len} tokens")

    # Use chunked prefill for contexts > 32K
    use_chunked = seq_len > 32768

    # === Dense prefill ===
    adapter.set_mode("dense")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if use_chunked:
        past_kv, first_token = chunked_prefill(model, inputs["input_ids"])
    else:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True)
        first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_kv = outputs.past_key_values
        del outputs
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    peak_prefill = torch.cuda.max_memory_allocated()
    print(f"  Prefill: {prefill_ms:.0f} ms, peak {peak_prefill/1e9:.2f} GB"
          + (" (chunked)" if use_chunked else ""))

    # Build tiered caches
    t0 = time.perf_counter()
    adapter.load_certified_cache(past_kv, layer_epsilons=LAYER_EPS, default_epsilon=1e-4)
    torch.cuda.synchronize()
    tiered_ms = (time.perf_counter() - t0) * 1000
    tiered_vram = sum(c.vram_bytes() for c in adapter.certified_state.tiered_caches.values())
    tiered_cpu = sum(c.cpu_bytes() for c in adapter.certified_state.tiered_caches.values())
    print(f"  Tiered: {tiered_vram/1e6:.0f} MB VRAM, {tiered_cpu/1e6:.0f} MB CPU ({tiered_ms:.0f} ms)")

    # Free prefill KV
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()
    post_free = torch.cuda.memory_allocated()
    print(f"  After free: {post_free/1e9:.2f} GB")

    # === Certified decode ===
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
    # Paper-1 attends every block; this is the fraction served from the
    # cheap INT8-key tail (not in adaptive top-K*). Read the new key with
    # a fallback to the legacy alias for older state objects.
    int8_tail_rate = stats.get("int8_tail_rate", stats.get("skip_rate", 0.0))

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

    # === Dense decode ===
    adapter.clear()
    adapter.set_mode("dense")
    gc.collect()
    torch.cuda.empty_cache()

    # Re-prefill for dense
    if use_chunked:
        past_kv_d, first_token_d = chunked_prefill(model, inputs["input_ids"])
    else:
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
    r = {
        "seq_len": seq_len,
        "gen_steps": gen_steps,
        "prefill_ms": prefill_ms,
        "peak_prefill_gb": peak_prefill / 1e9,
        "chunked_prefill": use_chunked,
        "dense_ms_per_step": dense_time / gen_steps * 1000,
        "dense_peak_gb": dense_peak / 1e9,
        "cert_ms_per_step": cert_time / gen_steps * 1000,
        "cert_peak_gb": cert_peak / 1e9,
        "speedup": speedup,
        "int8_tail_rate": int8_tail_rate,
        # Legacy alias (Paper-2 vocabulary). Every block is attended; this
        # is the fraction served from INT8 keys.
        "skip_rate": int8_tail_rate,
        "tiered_vram_mb": tiered_vram / 1e6,
        "tiered_cpu_mb": tiered_cpu / 1e6,
        "vram_saved_gb": (dense_peak - cert_peak) / 1e9,
        "dense_text": tokenizer.decode(dense_tokens, skip_special_tokens=True)[:80],
        "cert_text": tokenizer.decode(cert_tokens, skip_special_tokens=True)[:80],
    }

    print(f"  Dense:     {r['dense_ms_per_step']:.1f} ms/step, peak {r['dense_peak_gb']:.2f} GB")
    print(f"  Certified: {r['cert_ms_per_step']:.1f} ms/step, peak {r['cert_peak_gb']:.2f} GB")
    print(f"  Speedup:   {r['speedup']:+.1%}")
    print(f"  INT8-tail rate: {r['int8_tail_rate']:.1%} "
          f"(fraction served from cheap INT8-key path; every block attended)")
    print(f"  VRAM saved: {r['vram_saved_gb']:.2f} GB")
    return r


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_id} (INT8 weights)...")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto", token=token,
    )
    model.eval()
    print(f"Model VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB (INT8)")

    config = DotCacheConfig(head_dim=model.config.hidden_size // model.config.num_attention_heads)
    adapter = LlamaDotCacheModelAdapter(model, config)

    results = []
    gen_steps = 32

    for ctx_len, label in [(8192, "8K"), (16384, "16K"), (32768, "32K"), (49152, "48K"), (65536, "64K")]:
        print(f"\n{'='*60}")
        print(f"  {label} context, {gen_steps} decode steps (INT8 model)")
        print(f"{'='*60}")

        try:
            r = run_benchmark(model, adapter, tokenizer, ctx_len, gen_steps)
            r["context"] = label
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {str(e)[:100]}")
            results.append({"context": label, "error": "OOM"})

        adapter.clear()
        adapter.set_mode("dense")
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"CONTEXT SCALING (INT8 model, chunked prefill)")
    print(f"{'='*60}")
    print(f"{'Ctx':>5} {'Dense ms':>9} {'Cert ms':>8} {'Ratio':>6} {'INT8-tail':>10} {'Dense GB':>9} {'Cert GB':>8} {'Saved':>6}")
    for r in results:
        if "error" in r:
            print(f"  {r['context']:>3} {'OOM':>9}")
        else:
            ratio = r['cert_ms_per_step'] / r['dense_ms_per_step']
            print(f"  {r['context']:>3} {r['dense_ms_per_step']:>8.1f} {r['cert_ms_per_step']:>7.1f} "
                  f"{ratio:>5.1f}x {r['int8_tail_rate']:>9.1%} "
                  f"{r['dense_peak_gb']:>8.2f} {r['cert_peak_gb']:>7.2f} {r['vram_saved_gb']:>5.2f}")

    out_path = Path("benchmarks/results/certified_64k_int8model.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
