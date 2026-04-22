"""Profile a single certified decode step to find the bottleneck.

Breaks down: QKV projection, certified kernels, output projection, FFN+norms.
"""
from __future__ import annotations

import os
import sys
import gc
import time
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


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig
    from dotcache.kernels.certified_attention import certified_attention_layer
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()
    print(f"Model VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Prefill
    ctx_len = 8192
    prompt = build_context(ctx_len, question_idx=0)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"Prefill: {seq_len} tokens")

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Build tiered caches
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered_caches = create_tiered_cache_from_model(past_kv, layer_ids, block_size=16)
    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    config = model.config
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // num_q
    gqa = num_q // num_kv
    q_scale = 1.0 / (head_dim ** 0.5)

    # =========================================================
    # Profile 1: Individual kernel costs (outside forward pass)
    # =========================================================
    print(f"\n{'='*60}")
    print("KERNEL PROFILING (standalone, per layer)")
    print(f"{'='*60}")

    # Get a representative query from layer 15 (middle)
    with torch.inference_mode():
        dec = model(input_ids=first_token, past_key_values=past_kv, use_cache=True, output_hidden_states=True)

    # Profile each phase separately
    from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
    from dotcache.kernels.selective_attend_triton import selective_attend_multihead

    for lid in [0, 15, 31]:
        cache = tiered_caches[lid]
        attn = model.model.layers[lid].self_attn
        # Get q for this layer from hidden states
        with torch.inference_mode():
            q_all = attn.q_proj(dec.hidden_states[lid]).view(num_q, head_dim).to(torch.float32)

        eps = LAYER_EPS.get(lid, 1e-4)

        # Warmup
        for _ in range(5):
            certified_attention_layer(cache, q_all, gqa, q_scale, block_epsilon=eps)
        torch.cuda.synchronize()

        # Phase 1: Score + Certify
        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m_b, S_b, skip = fused_score_certify_multihead(
                cache.keys_int8, cache.keys_scale, q_all, cache.correction,
                gqa, cache.block_size, q_scale, block_epsilon=eps,
            )
        torch.cuda.synchronize()
        t_score = (time.perf_counter() - t0) / iters * 1e6

        # Phase 2: Selective Attend
        keys_deq = (cache.keys_int8.to(torch.float32).reshape(
            cache.kv_heads, cache.num_blocks, cache.block_size, cache.head_dim
        ) * cache.keys_scale[:, :, None, None]).reshape(cache.kv_heads, cache.num_tokens, cache.head_dim)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = selective_attend_multihead(
                keys_deq, cache.values_fp16.to(torch.float32),
                q_all, skip.to(torch.int32), gqa, cache.block_size, q_scale,
            )
        torch.cuda.synchronize()
        t_attend = (time.perf_counter() - t0) / iters * 1e6

        # INT8 dequant overhead
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            kd = (cache.keys_int8.to(torch.float32).reshape(
                cache.kv_heads, cache.num_blocks, cache.block_size, cache.head_dim
            ) * cache.keys_scale[:, :, None, None]).reshape(cache.kv_heads, cache.num_tokens, cache.head_dim)
        torch.cuda.synchronize()
        t_deq = (time.perf_counter() - t0) / iters * 1e6

        skip_rate = skip.sum().item() / (num_q * cache.num_blocks)
        print(f"\n  Layer {lid}: skip={skip_rate:.1%}")
        print(f"    Score+certify: {t_score:>8.1f} μs")
        print(f"    INT8 dequant:  {t_deq:>8.1f} μs")
        print(f"    Attend:        {t_attend:>8.1f} μs")
        print(f"    Total kernel:  {t_score + t_deq + t_attend:>8.1f} μs")

    # =========================================================
    # Profile 2: Full forward pass breakdown
    # =========================================================
    print(f"\n{'='*60}")
    print("FORWARD PASS BREAKDOWN")
    print(f"{'='*60}")

    # Dense step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.inference_mode():
            model(input_ids=first_token, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / 20 * 1e3
    print(f"  Dense forward:     {t_dense:.2f} ms")

    # Certified step (with adapter)
    dc_config = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, dc_config)
    adapter.load_certified_cache(past_kv, layer_epsilons=LAYER_EPS, default_epsilon=1e-4)
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device="cuda")

    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            model(input_ids=first_token, use_cache=False,
                  cache_position=cache_position, position_ids=cache_position.unsqueeze(0))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.inference_mode():
            model(input_ids=first_token, use_cache=False,
                  cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
    torch.cuda.synchronize()
    t_cert = (time.perf_counter() - t0) / 20 * 1e3
    print(f"  Certified forward: {t_cert:.2f} ms")
    print(f"  Overhead:          {t_cert - t_dense:.2f} ms ({(t_cert/t_dense - 1)*100:.0f}%)")

    # Estimate kernel vs rest
    # Sum kernel time across all 32 layers
    total_kernel_us = 0
    for lid in range(32):
        cache = tiered_caches[lid]
        eps = LAYER_EPS.get(lid, 1e-4)
        # Use layer 15 q as estimate (they're all similar)
        with torch.inference_mode():
            q_all = attn.q_proj(dec.hidden_states[15]).view(num_q, head_dim).to(torch.float32)
        total_kernel_us += t_score + t_deq + t_attend  # approximate from layer 31

    print(f"\n  Estimated kernel total (32 layers): {total_kernel_us/1000:.2f} ms")
    print(f"  Non-kernel overhead: {t_cert - total_kernel_us/1000:.2f} ms")
    print(f"  (FFN, norms, embedding, Python dispatch)")


if __name__ == "__main__":
    main()
