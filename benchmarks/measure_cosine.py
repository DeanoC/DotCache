"""Measure per-layer cosine similarity: certified vs dense attention.

Runs one decode step through both paths with the same queries,
compares attention outputs per layer.
"""
from __future__ import annotations

import os
import sys
import gc
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


def measure_cosine(model, tokenizer, ctx_len, v_mode="fp16"):
    """Measure per-layer cosine between certified and dense attention.

    Runs dense prefill, captures per-layer queries from a decode step,
    then runs certified attention on same queries and compares.
    """
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )
    from dotcache.kernels.certified_attention import certified_attention_layer

    config = model.config
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // num_q
    gqa = num_q // num_kv
    q_scale = 1.0 / (head_dim ** 0.5)
    num_layers = config.num_hidden_layers

    prompt = build_context(ctx_len, question_idx=0)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Dense prefill + one decode step with hidden states
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with torch.inference_mode():
        dec = model(
            input_ids=first_token, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True,
        )

    # Save hidden states (lightweight) and free heavy decode outputs
    hidden_states_cpu = [h.detach().cpu() for h in dec.hidden_states]
    del dec

    # Build tiered caches
    layer_ids = list(range(num_layers))
    if v_mode == "int4":
        caches = create_tiered_cache_int4v_from_model(past_kv, layer_ids)
    else:
        caches = create_tiered_cache_from_model(past_kv, layer_ids)

    # Free prefill KV (now in tiered caches)
    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # For each layer: extract dense attention output, run certified, compare
    # Process one KV head at a time to avoid OOM at long contexts
    per_layer = []
    for lid in range(num_layers):
        cache = caches[lid]
        eps = LAYER_EPS.get(lid, 1e-4)
        attn_wrapper = model.model.layers[lid].self_attn
        attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper

        with torch.inference_mode():
            h = hidden_states_cpu[lid].to("cuda")
            q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
            del h

        # Dense attention: compute per KV-head to limit VRAM
        out_dense = torch.empty(num_q, head_dim, dtype=torch.float32, device="cuda")
        for kv in range(num_kv):
            keys_h = cache.keys_fp16_cpu[kv].to(dtype=torch.float32, device="cuda")
            if cache.values_fp16 is not None:
                vals_h = cache.values_fp16[kv].to(torch.float32)
            elif cache.values_fp16_cpu is not None:
                vals_h = cache.values_fp16_cpu[kv].to(dtype=torch.float32, device="cuda")
            else:
                vals_h = cache.dequantise_int4_values()[kv]

            for qh in range(kv * gqa, (kv + 1) * gqa):
                s = torch.matmul(keys_h, q_all[qh]) * q_scale
                w = torch.softmax(s, dim=0)
                out_dense[qh] = w @ vals_h

            del keys_h, vals_h

        # Certified attention output
        out_cert, stats = certified_attention_layer(
            cache, q_all, gqa, q_scale, block_epsilon=eps, collect_stats=True,
        )

        cos = torch.nn.functional.cosine_similarity(out_dense, out_cert, dim=1)
        per_layer.append({
            "layer": lid,
            "cos_min": cos.min().item(),
            "cos_mean": cos.mean().item(),
            "skip_rate": stats["skip_rate"],
            "v_format": stats.get("v_format", v_mode),
        })

        del out_dense, out_cert
        gc.collect()
        torch.cuda.empty_cache()

    cos_mins = [l["cos_min"] for l in per_layer]
    return {
        "seq_len": seq_len,
        "v_mode": v_mode,
        "cos_min": min(cos_mins),
        "cos_mean": float(np.mean(cos_mins)),
        "cos_p5": float(np.percentile(cos_mins, 5)),
        "per_layer": per_layer,
    }


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    results = []

    for ctx_len, label in [(8192, "8K"), (16384, "16K"), (32768, "32K")]:
        for v_mode in ["fp16", "int4"]:
            print(f"\n{label} context, V={v_mode}:")
            r = measure_cosine(model, tokenizer, ctx_len, v_mode)
            r["context"] = label
            results.append(r)

            print(f"  cos_min={r['cos_min']:.6f}, cos_mean={r['cos_mean']:.6f}, cos_p5={r['cos_p5']:.6f}")

            # Per-layer detail for interesting layers
            for lid in [0, 15, 28, 31]:
                pl = r["per_layer"][lid]
                print(f"    L{lid:2d}: cos_min={pl['cos_min']:.6f}, skip={pl['skip_rate']:.1%}, v={pl['v_format']}")

            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"COSINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ctx':>4} {'V-mode':>6} {'cos_min':>9} {'cos_mean':>9} {'cos_p5':>8}")
    for r in results:
        print(f"  {r['context']:>2} {r['v_mode']:>6} {r['cos_min']:>9.6f} {r['cos_mean']:>9.6f} {r['cos_p5']:>8.6f}")

    out_path = Path("benchmarks/results/cosine_measurements.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
