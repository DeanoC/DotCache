"""Calibrate context-aware per-layer epsilon profile.

For each (context_length, layer), sweep epsilon from coarse to fine,
find the largest epsilon where per-layer cos_min ≥ target.

Output: an EpsilonProfile ready for production use.
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
from dotcache.kernels.epsilon_profile import EpsilonProfile

COS_TARGET = 0.999
EPSILON_CANDIDATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]


def calibrate_layer(
    model, past_kv, hidden_state, layer_id, num_q, num_kv, head_dim, gqa, q_scale,
    cache_factory, cos_target=COS_TARGET,
):
    """Find the largest epsilon for one layer where cos_min ≥ target."""
    from dotcache.kernels.certified_attention import certified_attention_layer

    attn_wrapper = model.model.layers[layer_id].self_attn
    attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper
    with torch.inference_mode():
        h = hidden_state.to("cuda")
        q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
        del h

    cache = cache_factory(layer_id)

    # Dense oracle (per KV-head to save memory)
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

    # Sweep epsilon from coarse to fine
    best_eps = EPSILON_CANDIDATES[-1]  # tightest fallback
    best_cos = 0.0
    best_skip = 0.0

    for eps in EPSILON_CANDIDATES:
        out_cert, stats = certified_attention_layer(
            cache, q_all, gqa, q_scale, block_epsilon=eps, collect_stats=True,
            v_tolerance=0.5,
        )
        cos = torch.nn.functional.cosine_similarity(out_dense, out_cert, dim=1)
        cos_min = cos.min().item()
        skip_rate = stats["skip_rate"]

        if cos_min >= cos_target:
            best_eps = eps
            best_cos = cos_min
            best_skip = skip_rate
            break  # First (largest) epsilon that passes — take it

    del out_dense, cache
    return best_eps, best_cos, best_skip


def calibrate_context(model, tokenizer, ctx_len, cos_target=COS_TARGET):
    """Calibrate all layers at one context length."""
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

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

    # Prefill
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # One decode step for hidden states
    with torch.inference_mode():
        dec = model(
            input_ids=first_token, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True,
        )
    hidden_states_cpu = [h.detach().cpu() for h in dec.hidden_states]
    del dec

    # Build tiered caches
    layer_ids = list(range(num_layers))
    caches = create_tiered_cache_from_model(past_kv, layer_ids)
    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()

    def cache_factory(lid):
        return caches[lid]

    # Calibrate each layer
    layer_results = {}
    for lid in range(num_layers):
        eps, cos, skip = calibrate_layer(
            model, None, hidden_states_cpu[lid], lid,
            num_q, num_kv, head_dim, gqa, q_scale,
            cache_factory, cos_target,
        )
        layer_results[lid] = {"epsilon": eps, "cos_min": cos, "skip_rate": skip}
        marker = "✓" if cos >= cos_target else "✗"
        print(f"    L{lid:2d}: ε={eps:.0e}, cos={cos:.6f}, skip={skip:.1%} {marker}")

    del caches, hidden_states_cpu
    gc.collect()
    torch.cuda.empty_cache()

    return seq_len, layer_results


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
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    profile = EpsilonProfile(num_layers=model.config.num_hidden_layers)
    all_results = {}

    for ctx_len in CONTEXT_LENGTHS:
        label = f"{ctx_len // 1024}K"
        print(f"\n{'='*50}")
        print(f"  Calibrating {label} (target cos ≥ {COS_TARGET})")
        print(f"{'='*50}")

        seq_len, layer_results = calibrate_context(model, tokenizer, ctx_len, COS_TARGET)
        all_results[label] = {"seq_len": seq_len, "layers": layer_results}

        # Store in profile
        eps_dict = {lid: r["epsilon"] for lid, r in layer_results.items()}
        profile.profiles[ctx_len] = eps_dict
        profile._sorted_lengths = sorted(profile.profiles.keys())

    # Summary
    print(f"\n{'='*50}")
    print(f"CALIBRATED EPSILON PROFILE (cos ≥ {COS_TARGET})")
    print(f"{'='*50}")
    print(profile.summary())

    # Show compact table
    print(f"\n{'Layer':>5}", end="")
    for ctx in CONTEXT_LENGTHS:
        print(f"  {ctx//1024:>4}K", end="")
    print()
    for lid in range(model.config.num_hidden_layers):
        print(f"  {lid:>3}", end="")
        for ctx in CONTEXT_LENGTHS:
            eps = profile.get_epsilon(ctx, lid)
            print(f"  {eps:>5.0e}", end="")
        print()

    # Save
    out_path = Path("benchmarks/results/calibrated_epsilon_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "cos_target": COS_TARGET,
        "context_lengths": CONTEXT_LENGTHS,
        "profile": {str(k): {str(lid): v for lid, v in eps.items()}
                     for k, eps in profile.profiles.items()},
        "per_context": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")

    # Also save as Python constant for easy import
    py_path = Path("dotcache/kernels/epsilon_profile_calibrated.py")
    with open(py_path, "w") as f:
        f.write(f'"""Auto-generated epsilon profile (cos ≥ {COS_TARGET})."""\n\n')
        f.write("CALIBRATED_PROFILE = {\n")
        for ctx in sorted(profile.profiles.keys()):
            f.write(f"    {ctx}: {{\n")
            eps_dict = profile.profiles[ctx]
            # Group consecutive layers with same epsilon
            groups = []
            current_eps = None
            current_start = 0
            for lid in range(model.config.num_hidden_layers):
                e = eps_dict.get(lid, 1e-4)
                if e != current_eps:
                    if current_eps is not None:
                        groups.append((current_start, lid - 1, current_eps))
                    current_eps = e
                    current_start = lid
            groups.append((current_start, model.config.num_hidden_layers - 1, current_eps))
            for start, end, e in groups:
                if start == end:
                    f.write(f"        {start}: {e},\n")
                else:
                    f.write(f"        **{{lid: {e} for lid in range({start}, {end + 1})}},\n")
            f.write(f"    }},\n")
        f.write("}\n")
    print(f"Python -> {py_path}")


if __name__ == "__main__":
    main()
