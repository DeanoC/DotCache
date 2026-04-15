"""LLaMA sparsity diagnostic with heterogeneous content.

Uses semantically diverse context (code + docs + paper + config)
instead of repetitive filler. Tests whether attention sparsity
increases when the context contains genuinely irrelevant segments.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.build_heterogeneous_context import build_context


def main():
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    block_size = args.block_size

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads
    gqa_group = num_q_heads // num_kv_heads

    # Test multiple question targets to see if sparsity depends on which segment is queried
    for q_idx, q_label in enumerate(["CUDA_RESULTS", "PAPER_SECTION", "BENCHMARK_REPORT"]):
        prompt = build_context(args.context_length, question_idx=q_idx)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_length).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        num_blocks = (seq_len + block_size - 1) // block_size

        print(f"\n{'='*60}")
        print(f"Question target: {q_label} ({seq_len} tokens, {num_blocks} blocks)")
        print(f"{'='*60}")

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, output_hidden_states=True)
        past_kv = outputs.past_key_values

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        with torch.inference_mode():
            decode_out = model(
                input_ids=next_token, past_key_values=past_kv,
                use_cache=True, output_hidden_states=True,
            )

        q_scale = 1.0 / (head_dim ** 0.5)
        all_deltas = []
        per_layer_deltas = {l: [] for l in range(num_layers)}

        for layer_idx in range(num_layers):
            if hasattr(past_kv, 'layers'):
                key_cache = past_kv.layers[layer_idx].keys[0]
            else:
                key_cache = past_kv[layer_idx][0][0]

            if hasattr(model, 'model'):
                attn_mod = model.model.layers[layer_idx].self_attn
            else:
                attn_mod = model.layers[layer_idx].self_attn

            hidden = decode_out.hidden_states[layer_idx]
            with torch.inference_mode():
                q_proj = attn_mod.q_proj(hidden)
                q = q_proj.view(num_q_heads, head_dim).to(torch.float32)

            for kv_head_idx in range(num_kv_heads):
                keys = key_cache[kv_head_idx, :num_blocks * block_size, :].to(dtype=torch.float32)
                keys_blocked = keys.reshape(num_blocks, block_size, head_dim)

                for q_head_idx in range(kv_head_idx * gqa_group, (kv_head_idx + 1) * gqa_group):
                    q_vec = q[q_head_idx]
                    logits = torch.matmul(keys_blocked, q_vec) * q_scale
                    block_maxes = logits.max(dim=1).values
                    m_global = block_maxes.max().item()
                    deltas = (block_maxes - m_global).cpu().numpy()
                    all_deltas.extend(deltas.tolist())
                    per_layer_deltas[layer_idx].extend(deltas.tolist())

        d = np.array(all_deltas)
        n = len(d)

        tier_a = np.sum(d > -5)
        tier_b = np.sum((d > -20) & (d <= -5))
        tier_c = np.sum(d <= -20)
        print(f"  Tier A: {tier_a/n:.1%}  Tier B: {tier_b/n:.1%}  Tier C: {tier_c/n:.1%}")

        # Per-layer for deep layers
        for layer_idx in [0, num_layers // 2, num_layers - 4, num_layers - 2, num_layers - 1]:
            ld = np.array(per_layer_deltas[layer_idx])
            ta = np.mean(ld > -5) * 100
            tc = np.mean(ld <= -20) * 100
            print(f"  layer {layer_idx:>2}: Tier_A={ta:>5.1f}%  Tier_C={tc:>5.1f}%")

        # Attention concentration (sample layer 31, head 14)
        if hasattr(past_kv, 'layers'):
            kc = past_kv.layers[num_layers - 1].keys[0]
        else:
            kc = past_kv[num_layers - 1][0][0]
        # head 14 maps to kv_head 14 // gqa_group
        kv_idx = 14 // gqa_group
        keys_sample = kc[kv_idx, :num_blocks * block_size, :].to(torch.float32)
        keys_bl = keys_sample.reshape(num_blocks, block_size, head_dim)

        if hasattr(model, 'model'):
            attn_last = model.model.layers[num_layers - 1].self_attn
        else:
            attn_last = model.layers[num_layers - 1].self_attn
        h_last = decode_out.hidden_states[num_layers - 1]
        with torch.inference_mode():
            q_last = attn_last.q_proj(h_last).view(num_q_heads, head_dim)[14].to(torch.float32)

        all_logits = torch.matmul(keys_sample[:num_blocks * block_size], q_last) * q_scale
        attn_w = torch.softmax(all_logits, dim=0)
        block_masses = attn_w.reshape(num_blocks, block_size).sum(dim=1).cpu().numpy()
        sorted_m = np.sort(block_masses)[::-1]
        cumsum = np.cumsum(sorted_m)
        print(f"  Layer {num_layers-1} head 14 attention concentration:")
        for pct in [0.5, 0.8, 0.9, 0.95]:
            k = int(np.searchsorted(cumsum, pct)) + 1
            print(f"    {pct:.0%} mass in top {k} blocks ({k/num_blocks:.1%})")


if __name__ == "__main__":
    main()
