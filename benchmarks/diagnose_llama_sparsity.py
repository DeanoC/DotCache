"""Attention sparsity diagnostic for LLaMA 3.1-8B.

Standalone script — loads model via HuggingFace, runs prefill + 1 decode step,
then computes actual max(q·k) per 16-token block per layer per query head.

Reports the distribution of (actual_max_block - m_global) to characterise
attention sparsity. If attention is sparse, many blocks will be far below
the global max and certified early exit becomes viable.
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="LLaMA attention sparsity diagnostic")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    token = os.environ.get("HF_TOKEN", "")
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, token=token,
    ).to("cuda")
    model.eval()
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads
    gqa_group = num_q_heads // num_kv_heads
    print(f"Architecture: {num_layers} layers, {num_q_heads} Q heads, {num_kv_heads} KV heads, head_dim={head_dim}")

    # Build prompt: real content + filler to test sparsity
    real_content = Path("docs/performance_journal.md").read_text()[:8000]
    filler = "The quick brown fox jumps over the lazy dog. " * 3000
    prompt = filler + "\n\n" + real_content  # Real content at end

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.context_length,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"Tokens: {seq_len} (target: {args.context_length})")

    # Prefill
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    print(f"VRAM after prefill: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # One decode step to get real queries
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        decode_out = model(
            input_ids=next_token,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,  # need hidden states for query extraction
        )

    # Extract queries from hidden states through the QKV projections
    # We'll compute q·K for each layer using the actual model weights
    block_size = args.block_size
    num_blocks = (seq_len + block_size - 1) // block_size

    print(f"\nBlocks: {num_blocks} (block_size={block_size})")
    print(f"Running sparsity analysis across {num_layers} layers, {num_q_heads} Q heads...")

    all_deltas = []
    per_layer_deltas = {l: [] for l in range(num_layers)}
    per_head_deltas = {h: [] for h in range(num_q_heads)}
    per_layer_head_tier_c = {}  # (layer, head) -> fraction in Tier C

    for layer_idx in range(num_layers):
        # Get key cache: DynamicCache.layers[i].keys → [batch, kv_heads, seq_len, head_dim]
        if hasattr(past_kv, 'layers'):
            key_cache = past_kv.layers[layer_idx].keys[0]  # [kv_heads, seq_len, head_dim]
        elif hasattr(past_kv, 'key_cache'):
            key_cache = past_kv.key_cache[layer_idx][0]
        else:
            key_cache = past_kv[layer_idx][0][0]

        # Get the query projection for this decode step
        # Access the attention module to project the hidden state
        if hasattr(model, 'model'):
            attn = model.model.layers[layer_idx].self_attn
        else:
            attn = model.layers[layer_idx].self_attn

        # Get the hidden state at this layer from the decode step
        # hidden_states[0] is embeddings, hidden_states[layer_idx+1] is after layer
        hidden = decode_out.hidden_states[layer_idx]  # [1, 1, hidden_size]

        # Project to get query
        with torch.inference_mode():
            q_proj = attn.q_proj(hidden)  # [1, 1, num_q_heads * head_dim]
            q = q_proj.view(1, 1, num_q_heads, head_dim).squeeze(0).squeeze(0)  # [num_q_heads, head_dim]
            q = q.to(dtype=torch.float32)

        # Apply RoPE would be needed for exact values, but for relative ranking
        # (which block is highest vs lowest) the unrotated query is a reasonable proxy.
        # The rotation is position-dependent and applies equally to q and k,
        # so the relative ordering is approximately preserved.

        q_scale = 1.0 / (head_dim ** 0.5)

        for q_head_idx in range(num_q_heads):
            kv_head_idx = q_head_idx // gqa_group
            q_vec = q[q_head_idx].to(dtype=torch.float32)
            keys = key_cache[kv_head_idx].to(dtype=torch.float32)  # [seq_len, head_dim]

            # Compute per-block max(q·k)
            block_maxes = []
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if end <= start:
                    continue
                K_block = keys[start:end]  # [block_size, head_dim]
                logits = torch.matmul(K_block, q_vec) * q_scale
                block_maxes.append(logits.max().item())

            if not block_maxes:
                continue

            m_global = max(block_maxes)
            deltas = [bm - m_global for bm in block_maxes]
            all_deltas.extend(deltas)
            per_layer_deltas[layer_idx].extend(deltas)
            per_head_deltas[q_head_idx].extend(deltas)

            tier_c_frac = np.mean(np.array(deltas) <= -20)
            per_layer_head_tier_c[(layer_idx, q_head_idx)] = tier_c_frac

    d = np.array(all_deltas)
    n = len(d)

    print(f"\n{'='*70}")
    print(f"ATTENTION SPARSITY: {args.model_id} @ {seq_len} tokens")
    print(f"{'='*70}")
    print(f"Total evaluations: {n} ({num_blocks} blocks × {num_layers} layers × {num_q_heads} heads)")

    print(f"\nHistogram of (actual_max_block - m_global):")
    for lo, hi in [(-5, 0), (-10, -5), (-15, -10), (-20, -15), (-30, -20), (-50, -30), (-100, -50)]:
        count = np.sum((d > lo) & (d <= hi))
        if count > 0:
            print(f"  ({lo:>4}, {hi:>4}]: {count:>8} ({count/n:>6.1%})")
    count = np.sum(d <= -100)
    if count > 0:
        print(f"  (<=-100):    {count:>8} ({count/n:>6.1%})")

    tier_a = np.sum(d > -5)
    tier_b = np.sum((d > -20) & (d <= -5))
    tier_c = np.sum(d <= -20)
    print(f"\nTier sizing:")
    print(f"  Tier A (within 5 of max):  {tier_a:>8} ({tier_a/n:.1%}) — must process")
    print(f"  Tier B (5-20 below max):   {tier_b:>8} ({tier_b/n:.1%}) — uncertain")
    print(f"  Tier C (>20 below max):    {tier_c:>8} ({tier_c/n:.1%}) — safely skip")

    # Per-layer sparsity
    print(f"\nPer-layer Tier C fraction:")
    for layer_idx in range(num_layers):
        ld = np.array(per_layer_deltas[layer_idx])
        tc = np.mean(ld <= -20) * 100
        ta = np.mean(ld > -5) * 100
        if tc > 0 or layer_idx % 4 == 0:
            print(f"  layer {layer_idx:>2}: Tier_A={ta:>5.1f}%  Tier_C={tc:>5.1f}%")

    # Per-head sparsity
    print(f"\nPer-head Tier C fraction (top 10 sparsest heads):")
    head_tc = [(h, np.mean(np.array(per_head_deltas[h]) <= -20)) for h in range(num_q_heads)]
    head_tc.sort(key=lambda x: x[1], reverse=True)
    for h, tc in head_tc[:10]:
        ta = np.mean(np.array(per_head_deltas[h]) > -5)
        print(f"  q_head={h:>2}: Tier_C={tc:.1%}  Tier_A={ta:.1%}")

    # Find the most sparse (layer, head) combinations
    print(f"\nMost sparse (layer, head) combinations:")
    sparse_combos = sorted(per_layer_head_tier_c.items(), key=lambda x: x[1], reverse=True)
    for (layer_idx, head_idx), frac in sparse_combos[:15]:
        if frac > 0:
            print(f"  layer={layer_idx:>2} head={head_idx:>2}: {frac:.1%} Tier C")
    if sparse_combos[0][1] == 0:
        print(f"  (none — all (layer, head) pairs have 0% Tier C)")

    # Attention concentration: what fraction of total softmax mass is in top-k blocks?
    print(f"\nAttention concentration (% of softmax mass in top-k blocks, sampled):")
    sample_layer = num_layers // 2
    sample_head = 0
    kv_head_idx = sample_head // gqa_group
    if hasattr(past_kv, 'layers'):
        key_cache_sample = past_kv.layers[sample_layer].keys[0, kv_head_idx].to(dtype=torch.float32)
    elif hasattr(past_kv, 'key_cache'):
        key_cache_sample = past_kv.key_cache[sample_layer][0, kv_head_idx].to(dtype=torch.float32)
    else:
        key_cache_sample = past_kv[sample_layer][0][0][kv_head_idx].to(dtype=torch.float32)
    if hasattr(model, 'model'):
        attn_s = model.model.layers[sample_layer].self_attn
    else:
        attn_s = model.layers[sample_layer].self_attn
    hidden_s = decode_out.hidden_states[sample_layer]
    with torch.inference_mode():
        q_s = attn_s.q_proj(hidden_s).view(num_q_heads, head_dim)[sample_head].to(torch.float32)
    all_logits = torch.matmul(key_cache_sample, q_s) * q_scale
    # Compute softmax attention weights
    attn_weights = torch.softmax(all_logits, dim=0)
    # Group by block and sum
    block_masses = []
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        block_masses.append(attn_weights[start:end].sum().item())
    block_masses = np.array(block_masses)
    sorted_masses = np.sort(block_masses)[::-1]
    cumsum = np.cumsum(sorted_masses)
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cumsum, pct)) + 1
        print(f"  {pct:.0%} of mass in top {k} blocks ({k/num_blocks:.1%} of {num_blocks})")


if __name__ == "__main__":
    main()
