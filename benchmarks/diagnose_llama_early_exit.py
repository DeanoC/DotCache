"""Simulate certified early exit on LLaMA 3.1-8B using exact per-block max scores.

For each layer × q_head, computes actual max(q·k) per block, then simulates
the streaming evaluation with the certified exit mechanism using exact scores.
Reports what fraction of blocks would be processed before certification.

Also measures the bound gap (using spherical bounds computed from block stats)
to see if the tighter bounds achievable on LLaMA's sparser attention would
enable practical early exit.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN", "")

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads
    gqa_group = num_q_heads // num_kv_heads
    block_size = 16
    mass_eps = 1e-3

    # Build prompt: filler + real content
    real_content = Path("docs/performance_journal.md").read_text()[:8000]
    filler = "The quick brown fox jumps over the lazy dog. " * 3000
    prompt = filler + "\n\n" + real_content

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    print(f"Tokens: {seq_len}, Blocks: {num_blocks}")

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values

    # One decode step for real queries
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        decode_out = model(
            input_ids=next_token, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True,
        )
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    q_scale = 1.0 / (head_dim ** 0.5)

    # Per-layer, per-head analysis
    exit_fractions = []  # fraction of blocks processed at cert exit
    bound_gaps = []
    per_layer_exit = {l: [] for l in range(num_layers)}

    for layer_idx in range(num_layers):
        # Key cache
        if hasattr(past_kv, 'layers'):
            key_cache = past_kv.layers[layer_idx].keys[0]  # [kv_heads, seq_len, head_dim]
        else:
            key_cache = past_kv[layer_idx][0][0]

        # Query from decode step
        if hasattr(model, 'model'):
            attn = model.model.layers[layer_idx].self_attn
        else:
            attn = model.layers[layer_idx].self_attn

        hidden = decode_out.hidden_states[layer_idx]
        with torch.inference_mode():
            q_proj = attn.q_proj(hidden)
            q = q_proj.view(1, 1, num_q_heads, head_dim).squeeze(0).squeeze(0).to(torch.float32)

        for q_head_idx in range(num_q_heads):
            kv_head_idx = q_head_idx // gqa_group
            q_vec = q[q_head_idx]
            keys = key_cache[kv_head_idx].to(dtype=torch.float32)  # [seq_len, head_dim]

            # Compute per-block: actual_max, centroid_sim, spherical_bound
            block_actual_max = []
            block_center_sim = []
            block_upper_bound = []
            block_token_counts = []

            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if end <= start:
                    continue
                K_block = keys[start:end]
                tok_count = end - start

                logits = torch.matmul(K_block, q_vec) * q_scale
                actual_max = logits.max().item()

                center = K_block.mean(dim=0)
                center_sim = (torch.dot(center, q_vec) * q_scale).item()
                radius = (K_block - center).norm(dim=-1).max().item()
                q_norm = q_vec.norm().item()
                upper = center_sim + q_norm * radius * abs(q_scale)

                block_actual_max.append(actual_max)
                block_center_sim.append(center_sim)
                block_upper_bound.append(upper)
                block_token_counts.append(tok_count)

            if not block_actual_max:
                continue

            n_blks = len(block_actual_max)
            actual_max_arr = np.array(block_actual_max)
            upper_arr = np.array(block_upper_bound)
            tok_counts = np.array(block_token_counts)

            # Bound gap
            gaps = upper_arr - actual_max_arr
            bound_gaps.extend(gaps.tolist())

            # --- Simulate streaming with EXACT scores ---
            # Process blocks in descending actual_max order
            sorted_idx = np.argsort(actual_max_arr)[::-1]
            m = float('-inf')
            l_sum = 0.0
            exit_block = n_blks  # default: process all

            processed_set = set()
            for i, idx in enumerate(sorted_idx):
                processed_set.add(idx)
                bmax = actual_max_arr[idx]
                tc = tok_counts[idx]

                if bmax > m:
                    l_sum = l_sum * np.exp(m - bmax) + tc
                    m = bmax
                else:
                    l_sum += tc * np.exp(bmax - m)

                # Check certificate every 16 blocks
                if (i + 1) % 16 == 0 or i == n_blks - 1:
                    # Residual mass from unprocessed blocks (using EXACT scores)
                    remaining_mass = 0.0
                    for j in range(n_blks):
                        if j not in processed_set:
                            remaining_mass += tok_counts[j] * np.exp(actual_max_arr[j] - m)

                    beta = remaining_mass / (l_sum + remaining_mass) if (l_sum + remaining_mass) > 0 else 0.0
                    if beta < mass_eps:
                        exit_block = i + 1
                        break

            exit_frac = exit_block / n_blks
            exit_fractions.append(exit_frac)
            per_layer_exit[layer_idx].append(exit_frac)

    # Results
    ef = np.array(exit_fractions)
    bg = np.array(bound_gaps)

    print(f"\n{'='*70}")
    print(f"LLaMA 3.1-8B CERTIFIED EARLY EXIT SIMULATION")
    print(f"{'='*70}")
    print(f"Evaluations: {len(ef)} (layer × head)")
    print(f"\nWith EXACT per-block scores (ideal case):")
    print(f"  Exit fraction: median={np.median(ef):.1%}  mean={np.mean(ef):.1%}  p10={np.percentile(ef, 10):.1%}  p90={np.percentile(ef, 90):.1%}")
    print(f"  Blocks skipped: {1 - np.mean(ef):.1%} average")

    print(f"\nPer-layer exit fraction (mean across heads):")
    for layer_idx in range(num_layers):
        le = np.array(per_layer_exit[layer_idx])
        if len(le) > 0:
            skip = 1 - np.mean(le)
            marker = " ***" if skip > 0.1 else ""
            if layer_idx % 4 == 0 or skip > 0.05:
                print(f"  layer {layer_idx:>2}: exit at {np.mean(le):.1%} of blocks (skip {skip:.1%}){marker}")

    print(f"\nBound gap (spherical U_b - actual_max):")
    print(f"  median={np.median(bg):.2f}  p90={np.percentile(bg, 90):.2f}  max={np.max(bg):.2f}")
    print(f"  exp(median gap) = {np.exp(np.median(bg)):.1f}")

    # Compare: if we used UPPER BOUNDS instead of exact scores
    print(f"\n--- Comparison: exact scores vs geometric bounds ---")
    exit_with_bounds = []
    for layer_idx in range(num_layers):
        if hasattr(past_kv, 'layers'):
            key_cache = past_kv.layers[layer_idx].keys[0]
        else:
            key_cache = past_kv[layer_idx][0][0]

        hidden = decode_out.hidden_states[layer_idx]
        with torch.inference_mode():
            q_proj = attn.q_proj(hidden)  # reusing last layer's attn, but fine for comparison
            q_all = q_proj.view(num_q_heads, head_dim).to(torch.float32)

        # Just sample 2 heads per layer for speed
        for q_head_idx in [0, num_q_heads - 1]:
            kv_head_idx = q_head_idx // gqa_group
            q_vec = q_all[q_head_idx]
            keys = key_cache[kv_head_idx].to(dtype=torch.float32)
            q_norm = q_vec.norm().item()

            block_actual = []
            block_upper = []
            block_tc = []
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if end <= start: continue
                K_block = keys[start:end]
                tc = end - start
                actual = (torch.matmul(K_block, q_vec) * q_scale).max().item()
                center = K_block.mean(dim=0)
                csim = (torch.dot(center, q_vec) * q_scale).item()
                r = (K_block - center).norm(dim=-1).max().item()
                ub = csim + q_norm * r * abs(q_scale)
                block_actual.append(actual)
                block_upper.append(ub)
                block_tc.append(tc)

            n_blks = len(block_actual)
            upper_arr = np.array(block_upper)
            actual_arr = np.array(block_actual)
            tc_arr = np.array(block_tc)

            # Simulate with bounds
            sorted_idx = np.argsort(actual_arr)[::-1]
            m = float('-inf')
            l_sum = 0.0
            exit_block = n_blks
            processed_set = set()

            for i, idx in enumerate(sorted_idx):
                processed_set.add(idx)
                bmax = actual_arr[idx]
                tc = tc_arr[idx]
                if bmax > m:
                    l_sum = l_sum * np.exp(m - bmax) + tc
                    m = bmax
                else:
                    l_sum += tc * np.exp(bmax - m)

                if (i + 1) % 16 == 0 or i == n_blks - 1:
                    # Residual using UPPER BOUNDS (not exact)
                    remaining_mass = 0.0
                    for j in range(n_blks):
                        if j not in processed_set:
                            remaining_mass += tc_arr[j] * np.exp(upper_arr[j] - m)
                    beta = remaining_mass / (l_sum + remaining_mass) if (l_sum + remaining_mass) > 0 else 0.0
                    if beta < mass_eps:
                        exit_block = i + 1
                        break

            exit_with_bounds.append(exit_block / n_blks)

    ewb = np.array(exit_with_bounds)
    print(f"  With geometric bounds: exit at {np.mean(ewb):.1%} of blocks (skip {1-np.mean(ewb):.1%})")
    print(f"  With exact scores:     exit at {np.mean(ef):.1%} of blocks (skip {1-np.mean(ef):.1%})")
    print(f"  Gap: bounds force {np.mean(ewb) - np.mean(ef):.1%} more blocks to be processed")


if __name__ == "__main__":
    main()
