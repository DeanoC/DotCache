"""Diagnose the gap between certified upper bounds and actual max logits.

For each (block, q_head), logs:
  1. q·μ_b — centroid dot product (base estimate)
  2. actual max(q·k) — ground truth
  3. U_b — the certified upper bound

This decomposes the total gap (U_b - actual_max) into:
  - centroid gap: actual_max - q·μ_b (how far the best key is from mean)
  - correction gap: U_b - actual_max (how much the geometric correction overshoots)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
)
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    load_qwen35_text_only_from_pretrained,
    _run_dense_prefill,
    _run_dense_decode_step,
    _clone_qwen35_past_key_values,
)
from dotcache.integrations.llama import _timed_call


def main():
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        "Qwen/Qwen3.5-0.8B", device="cuda", torch_dtype="float16",
    )
    dc = real_mixed_probe_dotcache_config()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(model, dc)
    config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
    config.enable_interval_bound = True
    config.enable_ellipsoidal_bound = True

    device = next(model.parameters()).device
    prompt = Path("docs/performance_journal.md").read_text()[:40000]
    ids, mask = _build_prompt_text_inputs(tokenizer, device=device, prompt_text=prompt, prompt_length=4096)
    print(f"Tokens: {ids.shape[1]}")

    # Prefill
    adapter.set_mode("dense")
    adapter.clear()
    pf, _ = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=ids, attention_mask=mask), device=device,
    )

    # Load persistent runtime
    pkv = _clone_qwen35_past_key_values(pf.past_key_values)
    adapter.persistent_serving_config = config
    adapter.clear()
    adapter.load_attention_subset_persistent_prefill_cache(pkv)
    adapter.set_mode("dotcache_attention_subset_persistent_experimental")
    adapter.configure_persistent_shortlist_policy_context(
        prompt_family=None, prompt_length=int(ids.shape[1]),
    )
    rt = adapter.require_persistent_hybrid_runtime_state()

    # Run one decode step to get the real query
    first_token = pf.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    dec_mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=1)
    cache_pos = torch.tensor([ids.shape[1]], dtype=torch.long, device=device)
    outputs, _ = _timed_call(
        lambda: _run_dense_decode_step(
            model, decode_input_ids=first_token, attention_mask=dec_mask,
            past_key_values=rt.model_past_key_values, cache_position=cache_pos,
        ),
        device=device,
    )
    rt.advance(outputs.past_key_values)

    # Now diagnose each FA layer
    q_to_kv = {qh: qh // (16 // 2) for qh in range(16)}  # 16 q_heads, 2 kv_heads, group=8

    for layer_id in sorted(rt.full_attention.layers.keys()):
        state = rt.full_attention.layers[layer_id]
        num_blocks = int(len(state.block_token_starts))
        kv_heads, total_tokens, head_dim = state.key_cache.shape

        # We need the actual query used for this layer. The adapter stores it
        # during _forward_persistent but doesn't expose it. Let's reconstruct
        # by calling score_blocks which uses the bound computation.
        # Instead, let's just pick q_head=0 and compute manually.

        # Get the centroid, radius, and key cache for layer
        print(f"\n=== Layer {layer_id} ({num_blocks} blocks, {total_tokens} tokens, head_dim={head_dim}) ===")

        # We don't have the actual query from the forward pass, but we can
        # use a representative one by taking the mean of the key centroids
        # as a proxy, or better, by extracting from the model directly.
        # For now, let's compute the gaps using a random query with realistic norm.
        # The key insight is the RELATIVE gaps, not absolute values.

        # Actually, let's compute exact max(q·k) for each block using the raw key cache,
        # and compare against the bound metadata. We use the key centroid as the query
        # to get a representative analysis.

        for q_head_idx in [0, 8]:  # Sample two q_heads from different KV groups
            kv_head_idx = q_to_kv[q_head_idx]

            # Use mean of all keys as a representative query direction
            all_keys = state.key_cache[kv_head_idx, :total_tokens, :].to(dtype=torch.float32)
            q_vec = all_keys.mean(dim=0)
            q_vec = q_vec / q_vec.norm() * 10.0  # Normalize and scale to realistic norm
            q_scale = 1.0 / (head_dim ** 0.5)

            center = state.block_k_center[:, kv_head_idx, :].to(device=device, dtype=torch.float32)
            radius = state.block_k_radius[:, kv_head_idx].to(device=device, dtype=torch.float32)
            q_norm = q_vec.norm().item()

            centroid_gaps = []
            correction_gaps = []
            total_gaps = []

            for block_idx in range(num_blocks):
                tok_start = int(state.block_token_starts[block_idx])
                tok_count = int(state.block_token_counts[block_idx])
                if tok_count <= 0:
                    continue

                # 1. q·μ_b (centroid dot product)
                center_sim = (torch.dot(center[block_idx], q_vec) * q_scale).item()

                # 2. actual max(q·k)
                key_slice = state.key_cache[kv_head_idx, tok_start:tok_start + tok_count, :].to(
                    dtype=torch.float32, device=device
                )
                actual_logits = (torch.matmul(key_slice, q_vec) * q_scale)
                actual_max = actual_logits.max().item()

                # 3. U_b (spherical bound)
                U_spherical = center_sim + q_norm * radius[block_idx].item() * abs(q_scale)

                # Interval bound
                k_min = state.block_k_min[block_idx, kv_head_idx, :].to(device=device, dtype=torch.float32)
                k_max = state.block_k_max[block_idx, kv_head_idx, :].to(device=device, dtype=torch.float32)
                q_scaled = q_vec * q_scale
                U_interval = torch.where(q_scaled >= 0, q_scaled * k_max, q_scaled * k_min).sum().item()

                # Ellipsoidal bound (if available)
                U_ellip = float("inf")
                if state.block_k_pc1 is not None:
                    pc1 = state.block_k_pc1[block_idx, kv_head_idx, :].to(device=device, dtype=torch.float32)
                    r_along = state.block_k_r_along[block_idx, kv_head_idx].item()
                    r_perp = state.block_k_r_perp[block_idx, kv_head_idx].item()
                    q_dot_pc1 = torch.dot(q_vec, pc1).item()
                    q_perp_norm = (q_norm**2 - q_dot_pc1**2)**0.5 if q_norm**2 > q_dot_pc1**2 else 0.0
                    U_ellip = center_sim + (abs(q_dot_pc1) * r_along + q_perp_norm * r_perp) * abs(q_scale)

                U_best = min(U_spherical, U_interval, U_ellip)

                centroid_gap = actual_max - center_sim
                correction_gap = U_best - actual_max
                total_gap = U_best - center_sim

                centroid_gaps.append(centroid_gap)
                correction_gaps.append(correction_gap)
                total_gaps.append(total_gap)

            centroid_gaps = np.array(centroid_gaps)
            correction_gaps = np.array(correction_gaps)
            total_gaps = np.array(total_gaps)

            print(f"\n  q_head={q_head_idx} (kv_head={kv_head_idx}), {len(centroid_gaps)} blocks:")
            print(f"    centroid gap  (actual_max - q·μ): median={np.median(centroid_gaps):.3f}  p90={np.percentile(centroid_gaps, 90):.3f}  max={np.max(centroid_gaps):.3f}")
            print(f"    correction gap (U_best - actual):  median={np.median(correction_gaps):.3f}  p90={np.percentile(correction_gaps, 90):.3f}  max={np.max(correction_gaps):.3f}")
            print(f"    total gap      (U_best - q·μ):    median={np.median(total_gaps):.3f}  p90={np.percentile(total_gaps, 90):.3f}  max={np.max(total_gaps):.3f}")
            print(f"    correction / total:                median={np.median(correction_gaps / np.maximum(total_gaps, 1e-10)):.1%}  mean={np.mean(correction_gaps / np.maximum(total_gaps, 1e-10)):.1%}")

            # Also show the bound-type breakdown
            violations = np.sum(correction_gaps < -1e-6)
            if violations > 0:
                print(f"    *** BOUND VIOLATIONS: {violations} blocks where U_best < actual_max ***")


if __name__ == "__main__":
    main()
