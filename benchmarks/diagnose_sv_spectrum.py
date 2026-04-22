"""Singular value spectrum and predicted bound gap at each PC count.

For a sample of blocks, computes:
  1. Full SVD of the 16×256 key matrix (rank ≤ 15)
  2. Singular values σ_1 through σ_15
  3. Predicted correction gap at k=1,2,4,8,12,15 PCs
  4. Metadata cost at each k
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
    _clone_qwen35_past_key_values,
)
from dotcache.integrations.llama import _timed_call


def _multi_pc_bound(
    q_vec: torch.Tensor,       # [head_dim]
    center: torch.Tensor,      # [head_dim]
    pcs: torch.Tensor,         # [k, head_dim]  (top-k right singular vectors)
    sigmas: torch.Tensor,      # [k]            (top-k singular values)
    r_residual: float,         # residual radius after k PCs
    q_scale: float,
) -> float:
    """Multi-PC ellipsoidal upper bound."""
    center_sim = (torch.dot(center, q_vec) * q_scale).item()
    q_dot_pcs = torch.matmul(pcs, q_vec)            # [k]
    aniso_term = (sigmas * q_dot_pcs.abs()).sum().item()
    q_norm = q_vec.norm().item()
    q_in_pcs_sq = (q_dot_pcs ** 2).sum().item()
    q_residual_norm = (q_norm**2 - q_in_pcs_sq)**0.5 if q_norm**2 > q_in_pcs_sq else 0.0
    residual_term = r_residual * q_residual_norm
    return center_sim + (aniso_term + residual_term) * abs(q_scale)


def main():
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        "Qwen/Qwen3.5-0.8B", device="cuda", torch_dtype="float16",
    )
    dc = real_mixed_probe_dotcache_config()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(model, dc)
    config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
    device = next(model.parameters()).device

    prompt = Path("docs/performance_journal.md").read_text()[:40000]
    ids, mask = _build_prompt_text_inputs(tokenizer, device=device, prompt_text=prompt, prompt_length=4096)
    print(f"Tokens: {ids.shape[1]}")

    adapter.set_mode("dense")
    adapter.clear()
    pf, _ = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=ids, attention_mask=mask), device=device,
    )
    pkv = _clone_qwen35_past_key_values(pf.past_key_values)
    adapter.persistent_serving_config = config
    adapter.clear()
    adapter.load_attention_subset_persistent_prefill_cache(pkv)
    adapter.set_mode("dotcache_attention_subset_persistent_experimental")
    adapter.configure_persistent_shortlist_policy_context(
        prompt_family=None, prompt_length=int(ids.shape[1]),
    )
    rt = adapter.require_persistent_hybrid_runtime_state()

    k_values = [1, 2, 4, 8, 12, 15]
    q_to_kv = {qh: qh // 8 for qh in range(16)}

    # Collect SVD spectrum and predicted gaps across layers
    all_sv_ratios = []
    all_gaps_by_k = {k: [] for k in k_values}

    for layer_id in sorted(rt.full_attention.layers.keys()):
        state = rt.full_attention.layers[layer_id]
        num_blocks = int(len(state.block_token_starts))
        kv_heads, total_tokens, head_dim = state.key_cache.shape

        print(f"\n=== Layer {layer_id} ({num_blocks} blocks) ===")

        for kv_head_idx in range(kv_heads):
            block_svs = []

            for block_idx in range(num_blocks):
                tok_start = int(state.block_token_starts[block_idx])
                tok_count = int(state.block_token_counts[block_idx])
                if tok_count <= 1:
                    continue

                # Key matrix for this block: [tok_count, head_dim]
                K = state.key_cache[kv_head_idx, tok_start:tok_start + tok_count, :].to(
                    dtype=torch.float32, device=device
                )
                center = K.mean(dim=0)
                K_centered = K - center

                # SVD
                U, S, Vh = torch.linalg.svd(K_centered, full_matrices=False)
                # S has min(tok_count, head_dim) values; for 16 tokens, rank ≤ 15
                svs = S.cpu().numpy()
                block_svs.append(svs)

            if not block_svs:
                continue

            # Pad to same length and stack
            max_rank = max(len(s) for s in block_svs)
            padded = np.array([np.pad(s, (0, max_rank - len(s))) for s in block_svs])

            # Singular value spectrum
            median_svs = np.median(padded, axis=0)
            print(f"\n  kv_head={kv_head_idx}: singular value spectrum (median across {len(block_svs)} blocks):")
            sv_str = "  ".join(f"σ{i+1}={median_svs[i]:.3f}" for i in range(min(15, len(median_svs))))
            print(f"    {sv_str}")
            if median_svs[0] > 0:
                ratio_str = "  ".join(
                    f"σ{i+1}/σ1={median_svs[i]/median_svs[0]:.3f}"
                    for i in range(min(15, len(median_svs)))
                )
                print(f"    {ratio_str}")
                for i in range(min(15, len(median_svs))):
                    all_sv_ratios.append(median_svs[i] / median_svs[0])

            # Predicted correction gap at each k
            # Use a representative query (mean of centroids)
            all_centers = state.block_k_center[:, kv_head_idx, :].to(device=device, dtype=torch.float32)
            q_vec = all_centers.mean(dim=0)
            q_vec = q_vec / q_vec.norm() * 10.0
            q_scale = 1.0 / (head_dim ** 0.5)

            for k in k_values:
                gaps = []
                for block_idx in range(num_blocks):
                    tok_start = int(state.block_token_starts[block_idx])
                    tok_count = int(state.block_token_counts[block_idx])
                    if tok_count <= 1:
                        continue

                    K = state.key_cache[kv_head_idx, tok_start:tok_start + tok_count, :].to(
                        dtype=torch.float32, device=device
                    )
                    center = K.mean(dim=0)
                    K_centered = K - center
                    U, S, Vh = torch.linalg.svd(K_centered, full_matrices=False)

                    actual_rank = min(int((S > 1e-6).sum().item()), len(S))
                    use_k = min(k, actual_rank)

                    pcs = Vh[:use_k]           # [use_k, head_dim]
                    sigmas = S[:use_k]          # [use_k]

                    # Residual radius: max ‖K_centered - K_projected‖ over tokens
                    if use_k < actual_rank:
                        K_proj = K_centered @ Vh[:use_k].T @ Vh[:use_k]
                        residual = K_centered - K_proj
                        r_residual = residual.norm(dim=-1).max().item()
                    else:
                        r_residual = 0.0

                    # Multi-PC bound
                    U_multi = _multi_pc_bound(q_vec, center, pcs, sigmas, r_residual, q_scale)

                    # Actual max
                    actual_logits = torch.matmul(K, q_vec) * q_scale
                    actual_max = actual_logits.max().item()

                    gaps.append(U_multi - actual_max)

                gaps = np.array(gaps)
                all_gaps_by_k[k].extend(gaps.tolist())

            # Print per-layer per-head gap summary
            print(f"\n  kv_head={kv_head_idx}: predicted correction gap by PC count:")
            print(f"    {'k':>4}  {'median':>8}  {'p90':>8}  {'max':>8}  {'metadata':>12}")
            for k in k_values:
                layer_gaps = all_gaps_by_k[k][-num_blocks:]  # last num_blocks entries
                layer_gaps = np.array(layer_gaps)
                meta_bytes = k * head_dim * 4 + k * 4 + 4  # k PCs (fp32) + k sigmas + r_residual
                print(f"    {k:>4}  {np.median(layer_gaps):>8.3f}  {np.percentile(layer_gaps, 90):>8.3f}  {np.max(layer_gaps):>8.3f}  {meta_bytes:>10} B")

    # Global summary
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY across all layers and KV heads")
    print("=" * 70)
    print(f"\n{'k':>4}  {'median gap':>12}  {'p90 gap':>10}  {'max gap':>10}  {'exp(med)':>10}  {'meta/block':>12}")
    for k in k_values:
        g = np.array(all_gaps_by_k[k])
        meta_bytes = k * 256 * 4 + k * 4 + 4
        med = np.median(g)
        print(f"  {k:>4}  {med:>12.3f}  {np.percentile(g, 90):>10.3f}  {np.max(g):>10.3f}  {np.exp(med):>10.1f}  {meta_bytes:>10} B")

    print(f"\nFor reference: current 1-PC correction gap median ≈ 9.0, exp(9) ≈ 8103")
    print(f"Target for early exit: gap < 2.0, exp(2) ≈ 7.4")


if __name__ == "__main__":
    main()
