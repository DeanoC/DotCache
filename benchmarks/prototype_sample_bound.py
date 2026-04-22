"""Prototype: sampling-based bound tightening.

For each block, store s representative keys. At query time:
  1. Compute q·k_i for each sample → max_sample (exact lower bound on actual_max)
  2. For the remaining (16-s) unseen keys, compute their centroid + radius
  3. Effective upper bound = max(max_sample, q·μ_unseen + ‖q‖·r_unseen)

This should be much tighter than the current geometric-only bound because:
  - max_sample ≈ actual_max (good samples hit the high-scoring key)
  - bound_on_unseen has a smaller radius (fewer keys, less spread)

We test multiple sampling strategies and sample counts.
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


def _select_samples_random(K: torch.Tensor, s: int, seed: int = 42) -> torch.Tensor:
    """Random sample indices."""
    g = torch.Generator(device=K.device).manual_seed(seed)
    perm = torch.randperm(K.shape[0], generator=g, device=K.device)
    return perm[:s]


def _select_samples_max_norm(K: torch.Tensor, s: int) -> torch.Tensor:
    """Keys with highest L2 norm."""
    norms = K.norm(dim=-1)
    return norms.topk(s).indices


def _select_samples_diverse(K: torch.Tensor, s: int) -> torch.Tensor:
    """Greedy farthest-point sampling from centroid."""
    center = K.mean(dim=0)
    dists = (K - center).norm(dim=-1)
    selected = [dists.argmax().item()]
    for _ in range(s - 1):
        min_dists = torch.stack([(K - K[idx]).norm(dim=-1) for idx in selected]).min(dim=0).values
        # Mask already selected
        for idx in selected:
            min_dists[idx] = -1.0
        selected.append(min_dists.argmax().item())
    return torch.tensor(selected, device=K.device, dtype=torch.long)


def _select_samples_extremal(K: torch.Tensor, s: int) -> torch.Tensor:
    """Keys at extremes: max/min along top PCA directions."""
    center = K.mean(dim=0)
    K_c = K - center
    # Top PCA direction
    _, _, Vh = torch.linalg.svd(K_c, full_matrices=False)
    selected = set()
    for pc_idx in range(min(s // 2, Vh.shape[0])):
        proj = K_c @ Vh[pc_idx]
        selected.add(proj.argmax().item())
        selected.add(proj.argmin().item())
        if len(selected) >= s:
            break
    # Fill remaining with max-norm if needed
    if len(selected) < s:
        norms = K.norm(dim=-1)
        for idx in norms.argsort(descending=True).tolist():
            selected.add(idx)
            if len(selected) >= s:
                break
    return torch.tensor(list(selected)[:s], device=K.device, dtype=torch.long)


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

    sample_counts = [1, 2, 4, 8]
    strategies = {
        "random": _select_samples_random,
        "max_norm": _select_samples_max_norm,
        "diverse": _select_samples_diverse,
        "extremal": _select_samples_extremal,
    }

    q_to_kv = {qh: qh // 8 for qh in range(16)}

    # Collect results across all layers
    results_by_strategy_count: dict[str, dict[int, list[float]]] = {
        name: {s: [] for s in sample_counts} for name in strategies
    }
    # Also collect current spherical bound gaps for reference
    current_gaps: list[float] = []

    for layer_id in sorted(rt.full_attention.layers.keys()):
        state = rt.full_attention.layers[layer_id]
        num_blocks = int(len(state.block_token_starts))
        kv_heads, total_tokens, head_dim = state.key_cache.shape

        for kv_head_idx in range(kv_heads):
            # Representative query: mean of centroids, normalized
            all_centers = state.block_k_center[:, kv_head_idx, :].to(device=device, dtype=torch.float32)
            q_vec = all_centers.mean(dim=0)
            q_vec = q_vec / q_vec.norm() * 10.0
            q_scale = 1.0 / (head_dim ** 0.5)
            q_norm = q_vec.norm().item()

            for block_idx in range(num_blocks):
                tok_start = int(state.block_token_starts[block_idx])
                tok_count = int(state.block_token_counts[block_idx])
                if tok_count <= 2:
                    continue

                K = state.key_cache[kv_head_idx, tok_start:tok_start + tok_count, :].to(
                    dtype=torch.float32, device=device
                )
                center = K.mean(dim=0)
                radius = (K - center).norm(dim=-1).max().item()

                # Actual max(q·k)
                actual_logits = torch.matmul(K, q_vec) * q_scale
                actual_max = actual_logits.max().item()

                # Current spherical bound
                center_sim = (torch.dot(center, q_vec) * q_scale).item()
                U_spherical = center_sim + q_norm * radius * abs(q_scale)
                current_gaps.append(U_spherical - actual_max)

                # Sample-based bounds
                for strat_name, strat_fn in strategies.items():
                    for s in sample_counts:
                        if s >= tok_count:
                            # All keys sampled → exact
                            results_by_strategy_count[strat_name][s].append(0.0)
                            continue

                        sample_indices = strat_fn(K, s)
                        K_samples = K[sample_indices]
                        max_sample = (torch.matmul(K_samples, q_vec) * q_scale).max().item()

                        # Unseen keys
                        all_indices = torch.arange(tok_count, device=device)
                        unseen_mask = torch.ones(tok_count, dtype=torch.bool, device=device)
                        unseen_mask[sample_indices] = False
                        K_unseen = K[unseen_mask]

                        if K_unseen.shape[0] == 0:
                            effective_bound = max_sample
                        else:
                            unseen_center = K_unseen.mean(dim=0)
                            unseen_radius = (K_unseen - unseen_center).norm(dim=-1).max().item()
                            unseen_center_sim = (torch.dot(unseen_center, q_vec) * q_scale).item()
                            bound_unseen = unseen_center_sim + q_norm * unseen_radius * abs(q_scale)
                            effective_bound = max(max_sample, bound_unseen)

                        gap = effective_bound - actual_max
                        results_by_strategy_count[strat_name][s].append(gap)

    # Print results
    print("\n" + "=" * 80)
    print("SAMPLING-BASED BOUND TIGHTENING RESULTS")
    print("=" * 80)

    cg = np.array(current_gaps)
    print(f"\nCurrent spherical bound (no sampling):")
    print(f"  Gap: median={np.median(cg):.3f}  p90={np.percentile(cg, 90):.3f}  max={np.max(cg):.3f}")
    print(f"  exp(median gap) = {np.exp(np.median(cg)):.1f}")

    print(f"\n{'Strategy':<12} {'s':>3} {'median':>8} {'p90':>8} {'max':>8} {'exp(med)':>10} {'meta/blk':>10}")
    print("-" * 70)
    for strat_name in strategies:
        for s in sample_counts:
            g = np.array(results_by_strategy_count[strat_name][s])
            meta = s * head_dim * 4  # s keys at fp32
            med = np.median(g)
            print(
                f"{strat_name:<12} {s:>3} {med:>8.3f} {np.percentile(g, 90):>8.3f} "
                f"{np.max(g):>8.3f} {np.exp(med):>10.1f} {meta:>9} B"
            )
        print()

    print(f"Target: gap < 2.0 → exp(2) = {np.exp(2.0):.1f}")
    print(f"Metadata cost: s=4 keys × {head_dim}d × 4B = {4 * head_dim * 4} B/block")
    print(f"  vs current centroid+radius = {head_dim * 4 + 4} B/block")

    # Violations check
    for strat_name in strategies:
        for s in sample_counts:
            g = np.array(results_by_strategy_count[strat_name][s])
            violations = np.sum(g < -1e-6)
            if violations > 0:
                print(f"  *** {strat_name} s={s}: {violations} VIOLATIONS (bound < actual_max) ***")


if __name__ == "__main__":
    main()
