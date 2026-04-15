# Bound Winner Analysis (2026-04-14)

Follow-on to `qwen35_bound_mode_compare_20260414.md`. After establishing that the interval
bound delivers +13.2% MPS speedup and the ellipsoidal bound costs −58.3%, this document
examines *why* via bound winner fractions, derives memory implications for long-context
CUDA deployment, and records the MPS overhead hypothesis with falsifiable CUDA predictions.

See `docs/qwen35_bound_cuda_results_20260414.md` for the CUDA results that test those predictions.

## Setup

- Model: Qwen/Qwen3.5-0.8B (kv_heads=2, head_dim=256, 6 full-attention layers)
- Corpus: 8-case broader public validation manifest
- Config: real-mixed Stage 9, 8 decode steps per case
- Lanes: `interval` (spherical + interval) and `interval_ellip` (all three)
- New metric: per-(block, q_head) winner count — which bound provides the tightest upper
  bound on each evaluation

## Bound Winner Fractions

### `interval` lane (196,608 evals)

| Case | spherical | interval |
|---|---|---|
| aae_stage_summary | 79% | 21% |
| bench_decode_code | 87% | 13% |
| benchmark_report | 71% | 29% |
| compressed_page_rfc | 79% | 21% |
| hip_call_flow | 81% | 19% |
| local_layer_profiles | 73% | 27% |
| test_attention_vs_dense | 74% | 26% |
| turboquant_comparison_plan | 73% | 27% |
| **Average** | **76.6%** | **23.4%** |

The interval bound fires on 1-in-4 evaluations. The cases where it wins most
(`benchmark_report`, `local_layer_profiles`, `turboquant`) have more diverse key
distributions; `bench_decode_code` at 13% is the most spherically-distributed.

### `interval_ellip` lane (219,983 evals)

| Case | spherical | interval | ellipsoidal |
|---|---|---|---|
| aae_stage_summary | 40% | 19% | 41% |
| bench_decode_code | 43% | 12% | 45% |
| benchmark_report | 38% | 25% | 37% |
| compressed_page_rfc | 40% | 19% | 41% |
| hip_call_flow | 45% | 18% | 37% |
| local_layer_profiles | 41% | 24% | 35% |
| test_attention_vs_dense | 40% | 24% | 37% |
| turboquant_comparison_plan | 38% | 23% | 39% |
| **Average** | **40.2%** | **20.9%** | **38.9%** |

With all three bounds active, the split is nearly even (~40/20/40). Ellipsoidal is
**genuinely tight** — it wins as many evaluations as spherical on the same data.

## Why the Interval Speedup Exceeds Its Win Rate

The interval bound wins 23.4% of evaluations yet delivers 13.2% faster decode. The
asymmetry is explained by *where* it wins: upper bounds are used to prioritise blocks for
certified streaming. The marginal blocks near the certified-stop threshold are the ones the
interval bound is most likely to tighten, because those are the blocks with intermediate
upper bounds — not the obvious sinks/recents (where spherical is already tight) nor the
clearly excluded far blocks (where the gap doesn't matter). Tightening 1-in-4 of the
*decision-boundary* blocks triggers earlier certified exit; the fraction of all evaluations
understates the impact on the stopping criterion.

## Memory Overhead

Model geometry: kv_heads=2, head_dim=256, float32 metadata, block_size=16, 6 FA layers.

### Per block (across all 6 FA layers)

| Bound | New tensors | Bytes/block |
|---|---|---|
| Interval | 3 × [B, 2, 256] | **36 KB** |
| Ellipsoidal (marginal) | 1 × [B, 2, 256] + 2 × [B, 2] | **12 KB** |
| Combined | 4 × [B, 2, 256] + 2 × [B, 2] | **48 KB** |

The ellipsoidal bound only requires the first principal component vector (`block_k_pc1`)
plus two scalars per KV head (`block_k_r_along`, `block_k_r_perp`). At 1/3 the memory
cost of the interval metadata, it is a compact representation.

### At representative context lengths

| Context | Interval | Ellipsoidal | Combined |
|---|---|---|---|
| 1.5K tok (97 blocks/layer) | 3.4 MB | 1.2 MB | 4.6 MB |
| 4K tok (250 blocks/layer) | 9.0 MB | 3.0 MB | 12.0 MB |
| 16K tok (1024 blocks/layer) | 36.0 MB | 12.1 MB | 48.1 MB |
| 64K tok (4096 blocks/layer) | 144 MB | 48.4 MB | 192 MB |
| 128K tok (8192 blocks/layer) | 288 MB | 96.8 MB | 385 MB |

These are overhead costs on top of the KV cache itself. At 128K tokens the combined
overhead is 385 MB, dominated by the interval tensors (288 MB). Ellipsoidal adds only 97 MB
on top.

### Memory efficiency vs win rate (1.5K context)

| Bound | Memory | Win rate | KB per win-rate point |
|---|---|---|---|
| Interval | 3.4 MB | 23.4% | 15 KB/pt |
| Ellipsoidal (marginal) | 1.2 MB | 38.9% | **3 KB/pt** |

**Ellipsoidal is 4.9× more memory-efficient per win-rate point than interval.** It delivers
more bound tightening per byte of metadata, because the PCA decomposition is a
geometrically richer representation than the axis-aligned K_min/K_max envelope.

## MPS Performance Results

### Bound mode comparison (8 cases, ~1.5K context)

| Lane | avg ms/step | vs spherical |
|---|---|---|
| `spherical_only` | 557.10 | baseline |
| `interval` | **483.50** | **+13.2%** |
| `interval_ellip` | 881.78 | −58.3% |

### Context-length scaling (MPS, ≤ 8K only — 16K/32K deferred to CUDA)

| tokens | blk/layer | spherical ms/step | interval ms/step | speedup | int_win_frac | cert_stop_rate |
|---|---|---|---|---|---|---|
| 2,048 | 128 | 763 | 906 | **−18.7%** | 23% | 6.2% |
| 4,096 | 256 | 1,353 | 1,423 | **−5.1%** | 16% | 6.2% |
| 8,192 | 512 | 3,445 | 4,002 | **−16.2%** | 21% | 6.2% |

The interval bound is slower at every MPS context length on this file set. Key observations:
- cert_stop_rate is constant at ~6.2% regardless of context length
- The MPS overhead per block (~1ms) dominates the saving on most files
- Interval win rates stay flat (16–23%) — the hypothesis of increasing anisotropy with context is not supported on these files

## MPS Overhead Hypothesis

The `interval_ellip` slowdown is consistent with **MPS kernel-launch overhead** being the
dominant cost. For `interval`, the extra op is one `where + sum` per head — cache-friendly
on Apple GPU. For `ellipsoidal`, five distinct ops (matmul, sq, clamp, sqrt, mul) per head
each trigger a separate Metal kernel dispatch with ~5–20 µs overhead.

At 100 blocks × 6 FA layers × 8 q-heads × 5 ops × 5 µs = **240 ms/step** — consistent
with the observed regression.

On CUDA, these small ops can be fused or executed with much lower per-op overhead. The
device-aware dispatch (`_use_batched = device.startswith("cuda")`) is already implemented:
CUDA takes a single `[blocks, q_heads]` batched einsum; MPS/CPU stays per-head.

## CUDA Predictions (Made Before CUDA Run)

| Lane | Prediction | Reasoning |
|---|---|---|
| `interval` | **−8% to −15% vs spherical** | Interval bound is tighter on real keys; CUDA makes O(d) multiply cheap; cert exits fire earlier |
| `interval_ellip` | **0% to −10% vs spherical** | Ellipsoidal cost is low on CUDA (fused matmul); anisotropic tightening yields more cert stops |
| cert_stop_rate | **≈ 6.2% at all lengths** | Determined by epsilons and data, not hardware |
| ms/step delta | **grows monotonically** with context | More blocks to skip × lower CUDA cost/block |

**See `docs/qwen35_bound_cuda_results_20260414.md` for actual CUDA outcomes.**

## Ellipsoidal Fix History (MPS)

Two optimisations were applied:

1. **`center_sim` reuse** — `_compute_ellipsoidal_upper_bound` now accepts optional
   `center_sim=` param; the MPS loop passes the already-computed centroid dot product,
   eliminating one redundant `matmul(center, query_vec)` per head. Effect: negligible
   (the bottleneck is the remaining 4 ops, not the centroid matmul).

2. **Device-aware batched CUDA path** — `_compute_interval_upper_bound_batched` and
   `_compute_ellipsoidal_upper_bound_batched` pre-compute all heads in a single
   `[blocks, q_heads, head_dim]` einsum before the per-head loop. Loop body on CUDA
   just indexes (`_upper_I_all[:, q_head_idx]`). On MPS the per-head path is unchanged.

## Artefacts

| File | Description |
|---|---|
| `benchmarks/bench_paper_mode_compare.py` | 3-lane mode comparison benchmark |
| `benchmarks/bench_paper_context_scaling.py` | Context-length scaling benchmark (1K–32K) |
| `benchmarks/results/qwen35_bound_mode_compare_20260414_mps/` | MPS mode-compare results |
| `benchmarks/results/qwen35_bound_winners_20260414_mps/` | MPS bound winner fraction data |
| `benchmarks/results/qwen35_bound_context_scaling_20260414_mps/` | MPS context scaling |
| `benchmarks/results/qwen35_ellip_fix_20260414_mps/` | Post-fix MPS benchmark |
| `docs/qwen35_bound_cuda_results_20260414.md` | CUDA results |
