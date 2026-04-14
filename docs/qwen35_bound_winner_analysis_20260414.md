# Bound Mode Winner Analysis — MPS Results and CUDA Predictions (2026-04-14)

This document records the MPS bound-mode comparison results, explains the MPS
overhead hypothesis, and makes falsifiable predictions for the CUDA run.

## MPS results summary

Source: `docs/qwen35_bound_mode_compare_20260414.md` /
`benchmarks/results/qwen35_bound_mode_compare_20260414_mps/`

| Lane | avg ms/step | vs spherical | cert_stop_blocks/case |
|---|---|---|---|
| `spherical_only` | 538.36 | baseline | 384.0 |
| `interval` | **507.82** | **−5.7%** | 384.0 |
| `interval_ellip` | 830.22 | +54.2% | 384.0 |

Key observations:
- `interval` is **5.7% faster** than `spherical_only` on MPS despite identical
  `cert_stop_blocks` counts — the timing difference is in the score-computation
  kernel, not in certified early-exit depth.
- `interval_ellip` is **54.2% slower** due to per-head kernel launch overhead on MPS
  (matmul, **2, sqrt, mul per head × 28 FA layers × up to 100 blocks × 8 steps).
- `cert_stop_blocks` is identical for all three lanes (384 summed over layers).
  cert_stop_rate ≈ 6.2% of (step × layer) pairs regardless of bound mode.

### Context-scaling observations (MPS, contexts ≤ 8K only)

The MPS context-scaling sweep was limited to ≤ 8192 tokens because dense SDPA
prefill OOM'd at 16K/32K.  Within the tested range:

- cert_stop_rate was constant at ~6.2% across all context lengths.
- ms/step saving from `interval` was noisy but on average positive.
- No clear monotonic growth in the delta with context length.

## MPS overhead hypothesis

The `interval_ellip` result is consistent with **MPS kernel-launch overhead**
being the dominant cost at the block level.  Each certified bound evaluation on MPS
involves O(num_heads × num_blocks) small tensor ops.  For `interval` these map to
one `where + sum` per head, which happens to be cache-friendly on the Apple GPU.
For `ellipsoidal` these are five distinct ops (matmul, sq, clamp, sqrt, mul) per
head, each triggering a separate Metal kernel dispatch with fixed ~5–20 µs overhead.

At 100 blocks × 28 layers × 8 q-heads, even 5 µs/dispatch overhead would add:
  100 × 28 × 8 × 5 ops × 5 µs = **560 ms/step** overhead — consistent with the
  observed +292 ms regression (+54.2%).

On CUDA (via cuDNN / cuBLAS fused kernels), these small ops can be fused or
executed with much lower per-op overhead.  Prediction: `interval_ellip` will be
competitive or faster than `spherical_only` on CUDA.

## CUDA predictions

### Bound mode comparison (same corpus, CUDA)

| Lane | Prediction | Reasoning |
|---|---|---|
| `spherical_only` | baseline — fastest absolute time | No change in compute pattern |
| `interval` | **−8% to −15% vs spherical** | Interval bound is tighter on real keys; CUDA kernel fusion makes the O(d) multiply cheap; certified exits should fire earlier |
| `interval_ellip` | **0% to −10% vs spherical** | Ellipsoidal cost is low on CUDA (fused matmul); anisotropic tightening should yield more cert stops |

If `interval_ellip` is still significantly negative on CUDA despite batched path,
see the note at the bottom on the per-kv-head loop optimisation.

### cert_stop_rate (context scaling, CUDA)

**Prediction**: cert_stop_rate ≈ 6.2% at every context length (1K–32K).

Cert_stop_rate is determined by the epsilons (`mass_eps=1e-3`, `value_eps=1e-3`)
and the data distribution — not by the hardware or the context length.  The same
block selection and residual-certificate machinery runs on CUDA as on MPS.

### ms/step delta growth with context length (CUDA)

**Prediction**: the absolute ms/step saving from `interval` vs `spherical_only`
**grows monotonically** with context length.

Reasoning:
1. At longer context, each block contains more tokens and the key distribution
   is richer — the interval bound is tighter relative to the sphere.
2. More blocks per context = more bound evaluations per step = larger absolute cost difference.
3. On MPS the delta was noisy due to kernel-launch overhead dominating;
   on CUDA kernel throughput dominates, so the signal is cleaner.

If the delta is *not* monotone on CUDA, that would suggest the interval bound is
not significantly tighter than the sphere for this model's key distribution at
large context — the ball bound was already tight.

### interval_ellip overhead root cause (if still slow on CUDA)

If `interval_ellip` is still significantly negative after switching to the batched
path (`_compute_ellipsoidal_upper_bound_batched`), the likely cause is the gather
```
state.block_k_pc1[:, q_to_kv_t, :]   # [num_blocks, num_q_heads, head_dim]
```
which expands kv_heads (2) → q_heads (8) before the einsum, creating a
`[num_q_heads, num_blocks, head_dim]` intermediate tensor.  For 32K context
(2000 blocks) × 8 q_heads × 256 head_dim × float32 = 16 MB — fine for HBM.

But if the bottleneck is still the intermediate, the fix is to loop over
**kv_heads** (2 iterations) instead of **q_heads** (8 iterations), scatter results:
```python
upper_E = torch.full((num_blocks,), float("-inf"), ...)
for kv_idx in range(num_kv_heads):
    q_mask = (q_to_kv_t == kv_idx)
    q_sub = query_tensor[q_mask]          # [q_per_kv, head_dim]
    # matmul against [num_blocks, head_dim] — no expansion
    ...
    upper_E = torch.maximum(upper_E, sub_upper.max(dim=0).values)
```
This halves the peak intermediate tensor size and may improve L2/HBM efficiency.

## How to run the CUDA benchmarks

### Bound mode comparison (3-lane)

```bash
python benchmarks/bench_qwen35_bound_mode_compare.py \
    --device cuda --backend torch_cuda \
    --decode-steps 8 \
    --output-json benchmarks/results/qwen35_bound_mode_compare_YYYYMMDD_cuda/qwen35_bound_mode_compare.json \
    --output-md  benchmarks/results/qwen35_bound_mode_compare_YYYYMMDD_cuda/qwen35_bound_mode_compare.md
```

### Context-length scaling (1K–32K)

```bash
# Full run (no OOM expected on CUDA with HBM):
python benchmarks/bench_qwen35_bound_context_scaling.py \
    --device cuda --backend torch_cuda \
    --decode-steps 16 \
    --output-json benchmarks/results/qwen35_bound_context_scaling_YYYYMMDD_cuda/context_scaling.json \
    --output-md  benchmarks/results/qwen35_bound_context_scaling_YYYYMMDD_cuda/context_scaling.md

# If VRAM is tight, skip dense reference at 32K:
python benchmarks/bench_qwen35_bound_context_scaling.py \
    --device cuda --backend torch_cuda \
    --decode-steps 16 \
    --max-dense-length 16384 \
    --output-json benchmarks/results/qwen35_bound_context_scaling_YYYYMMDD_cuda/context_scaling.json \
    --output-md  benchmarks/results/qwen35_bound_context_scaling_YYYYMMDD_cuda/context_scaling.md
```

## Artefacts

| File | Description |
|---|---|
| `benchmarks/bench_qwen35_bound_mode_compare.py` | 3-lane mode comparison benchmark |
| `benchmarks/bench_qwen35_bound_context_scaling.py` | Context-length scaling benchmark (1K–32K) |
| `docs/qwen35_bound_mode_compare_20260414.md` | MPS mode-comparison results |
| `docs/qwen35_bound_winner_analysis_20260414.md` | This document |
| `docs/qwen35_bound_cuda_results_YYYYMMDD.md` | CUDA results (to be created after CUDA run) |
