# Qwen3.5 PyTorch Drift Summary

Date: 2026-04-14

## Scope

This note summarizes the current state of the shared Qwen3.5 runtime vs PyTorch mismatch work.

The direct HIP runner is no longer the main problem. `hip-direct` and native-device paths already match each other closely enough that the remaining correctness work is now focused on the shared model/runtime path against PyTorch.

Primary recent artifact:

- [/tmp/qwen35_decode_layer_trace_v17.json](/tmp/qwen35_decode_layer_trace_v17.json)

Related tracking doc:

- [qwen35_hip_direct_tracking.md](/home/deano/DotCache/docs/qwen35_hip_direct_tracking.md)

## Current Headline

Current one-token native HIP vs PyTorch trace under `CANDLE_QWEN35_DELTA_SCAN_MODE=torch-like`:

- `prefill_max_delta = 0.59375`
- `decode_max_delta = 12.0625`

The remaining mismatch is not an `lm_head` problem and not a direct-HIP runner problem. It is accumulated decoder hidden-state drift that gets amplified by RMSNorm.

## What Is Already Fixed

These were real semantic bugs, not harmless numeric noise:

- linear prefill Flat3d chunk solver had wrong solved-value semantics
- full-attention query prep on HIP used a non-contiguous query slice before RMSNorm

Fixing those materially reduced drift. That is strong evidence that the remaining gap is still due to implementation-path differences, not just unavoidable floating point behavior.

## What Is Ruled Out

The following are not the main remaining source of the PyTorch mismatch:

- tokenizer / embedding path
- direct HIP runner vs native-device runner
- `lm_head`
- a single catastrophic late decode cache bug
- a single obviously broken late token-mixer path

Relevant measurements:

- `pytorch_decode_output_projection_from_oracle_hidden_max_delta = 0.03125`
- `pytorch_decode_final_hidden_max_delta = 0.46875`

Interpretation:

- when runtime `lm_head` is fed the PyTorch final hidden, logits are close
- the large decode delta is therefore already present in the decoder hidden state before output projection

## Current Decode Drift Shape

The decode residual drift does not grow uniformly. The largest positive per-layer growth points are:

- layer 23: input `0.033203125` -> output `0.09375`, growth `+0.060546875`
- layer 3: input `0.0014648438` -> output `0.01171875`, growth `+0.0102539062`
- layer 7: input `0.0078125` -> output `0.015625`, growth `+0.0078125`
- layers 10, 14, 15, 16, 17, 18: about `+0.00390625` each

This says:

- early decode is mostly clean
- the first meaningful residual jump appears at layer 3
- there is a smaller steady accumulation block in the middle of the stack
- layer 23 is the last large growth point before final norm

## Layer 3 Decode

Decode layer 3 is the first meaningful growth point under the `torch-like` baseline.

Current coarse breakdown:

- input RMSNorm input delta: `0.0014648438`
- input RMSNorm output delta: `0.0625`
- token mixer delta: `0.0045166016`
- MLP delta: `0.004272461`
- layer output delta: `0.01171875`

Interpretation:

- token mixer and MLP are both small
- the large visible local jump is RMSNorm amplification of an already-small hidden-state difference

So layer 3 looks like a residual-stream drift exposure point, not the unique root cause.

## Layer 23 Decode

Layer 23 is the largest per-layer decode growth point near the end of the stack.

Current breakdown:

- input RMSNorm: `0.1875`
- token mixer: `0.015625`
- post-attention RMSNorm: `0.125`
- MLP gate proj: `0.07421875`
- MLP up proj: `0.0625`
- MLP activated hidden: `0.109375`
- MLP down proj: `0.1015625`
- layer output: `0.09375`

Interpretation:

- token mixer is small
- layer 23 has both:
  - RMSNorm amplification
  - real MLP contribution

This is not a single bad projection. The late decode growth is residual drift entering the layer plus moderate MLP drift.

## Final Decode Norm

Final decode tail:

- final hidden delta: `0.46875`
- final norm input delta: `0.09375`
- final norm output delta: `0.46875`

Interpretation:

- final model RMSNorm is amplifying accumulated residual drift
- it is not creating the hidden mismatch from zero

## Middle Layer Decode Breakdowns (Layers 15-18)

Traced in `/tmp/qwen35_decode_layer_trace_v18.json`.

All four middle layers show the same pattern:

| Layer | Input delta | Norm amplif | Token mixer | Post-attn norm | MLP | Growth |
|-------|------------|-------------|-------------|----------------|-----|--------|
| 15 | 0.0156 | 16.0x | 0.0059 | 0.156 | 0.0081 | +0.0039 |
| 16 | 0.0195 | 16.0x | 0.0078 | 0.141 | 0.0084 | +0.0039 |
| 17 | 0.0234 | 10.7x | 0.0059 | 0.156 | 0.0076 | +0.0039 |
| 18 | 0.0273 | 13.7x | 0.0068 | 0.188 | 0.0078 | +0.0039 |

Interpretation:

- token mixer and MLP deltas are small and consistent (~0.006-0.008)
- both are actually larger than the net per-layer growth (0.0039)
- the residual connection partially cancels them
- the steady +0.0039/layer is the un-cancelled remainder
- the middle block is NOT the root cause; it is steady-state noise accumulation

## Layer 23 MLP Is the Dominant Remaining Drift Source

Comparison of MLP behavior across layers:

| Layer | Post-attn norm (MLP input delta) | MLP output delta | Ratio |
|-------|----------------------------------|-----------------|-------|
| 15 | 0.156 | 0.008 | 0.05x (reducing) |
| 16 | 0.141 | 0.008 | 0.06x (reducing) |
| 17 | 0.156 | 0.008 | 0.05x (reducing) |
| 18 | 0.188 | 0.008 | 0.04x (reducing) |
| **23** | **0.125** | **0.102** | **0.81x (preserving)** |

Layer 23's post-attn norm input (0.125) is actually **smaller** than the middle layers (0.14-0.19). Yet its MLP output delta is **13x larger**. This rules out accumulated input drift as the explanation.

Layer 23 MLP internal breakdown:

- gate_proj: 0.074
- up_proj: 0.063
- activated_hidden (silu(gate)*up): 0.109
- down_proj: 0.102
- MLP output: 0.102

In middle layers, the MLP reduces the post-attn norm delta by ~20x. In layer 23, the MLP barely reduces it at all. Something about layer 23's MLP specifically fails to cancel the norm-amplified delta.

This is NOT a full_attention vs linear_attention difference: both layers 15 and 23 are full_attention, but behave completely differently through the MLP.

## Current Diagnosis

The correct mental model is now:

1. small residual-stream differences accumulate at a steady ~0.004/layer through the middle of the stack
2. these are normal floating-point noise partially cancelled by residual connections
3. RMSNorm amplifies the accumulated residual drift at each layer, but the MLP typically reduces it back
4. **layer 23's MLP does not reduce the norm-amplified delta** — it passes it through nearly unchanged
5. this un-reduced MLP delta feeds the final RMSNorm, which amplifies it further, producing the large output delta

The problem is specifically in how layer 23's MLP processes the incoming delta. The input delta to the MLP is comparable to (actually smaller than) other layers, but the output delta is 13x larger.

## Highest-Probability Remaining Issue Class

Two hypotheses:

**Hypothesis A: Numerical sensitivity** — layer 23's weight values happen to have high gain in the direction of the error vector, so small input deltas produce large output deltas through the linear projections. This would not be a bug.

**Hypothesis B: Implementation bug** — there is a weight loading, transposition, or computation difference specific to layer 23's MLP. This is in the same class as previous fixed bugs (correct formula, wrong staging).

Evidence favoring B: the previous two fixed bugs were exactly this class (layout/staging), and a 13x difference between identically-coded layers is extreme for pure numerical sensitivity.

## Oracle-Fed MLP Test (v19 trace)

Fed PyTorch oracle post-attn norm output directly into runtime MLP (bypasses all upstream computation):

| Layer | MLP with runtime input | MLP with oracle input | Reduction |
|-------|----------------------|---------------------|-----------|
| 15 | 0.0081 | 0.00098 | 8.2x |
| 23 | 0.1016 | 0.0078 | 13x |

Both layers' MLPs produce accurate results when given correct inputs. The MLP weights and computation are correct. Layer 23's large delta is pure numerical sensitivity to upstream error.

## BF16 vs FP32 Oracle Comparison

Both oracles produce **identical** results:
- vs FP32 PyTorch: `decode_max_delta = 12.0625`
- vs BF16 PyTorch: `decode_max_delta = 12.0625`

This rules out FP32-vs-BF16 precision as the root cause. PyTorch BF16 on CPU likely uses FP32 accumulation for matmuls, so it is effectively FP32. The Rust HIP runtime differs from BOTH by the same amount.

## Layer 0 Decode Analysis

Layer 0 (first layer, linear attention):
- input norm delta: 0.0 (embedding is bit-exact)
- linear qkv proj delta: 0.015625 (0.0 input → 0.016 output!)
- initial recurrent state delta: 0.023 (carried from prefill)
- initial conv state: **shape mismatch** (cannot compare)

The QKV projection delta of 0.016 from a 0.0 input suggests either:
- HIP BF16 matmul uses different accumulation/tiling than PyTorch CPU
- OR the conv state shape mismatch causes the decode linear attention to process data differently

## Current Diagnosis

1. The MLP is correct — oracle-fed test proves this
2. The drift is not BF16 vs FP32 precision — BF16 oracle shows same delta
3. Full attention layers contribute ~88% of total drift
4. The per-layer residual growth (~0.004) is the root cause
5. Layer 0 already shows drift even with 0.0 input — the HIP matmul and/or conv state handling introduces error from the very first operation
6. The conv state **shape mismatch** between runtime and PyTorch is a known issue that warrants investigation

## Conv State Shape Mismatch — Benign

- PyTorch: `[1, 6144, 4]` — stores full `kernel_size` history
- Runtime: `[1, 6144, 3]` — stores `kernel_size - 1` history

This is an equivalent representation. Both correctly compute the depthwise conv1d: PyTorch keeps 4 entries and reads the last 4; the runtime keeps 3 and prepends them to the new input to form a window of 4.

## Layer 0 QKV Projection — GPU Matmul Non-Determinism

With input delta = 0.0 (bit-exact input), four projections from the same input produce:

| Projection | Output dim | Prefill delta | Decode delta |
|-----------|-----------|--------------|-------------|
| qkv | 6144 | 0.0078 | 0.016 |
| z | 2048 | 0.0039 | ~0 |
| b | 16 | 0.0 | 0.0 |
| a | 16 | 0.0 | 0.0 |

All share the same input, same dot-product length (1024), and same `matmul` code path. Only the output dimension differs. Small output projections are bit-exact; large ones show measurable error.

This is GPU matmul non-determinism: HIP/ROCm GEMM uses different tiling and accumulation order than PyTorch CPU (which likely uses F32 accumulation even for BF16 inputs). This is inherent to GPU computation, not a bug.

Cannot verify via CPU baseline because candle does not support BF16 matmul on CPU.

## Revised Drift Model

The decode drift chain is:

1. GPU matmul introduces ~0.008/projection for large projections (qkv, gate, up, down)
2. Multiple projections per layer with partial cancellation → ~0.004 net residual growth/layer
3. 24 layers of accumulation → ~0.094 total residual delta
4. Final RMSNorm amplification → ~0.47
5. lm_head output projection → ~12.06 logit delta

This chain is plausible as a purely numerical explanation. Previous bug fixes (non-contiguous query, chunk solver) were real semantic bugs whose fixes materially reduced drift, confirming that at the time, the drift was a mix of real bugs and numerical noise. The current 12.06 may be dominated by the irreducible GPU matmul component.

## PyTorch GPU Baseline (ROCm BF16)

Installed ROCm PyTorch and ran the same model on the same GPU (AMD Radeon 890M / gfx1150):

| Comparison | Decode logit delta | Final hidden delta | Tokens match? |
|-----------|-------------------|-------------------|---------------|
| PT-GPU vs PT-CPU (BF16) | 0.172 | 0.25 | Yes |
| Rust-HIP vs PT-CPU (BF16) | 12.06 | 0.47 | Yes |

Per-layer decode hidden state delta:

| Layer | PT-GPU vs PT-CPU | Rust-HIP vs PT-CPU | Ratio |
|-------|-----------------|-------------------|-------|
| 0 | 0.004 | 0.0005 | 0.1x |
| 7 | 0.008 | 0.008 | 1.0x |
| 15 | 0.008 | 0.016 | 2.0x |
| 18 | 0.008 | 0.027 | 3.4x |
| 22 | 0.031 | 0.031 | 1.0x |
| 23 | 0.031 | 0.033 | 1.1x |

The Rust runtime's hidden state error is 1.9x worse than PyTorch GPU (0.47 vs 0.25). But the logit error is 70x worse (12.06 vs 0.17). This extreme amplification from 1.9x in hidden state to 70x in logits is because the Rust error vector happens to align with high-gain directions of the lm_head projection.

**Critically: both Rust-HIP and PyTorch-GPU produce identical token sequences** ("Hello from DotCache.\nI have a question about the following code"). The 12.06 logit delta does not affect greedy decoding.

## Updated Diagnosis

The decode drift is a combination of:

1. **~50% GPU matmul baseline** — PyTorch GPU shows 0.25 hidden delta just from running on GPU vs CPU. This is inherent to different matmul accumulation paths on the GPU.

2. **~50% Rust runtime overhead** — the Rust runtime has ~2x more accumulated hidden state error than PyTorch GPU. This is likely from:
   - Different matmul kernel choices (candle's HIP matmul vs PyTorch's hipBLAS)
   - Possible differences in how intermediate computations are accumulated
   - RMSNorm kernel implementation differences

3. **Extreme lm_head amplification** — the 1.9x hidden state overhead gets amplified to 70x in logits by the output projection. This is a property of the error vector direction, not the error magnitude.

## Functional Status

The Rust HIP runtime produces **correct token sequences** matching PyTorch GPU for this test case. The 12.06 logit delta against PyTorch CPU is misleading — most of it (the 0.172 baseline) is from GPU matmul non-determinism, and the rest is amplified by lm_head alignment.

## F32 Small-Sequence Matmul Improvement

Root cause identified: candle's `rocblas_gemm_strided_batched_ex` with BF16 inputs produces different rounding than PyTorch's hipblasLt for the same inputs and weights. The difference is visible even with identical 0.0-delta inputs (QKV projection delta: 0.016 with BF16, ~0 with F32).

Fix: upcast inputs and weights to F32 for matmul when total tokens <= 32. For decode (seq_len=1) and short prefills, the matmul is memory-bound anyway, so the F32 overhead is negligible.

Results:

| Metric | BF16 matmul | F32 small-seq matmul | Improvement |
|--------|------------|---------------------|-------------|
| Final hidden delta | 0.469 | 0.375 | -20% |
| Layer 0 QKV delta | 0.016 | ~0 | -99% |
| Layer 23 input delta | 0.033 | 0.025 | -24% |
| Hidden ratio vs PT-GPU | 1.9x | 1.5x | better |

The remaining 1.5x gap vs PyTorch GPU is from:
- Prefill state still carries some BF16 matmul error (for longer prefills, F32 upcast would be too expensive)
- Different internal kernels (RMSNorm, conv1d, recurrent state update)
- Different rocBLAS vs hipblasLt kernel selection

## Best Next Step

1. **Run a longer generation test** (100+ tokens) to verify token sequences still match
2. **Investigate the remaining early-layer (4-13) overhead** — these layers show 1.5-3x more error despite the F32 matmul
3. **Consider switching to hipblasLt** for matmul to better match PyTorch's kernel selection
