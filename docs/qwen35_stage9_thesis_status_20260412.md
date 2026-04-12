# Qwen3.5 Stage 9 Thesis Status (2026-04-12)

This note is the short status checkpoint for the current Qwen3.5 AAE/DotCache line.

## Core claim

The core thesis now looks supported on Apple MPS:

- certified streaming in the real serving loop works
- safe fallback behavior remains intact on the tested prompts
- key-side mixed `M0` execution can be reattached to certified streaming
- the resulting real mixed Stage 9 `bias` path is faster than both:
  - the conservative certified baseline
  - the earlier non-`M0` Stage 9 path

This is not yet a claim of bit-identical logits to dense attention. The supported claim remains:

- stable greedy decode behavior with exact-match outputs on the tested corpora
- materially lower decode latency than the conservative exact-serving lane

## Best current evidence

### Private uncapped 10-prompt longdecode

Checked-in reference points:

- conservative certified:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_20260411_large_promptfiles_uncapped_longdecode_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260411_large_promptfiles_uncapped_longdecode_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `2457.43 ms/step`
- non-`M0` Stage 9 packed certified:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_20260411_large_promptfiles_uncapped_longdecode_stage9_metal_packed_streaming_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260411_large_promptfiles_uncapped_longdecode_stage9_metal_packed_streaming_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `1232.17 ms/step`
- real mixed Stage 9:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_large_promptfiles_uncapped_longdecode/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_large_promptfiles_uncapped_longdecode/qwen35_persistent_real_mixed_probe.md)
  - bias `1184.73 ms/step`

So the current real mixed Stage 9 path is:

- about `3.85%` faster than the non-`M0` Stage 9 path
- about `51.8%` faster than the conservative certified baseline

### Portable repo-local corpora

These are the current public, cross-machine reference points for CUDA comparison:

- large:
  - refreshed:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
    - bias `1407.44 ms/step`
  - older checked-in reference:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps/qwen35_persistent_real_mixed_probe.md)
    - bias `1432.34 ms/step`
- broad:
  - refreshed:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
    - bias `1627.72 ms/step`
  - older checked-in reference:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps/qwen35_persistent_real_mixed_probe.md)
    - bias `1679.00 ms/step`
- external:
  - refreshed:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
    - bias `843.77 ms/step`
  - older checked-in reference:
    - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps/qwen35_persistent_real_mixed_probe.md)
    - bias `918.66 ms/step`

On these portable corpora:

- `bias` beats `hand` on every checked-in case
- bias/hand exact-match stays `1.0`
- the current runtime is genuinely executing heavy key-side `M0`, not only falling back to all-`M3`

The refreshed portable MPS numbers are also useful because they confirm that the newer real-mixed runtime optimizations materially improved the public cross-machine reference set, so older portable MPS bundles should be treated as stale comparison anchors rather than the final MPS read.

## Spec status

For the core AAE spec, the repo now appears close enough to call the core thesis demonstrated on MPS.

### Effectively in place

- Sections `2-3`
  - block metadata, ordering signals, and fallback structure are real
- Sections `7-8`
  - conservative execution and mixed key-side execution are both real in the serving loop

### Materially implemented now

- Sections `4-6`
  - certified streaming is now part of the actual serving path
  - safe stop happens during execution, not only as a post-hoc selection check
  - conservative certified streaming is benchmarked and useful
  - certified streaming is reattached to Stage 9 mixed execution

### Still out of scope or unfinished

- value-side `M0`
- learned ordering/scoring
- stronger certification on the hardest residual tails

These still matter, but they are no longer required to support the current thesis.

## What still needs investigation for confidence

### 1. Cross-device reproduction

This is the biggest remaining confidence gap.

We want CUDA to tell us whether the same shape holds:

- conservative certified helps
- non-`M0` Stage 9 helps more
- real mixed Stage 9 `bias` is the serving winner

If CUDA reproduces that ordering, the thesis becomes much stronger.

### 2. Portable-corpus robustness

The repo-local manifests are now good enough for cross-machine comparison, but we still want to know whether the same win shape holds across a broader family of public prompt styles.

The practical question is not only "does it win on one pack?" but:

- does it keep winning on docs, code-heavy prompts, design docs, and harder long-context mixes
- does `bias` stay the default winner

### 3. Repeatability

The current numbers are strong, but the thesis should rest on stable runs rather than one especially lucky point.

What we still want:

- repeat runs on the portable corpora
- simple variance summaries for large, broad, and external

Early local read on the current code path, all on portable MPS with `3` repeats each:

- external:
  - bias ms/step values:
    - `809.79`
    - `836.33`
    - `836.05`
  - mean: `827.39`
  - population stdev: `12.45`
- broad:
  - bias ms/step values:
    - `1663.30`
    - `1653.62`
    - `1676.22`
  - mean: `1664.38`
  - population stdev: `9.26`
- large:
  - bias ms/step values:
    - `1383.38`
    - `1377.95`
    - `1383.41`
  - mean: `1381.58`
  - population stdev: `2.57`

That is encouragingly tight for an early repeatability pass, but the checked-in portable MPS bundles still predate the latest runtime optimizations. They should be refreshed once we are ready to promote the portable corpus to the new baseline.

### 4. Exact-key fallback frontier

Recent telemetry clarified something important:

- some remaining exact-key fallback blocks are not just a correctness tax
- in at least one portable external case, forcing those blocks into `M0` made performance worse

So the next confidence question is:

- when is exact-key fallback the right fast path
- when is `M0` the right fast path

That is now more of a runtime-policy question than a kernel question.

First focused portable-external study:

- benchmark artifact:
  - [benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external/qwen35_persistent_exact_key_frontier.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external/qwen35_persistent_exact_key_frontier.md)
- baseline:
  - exact-key fallback is entirely concentrated in layer `15`
  - `8` exact-key blocks per case
  - baseline bias `841.94 ms/step`
- per-layer threshold sweep on layer `15`:
  - `0.22` removes exact-key fallback, preserves exact-match, but regresses badly to `1092.89 ms/step`
  - `0.24` also removes exact-key fallback, preserves exact-match, and is near-neutral at `847.62 ms/step`

So the current read is:

- the exact-key fallback frontier is narrow and localizable
- it is not a simple "remove fallback and go faster" story
- some boundary fallback looks genuinely performance-positive
- the next useful work there is likely a cost-aware mixed policy, not a globally looser gate

### 5. Hard-case explanation

The system should still have a clean story for the remaining hard prompts and layers:

- which cases still stop late
- whether the blocker is ordering, certification slack, or mixed-mode gating
- whether the system degrades safely when it cannot stop early

## Current practical conclusion

The current state is good enough to say:

- the core Qwen3.5 Stage 9 thesis is supported on MPS
- conservative certified streaming is real
- real mixed key-side `M0` execution is real
- the best real mixed `bias` path is now the measured serving winner on the main MPS benchmarks

The remaining work is mainly about confidence, portability, and understanding the residual hard frontier, not about proving basic viability anymore.

## CUDA read so far

Initial CUDA reproduction on the portable repo-local corpora confirms the core algorithmic story, but not the same winner ordering as MPS.

Current CUDA portable real-mixed results reported against the same portable manifests:

- large:
  - hand `617.78 ms/step`
  - bias `611.34 ms/step`
  - exact-match `1.0`
- broad:
  - hand `752.07 ms/step`
  - bias `750.07 ms/step`
  - exact-match `1.0`
- external:
  - hand `378.04 ms/step`
  - bias `376.70 ms/step`
  - exact-match `1.0`

Portable CUDA comparison baselines reported on the same corpora:

- non-`M0` Stage 9 bias:
  - large `468.69`
  - broad `632.12`
  - external `192.54`
- conservative certified bias:
  - large `652.91`
  - broad `792.60`
  - external `380.85`

The important conclusion is:

- CUDA reproduces correctness and viability
- but CUDA does not currently reproduce the MPS winner ordering
- on the current portable corpus, non-`M0` Stage 9 is the serving winner on CUDA

The reported reason is also useful:

- this is not explained by accidental `M3` fallback
- the real mixed CUDA path is executing `M0` only
- the current loss is the remaining `direct_m0` gather/score/final-mix cost on CUDA

So the current thesis should now be stated precisely:

- the algorithmic thesis is supported across MPS and CUDA
- the best Stage 9 execution policy is still backend-dependent today

That is a strong research result, not a failure. It means the method transfers, while the best runtime realization still depends on backend-specific systems work.
