# Qwen3.5 Stage 9 Thesis Status (2026-04-12)

This note is the short status checkpoint for the current Qwen3.5 AAE/DotCache line.

## Core claim

The core thesis now looks supported on Apple MPS and on the current CUDA portable-corpus lane:

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

The portable MPS policy lanes are now fully refreshed on the current runtime too:

- real mixed:
  - large `1407.44`
  - broad `1627.72`
  - external `843.77`
- non-`M0` Stage 9:
  - large `3784.86`
  - broad `4234.40`
  - external `1700.51`
- conservative certified:
  - large `2054.42`
  - broad `2828.78`
  - external `1340.05`

So on the portable MPS set, the current real-mixed path is not just the winner in principle. It is comfortably ahead of both refreshed alternative policy lanes across all three corpora.

The refreshed portable MPS numbers are also useful because they confirm that the newer real-mixed runtime optimizations materially improved the public cross-machine reference set, so older portable MPS bundles should be treated as stale comparison anchors rather than the final MPS read.

### Practical MPS machine limit

One useful negative result is now clear on this machine:

- `Qwen/Qwen3.5-4B` can load and run the Stage 9 real-mixed path on MPS
- but it causes heavy paging and is not a practical benchmark lane on this host

First portable-external `4B` feasibility run on this box:

- real mixed `bias`: `22607.39 ms/step`
- real mixed `hand`: `38160.86 ms/step`
- exact-match: `1.0`

That is useful as a rough feasibility check only. It should not be treated as a serious serving benchmark for this machine, and follow-on `4B` diagnostics with extra timing were unstable enough that they are not good research signals either.

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

This has improved materially.

CUDA now supports the same high-level serving-shape claim on the portable corpus:

- conservative certified helps
- real mixed Stage 9 `bias` now beats the same-tree non-`M0` Stage 9 baseline on `large`, `broad`, and `external`

So the remaining CUDA confidence question is no longer “does real mixed ever win?” or even “can `external` be flipped?” It is:

- how much headroom remains after the latest general-path cleanup
- and whether the same updated ordering holds on future corpora and reruns

### 2. Portable-corpus robustness

The repo-local manifests are now good enough for cross-machine comparison, but we still want to know whether the same win shape holds across a broader family of public prompt styles.

The practical question is not only "does it win on one pack?" but:

- does it keep winning on docs, code-heavy prompts, design docs, and harder long-context mixes
- does `bias` stay the default winner

New local read on a broader repo-local public-validation mix:

- manifest:
  - [benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json)
- result note:
  - [docs/qwen35_stage9_public_validation_20260412.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/qwen35_stage9_public_validation_20260412.md)

That run mixed roadmap docs, planning docs, backend notes, and code-heavy benchmark source files.

Current MPS `bias` results on that validation set:

- real mixed: `552.24 ms/step`
- conservative certified: `1434.78`
- non-`M0` Stage 9: `2273.01`

So on this broader public-validation corpus, the winner ordering still holds cleanly:

- real mixed remains the serving winner
- conservative certified remains the middle lane
- non-`M0` Stage 9 remains the slowest of the three

That materially strengthens the current confidence story on MPS, because the win is no longer resting only on the earlier large / broad / external manifest family.

Quick repeatability read on the real-mixed public-validation lane:

- bias ms/step values:
  - `552.24`
  - `591.98`
  - `590.81`
- mean: `578.34`
- population stdev: `18.46`

So even with some run-to-run spread, the public-validation winner ordering remains comfortably intact.

Same-tree CUDA public-validation read on that same corpus is now also checked in:

- real mixed `bias`: `327.70 ms/step`
- conservative certified `333.35 ms/step`
- non-`M0` Stage 9 `339.66 ms/step`
- exact-match still `1.0`
- real mixed remains the same-tree winner

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

Second focused portable-broad study:

- benchmark artifact:
  - [benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad/qwen35_persistent_exact_key_frontier.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad/qwen35_persistent_exact_key_frontier.md)
- baseline:
  - exact-key fallback is again entirely concentrated in layer `15`
  - `8` exact-key blocks per case
  - baseline bias `1864.58 ms/step`
- per-layer threshold sweep on layer `15`:
  - `0.20` keeps the fallback frontier unchanged, preserves exact-match, and improves to `1772.44 ms/step`
  - `0.22` removes exact-key fallback, preserves exact-match, but regresses badly to `2205.63 ms/step`
  - `0.24` removes exact-key fallback, preserves exact-match, and improves materially to `1693.05 ms/step`

So the cross-corpus read is now a little richer:

- the frontier remains narrow and localizable
- layer `15` is the important boundary on both `external` and `broad`
- but the best policy is corpus-sensitive
- removing exact-key fallback can be:
  - harmful on `external`
  - helpful on `broad`

That makes the next useful policy shape more likely to be cost-aware and context-sensitive rather than a single global threshold bump.

Live policy compare result on the real mixed Stage 9 lane:

- benchmark artifact:
  - [benchmarks/results/qwen35_persistent_exact_key_live_policy_compare_20260412/qwen35_persistent_exact_key_live_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_live_policy_compare_20260412/qwen35_persistent_exact_key_live_policy_compare.md)

Across the portable `external`, `broad`, `large`, and the newer public-validation manifest, the current baseline policy still ranked best in the live runtime:

- baseline: `683.99 ms/step`
- layer15_always_024: `694.62`
- layer15_len_ge_1800_024: `723.10`
- layer15_code_or_len_ge_1800_024: `756.63`

All of those alternatives preserved exact-match vs baseline, but none improved runtime.

That is a useful negative result:

- the cheap layer-15 heuristics are not good enough to promote into the runtime
- the offline frontier signal was directionally interesting but not sufficient for live policy choice
- if we revisit this frontier, it should be with a stronger cost model or richer runtime features, not a simple threshold heuristic

Third focused portable-large study:

- benchmark artifact:
  - [benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large/qwen35_persistent_exact_key_frontier.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large/qwen35_persistent_exact_key_frontier.md)
- baseline:
  - exact-key fallback is again entirely concentrated in layer `15`
  - `8` exact-key blocks per case
  - baseline bias `1656.64 ms/step`
- per-layer threshold sweep on layer `15`:
  - `0.20` keeps the fallback frontier unchanged, preserves exact-match, and improves to `1515.22 ms/step`
  - `0.22` removes exact-key fallback, preserves exact-match, but regresses badly to `2305.68 ms/step`
  - `0.24` removes exact-key fallback, preserves exact-match, and is effectively neutral at `1661.59 ms/step`

So the exact-key frontier picture is now pretty consistent in structure and inconsistent in the way that actually matters:

- the frontier is narrow on all three portable corpora
- it is always layer `15`
- it is always about `8` exact-key blocks per case
- but the best treatment is workload-dependent

Current pattern:

- `external`: removing the frontier is slightly worse
- `broad`: removing the frontier at `0.24` is materially better
- `large`: full removal at `0.24` is neutral, while a lighter `0.20` adjustment improves performance without removing fallback

That strengthens the case for a cost-aware mixed policy instead of a single threshold rule. The frontier is simple enough to target, but not simple enough to flatten globally.

Cheap policy study across the checked-in portable frontier artifacts:

- study artifact:
  - [benchmarks/results/qwen35_persistent_exact_key_policy_study_20260412_repo_frontiers/qwen35_persistent_exact_key_policy_study.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_policy_study_20260412_repo_frontiers/qwen35_persistent_exact_key_policy_study.md)
- policies compared:
  - baseline
  - always `layer 15 -> 0.20`
  - always `layer 15 -> 0.24`
  - simple prompt-length splits between `0.20` and `0.24`
- best simple global policy on the saved portable frontier set:
  - always `layer 15 -> 0.20`
  - aggregate average `1489.86 ms/step`
  - versus baseline aggregate average `1593.67 ms/step`

So the current best cheap recommendation is surprisingly simple:

- do not try to remove the layer-15 frontier globally
- instead, keep the frontier and lower layer `15` to `0.20`
- that beats both the current baseline and the prompt-length split heuristics on the checked-in portable studies

That is not the final policy yet, but it is the strongest current small-policy candidate to test in the live runtime.

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

Updated CUDA reproduction on the portable repo-local corpora now confirms more of the systems story too.

Current CUDA portable real-mixed `bias` results:

- large:
  - bias `399.74 ms/step`
  - exact-match `1.0`
- broad:
  - bias `460.70 ms/step`
  - exact-match `1.0`
- external:
  - bias `252.42 ms/step`
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
- and now partially reproduces the MPS winner ordering
- real mixed now wins on `large` and `broad`
- `external` remains the current holdout where non-`M0` Stage 9 still wins

The reported reason is also useful:

- this is not explained by accidental `M3` fallback
- the real mixed CUDA path is executing `M0` only
- the remaining CUDA hotspot is now concentrated in `external`, especially `final_mix`

So the current thesis should now be stated precisely:

- the algorithmic thesis is supported across MPS and CUDA
- the best Stage 9 execution policy is still somewhat backend-dependent today
- but the gap is now much narrower and more localized than before

That is a strong research result, not a failure. It means the method transfers, while the best runtime realization still depends on backend-specific systems work.
