# Qwen3.5 Stage 9 `state_cache_roadmap` First-Divergence Localization (2026-04-13)

This note isolates the strongest current public repro for the broader DotCache-family vs dense boundary.

Single-case repro manifest:

- [benchmarks/manifests/qwen35_stage9_state_cache_roadmap_first_divergence_20260413.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_state_cache_roadmap_first_divergence_20260413.json)

Source artifact used for the current read:

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.json)

## Why this case

`state_cache_roadmap` is the best current structural repro because:

- the serving-family divergence starts extremely early
- it shows a non-trivial mixed frontier
- it reproduces the broader DotCache-family vs dense boundary cleanly

## Current first-divergence read

Prompt:

- [state_cache_roadmap.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/state_cache_roadmap.md)
- prompt length: `1345`

Generated IDs:

- dense:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- hand / real mixed:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`

Earliest divergence:

- shared prefix length vs dense: `1`
- first divergent token index: `2`

This means the repro becomes informative almost immediately after the first generated token.

## Execution shape on the targeted MPS real-mixed run

- processed blocks: `85`
- checkpoints: `6`
- executed `M0` blocks: `4032`
- executed all-`M3` blocks: `48`
- executed exact-key `M3` blocks: `56`

That makes this a better debugging repro than `performance_journal`, which diverges later and stays closer to pure `M0`.

## Current interpretation

The important read is not just that this case diverges from dense. It is that:

- real mixed matches hand exactly
- non-`M0` and conservative also line up with the same serving-family output
- so the first-divergence question is now:
  - what internal quantity separates the broader DotCache-family path from dense by token `2` on this prompt?

## Recommended next debugging step

Use this manifest as the single canonical repro for step-local debugging:

1. compare dense vs serving-family at generated token `2`
2. inspect the earliest internal quantity that differs
3. keep the comparison lane-by-lane, but do not widen back to multi-case benchmarking until the first divergence is explained
