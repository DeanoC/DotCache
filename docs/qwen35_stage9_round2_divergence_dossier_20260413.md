# Qwen3.5 Stage 9 Round 2 Divergence Dossier (2026-04-13)

This note is the compact debugging dossier for the round-2 public-validation dense divergence.

Primary source notes:

- [qwen35_stage9_public_validation_round2_20260413.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/qwen35_stage9_public_validation_round2_20260413.md)
- [qwen35_stage9_public_validation_round2_dense_sanity_20260413.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/qwen35_stage9_public_validation_round2_dense_sanity_20260413.md)

## Cases

The two round-2 cases that diverged from dense are:

- `performance_journal`
- `state_cache_roadmap`

Update (fixed-tree post-handoff interpretation):

- `state_cache_roadmap` now matches dense in the fixed-tree suites
- `performance_journal` is narrowed to a late tie-boundary behavior on CUDA runs
- `submission_execution_plan` is no longer part of the original two-case suspect pair, and MPS post-fix reruns now show it matching dense

## What diverged (historical context)

On both cases:

- `performance_journal` and `state_cache_roadmap` diverged from dense in early round-2 checked runs
- real mixed `bias`, hand, non-`M0` Stage 9, and conservative certified all matched each other in those serving-family runs
- this divergence is now resolved for `state_cache_roadmap` after the handoff fix
- `submission_execution_plan` is resolved in MPS post-fix reruns; CUDA status is outside the scope of this dossier
- residual `performance_journal` behavior is now treated as a late tie-boundary residual rather than a mixed-path correctness failure

That means the current problem framing is:

- shared DotCache-family vs dense boundary

not:

- Stage 9 mixed-only regression

## Prefix behavior

### `performance_journal`

- serving-family prefix match vs dense: `4`
- first divergent region starts after the shared prefix
- real mixed and hand both generate:
  - `[198, 220, 471, 1510, 8412, 1551, 7408, 63]`
- dense generates:
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`

### `state_cache_roadmap`

- serving-family prefix match vs dense: `1`
- divergence begins much earlier than `performance_journal`
- real mixed and hand both generate:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- dense generates:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`

## Execution-shape clues from MPS

### `performance_journal`

- targeted real mixed still stays almost entirely mixed:
  - executed `M0` blocks: `6192`
  - executed all-`M3` blocks: `0`
  - exact-key `M3` blocks: `8`

### `state_cache_roadmap`

- targeted real mixed shows a more interesting frontier:
  - executed `M0` blocks: `4032`
  - executed all-`M3` blocks: `48`
  - exact-key `M3` blocks: `56`

This makes `state_cache_roadmap` the stronger structural debugging case.

## Working hypotheses

- the divergence is tied to a broader DotCache-family approximation boundary rather than Stage 9 mixed specifically
- the boundary may correlate with highly structured markdown/planning text rather than only raw prompt length
- `state_cache_roadmap` may be the more informative reproducer because it introduces a visible exact-key / all-`M3` frontier

## Recommended next checks

- reproduce the same two cases on CUDA and compare lane-by-lane against dense
- check whether the same pair still diverges in the same way
- if yes, localize the first internal quantity that differs from dense rather than tuning policy first
