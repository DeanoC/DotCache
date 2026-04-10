# Qwen3.5 Persistent Shortlist Runtime Candidate

- recommended policy: `layer_id + kv_head_id + step_bucket`
- policy groups: 36
- policy path: `/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_shortlist_runtime_candidate_20260410_4096_nonsynthetic_longdecode/persistent_shortlist_policy.json`

## In-Corpus Replay

- hand-tuned avg abs: 0.0395963
- runtime-candidate avg abs: 0.0394641
- hand-tuned max abs: 0.2657154
- runtime-candidate max abs: 0.2657154
- avg selected tokens: 2339.75

## Held-Out Generalization

- runtime-candidate 36-group top-1: 0.625
- runtime-candidate 36-group chosen-safe rate: 0.743
- runtime-candidate 36-group missing bucket rate: 0.000
- prompt-family 108-group missing bucket rate: 1.000
- prompt-family 108-group fallback rate: 1.000

The 36-group table is the preferred runtime candidate because it preserves the in-corpus replay win while still resolving buckets on unseen prompt families. The 108-group prompt-family table does not generalize: on held-out families it misses every bucket and falls back entirely.
