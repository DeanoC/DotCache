# Qwen3.5 Stage 9 Broader Public Validation (2026-04-14)

This note describes the intent and design of the broader public validation corpus for the Stage 9
cross-backend story.

## Motivation

Rounds 1 and 2 of public validation (2026-04-12 and 2026-04-13) established the winner ordering on
MPS and CUDA across two independent 8-case corpora. However, both of those corpora drew heavily from
Stage 9 documentation and evaluation artifacts. For the cross-backend story to feel
publication-grade rather than "strong internal evidence", the validation should extend to a broader
family of content types with no overlap to the earlier rounds.

The goal of this corpus is:

- use only files not present in rounds 1 or 2
- cover a wider spread of content types: results docs, profiling/numeric data, architecture docs,
  RFC-style design docs, stage progress summaries, technical comparison plans, core benchmark
  infrastructure code, and test code
- keep prompt lengths at 1024 or 1536 throughout (Mac Mini MPS host constraint)

## Manifest

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_broader_20260414.json](../benchmarks/manifests/qwen35_stage9_repo_public_validation_broader_20260414.json)

## Cases

| Case tag | File | Prompt length | Content type |
|---|---|---|---|
| `benchmark_report` | `docs/benchmark_report.md` | 1536 | Results/analysis doc |
| `local_layer_profiles` | `docs/local_layer_profiles.md` | 1024 | Profiling/numeric data |
| `hip_call_flow` | `docs/qwen35_hip_prompt_to_output_call_flow.md` | 1024 | Architecture doc |
| `compressed_page_rfc` | `docs/dotcache_compressed_page_test_readiness_rfc.md` | 1024 | RFC/design doc |
| `aae_stage_summary` | `docs/aae_dotcache_stage_progress_summary_20260413.md` | 1536 | Stage progress summary |
| `turboquant_comparison_plan` | `docs/turboquant_comparison_plan.md` | 1024 | Technical comparison plan |
| `bench_decode_code` | `benchmarks/bench_decode.py` | 1536 | Core infra benchmark code |
| `test_attention_vs_dense` | `tests/test_attention_vs_dense.py` | 1024 | Test code |

## What this adds

Compared to rounds 1 and 2, this corpus introduces:

- profiling/numeric data files not previously used
- architecture and call-flow documentation
- RFC-style design documents
- the newest stage progress summary prose (written after the round-2 divergence investigation)
- core decode benchmark infrastructure (not just specialized probe code)
- an attention-vs-dense test file (different from the real-mixed-probe test used in round 2)

## Results

### MPS

Result bundles:

- real mixed:
  [benchmarks/results/qwen35_persistent_real_mixed_probe_20260414_repo_promptfiles_public_validation_broader_mps/qwen35_persistent_real_mixed_probe.md](../benchmarks/results/qwen35_persistent_real_mixed_probe_20260414_repo_promptfiles_public_validation_broader_mps/qwen35_persistent_real_mixed_probe.md)
  - bias `404.13 ms/step`
  - hand `474.61`
  - exact-match vs hand `1.0`
  - bias beats hand on all 8 cases
- non-`M0` Stage 9:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260414_repo_promptfiles_public_validation_broader_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260414_repo_promptfiles_public_validation_broader_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `1352.13 ms/step`
  - bias vs dense exact-match `1.0`
  - bias vs hand exact-match `1.0`
- conservative certified:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260414_repo_promptfiles_public_validation_broader_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260414_repo_promptfiles_public_validation_broader_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `913.96 ms/step`
  - bias vs dense exact-match `1.0`
  - bias vs hand exact-match `1.0`

### MPS winner ordering

- real mixed `404.13`
- conservative certified `913.96`
- non-`M0` Stage 9 `1352.13`

Real mixed is:
- about `55.8%` faster than conservative certified
- about `70.1%` faster than non-`M0` Stage 9

Notably, all three lanes match dense exactly on this corpus (exact-match `1.0` across the board),
which is a stronger result than rounds 1 and 2 where non-`M0` and conservative only matched dense
at `0.75`.

### CUDA

- real mixed: TBD
- conservative certified: TBD
- non-`M0` Stage 9: TBD

## Interpretation

The winner ordering holds cleanly on this third independent corpus. The real mixed path remains the
serving winner by a large margin, and all three lanes preserve exact-match vs dense on the new
content types.

The `1.0` dense exact-match across all lanes on this corpus (versus `0.75` in round 2) is a useful
positive signal: the broader content mix did not introduce the kind of boundary cases that round 2
exposed. That does not mean such cases cannot exist, but it does mean the system degrades safely and
consistently across a wide range of public repo-local content.
