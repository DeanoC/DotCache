# Qwen3.5 CUDA Grouped Follow-Up Plan

This note defines the next narrow CUDA pass now that grouped shortlist decode is operational after the mixed-signature bucketing patch.

## Goal

Answer two concrete questions:

1. Does forced grouped shortlist decode preserve quality acceptably at `32768` and `49152`?
2. Is the near-parity serving result reproducible across a few clean single-shot reruns?

## Runs To Execute

### A. Forced-Grouped Quality Tail

Run:

```bash
scripts/run_qwen35_cuda_shortlist_large_context_forced_grouped_quality_tail.sh
```

Default output:

```text
benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl
```

This runs:

- `exact`
- `shortlist_base`
- `shortlist_l23_ctx`

at:

- `32768`
- `49152`

with:

- `DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING=1`
- `quality_mode=loss_tail`
- `quality_eval_steps=4`

Primary readout:

- `teacher_forced_loss_delta`
- `teacher_forced_logit_max_abs_error`
- `decode_path_counts`
- any timeout / missing-row failures

Decision rule:

- if forced-grouped shortlist quality is materially better than the default shortlist quality tail, grouped CUDA becomes a stronger candidate default
- if it is similar or worse, grouped CUDA is still mostly a serving-path cleanup, not yet a quality rescue

### B. Serving Reproducibility

Run:

```bash
scripts/run_qwen35_cuda_shortlist_large_context_repro_serving.sh
```

Default output directory:

```text
benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/
```

The wrapper performs `3` single-shot repeats for each of:

- `default`
- `forced_grouped`

and each repeat covers:

- `shortlist_base`
- `shortlist_l23_ctx`

at:

- `32768`
- `49152`

Override repeat count if needed:

```bash
REPEATS=5 scripts/run_qwen35_cuda_shortlist_large_context_repro_serving.sh
```

Expected files:

- `default_repeat1.jsonl`
- `default_repeat2.jsonl`
- `default_repeat3.jsonl`
- `forced_grouped_repeat1.jsonl`
- `forced_grouped_repeat2.jsonl`
- `forced_grouped_repeat3.jsonl`

Primary readout:

- `dotcache_decode_ms_per_step`
- `decode_path_counts`
- selected page counts
- row-to-row variance
- any `NoExactRow` / timeout anomalies

Decision rule:

- if forced-grouped shortlist remains near default across repeats, grouped CUDA is stable enough to discuss as an operational alternate path
- if the gap swings widely, treat the current near-parity result as encouraging but not yet publication-grade

## What To Write Into The Journal

- one compact table for forced-grouped quality tail vs the existing default quality tail
- one compact reproducibility table with mean, min, and max decode ms/step for each context/case/mode
- any wrapper-level misses called out separately from backend failures

## What This Unlocks Next

- if grouped quality is competitive and serving is reproducible, the next paper update can describe grouped CUDA as a real alternate execution path rather than a backend debugging lane
- if not, the next engineering target should be timing splits inside selector / score / mix so we know exactly where grouped CUDA still loses
