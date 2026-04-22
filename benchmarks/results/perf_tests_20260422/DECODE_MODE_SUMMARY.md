# Decode-mode experiment — per-token cache trace + pg19 teacher-forced

**Purpose.** Test whether the certified path's cache hit rate depends on the
decode mode (argmax vs teacher-forced) or on the prompt content (pg19 vs
filler). The motivation was an apparent 99.9% zero-pagein result on PG-19 in
the main Test 3 table at `cap=64`, which did not reproduce on niah/ruler.

**Spoiler.** The 99.9% number was a telemetry bug (see the [correction
section](#correction) below). Once fixed, pg19 behaves identically to niah
and ruler under both decode modes: flat ~2% cache hit rate at cap=64, ~470
MB/token H2D traffic. **Decode mode and prompt content don't measurably
shape the cache curve on this model at 8K context.** The real variable
is cache capacity vs corpus size.

## What we measured

Two per-token traces, same script, same cache capacity (cap=64), same
cert config (tau_cov=0.995, ranking_fallback on, etc.), 256 decode
tokens from an 8K pg19 prefill:

- `per_token_trace_pg19_cap64.json` — argmax decode
- `per_token_trace_pg19_cap64_teacher_forced.json` — pg19 ground-truth
  tokens fed in each step instead of argmax

Both traces snapshot cumulative `_fp16_key_cache_hits/misses/h2d_bytes`
from all 32 layer caches directly between decode steps, bypassing
`step_stats` (which is what the broken pg19 Test 3 telemetry relied on).

## Result — two identical curves

Both traces show:

| Step | Argmax hit rate | Teacher-forced hit rate |
|---|---|---|
| 0 | 0.3% | 0.3% |
| 16 | 2.0% | 2.0% |
| 32 | 1.8% | 1.8% |
| 64 | 2.0% | 2.1% |
| 128 | 2.3% | 2.6% |
| 192 | 2.6% | 2.5% |
| 240 | 2.0% | 2.5% |

H2D MB/step stays ~520 MB throughout on both. There's no "concentrated
→ scattered" drift, no "teacher-forced concentrated vs argmax scattered"
split. Both decode modes produce essentially identical cache behaviour.

## What this implies

- **Llama-3.1-8B's certified attention at 8K context does not have locality
  at the 16-token block granularity for this model.** The per-step top-K
  union across 32 Q-heads is ~180 blocks, and those 180 blocks shift
  meaningfully step-to-step — there is no small hot working set.
- **The cache is a pure bandwidth-hiding tool**, not a locality-exploiting
  structure on this model. It helps only when capacity ≥ corpus (see the
  capacity sweep at `cache_sweep/SUMMARY.md` — knee at exactly 512 blocks).
- **Decode mode (argmax vs teacher-forced) is not a variable** worth
  instrumenting in the paper; any narrative about "concentrated attention
  regimes" should be dropped for this model/context.

## Paper implications

§9.8 of the preprint previously claimed 99.9% zero-pagein on PG-19 as
evidence that the tiered architecture is near-free on in-distribution
workloads. That claim must be withdrawn. The accurate statement is:

> The tiered architecture's cost is dominated by H2D page-in bandwidth,
> which is near-uniform across benchmarks at any cache capacity below the
> corpus size. Above corpus size, H2D collapses to near-zero; below, it
> is ~470 MB/token regardless of workload.

The capacity sweep at `cache_sweep/SUMMARY.md` is the correct evidence
for the bandwidth story: 5.71% hit rate at cap=384, 99.62% at cap=512
(exactly the corpus size). That's the knee.

## Correction — why the 99.9% number was wrong

`PageinTelemetry.record_step` tracks a cursor into
`CertifiedAttentionState.step_stats` and slices from cursor to end each
call to capture "stats appended since last record". This works when
`step_stats` grows monotonically across decode steps (the niah / ruler
harness pattern). But `pg19_perplexity.py` has its own pre-existing
drain pattern — `aggregate_step_stats()` then `clear_step_stats()` —
that ran *after* my record_step each iteration. On the next iteration
the cursor was past the end of the freshly-cleared (then refilled)
list; my slice returned empty; the per-step telemetry recorded zero
H2D for 1637 of 1638 steps. The `99.9% zero-pagein` number was
`1637 / 1638` — an artefact of every step after the first silently
logging empty data.

Fixed in `d9e87084`: added `_clear_seq` counter on
`CertifiedAttentionState`, incremented in `clear_step_stats()`;
`PageinTelemetry` watches for advances and resets its cursor to 0
before slicing. The re-run of Test 3 pg19 now reports `pct_zero_pagein
= 0%`, matching niah/ruler.

The per-token traces (`bench_per_token_cache_trace.py`) were written
after this bug was introduced but they snapshot cache counters
**directly on each layer cache**, never going through `step_stats`.
So they were unaffected, and their flat-2%-hit curves are the correct
picture that should have been visible all along.
